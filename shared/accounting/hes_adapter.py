"""Adapt existing HES fiscal outputs to canonical household P&L component facts.

The adapter is path-parameterized and never writes microdata.  It supports the
pooled 2016-2018 and 2021-2023 production artifacts.
"""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd

from contracts.keys import household_key
from contracts.validate import validate_household_record
from .pnl import KEY_COLUMNS, load_component_catalog, validate_component_facts, validate_national_controls

SOURCE_KEYS = ["year", "s_seker", "misparmb"]


def _amount(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)


def _fact_frame(
    households: pd.DataFrame,
    catalog: pd.DataFrame,
    component_id: str,
    amount: pd.Series | np.ndarray,
    representation: str,
    source_dataset_id: str,
    method_version: str,
    national_control_id: str = "",
    embedded_tax_basis: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    meta = catalog.set_index("component_id").loc[component_id]
    result = pd.DataFrame({
        "household_key": households.household_key,
        "reference_year": households.reference_year,
        "component_id": component_id,
        "representation": representation,
        "method_version": method_version,
        "amount_nis": np.asarray(amount, dtype=float),
        "accounting_side": meta.accounting_side,
        "value_basis": meta.value_basis,
        "economic_function": meta.economic_function,
        "payer_funder": meta.payer_funder,
        "delivery_value_type": meta.delivery_value_type,
        "frequency_basis": "monthly_flow",
        "price_basis": "nominal",
        "price_year": households.reference_year,
        "evidence_status": "observed" if representation == "reported" else "macro_calibrated",
        "source_dataset_id": source_dataset_id,
        "national_control_id": national_control_id,
        "allocation_factor_id": "",
        "quality_flags": "{}",
    })
    result["embedded_tax_basis_nis_monthly"] = (
        np.nan if embedded_tax_basis is None else np.asarray(embedded_tax_basis, dtype=float)
    )
    return result


LOCKED_TOTALS = {
    2018: {"tax": 405_471_420_000.0, "transfer": 267_750_000_000.0},
    2023: {"tax": 546_006_600_000.0, "transfer": 394_732_394_501.5164},
}


def build_hes_fiscal_component_facts(
    tax_pickle: str | Path,
    transfer_pickle: str | Path,
    target_year: int,
    *,
    embedded_basis_pickle: str | Path | None = None,
    control_source_path: str | Path,
    catalog_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return canonical fiscal facts and verify the independent locked totals.

    Market income, private consumption and financing flows are intentionally not
    reconstructed here; they come from the household-balance-sheet P&L adapter.
    """
    tax = pd.read_pickle(tax_pickle)
    transfer = pd.read_pickle(transfer_pickle)
    for name, frame in [("tax", tax), ("transfer", transfer)]:
        missing = sorted(set(SOURCE_KEYS + ["weight_adj", "sector"]) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} artifact is missing columns: {missing}")
        if frame.duplicated(SOURCE_KEYS).any():
            raise ValueError(f"{name} artifact has duplicate household keys")
    joined = tax.merge(
        transfer[SOURCE_KEYS + [c for c in transfer.columns if c.startswith("tr_")]],
        on=SOURCE_KEYS, how="inner", validate="one_to_one", indicator=True,
    )
    if len(joined) != len(tax) or len(joined) != len(transfer) or joined._merge.ne("both").any():
        raise ValueError("tax and transfer artifacts do not contain the same households")
    joined = joined.drop(columns="_merge").reset_index(drop=True)
    basis_path = Path(embedded_basis_pickle) if embedded_basis_pickle else Path(tax_pickle).with_name(f"_embedded_tax_basis_{target_year}.pkl")
    basis = pd.read_pickle(basis_path)
    if basis.duplicated(SOURCE_KEYS).any():
        raise ValueError("embedded-tax basis export has duplicate household keys")
    basis_columns = [c for c in basis.columns if c.startswith("tax_embedded_basis_")]
    joined = joined.merge(basis[SOURCE_KEYS + basis_columns], on=SOURCE_KEYS, how="left", validate="one_to_one")

    households = pd.DataFrame({
        "spine_id": "hes",
        "reference_year": int(target_year),
        "source_wave": joined.year.astype(int),
        "source_year": joined.year.astype(int),
        "source_survey_id": joined.s_seker.astype(int),
        "source_household_id": joined.misparmb.astype(int),
        "survey_weight": _amount(joined, "weight_adj"),
        "household_size": (
            pd.to_numeric(joined.n_adults, errors="coerce").fillna(0)
            + pd.to_numeric(joined.nefashotad18, errors="coerce").fillna(0)
        ).clip(lower=1).round().astype(int),
        "sector_group": joined.sector.astype(str),
    })
    source_ids = joined.year.astype(str) + ":" + joined.s_seker.astype(str) + ":" + joined.misparmb.astype(str)
    households["household_key"] = [household_key("hes", target_year, value) for value in source_ids]
    households = households[[
        "spine_id", "reference_year", "household_key", "source_wave", "survey_weight",
        "household_size", "sector_group", "source_year", "source_survey_id",
        "source_household_id",
    ]]
    for record in households.to_dict("records"):
        validate_household_record(record)

    catalog = load_component_catalog(catalog_path)
    source_id = f"hes.pooled.{target_year}"
    fiscal_source_id = f"fiscal.allocation.{target_year}"
    method = f"hes-pnl-{target_year}-v1"
    frames: list[pd.DataFrame] = []

    reported = {
        "expense.government.tax.income_capital_gains": _amount(joined, "t211"),
        "expense.government.tax.social_contribution": _amount(joined, "t212") + _amount(joined, "t213"),
        "income.government.cash_benefit.nii": _amount(joined, "i141"),
        "income.government.cash_benefit.other": _amount(joined, "i142"),
    }
    for component_id, amount in reported.items():
        frames.append(_fact_frame(households, catalog, component_id, amount, "reported", source_id, method))

    legacy_rows = catalog.loc[catalog.get("legacy_code", pd.Series(index=catalog.index, dtype=object)).notna()]
    legacy_to_component = dict(zip(legacy_rows.legacy_code, legacy_rows.component_id))
    parity_rows = []
    control_rows = []
    for legacy_code, component_id in legacy_to_component.items():
        if legacy_code.startswith("2."):
            source_column = f"tax_{legacy_code}"
        elif legacy_code == "wel":
            source_column = "tr_wel"
        else:
            source_column = f"tr_{legacy_code}"
        if source_column not in joined.columns:
            raise ValueError(f"production artifact lacks {source_column} for {component_id}")
        amount = _amount(joined, source_column)
        control_kind = "tax" if legacy_code.startswith("2.") else "transfer"
        control_id = f"hes.{target_year}.{control_kind}.total"
        meta = catalog.set_index("component_id").loc[component_id]
        is_embedded = bool(meta.get("embedded_in_purchaser_price", False))
        basis_column = f"tax_embedded_basis_{legacy_code}"
        embedded_basis = _amount(joined, basis_column) if is_embedded and basis_column in joined else None
        frames.append(_fact_frame(households, catalog, component_id, amount, "allocated", fiscal_source_id, method, control_id, embedded_basis))
        if component_id.startswith("income.government.in_kind."):
            use_id = component_id.replace("income.government.in_kind.", "expense.public.in_kind.")
            frames.append(_fact_frame(households, catalog, use_id, amount, "allocated", fiscal_source_id, method, control_id))
        expected = float((amount * households.survey_weight).sum() * 12)
        parity_rows.append({"reference_year": target_year, "component_id": component_id, "source_column": source_column, "source_annual_nis": expected})

    facts = pd.concat(frames, ignore_index=True)
    validate_component_facts(facts, catalog)
    try:
        locked = LOCKED_TOTALS[int(target_year)]
    except KeyError as exc:
        raise ValueError(f"no independent locked fiscal totals for {target_year}") from exc
    controls = pd.DataFrame([
        {
            "national_control_id": f"hes.{target_year}.tax.total", "control_year": int(target_year),
            "component_id": "expense.government.tax", "amount_nis_annual": locked["tax"],
            "population_scope": "HES pooled household population", "source_dataset_id": "legacy.fiscal.control.lock",
            "source_reference": f"locked production tax total {target_year}", "method_version": "legacy-lock-v1",
        },
        {
            "national_control_id": f"hes.{target_year}.transfer.total", "control_year": int(target_year),
            "component_id": "income", "amount_nis_annual": locked["transfer"],
            "population_scope": "HES pooled household population", "source_dataset_id": "legacy.fiscal.control.lock",
            "source_reference": f"locked production transfer total {target_year}", "method_version": "legacy-lock-v1",
        },
    ])
    source_path = Path(control_source_path)
    source_checksum = sha256(source_path.read_bytes()).hexdigest()
    controls["source_checksum"] = source_checksum
    controls["source_reference"] = source_path.name
    validate_national_controls(facts, households, controls)

    weights = households[KEY_COLUMNS + ["survey_weight"]]
    allocated_resource = facts.loc[
        facts.representation.eq("allocated") & ~facts.component_id.str.startswith("expense.public.in_kind.")
    ].merge(weights, on=KEY_COLUMNS, validate="many_to_one")
    actual = allocated_resource.assign(actual_annual_nis=lambda x: x.amount_nis * x.survey_weight * 12).groupby("component_id", as_index=False).actual_annual_nis.sum()
    parity = pd.DataFrame(parity_rows).merge(actual, on="component_id", validate="one_to_one")
    parity["gap_nis"] = parity.actual_annual_nis - parity.source_annual_nis
    if not np.allclose(parity.actual_annual_nis, parity.source_annual_nis, atol=1.0, rtol=0):
        raise ValueError("canonical allocated facts do not reproduce source fiscal columns")
    return households, facts, controls, parity
