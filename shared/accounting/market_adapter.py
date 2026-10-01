"""Adapt the additive balance-sheet HES P&L export to canonical market facts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from contracts.keys import household_key
from contracts.validate import validate_household_record, validate_market_fact
from .pnl import load_component_catalog, validate_component_facts


REQUIRED_COLUMNS = {
    "year", "s_seker", "misparmb", "weight", "household_size", "demographic_group",
    "labor_income", "pension_income", "rental_property_income",
    "reported_interest_dividend_income", "other_income", "imputed_owner_rent",
    "transfers_income", "government_cash_benefits_reported", "private_transfer_income",
    "cash_consumption", "other_cash_consumption", "private_health_expense",
    "private_education_expense", "rent_paid", "imputed_housing_consumption",
    "private_transfers_paid", "mortgage_interest", "mortgage_principal",
    "pension_contributions", "training_fund_contributions",
    "provident_fund_contributions", "life_exec_insurance_contributions",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"market export contains invalid values in {column}")
    return values.astype(float)


def _fact_frame(
    households: pd.DataFrame,
    catalog: pd.DataFrame,
    component_id: str,
    amount: pd.Series | np.ndarray,
    target_year: int,
) -> pd.DataFrame:
    meta = catalog.set_index("component_id").loc[component_id]
    return pd.DataFrame({
        "household_key": households.household_key,
        "reference_year": int(target_year),
        "component_id": component_id,
        "representation": "reported",
        "method_version": f"hes-market-pnl-{target_year}-v1",
        "amount_nis": np.asarray(amount, dtype=float),
        "accounting_side": meta.accounting_side,
        "value_basis": meta.value_basis,
        "economic_function": meta.economic_function,
        "payer_funder": meta.payer_funder,
        "delivery_value_type": meta.delivery_value_type,
        "frequency_basis": "monthly_flow",
        "price_basis": "nominal",
        "price_year": int(target_year),
        "evidence_status": "derived",
        "source_dataset_id": f"hes.balance_sheet.market_pnl.{target_year}",
        "national_control_id": "",
        "allocation_factor_id": "",
        "quality_flags": "{}",
        "embedded_tax_basis_nis_monthly": np.nan,
    })


def build_hes_market_component_facts(
    market_export_pickle: str | Path,
    target_year: int,
    *,
    catalog_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return canonical HES household records, market facts and exact parity checks."""
    source = pd.read_pickle(market_export_pickle).reset_index(drop=True)
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"market P&L export is missing columns: {missing}")
    source_keys = ["year", "s_seker", "misparmb"]
    if source.duplicated(source_keys).any():
        raise ValueError("market P&L export has duplicate household keys")

    households = pd.DataFrame({
        "spine_id": "hes",
        "reference_year": int(target_year),
        "source_wave": _numeric(source, "year").round().astype(int),
        "source_year": _numeric(source, "year").round().astype(int),
        "source_survey_id": _numeric(source, "s_seker").round().astype(int),
        "source_household_id": _numeric(source, "misparmb").round().astype(int),
        "survey_weight": _numeric(source, "weight"),
        "household_size": _numeric(source, "household_size").clip(lower=1).round().astype(int),
        "sector_group": source.demographic_group.astype(str),
    })
    source_ids = (
        households.source_year.astype(str) + ":" + households.source_survey_id.astype(str)
        + ":" + households.source_household_id.astype(str)
    )
    households["household_key"] = [household_key("hes", target_year, value) for value in source_ids]
    households = households[[
        "spine_id", "reference_year", "household_key", "source_wave", "survey_weight",
        "household_size", "sector_group", "source_year", "source_survey_id",
        "source_household_id",
    ]]
    for record in households.to_dict("records"):
        validate_household_record(record)

    catalog = load_component_catalog(catalog_path)
    frames: list[pd.DataFrame] = []
    direct = {
        "income.market.pension_receipt": "pension_income",
        "income.market.rent": "rental_property_income",
        "income.market.capital": "reported_interest_dividend_income",
        # The resource uses the same purchaser-price housing service as the
        # matched imputed expense. The separate HES i121012 field is retained
        # in the export for lineage but is not mixed into this paired entry.
        "income.imputed.owner_occupied_rent": "imputed_housing_consumption",
        "expense.private.cash.education": "private_education_expense",
        "expense.private.cash.housing_rent": "rent_paid",
        "expense.private.cash.private_transfer": "private_transfers_paid",
        "expense.private.cash.mortgage_interest": "mortgage_interest",
        "expense.private.imputed.owner_occupied_rent": "imputed_housing_consumption",
        "financing.mortgage_principal": "mortgage_principal",
    }
    for component_id, source_column in direct.items():
        values = _numeric(source, source_column)
        if (values < 0).any():
            raise ValueError(f"{source_column} must be non-negative; use an explicit contra component")
        frames.append(_fact_frame(households, catalog, component_id, values, target_year))

    labor = _numeric(source, "labor_income")
    frames.append(_fact_frame(households, catalog, "income.market.labor", labor.clip(lower=0), target_year))
    frames.append(_fact_frame(
        households, catalog, "income.market.self_employment_loss", (-labor).clip(lower=0), target_year
    ))
    private_transfer = _numeric(source, "private_transfer_income")
    frames.append(_fact_frame(
        households, catalog, "income.market.private_transfer", private_transfer.clip(lower=0), target_year
    ))
    frames.append(_fact_frame(
        households, catalog, "income.market.private_transfer_loss", (-private_transfer).clip(lower=0), target_year
    ))
    other = _numeric(source, "other_income")
    frames.append(_fact_frame(households, catalog, "income.market.other", other.clip(lower=0), target_year))
    frames.append(_fact_frame(households, catalog, "income.market.other_loss", (-other).clip(lower=0), target_year))
    other_cash = _numeric(source, "other_cash_consumption")
    frames.append(_fact_frame(
        households, catalog, "expense.private.cash.other", other_cash.clip(lower=0), target_year
    ))
    frames.append(_fact_frame(
        households, catalog, "expense.private.cash.reconciliation", (-other_cash).clip(lower=0), target_year
    ))
    private_health = _numeric(source, "private_health_expense")
    frames.append(_fact_frame(
        households, catalog, "expense.private.cash.health", private_health.clip(lower=0), target_year
    ))
    frames.append(_fact_frame(
        households, catalog, "expense.private.cash.health_rebate", (-private_health).clip(lower=0), target_year
    ))
    retirement_contribution = sum(
        (_numeric(source, column) for column in [
            "pension_contributions", "training_fund_contributions",
            "provident_fund_contributions", "life_exec_insurance_contributions",
        ]),
        start=pd.Series(0.0, index=source.index),
    )
    if (retirement_contribution < 0).any():
        raise ValueError("retirement contributions must be non-negative")
    frames.append(_fact_frame(
        households, catalog, "financing.pension_contribution", retirement_contribution, target_year
    ))

    facts = pd.concat(frames, ignore_index=True)
    validate_component_facts(facts, catalog)
    catalog_dict = catalog.set_index("component_id").to_dict("index")
    for metadata in catalog_dict.values():
        if pd.isna(metadata.get("parent_component_id")):
            metadata["parent_component_id"] = None
    for record in facts.to_dict("records"):
        validate_market_fact(record, catalog_dict)

    emitted_market_income = (
        _numeric(source, "labor_income") + _numeric(source, "pension_income")
        + _numeric(source, "rental_property_income")
        + _numeric(source, "reported_interest_dividend_income")
        + other + _numeric(source, "private_transfer_income")
        + _numeric(source, "imputed_housing_consumption")
    )
    legacy_market_income = (
        _numeric(source, "labor_income") + _numeric(source, "pension_income")
        + _numeric(source, "rental_property_income")
        + _numeric(source, "reported_interest_dividend_income")
        + other + _numeric(source, "transfers_income")
        - _numeric(source, "government_cash_benefits_reported")
        + _numeric(source, "imputed_housing_consumption")
    )
    emitted_expense = (
        _numeric(source, "other_cash_consumption") + _numeric(source, "private_health_expense")
        + _numeric(source, "private_education_expense") + _numeric(source, "rent_paid")
        + _numeric(source, "private_transfers_paid") + _numeric(source, "mortgage_interest")
        + _numeric(source, "imputed_housing_consumption")
    )
    legacy_expense = (
        _numeric(source, "cash_consumption") + _numeric(source, "private_transfers_paid")
        + _numeric(source, "mortgage_interest") + _numeric(source, "imputed_housing_consumption")
    )
    parity = pd.DataFrame({
        "household_key": households.household_key,
        "reference_year": int(target_year),
        "market_income_gap_nis_monthly": emitted_market_income - legacy_market_income,
        "private_expense_gap_nis_monthly": emitted_expense - legacy_expense,
    })
    gap_columns = ["market_income_gap_nis_monthly", "private_expense_gap_nis_monthly"]
    if not np.allclose(parity[gap_columns], 0.0, atol=1e-8, rtol=0):
        raise ValueError(f"market adapter does not preserve source P&L: {parity[gap_columns].abs().max().to_dict()}")
    return households, facts, parity
