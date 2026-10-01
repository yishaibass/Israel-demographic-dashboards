"""Canonical household P&L view selection and accounting identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from contracts.validate import component_dimensions

PNL_VIEWS = ("reported", "allocated_extended")
KEY_COLUMNS = ["household_key", "reference_year"]
FACT_KEY = KEY_COLUMNS + ["component_id", "representation", "method_version"]
DIMENSION_COLUMNS = ["economic_function", "payer_funder", "delivery_value_type"]
FACT_REQUIRED = FACT_KEY + ["amount_nis", "accounting_side", "value_basis", *DIMENSION_COLUMNS, "frequency_basis", "price_basis", "price_year", "evidence_status", "source_dataset_id"]
FISCAL_CASH_PREFIXES = ("income.government.cash_benefit", "expense.government.tax")
PUBLIC_IN_KIND_PREFIXES = ("income.government.in_kind.", "expense.public.in_kind.")
GROSS_PURCHASE_COMPONENTS = {
    "expense.private.cash.other", "expense.private.cash.health", "expense.private.cash.education"
}
CONSUMPTION_TAX_CONTRA = "expense.private.cash.indirect_tax_reclassification"
PAIR_COMPONENTS = {"income.imputed.owner_occupied_rent": "expense.private.imputed.owner_occupied_rent"}


def _default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts" / "definitions" / "component_catalog.json"


def load_component_catalog(path: str | Path | None = None) -> pd.DataFrame:
    payload = json.loads((Path(path) if path else _default_catalog_path()).read_text(encoding="utf-8"))
    rows = payload["components"]
    by_id = {row["component_id"]: row for row in rows}
    resolved_rows = []
    for row in rows:
        resolved = dict(row)
        resolved.update(component_dimensions(by_id, row["component_id"]))
        resolved_rows.append(resolved)
    catalog = pd.DataFrame(resolved_rows)
    required = {"component_id", "parent_component_id", "label", "accounting_side", "value_basis"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"component catalog is missing columns: {missing}")
    if catalog.component_id.duplicated().any():
        raise ValueError("component catalog contains duplicate component_id values")
    return catalog


def validate_component_facts(facts: pd.DataFrame, catalog: pd.DataFrame) -> None:
    missing = sorted(set(FACT_REQUIRED) - set(facts.columns))
    if missing:
        raise ValueError(f"component facts are missing columns: {missing}")
    lineage = FACT_KEY + ["accounting_side", "value_basis", *DIMENSION_COLUMNS, "evidence_status", "source_dataset_id"]
    if facts[lineage].isna().any().any():
        raise ValueError("component fact keys and lineage fields cannot be null")
    if facts.duplicated(FACT_KEY).any():
        raise ValueError("component facts are not unique at the canonical fact grain")
    if not set(facts.representation).issubset({"reported", "allocated"}):
        raise ValueError("representation must be reported or allocated")
    amounts = pd.to_numeric(facts.amount_nis, errors="coerce")
    if amounts.isna().any() or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("amount_nis must be finite and non-negative")
    if not set(facts.frequency_basis).issubset({"monthly_flow", "annual_flow", "point_in_time_stock"}):
        raise ValueError("invalid frequency_basis")
    if facts.price_year.isna().any() or facts.price_basis.isna().any():
        raise ValueError("price basis and year are required")
    metadata = catalog.set_index("component_id")[["accounting_side", "value_basis", *DIMENSION_COLUMNS]]
    unknown = sorted(set(facts.component_id) - set(metadata.index))
    if unknown:
        raise ValueError(f"unknown component_id values: {unknown}")
    checked = facts.merge(metadata, left_on="component_id", right_index=True, suffixes=("", "_catalog"), validate="many_to_one")
    mismatch = (checked.accounting_side != checked.accounting_side_catalog) | (checked.value_basis != checked.value_basis_catalog)
    for column in DIMENSION_COLUMNS:
        mismatch |= checked[column] != checked[f"{column}_catalog"]
    if mismatch.any():
        raise ValueError(f"fact metadata does not match component catalog: {checked.loc[mismatch, 'component_id'].unique().tolist()}")

    parent = catalog.set_index("component_id").parent_component_id.to_dict()
    for group_key, group in facts.groupby(KEY_COLUMNS + ["representation"]):
        present = set(group.component_id)
        for component_id in present:
            ancestor = parent.get(component_id)
            while pd.notna(ancestor):
                if ancestor in present:
                    raise ValueError(f"parent and child component facts coexist for {group_key}: {ancestor}, {component_id}")
                ancestor = parent.get(ancestor)


def _starts_with(series: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    return series.astype(str).str.startswith(prefixes)


def _select_view(facts: pd.DataFrame, view: str, catalog: pd.DataFrame) -> pd.DataFrame:
    if view not in PNL_VIEWS:
        raise ValueError(f"view must be one of {PNL_VIEWS}; received {view!r}")
    included_ids = set(catalog.loc[catalog.household_pnl_treatment.eq("included"), "component_id"])
    pnl = facts.loc[
        facts.accounting_side.isin(["income", "expense"]) & facts.component_id.isin(included_ids)
    ].copy()
    fiscal_cash = _starts_with(pnl.component_id, FISCAL_CASH_PREFIXES)
    public_in_kind = _starts_with(pnl.component_id, PUBLIC_IN_KIND_PREFIXES)
    if view == "reported":
        return pnl.loc[pnl.representation.eq("reported")].copy()

    reported_fiscal = pnl.loc[fiscal_cash & pnl.representation.eq("reported"), KEY_COLUMNS + ["component_id"]].copy()
    allocated_fiscal = pnl.loc[fiscal_cash & pnl.representation.eq("allocated"), KEY_COLUMNS + ["component_id"]].copy()
    def replacement_root(component_id: str) -> str:
        return next(root for root in FISCAL_CASH_PREFIXES if component_id.startswith(root))
    reported_groups = {(h, y, replacement_root(c)) for h, y, c in reported_fiscal.itertuples(index=False, name=None)}
    allocated_groups = {(h, y, replacement_root(c)) for h, y, c in allocated_fiscal.itertuples(index=False, name=None)}
    missing = sorted(reported_groups - allocated_groups)
    if missing:
        raise ValueError(f"allocated representation missing for reported fiscal roots: {missing}")

    select_allocated = fiscal_cash | public_in_kind
    keep = (select_allocated & pnl.representation.eq("allocated")) | (~select_allocated & pnl.representation.eq("reported"))
    return pnl.loc[keep].copy()


def _validate_pairs(detail: pd.DataFrame) -> None:
    indexed = detail.groupby(KEY_COLUMNS + ["component_id"], as_index=False).amount_nis_monthly.sum()
    pairs = dict(PAIR_COMPONENTS)
    for resource_id in indexed.loc[indexed.component_id.str.startswith("income.government.in_kind."), "component_id"].unique():
        suffix = resource_id.removeprefix("income.government.in_kind.")
        pairs[resource_id] = f"expense.public.in_kind.{suffix}"
    for resource_id, use_id in pairs.items():
        resource = indexed.loc[indexed.component_id.eq(resource_id), KEY_COLUMNS + ["amount_nis_monthly"]]
        use = indexed.loc[indexed.component_id.eq(use_id), KEY_COLUMNS + ["amount_nis_monthly"]]
        if resource.empty and use.empty:
            continue
        paired = resource.merge(use, on=KEY_COLUMNS, how="outer", suffixes=("_resource", "_use")).fillna(0.0)
        if not np.allclose(paired.amount_nis_monthly_resource, paired.amount_nis_monthly_use, atol=1e-6, rtol=0):
            raise ValueError(f"resource/use pair does not balance: {resource_id}, {use_id}")


def build_household_pnl(facts: pd.DataFrame, view: str, *, catalog_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one monthly-NIS household P&L view and its additive detail."""
    catalog = load_component_catalog(catalog_path)
    validate_component_facts(facts, catalog)
    detail = _select_view(facts, view, catalog)
    presentation = catalog.set_index("component_id")[["presentation_section"]]
    detail = detail.merge(presentation, left_on="component_id", right_index=True, validate="many_to_one")
    if detail.frequency_basis.eq("point_in_time_stock").any():
        raise ValueError("point-in-time stocks cannot enter the P&L")
    factor = detail.frequency_basis.map({"monthly_flow": 1.0, "annual_flow": 1 / 12.0})
    if factor.isna().any():
        raise ValueError("P&L facts must be monthly or annual flows")
    detail["amount_nis_monthly"] = pd.to_numeric(detail.amount_nis) * factor
    _validate_pairs(detail)
    income_sign = np.where(detail.value_basis.eq("contra"), -1.0, 1.0)
    detail["income_nis_monthly"] = np.where(detail.accounting_side.eq("income"), detail.amount_nis_monthly * income_sign, 0.0)
    expense_sign = np.where(detail.value_basis.eq("contra"), -1.0, 1.0)
    detail["expense_nis_monthly"] = np.where(detail.accounting_side.eq("expense"), detail.amount_nis_monthly * expense_sign, 0.0)
    detail["net_nis_monthly"] = detail.income_nis_monthly - detail.expense_nis_monthly
    detail["pnl_view"] = view

    if view == "allocated_extended":
        embedded_ids = set(
            catalog.loc[catalog.get("embedded_in_purchaser_price", False).fillna(False).astype(bool), "component_id"]
        )
        embedded = detail.loc[detail.component_id.isin(embedded_ids)]
        if not embedded.empty:
            if "embedded_tax_basis_nis_monthly" not in embedded.columns:
                raise ValueError("embedded tax basis is required for allocated_extended view")
            basis = pd.to_numeric(embedded.embedded_tax_basis_nis_monthly, errors="coerce")
            if basis.isna().any() or (basis < 0).any() or not np.isfinite(basis).all():
                raise ValueError("embedded tax basis must be finite and non-negative")
            embedded = embedded.assign(_embedded_basis=basis)
            gross = detail.loc[detail.component_id.isin(GROSS_PURCHASE_COMPONENTS)].groupby(KEY_COLUMNS, as_index=False).amount_nis_monthly.sum().rename(columns={"amount_nis_monthly": "gross_purchase_nis"})
            candidates = embedded.assign(_candidate=lambda x: np.minimum(x.amount_nis_monthly, x._embedded_basis))
            candidate_total = candidates.groupby(KEY_COLUMNS, as_index=False)._candidate.sum()
            contra = candidate_total.merge(gross, on=KEY_COLUMNS, how="left").fillna({"gross_purchase_nis": 0.0})
            contra["amount_nis_monthly"] = np.minimum(contra._candidate, contra.gross_purchase_nis).clip(lower=0.0)
            contra = contra.drop(columns=["_candidate", "gross_purchase_nis"])
            contra["component_id"] = CONSUMPTION_TAX_CONTRA
            contra["amount_nis"] = contra["amount_nis_monthly"]
            contra["representation"] = "allocated"
            contra["method_version"] = "accounting-rule-v1"
            contra["accounting_side"] = "expense"
            contra["value_basis"] = "contra"
            contra_meta = catalog.set_index("component_id").loc[CONSUMPTION_TAX_CONTRA]
            for column in DIMENSION_COLUMNS:
                contra[column] = contra_meta[column]
            contra["presentation_section"] = contra_meta.presentation_section
            contra["frequency_basis"] = "monthly_flow"
            contra["evidence_status"] = "derived"
            contra["source_dataset_id"] = "derived.accounting"
            contra["national_control_id"] = ""
            contra["quality_flags"] = "[]"
            contra["income_nis_monthly"] = 0.0
            contra["expense_nis_monthly"] = -contra.amount_nis_monthly
            contra["net_nis_monthly"] = contra.amount_nis_monthly
            contra["pnl_view"] = view
            detail = pd.concat([detail, contra.reindex(columns=detail.columns)], ignore_index=True)

    value_columns = ["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]
    totals = detail.groupby(KEY_COLUMNS, as_index=False)[value_columns].sum()
    totals = facts[KEY_COLUMNS].drop_duplicates().merge(totals, on=KEY_COLUMNS, how="left").fillna({c: 0.0 for c in value_columns})
    totals["pnl_view"] = view
    return totals, detail


def validate_national_controls(facts: pd.DataFrame, households: pd.DataFrame, controls: pd.DataFrame, *, tolerance_nis: float = 1.0) -> pd.DataFrame:
    """Reconcile controlled allocated facts to weighted annual national totals."""
    for name, frame, required in [
        ("households", households, KEY_COLUMNS + ["survey_weight"]),
        ("controls", controls, ["national_control_id", "control_year", "amount_nis_annual"]),
    ]:
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    if households.duplicated(KEY_COLUMNS).any() or households.survey_weight.isna().any() or (households.survey_weight < 0).any():
        raise ValueError("household weights must be unique, present and non-negative")
    if "national_control_id" not in facts.columns:
        raise ValueError("component facts lack national_control_id")
    has_control = facts.national_control_id.fillna("").ne("")
    matched_public_use = facts.component_id.astype(str).str.startswith("expense.public.in_kind.")
    contra = facts.value_basis.eq("contra")
    # Public-service controls measure the benefit/resource once. The explicit
    # matched-use fact exists for the household P&L identity, not a second claim
    # on the national budget.
    controlled = facts.loc[facts.representation.eq("allocated") & has_control & ~matched_public_use & ~contra]
    weighted = controlled.merge(households[KEY_COLUMNS + ["survey_weight"]], on=KEY_COLUMNS, how="left", validate="many_to_one", indicator=True)
    if weighted.survey_weight.isna().any() or weighted._merge.ne("both").any():
        raise ValueError("controlled facts have missing household weights")
    annual_factor = weighted.frequency_basis.map({"monthly_flow": 12.0, "annual_flow": 1.0})
    if annual_factor.isna().any():
        raise ValueError("controlled fiscal facts must be flow amounts")
    actual = weighted.assign(actual_nis_annual=lambda x: x.amount_nis * annual_factor * x.survey_weight).groupby(["national_control_id", "reference_year"], as_index=False).actual_nis_annual.sum().rename(columns={"reference_year": "control_year"})
    result = controls[["national_control_id", "control_year", "amount_nis_annual"]].merge(actual, on=["national_control_id", "control_year"], how="left", validate="one_to_one")
    result["actual_nis_annual"] = result.actual_nis_annual.fillna(0.0)
    result["gap_nis"] = result.actual_nis_annual - result.amount_nis_annual
    result["passes"] = result.gap_nis.abs() <= tolerance_nis
    if not result.passes.all():
        failed = result.loc[~result.passes, ["national_control_id", "control_year", "gap_nis"]]
        raise ValueError(f"national controls do not reconcile: {failed.to_dict('records')}")
    return result


def summarize_components(detail: pd.DataFrame, group_columns: Iterable[str] = ()) -> pd.DataFrame:
    groups = list(group_columns) + ["component_id"]
    values = ["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]
    return detail.groupby(groups, as_index=False, dropna=False)[values].sum()


def summarize_functional_pnl(detail: pd.DataFrame, group_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Aggregate presentation lines by economic function, then payer/delivery.

    Public-service resources retain their functional detail but share the single
    income presentation line ``public_services_received``. Matched uses appear
    in the corresponding functional expense section.
    """
    required = {
        "accounting_side", "economic_function", "payer_funder",
        "delivery_value_type", "presentation_section",
        "income_nis_monthly", "expense_nis_monthly", "net_nis_monthly",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"P&L detail lacks functional rollup fields: {missing}")
    rolled = detail.copy()
    public_resource = (
        rolled.accounting_side.eq("income")
        & rolled.payer_funder.eq("government")
        & rolled.delivery_value_type.eq("in_kind")
    )
    rolled.loc[public_resource, "presentation_section"] = "public_services_received"
    # Presentation has one public-services resource line. Functional detail is
    # retained on ``detail``; the presentation copy intentionally collapses it.
    rolled.loc[public_resource, "economic_function"] = "all_public_services"
    groups = list(group_columns) + [
        "accounting_side", "presentation_section", "economic_function",
        "payer_funder", "delivery_value_type",
    ]
    values = ["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]
    return rolled.groupby(groups, as_index=False, dropna=False)[values].sum()
