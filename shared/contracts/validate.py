"""Dependency-free validation of the shared canonical definitions."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any


DEFINITION_DIR = Path(__file__).with_name("definitions")
REQUIRED_DEFINITIONS = (
    "spines.json",
    "field_registry.json",
    "component_catalog.json",
    "record_contracts.json",
)
VALID_GRAINS = {"household", "person", "geography", "household_component", "factor", "national"}
VALID_TYPES = {"string", "integer", "number", "boolean", "date", "json"}
VALID_SIDES = {"income", "expense", "asset", "liability", "financing", "memo"}
VALID_VALUE_BASES = {
    "cash", "imputed", "allocated", "stock", "asset_flow", "liability_flow", "contra", "rate", "count"
}
VALID_EVIDENCE = {"observed", "modeled", "transported", "macro_calibrated", "derived"}
VALID_REPRESENTATIONS = {"reported", "allocated"}
VALID_FREQUENCIES = {"monthly_flow", "annual_flow", "point_in_time_stock"}
VALID_PRICE_BASES = {"nominal", "real"}
DIMENSION_FIELDS = ("economic_function", "payer_funder", "delivery_value_type")
PERSONAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|^/home/)", re.I)


class ContractError(ValueError):
    """Raised when shared definitions are internally inconsistent."""


def _reject_personal_path(value: object, field: str) -> None:
    if isinstance(value, str) and PERSONAL_PATH.search(value):
        raise ContractError(f"personal path is not allowed in {field}")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_definitions(directory: Path = DEFINITION_DIR) -> dict[str, Any]:
    return {name.removesuffix(".json"): _read_json(directory / name) for name in REQUIRED_DEFINITIONS}


def _unique(records: list[dict[str, Any]], key: str, label: str) -> set[str]:
    values = [str(record[key]) for record in records]
    if len(values) != len(set(values)):
        raise ContractError(f"duplicate {label}")
    return set(values)


def _validate_parent_graph(components: list[dict[str, Any]], ids: set[str]) -> None:
    parents = {row["component_id"]: row.get("parent_component_id") for row in components}
    for component_id, parent_id in parents.items():
        if parent_id is not None and parent_id not in ids:
            raise ContractError(f"unknown parent {parent_id} for {component_id}")
        seen: set[str] = set()
        current: str | None = component_id
        while current is not None:
            if current in seen:
                raise ContractError(f"component hierarchy cycle at {component_id}")
            seen.add(current)
            current = parents.get(current)


def component_dimensions(
    component_catalog: dict[str, dict[str, Any]], component_id: str
) -> dict[str, str]:
    """Resolve inherited functional dimensions for one component."""
    if component_id not in component_catalog:
        raise ContractError(f"unknown component_id: {component_id}")
    chain: list[dict[str, Any]] = []
    current: str | None = component_id
    while current is not None:
        row = component_catalog[current]
        chain.append(row)
        parent = row.get("parent_component_id")
        if parent is None or parent == "" or (isinstance(parent, float) and math.isnan(parent)):
            current = None
        else:
            current = str(parent)
    resolved: dict[str, str] = {}
    for row in reversed(chain):
        for field in (*DIMENSION_FIELDS, "presentation_section", "household_pnl_treatment"):
            if field in row:
                resolved[field] = row[field]
    missing = set(DIMENSION_FIELDS) - set(resolved)
    if missing:
        raise ContractError(f"component dimensions missing for {component_id}: {sorted(missing)}")
    return resolved


def validate_definitions(definitions: dict[str, Any] | None = None) -> dict[str, int]:
    data = definitions or load_definitions()
    spines = data["spines"]["spines"]
    fields = data["field_registry"]["fields"]
    components = data["component_catalog"]["components"]
    contracts = data["record_contracts"]["contracts"]

    spine_ids = _unique(spines, "spine_id", "spine_id")
    if spine_ids != {"hes", "longitudinal", "census"}:
        raise ContractError("the three canonical spines must be hes, longitudinal, and census")

    field_ids = _unique(fields, "field_id", "field_id")
    for field in fields:
        if field["grain"] not in VALID_GRAINS:
            raise ContractError(f"invalid grain for {field['field_id']}")
        if field["data_type"] not in VALID_TYPES:
            raise ContractError(f"invalid data type for {field['field_id']}")
        availability = field.get("availability", {})
        if set(availability) != spine_ids:
            raise ContractError(f"availability must cover all spines for {field['field_id']}")
        if not set(availability.values()) <= {"native", "derived", "modeled", "not_available"}:
            raise ContractError(f"invalid availability status for {field['field_id']}")

    component_ids = _unique(components, "component_id", "component_id")
    _validate_parent_graph(components, component_ids)
    component_catalog = {row["component_id"]: row for row in components}
    legacy_codes = [row["legacy_code"] for row in components if "legacy_code" in row]
    if len(legacy_codes) != len(set(legacy_codes)):
        raise ContractError("duplicate legacy component code")
    for component in components:
        if component["accounting_side"] not in VALID_SIDES:
            raise ContractError(f"invalid accounting side for {component['component_id']}")
        if component["value_basis"] not in VALID_VALUE_BASES:
            raise ContractError(f"invalid value basis for {component['component_id']}")
        allowed_bases = set(component.get("allowed_value_bases", [component["value_basis"]]))
        if not allowed_bases or not allowed_bases <= VALID_VALUE_BASES:
            raise ContractError(f"invalid allowed value bases for {component['component_id']}")
        if component.get("parent_component_id") == "expense.government.tax" and not isinstance(
            component.get("embedded_in_purchaser_price"), bool
        ):
            raise ContractError(f"tax leaf missing explicit embedded-tax flag: {component['component_id']}")
        resolved_dimensions = component_dimensions(component_catalog, component["component_id"])
        for field in DIMENSION_FIELDS:
            allowed = set(data["component_catalog"]["dimension_values"][field])
            if resolved_dimensions[field] not in allowed:
                raise ContractError(f"invalid {field} for {component['component_id']}")
        presentation = resolved_dimensions.get("presentation_section")
        if presentation not in set(data["component_catalog"]["dimension_values"]["presentation_section"]):
            raise ContractError(f"invalid presentation_section for {component['component_id']}")
    views = data["component_catalog"].get("views", {})
    replacement_roots = set(views.get("allocated_extended", {}).get("replacement_roots", []))
    if not replacement_roots <= component_ids:
        raise ContractError("view replacement root is not a known component")
    resource_prefix = "income.government.in_kind."
    use_prefix = "expense.public.in_kind."
    resources = {item.removeprefix(resource_prefix) for item in component_ids if item.startswith(resource_prefix)}
    uses = {item.removeprefix(use_prefix) for item in component_ids if item.startswith(use_prefix)}
    if resources != uses:
        raise ContractError("public in-kind resource and use leaves must be paired")
    for suffix in resources:
        resource = component_dimensions(component_catalog, f"{resource_prefix}{suffix}")
        use = component_dimensions(component_catalog, f"{use_prefix}{suffix}")
        for field in DIMENSION_FIELDS:
            if resource[field] != use[field]:
                raise ContractError(f"public in-kind pair dimension mismatch for {suffix}: {field}")
        if resource.get("presentation_section") != "public_services_received":
            raise ContractError(f"public in-kind resource must roll to public_services_received: {suffix}")
        if use.get("presentation_section") in {"public_services_received", "excluded_collective"}:
            raise ContractError(f"household public-service use lacks a functional expense section: {suffix}")
        if resource.get("household_pnl_treatment") != "included" or use.get(
            "household_pnl_treatment"
        ) != "included":
            raise ContractError(f"paired household public service must remain included: {suffix}")
    collective = component_dimensions(component_catalog, "memo.government.collective_service")
    if (
        collective["economic_function"] != "collective_unattributed"
        or collective.get("presentation_section") != "excluded_collective"
        or collective.get("household_pnl_treatment") != "excluded_collective"
    ):
        raise ContractError("collective services must be memo-only and excluded from household P&L")

    contract_ids = _unique(contracts, "contract_id", "contract_id")
    for contract in contracts:
        required = set(contract["required_fields"])
        optional = set(contract.get("optional_fields", []))
        unknown = (required | optional) - field_ids
        if unknown:
            raise ContractError(f"unknown fields in {contract['contract_id']}: {sorted(unknown)}")
        if required & optional:
            raise ContractError(f"required and optional overlap in {contract['contract_id']}")

    return {
        "spines": len(spine_ids),
        "fields": len(field_ids),
        "components": len(component_ids),
        "contracts": len(contract_ids),
    }


def validate_component_fact(
    record: dict[str, Any],
    component_catalog: dict[str, dict[str, Any]] | None = None,
    national_control_ids: set[str] | None = None,
) -> None:
    required = {
        "household_key",
        "reference_year",
        "component_id",
        "amount_nis",
        "accounting_side",
        "value_basis",
        "frequency_basis",
        "price_basis",
        "price_year",
        "evidence_status",
        "representation",
        "economic_function",
        "payer_funder",
        "delivery_value_type",
        "method_version",
        "source_dataset_id",
    }
    missing = required - set(record)
    if missing:
        raise ContractError(f"component fact missing: {sorted(missing)}")
    component = None
    if component_catalog is not None:
        component = component_catalog.get(record["component_id"])
        if component is None:
            raise ContractError(f"unknown component_id: {record['component_id']}")
    if record["accounting_side"] not in VALID_SIDES:
        raise ContractError("invalid accounting_side")
    if record["value_basis"] not in VALID_VALUE_BASES:
        raise ContractError("invalid value_basis")
    if record["evidence_status"] not in VALID_EVIDENCE:
        raise ContractError("invalid evidence_status")
    if record["representation"] not in VALID_REPRESENTATIONS:
        raise ContractError("invalid representation")
    amount = record["amount_nis"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
        raise ContractError("amount_nis must be finite and numeric")
    if amount < 0:
        raise ContractError("amount_nis must be non-negative")
    if record["frequency_basis"] not in VALID_FREQUENCIES:
        raise ContractError("invalid frequency_basis")
    if record["price_basis"] not in VALID_PRICE_BASES:
        raise ContractError("invalid price_basis")
    price_year = record["price_year"]
    if isinstance(price_year, bool) or not isinstance(price_year, int) or not 1900 <= price_year <= 2100:
        raise ContractError("price_year must be a plausible integer year")
    if not str(record["method_version"]).strip() or not str(record["source_dataset_id"]).strip():
        raise ContractError("method_version and source_dataset_id must be non-empty")
    _reject_personal_path(record["method_version"], "method_version")
    _reject_personal_path(record["source_dataset_id"], "source_dataset_id")
    if component is not None:
        if record["accounting_side"] != component["accounting_side"]:
            raise ContractError("accounting_side does not match component catalog")
        allowed_bases = set(component.get("allowed_value_bases", [component["value_basis"]]))
        if record["value_basis"] not in allowed_bases:
            raise ContractError("value_basis does not match component catalog")
        resolved_dimensions = component_dimensions(component_catalog, record["component_id"])
        for field in DIMENSION_FIELDS:
            if record[field] != resolved_dimensions[field]:
                raise ContractError(f"{field} does not match component catalog")
        if component["value_basis"] == "stock" and record["frequency_basis"] != "point_in_time_stock":
            raise ContractError("stock component requires point_in_time_stock frequency")
        if component["accounting_side"] in {"income", "expense", "financing"} and record[
            "frequency_basis"
        ] == "point_in_time_stock":
            raise ContractError("flow component cannot use point_in_time_stock frequency")
        if component.get("embedded_in_purchaser_price") and record["representation"] == "allocated":
            basis = record.get("embedded_tax_basis_nis_monthly")
            if (
                isinstance(basis, bool)
                or not isinstance(basis, (int, float))
                or not math.isfinite(basis)
                or basis < 0
            ):
                raise ContractError("allocated embedded tax requires a non-negative finite embedded-tax basis")
    control_id = record.get("national_control_id")
    if control_id and national_control_ids is not None and control_id not in national_control_ids:
        raise ContractError(f"unknown national_control_id: {control_id}")


def validate_household_record(record: dict[str, Any]) -> None:
    required = {"spine_id", "reference_year", "household_key", "survey_weight", "household_size"}
    missing = required - set(record)
    if missing:
        raise ContractError(f"household record missing: {sorted(missing)}")
    if record["spine_id"] not in {"hes", "longitudinal", "census"}:
        raise ContractError("invalid household spine_id")
    year = record["reference_year"]
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise ContractError("invalid household reference_year")
    if not re.fullmatch(r"hh_[0-9a-f]{24}", str(record["household_key"])):
        raise ContractError("household_key must be an opaque canonical key")
    weight = record["survey_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ContractError("survey_weight must be positive and finite")
    size = record["household_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ContractError("household_size must be a positive integer")


def validate_market_fact(record: dict[str, Any], component_catalog: dict[str, dict[str, Any]]) -> None:
    validate_component_fact(record, component_catalog)
    if record["representation"] != "reported":
        raise ContractError("market fact must use reported representation")
    allowed = ("income.market.", "income.imputed.", "expense.private.", "financing.")
    if not record["component_id"].startswith(allowed):
        raise ContractError("component is outside the HES market-fact domain")
    if record.get("national_control_id"):
        raise ContractError("reported market fact cannot reference a fiscal national control")


def validate_fiscal_fact(
    record: dict[str, Any],
    component_catalog: dict[str, dict[str, Any]],
    national_control_ids: set[str],
) -> None:
    validate_component_fact(record, component_catalog, national_control_ids)
    if record["representation"] != "allocated":
        raise ContractError("fiscal fact must use allocated representation")
    if not record.get("national_control_id"):
        raise ContractError("fiscal fact requires national_control_id")
    allowed = ("income.government.", "expense.government.", "expense.public.in_kind.")
    if not record["component_id"].startswith(allowed):
        raise ContractError("component is outside the fiscal-allocation domain")
