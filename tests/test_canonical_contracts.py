from __future__ import annotations

import copy
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.contracts import (
    ContractError,
    component_dimensions,
    household_key,
    load_definitions,
    person_key,
    validate_definitions,
    validate_household_record,
    validate_market_fact,
)
from shared.contracts.validate import validate_component_fact


def fact_dimensions(catalog: dict[str, dict], component_id: str) -> dict[str, str]:
    resolved = component_dimensions(catalog, component_id)
    return {
        key: resolved[key]
        for key in ("economic_function", "payer_funder", "delivery_value_type")
    }


class CanonicalContractTests(unittest.TestCase):
    def test_definitions_are_internally_consistent(self) -> None:
        counts = validate_definitions()
        self.assertEqual(counts["spines"], 3)
        self.assertGreaterEqual(counts["fields"], 30)
        self.assertGreaterEqual(counts["components"], 25)
        self.assertEqual(counts["contracts"], 7)

    def test_keys_are_deterministic_and_spine_local(self) -> None:
        first = household_key("hes", 2023, "123")
        self.assertEqual(first, household_key("hes", 2023, "123"))
        self.assertNotEqual(first, household_key("longitudinal", 2023, "123"))
        self.assertNotEqual(first, household_key("hes", 2022, "123"))
        self.assertRegex(first, r"^hh_[0-9a-f]{24}$")
        person = person_key("hes", 2023, "123", "2")
        self.assertRegex(person, r"^person_[0-9a-f]{24}$")
        self.assertNotIn("123", person)

    def test_household_export_requires_opaque_key_and_positive_weight(self) -> None:
        row = {
            "spine_id": "hes",
            "reference_year": 2023,
            "household_key": household_key("hes", 2023, "123"),
            "survey_weight": 17.4,
            "household_size": 3,
        }
        validate_household_record(row)
        row["household_key"] = "123"
        with self.assertRaises(ContractError):
            validate_household_record(row)

    def test_unknown_component_parent_fails(self) -> None:
        definitions = load_definitions()
        broken = copy.deepcopy(definitions)
        broken["component_catalog"]["components"][0]["parent_component_id"] = "missing"
        with self.assertRaises(ContractError):
            validate_definitions(broken)

    def test_fiscal_detail_and_in_kind_pairs_are_preserved(self) -> None:
        definitions = load_definitions()
        components = definitions["component_catalog"]["components"]
        legacy_codes = {row.get("legacy_code") for row in components} - {None}
        tax_codes = {code for code in legacy_codes if code.startswith("2.")}
        self.assertEqual(
            tax_codes,
            {"2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.8b", "2.9", "2.10", "2.11", "2.12", "2.13", "2.14"},
        )
        self.assertTrue({"3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "wel"} <= legacy_codes)
        ids = {row["component_id"] for row in components}
        for service in ("health", "education", "welfare", "transport", "religion", "culture_sport", "public_housing", "other"):
            self.assertIn(f"income.government.in_kind.{service}", ids)
            self.assertIn(f"expense.public.in_kind.{service}", ids)

    def test_public_service_resources_roll_up_once_and_uses_roll_by_function(self) -> None:
        definitions = load_definitions()
        catalog = {
            row["component_id"]: row for row in definitions["component_catalog"]["components"]
        }
        expected_sections = {
            "health": "health",
            "education.school": "education_children",
            "education.higher": "education_children",
            "education.postsecondary": "education_children",
            "welfare": "social_care",
            "transport": "transport",
            "religion": "culture_religion_recreation",
            "culture_sport": "culture_religion_recreation",
            "public_housing": "housing",
            "other": "other_consumption",
        }
        for suffix, expense_section in expected_sections.items():
            resource = component_dimensions(catalog, f"income.government.in_kind.{suffix}")
            use = component_dimensions(catalog, f"expense.public.in_kind.{suffix}")
            self.assertEqual(resource["economic_function"], use["economic_function"])
            self.assertEqual(resource["payer_funder"], "government")
            self.assertEqual(use["payer_funder"], "government")
            self.assertEqual(resource["delivery_value_type"], "in_kind")
            self.assertEqual(use["delivery_value_type"], "in_kind")
            self.assertEqual(resource["presentation_section"], "public_services_received")
            self.assertEqual(use["presentation_section"], expense_section)
            self.assertEqual(use["household_pnl_treatment"], "included")

    def test_collective_service_is_memo_only(self) -> None:
        definitions = load_definitions()
        catalog = {
            row["component_id"]: row for row in definitions["component_catalog"]["components"]
        }
        collective = component_dimensions(catalog, "memo.government.collective_service")
        self.assertEqual(collective["economic_function"], "collective_unattributed")
        self.assertEqual(collective["presentation_section"], "excluded_collective")
        self.assertEqual(collective["household_pnl_treatment"], "excluded_collective")

    def test_dimension_inheritance_accepts_dataframe_nan_root(self) -> None:
        definitions = load_definitions()
        catalog = {
            row["component_id"]: dict(row)
            for row in definitions["component_catalog"]["components"]
        }
        catalog["income"]["parent_component_id"] = float("nan")
        resolved = component_dimensions(catalog, "income.market.labor")
        self.assertEqual(resolved["economic_function"], "household_resources")
        self.assertEqual(resolved["payer_funder"], "employer")

    def test_signed_market_residual_has_explicit_cash_and_contra_leaves(self) -> None:
        definitions = load_definitions()
        catalog = {
            row["component_id"]: row for row in definitions["component_catalog"]["components"]
        }
        self.assertEqual(catalog["income.market.other"]["value_basis"], "cash")
        self.assertEqual(catalog["income.market.other_loss"]["value_basis"], "contra")
        self.assertEqual(catalog["income.market.other_loss"]["accounting_side"], "income")
        self.assertEqual(catalog["income.market.private_transfer_loss"]["value_basis"], "contra")
        self.assertEqual(catalog["expense.private.cash.reconciliation"]["value_basis"], "contra")
        self.assertEqual(catalog["expense.private.cash.health_rebate"]["value_basis"], "contra")

    def test_component_fact_requires_explicit_basis_and_evidence(self) -> None:
        definitions = load_definitions()
        component_catalog = {
            row["component_id"]: row for row in definitions["component_catalog"]["components"]
        }
        fact = {
            "household_key": household_key("hes", 2023, "123"),
            "reference_year": 2023,
            "component_id": "income.market.labor",
            "amount_nis": 12000.0,
            "accounting_side": "income",
            "value_basis": "cash",
            "frequency_basis": "monthly_flow",
            "price_basis": "nominal",
            "price_year": 2023,
            "evidence_status": "observed",
            "representation": "reported",
            "method_version": "hes_2023_v1",
            "source_dataset_id": "cbs_hes_2016_2018_2021_2023",
            **fact_dimensions(component_catalog, "income.market.labor"),
        }
        validate_component_fact(fact, component_catalog)
        del fact["evidence_status"]
        with self.assertRaises(ContractError):
            validate_component_fact(fact, component_catalog)

    def test_component_fact_rejects_negative_or_wrong_side(self) -> None:
        definitions = load_definitions()
        catalog = {row["component_id"]: row for row in definitions["component_catalog"]["components"]}
        fact = {
            "household_key": household_key("hes", 2023, "123"),
            "reference_year": 2023,
            "component_id": "income.market.labor",
            "amount_nis": -1.0,
            "accounting_side": "income",
            "value_basis": "cash",
            "frequency_basis": "monthly_flow",
            "price_basis": "nominal",
            "price_year": 2023,
            "evidence_status": "observed",
            "representation": "reported",
            "method_version": "hes_2023_v1",
            "source_dataset_id": "cbs_hes_2016_2018_2021_2023",
            **fact_dimensions(catalog, "income.market.labor"),
        }
        with self.assertRaises(ContractError):
            validate_component_fact(fact, catalog)

    def test_market_fact_rejects_allocated_or_personal_path_lineage(self) -> None:
        definitions = load_definitions()
        catalog = {row["component_id"]: row for row in definitions["component_catalog"]["components"]}
        fact = {
            "household_key": household_key("hes", 2023, "123"),
            "reference_year": 2023,
            "component_id": "income.market.labor",
            "amount_nis": 10000.0,
            "accounting_side": "income",
            "value_basis": "cash",
            "frequency_basis": "monthly_flow",
            "price_basis": "nominal",
            "price_year": 2023,
            "evidence_status": "observed",
            "representation": "reported",
            "method_version": "hes_market_v1",
            "source_dataset_id": "cbs_hes_2016_2018_2021_2023",
            **fact_dimensions(catalog, "income.market.labor"),
        }
        validate_market_fact(fact, catalog)
        fact["source_dataset_id"] = "C:" + r"\Users\example\hes.csv"
        with self.assertRaises(ContractError):
            validate_market_fact(fact, catalog)
        fact["amount_nis"] = 1.0
        fact["accounting_side"] = "expense"
        with self.assertRaises(ContractError):
            validate_component_fact(fact, catalog)

    def test_allocated_embedded_tax_requires_precalibration_basis(self) -> None:
        definitions = load_definitions()
        catalog = {row["component_id"]: row for row in definitions["component_catalog"]["components"]}
        fact = {
            "household_key": household_key("hes", 2023, "123"),
            "reference_year": 2023,
            "component_id": "expense.government.tax.vat",
            "amount_nis": 900.0,
            "accounting_side": "expense",
            "value_basis": "cash",
            "frequency_basis": "monthly_flow",
            "price_basis": "nominal",
            "price_year": 2023,
            "evidence_status": "macro_calibrated",
            "representation": "allocated",
            "method_version": "fiscal_2023_v1",
            "source_dataset_id": "fiscal_state_revenue_2023",
            **fact_dimensions(catalog, "expense.government.tax.vat"),
        }
        with self.assertRaises(ContractError):
            validate_component_fact(fact, catalog)
        fact["embedded_tax_basis_nis_monthly"] = 300.0
        validate_component_fact(fact, catalog)

    def test_contract_files_contain_no_personal_paths(self) -> None:
        personal = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", re.I)
        contract_root = ROOT / "shared" / "contracts"
        for path in contract_root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md"}:
                self.assertIsNone(personal.search(path.read_text(encoding="utf-8")), str(path))


if __name__ == "__main__":
    unittest.main()
