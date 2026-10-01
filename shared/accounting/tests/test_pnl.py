import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ACCOUNTING_DIR = Path(__file__).resolve().parents[1]
SHARED_DIR = ACCOUNTING_DIR.parent
sys.path.insert(0, str(SHARED_DIR))

from accounting.pnl import build_household_pnl, summarize_functional_pnl, validate_national_controls
from contracts.validate import component_dimensions, validate_component_fact, validate_definitions

CATALOG_PATH = SHARED_DIR / "contracts" / "definitions" / "component_catalog.json"
CATALOG = {row["component_id"]: row for row in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["components"]}


def fact(component_id, amount, representation="reported", control="", embedded_basis=None):
    meta = CATALOG[component_id]
    dimensions = component_dimensions(CATALOG, component_id)
    frequency = "point_in_time_stock" if meta["value_basis"] == "stock" else "monthly_flow"
    return {
        "household_key": "hes/2023/1", "reference_year": 2023,
        "component_id": component_id, "representation": representation,
        "method_version": "fixture-v1", "amount_nis": amount,
        "accounting_side": meta["accounting_side"], "value_basis": meta["value_basis"],
        "economic_function": dimensions["economic_function"],
        "payer_funder": dimensions["payer_funder"],
        "delivery_value_type": dimensions["delivery_value_type"],
        "frequency_basis": frequency, "price_basis": "nominal", "price_year": 2023,
        "evidence_status": "observed" if representation == "reported" else "macro_calibrated",
        "source_dataset_id": "fixture.hes", "national_control_id": control,
        "embedded_tax_basis_nis_monthly": embedded_basis,
    }


def fixture():
    rows = [
        fact("income.market.labor", 10_000), fact("income.market.pension_receipt", 500),
        fact("income.imputed.owner_occupied_rent", 2_000),
        fact("expense.private.cash.other", 5_500), fact("expense.private.cash.health", 500),
        fact("expense.private.cash.education", 300), fact("expense.private.cash.mortgage_interest", 200),
        fact("expense.private.imputed.owner_occupied_rent", 2_000),
        fact("financing.pension_contribution", 1_100), fact("financing.mortgage_principal", 800),
        fact("asset.primary_residence", 2_000_000),
        fact("expense.government.tax.income_capital_gains", 1_200),
        fact("expense.government.tax.social_contribution", 400),
        fact("income.government.cash_benefit.old_age", 700),
        fact("expense.government.tax.income_capital_gains", 1_800, "allocated", "tax_income_2023"),
        fact("expense.government.tax.social_contribution", 500, "allocated"),
        fact("expense.government.tax.vat", 900, "allocated", embedded_basis=700),
        fact("expense.government.tax.corporate", 100, "allocated"),
        fact("income.government.cash_benefit.old_age", 1_000, "allocated"),
        fact("income.government.in_kind.health", 1_500, "allocated"),
        fact("expense.public.in_kind.health", 1_500, "allocated"),
        fact("income.government.in_kind.education", 1_000, "allocated"),
        fact("expense.public.in_kind.education", 1_000, "allocated"),
        fact("income.government.in_kind.welfare", 200, "allocated"),
        fact("expense.public.in_kind.welfare", 200, "allocated"),
    ]
    return pd.DataFrame(rows)


class PnlAccountingTest(unittest.TestCase):
    def test_canonical_contract_interoperability(self):
        validate_definitions()
        for record in fixture().to_dict("records"):
            validate_component_fact(record, CATALOG, {"tax_income_2023"})

    def test_reported_view_preserves_reported_tax_and_cash_benefit(self):
        totals, detail = build_household_pnl(fixture(), "reported")
        row = totals.iloc[0]
        self.assertEqual(row.income_nis_monthly, 13_200)
        self.assertEqual(row.expense_nis_monthly, 10_100)
        self.assertEqual(row.net_nis_monthly, 3_100)
        self.assertTrue((detail.representation == "reported").all())

    def test_allocated_extended_replaces_and_does_not_double_count(self):
        totals, detail = build_household_pnl(fixture(), "allocated_extended")
        row = totals.iloc[0]
        self.assertEqual(row.income_nis_monthly, 16_200)
        self.assertEqual(row.expense_nis_monthly, 13_800)
        self.assertEqual(row.net_nis_monthly, 2_400)
        tax = detail.loc[detail.component_id.eq("expense.government.tax.income_capital_gains")]
        self.assertEqual(tax.amount_nis.sum(), 1_800)
        contra = detail.loc[detail.component_id.eq("expense.private.cash.indirect_tax_reclassification")]
        self.assertEqual(contra.expense_nis_monthly.sum(), -700)

    def test_private_public_education_and_health_are_distinct(self):
        _, detail = build_household_pnl(fixture(), "allocated_extended")
        expected = {
            "expense.private.cash.health", "income.government.in_kind.health", "expense.public.in_kind.health",
            "expense.private.cash.education", "income.government.in_kind.education", "expense.public.in_kind.education",
        }
        self.assertTrue(expected.issubset(set(detail.component_id)))

    def test_function_first_rollup_and_single_public_services_resource_line(self):
        _, detail = build_household_pnl(fixture(), "allocated_extended")
        rolled = summarize_functional_pnl(detail)
        public_resource = rolled.loc[
            rolled.presentation_section.eq("public_services_received")
            & rolled.accounting_side.eq("income")
        ]
        self.assertEqual(len(public_resource), 1)
        self.assertEqual(public_resource.iloc[0].income_nis_monthly, 2_700)
        self.assertEqual(public_resource.iloc[0].economic_function, "all_public_services")

        health_uses = rolled.loc[
            rolled.accounting_side.eq("expense")
            & rolled.economic_function.eq("health")
        ]
        self.assertEqual(set(health_uses.payer_funder), {"household", "government"})
        education_uses = rolled.loc[
            rolled.accounting_side.eq("expense")
            & rolled.economic_function.eq("education_children")
        ]
        self.assertEqual(set(education_uses.payer_funder), {"household", "government"})
        self.assertAlmostEqual(rolled.income_nis_monthly.sum(), detail.income_nis_monthly.sum())
        self.assertAlmostEqual(rolled.expense_nis_monthly.sum(), detail.expense_nis_monthly.sum())

    def test_collective_memo_is_excluded_from_household_pnl(self):
        data = pd.concat([
            fixture(),
            pd.DataFrame([fact("memo.government.collective_service", 999, "allocated")]),
        ], ignore_index=True)
        totals, detail = build_household_pnl(data, "allocated_extended")
        baseline, _ = build_household_pnl(fixture(), "allocated_extended")
        self.assertNotIn("memo.government.collective_service", set(detail.component_id))
        pd.testing.assert_frame_equal(totals, baseline)

    def test_pension_and_mortgage_flow_treatment(self):
        _, detail = build_household_pnl(fixture(), "reported")
        self.assertIn("income.market.pension_receipt", set(detail.component_id))
        self.assertIn("expense.private.cash.mortgage_interest", set(detail.component_id))
        self.assertNotIn("financing.pension_contribution", set(detail.component_id))
        self.assertNotIn("financing.mortgage_principal", set(detail.component_id))

    def test_missing_allocated_replacement_fails(self):
        data = fixture()
        allocated_tax = data.component_id.str.startswith("expense.government.tax.") & data.representation.eq("allocated")
        broken = data.loc[~allocated_tax]
        with self.assertRaisesRegex(ValueError, "allocated representation missing"):
            build_household_pnl(broken, "allocated_extended")

    def test_negative_amount_fails(self):
        broken = fixture()
        broken.loc[0, "amount_nis"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_household_pnl(broken, "reported")

    def test_national_control_reconciliation(self):
        facts = pd.DataFrame([fact("expense.government.tax.income_capital_gains", 100, "allocated", "tax_income_2023")])
        households = pd.DataFrame([{"household_key": "hes/2023/1", "reference_year": 2023, "survey_weight": 2.0}])
        controls = pd.DataFrame([{"national_control_id": "tax_income_2023", "control_year": 2023, "amount_nis_annual": 2_400.0}])
        self.assertTrue(validate_national_controls(facts, households, controls).passes.all())

    def test_public_service_control_counts_resource_once(self):
        resource = fact("income.government.in_kind.health", 100, "allocated", "health_2023")
        matched_use = fact("expense.public.in_kind.health", 100, "allocated", "health_2023")
        facts = pd.DataFrame([resource, matched_use])
        households = pd.DataFrame([{"household_key": "hes/2023/1", "reference_year": 2023, "survey_weight": 2.0}])
        controls = pd.DataFrame([{"national_control_id": "health_2023", "control_year": 2023, "amount_nis_annual": 2_400.0}])
        result = validate_national_controls(facts, households, controls)
        self.assertEqual(result.iloc[0].actual_nis_annual, 2_400.0)

    def test_indirect_tax_contra_is_bounded_by_basis_and_gross(self):
        data = fixture()
        data.loc[data.component_id.eq("expense.private.cash.other"), "amount_nis"] = 100
        data.loc[data.component_id.isin(["expense.private.cash.health", "expense.private.cash.education"]), "amount_nis"] = 0
        _, detail = build_household_pnl(data, "allocated_extended")
        contra = detail.loc[detail.component_id.eq("expense.private.cash.indirect_tax_reclassification"), "amount_nis_monthly"].sum()
        self.assertEqual(contra, 100)
        self.assertLessEqual(contra, 900)
        self.assertLessEqual(contra, 700)

    def test_zero_spending_yields_zero_contra_and_preserves_full_tax(self):
        data = fixture()
        data.loc[data.component_id.isin(["expense.private.cash.other", "expense.private.cash.health", "expense.private.cash.education"]), "amount_nis"] = 0
        _, detail = build_household_pnl(data, "allocated_extended")
        self.assertEqual(detail.loc[detail.component_id.eq("expense.private.cash.indirect_tax_reclassification"), "amount_nis_monthly"].sum(), 0)
        self.assertEqual(detail.loc[detail.component_id.eq("expense.government.tax.vat"), "amount_nis"].sum(), 900)

    def test_missing_embedded_basis_fails_extended_view(self):
        data = fixture()
        data.loc[data.component_id.eq("expense.government.tax.vat"), "embedded_tax_basis_nis_monthly"] = float("nan")
        with self.assertRaisesRegex(ValueError, "embedded tax basis"):
            build_household_pnl(data, "allocated_extended")


if __name__ == "__main__":
    unittest.main()
