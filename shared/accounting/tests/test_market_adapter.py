from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ACCOUNTING_DIR = Path(__file__).resolve().parents[1]
SHARED_DIR = ACCOUNTING_DIR.parent
sys.path.insert(0, str(SHARED_DIR))

from accounting.market_adapter import build_hes_market_component_facts
from accounting.pnl import build_household_pnl


class MarketAdapterTest(unittest.TestCase):
    def test_signed_residuals_preserve_source_pnl(self):
        row = {
            "year": 2023, "s_seker": 2023, "misparmb": 1,
            "survey_year": 2023, "household_id": 1, "weight": 2.0,
            "household_size": 3, "demographic_group": "fixture",
            "labor_income": -100.0, "pension_income": 10.0,
            "rental_property_income": 5.0, "reported_interest_dividend_income": 2.0,
            "other_income": -10.0, "imputed_owner_rent": 20.0,
            "transfers_income": 4.5, "government_cash_benefits_reported": 5.0,
            "private_transfer_income": -0.5, "cash_consumption": 30.0,
            "other_cash_consumption": -5.0, "private_health_expense": 20.0,
            "private_education_expense": 10.0, "rent_paid": 5.0,
            "imputed_housing_consumption": 20.0, "private_transfers_paid": 1.0,
            "mortgage_interest": 2.0, "mortgage_principal": 3.0,
            "pension_contributions": 4.0, "training_fund_contributions": 1.0,
            "provident_fund_contributions": 1.0,
            "life_exec_insurance_contributions": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.pkl"
            pd.DataFrame([row]).to_pickle(path)
            households, facts, parity = build_hes_market_component_facts(path, 2023)
        self.assertEqual(len(households), 1)
        self.assertEqual(
            tuple(households.loc[0, ["source_year", "source_survey_id", "source_household_id"]]),
            (2023, 2023, 1),
        )
        self.assertLessEqual(parity.filter(like="gap_nis").abs().to_numpy().max(), 1e-8)
        amounts = facts.set_index("component_id").amount_nis
        self.assertEqual(amounts["income.market.self_employment_loss"], 100.0)
        self.assertEqual(amounts["income.market.private_transfer_loss"], 0.5)
        self.assertEqual(amounts["expense.private.cash.reconciliation"], 5.0)
        totals, _ = build_household_pnl(facts, "reported")
        self.assertAlmostEqual(totals.income_nis_monthly.iloc[0], -73.5)
        self.assertAlmostEqual(totals.expense_nis_monthly.iloc[0], 53.0)


if __name__ == "__main__":
    unittest.main()
