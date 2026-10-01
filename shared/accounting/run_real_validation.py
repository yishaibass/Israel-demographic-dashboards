"""Run the real HES market/fiscal interoperability gate without writing microdata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hes_adapter import build_hes_fiscal_component_facts
from .market_adapter import build_hes_market_component_facts
from .pnl import build_household_pnl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--market-export", type=Path, required=True)
    parser.add_argument("--tax", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--embedded-basis", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    args = parser.parse_args()

    market_hh, market_facts, market_parity = build_hes_market_component_facts(
        args.market_export, args.year
    )
    fiscal_hh, fiscal_facts, controls, fiscal_parity = build_hes_fiscal_component_facts(
        args.tax, args.transfer, args.year,
        embedded_basis_pickle=args.embedded_basis,
        control_source_path=args.control_source,
    )
    matched = market_hh.merge(
        fiscal_hh, on=["household_key", "reference_year"], how="outer",
        suffixes=("_market", "_fiscal"), indicator=True, validate="one_to_one",
    )
    if matched._merge.ne("both").any():
        raise ValueError(f"market/fiscal household keys differ: {matched._merge.value_counts().to_dict()}")
    source_key_columns = ["source_year", "source_survey_id", "source_household_id"]
    source_key_mismatches = pd.Series(False, index=matched.index)
    for column in source_key_columns:
        source_key_mismatches |= matched[f"{column}_market"].ne(matched[f"{column}_fiscal"])
    if source_key_mismatches.any():
        raise ValueError(
            "market/fiscal source-key tuples differ; "
            f"mismatched households={int(source_key_mismatches.sum())}"
        )
    weight_gap = (matched.survey_weight_market - matched.survey_weight_fiscal).abs()
    if not np.allclose(matched.survey_weight_market, matched.survey_weight_fiscal, atol=1e-8, rtol=0):
        raise ValueError(f"market/fiscal weights differ; max gap={weight_gap.max()}")

    facts = pd.concat([market_facts, fiscal_facts], ignore_index=True)
    reported, reported_detail = build_household_pnl(facts, "reported")
    extended, extended_detail = build_household_pnl(facts, "allocated_extended")
    if reported[["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]].isna().any().any():
        raise ValueError("reported P&L contains missing totals")
    if extended[["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]].isna().any().any():
        raise ValueError("extended P&L contains missing totals")
    for name, frame in [("reported", reported), ("extended", extended)]:
        identity_gap = frame.income_nis_monthly - frame.expense_nis_monthly - frame.net_nis_monthly
        if not np.allclose(identity_gap, 0.0, atol=1e-8, rtol=0):
            raise ValueError(f"{name} P&L identity fails")

    weights = market_hh.set_index(["household_key", "reference_year"]).survey_weight
    def weighted_monthly(frame: pd.DataFrame, column: str) -> float:
        indexed = frame.set_index(["household_key", "reference_year"])
        return float((indexed[column] * weights).sum() / weights.sum())

    result = {
        "year": args.year,
        "households": len(market_hh),
        "market_facts": len(market_facts),
        "fiscal_facts": len(fiscal_facts),
        "combined_facts": len(facts),
        "max_weight_gap": float(weight_gap.max()),
        "source_key_rows_match": int((~source_key_mismatches).sum()),
        "source_key_mismatches": int(source_key_mismatches.sum()),
        "max_market_income_parity_gap": float(market_parity.market_income_gap_nis_monthly.abs().max()),
        "max_private_expense_parity_gap": float(market_parity.private_expense_gap_nis_monthly.abs().max()),
        "max_fiscal_component_parity_gap_nis_annual": float(fiscal_parity.gap_nis.abs().max()),
        "reported_average_monthly": {
            key: weighted_monthly(reported, key)
            for key in ["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]
        },
        "allocated_extended_average_monthly": {
            key: weighted_monthly(extended, key)
            for key in ["income_nis_monthly", "expense_nis_monthly", "net_nis_monthly"]
        },
        "reported_detail_rows": len(reported_detail),
        "allocated_extended_detail_rows": len(extended_detail),
        "controls": controls[["national_control_id", "amount_nis_annual"]].to_dict("records"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
