"""Export 2016-2018 pooled HES household market-P&L inputs in 2018 NIS.

This is an additive source adapter. It reads the original HES modules and the
locked pooled-weight file, and does not alter fiscal-allocation artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GDP_PC = {2016: 131467.0, 2017: 138702.0, 2018: 147208.0}
BASE_YEAR = 2018
CHILDCARE_CODES = {371013, 371021, 371039, 371047, 371161}
MISSING_CODES = {99999999, 99999998, 999999, 999998, 9999, 99, 9}


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("cp1255", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {path.name}")


def release(root: Path, year: int) -> Path:
    candidates = sorted(path for path in root.glob(f"H{year}*") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"expected one HES release directory for {year}; found {len(candidates)}")
    return candidates[0]


def module_path(folder: Path, module: str) -> Path:
    candidates = [
        path for path in folder.glob("*.csv")
        if f"data{module}".lower() in path.name.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one {module} CSV in {folder.name}; found {len(candidates)}")
    return candidates[0]


def normalise(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = frame.rename(columns={column: column.lower() for column in frame.columns})
    lower = {column.lower(): column for column in frame.columns}
    rename = {}
    for canonical in [
        "s_seker", "misparmb", "weight", "y_kalkali", "age_group",
        "prodcode", "schum", "nefashot", "nefashotad18",
    ]:
        if canonical in lower:
            rename[lower[canonical]] = canonical
    frame = frame.rename(columns=rename)
    frame["year"] = year
    return frame


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def clean_financial(values: pd.Series) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    return result.where(~result.isin(MISSING_CODES)).fillna(0.0)


def age_band(age_group: pd.Series) -> pd.Series:
    midpoints = {
        1: 2, 2: 7, 3: 12, 4: 16, 5: 21, 6: 27, 7: 32, 8: 37,
        9: 42, 10: 47, 11: 52, 12: 57, 13: 60.5, 14: 63,
        15: 65.5, 16: 68.5, 17: 72.5, 18: 77, 19: 82, 20: 87,
    }
    age = pd.to_numeric(age_group, errors="coerce").map(midpoints)
    return pd.cut(age, [-np.inf, 29.999, 39.999, 49.999, 64.999, np.inf],
                  labels=["<30", "30-39", "40-49", "50-64", "65+"]).astype(object)


def expense_category(code: str) -> str:
    number = int(code)
    if number in CHILDCARE_CODES:
        return "Childcare"
    if code.startswith("36"):
        return "Health"
    if code.startswith("371"):
        return "Education"
    return "Other"


def load_source_modules(hes_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modules: dict[str, list[pd.DataFrame]] = {name: [] for name in ["mb", "prat", "incmissim", "house", "prod"]}
    for year in GDP_PC:
        folder = release(hes_root, year)
        factor = GDP_PC[BASE_YEAR] / GDP_PC[year]
        for name in modules:
            frame = normalise(read_csv(module_path(folder, name)), year)
            source_survey = pd.to_numeric(frame["s_seker"], errors="coerce")
            if source_survey.isna().any() or not source_survey.eq(year).all():
                raise ValueError(f"HES {year} {name} has an unexpected S_Seker value")
            if name in {"mb", "incmissim", "house"}:
                for column in frame.columns:
                    if column in {"s_seker", "misparmb", "year"}:
                        continue
                    if column.lower().startswith(("i", "t", "c", "s")) or column.lower() in {"total_net", "net"}:
                        frame[column] = clean_financial(frame[column]) * factor
            if name == "prod":
                frame["schum"] = clean_financial(frame["schum"]) * factor
            modules[name].append(frame)
    return tuple(pd.concat(modules[name], ignore_index=True, sort=False) for name in ["mb", "prat", "incmissim", "house", "prod"])


def build(hes_root: Path, pooled_weights: Path, mortgage_shares: Path) -> pd.DataFrame:
    mb, people, income, house, product = load_source_modules(hes_root)
    keys = ["year", "s_seker", "misparmb"]
    weights = pd.read_pickle(pooled_weights)
    required_weights = keys + ["weight_adj", "nefashot", "sector"]
    missing = sorted(set(required_weights) - set(weights.columns))
    if missing:
        raise ValueError(f"pooled weights are missing columns: {missing}")
    if weights.duplicated(keys).any():
        raise ValueError("pooled weights contain duplicate household keys")

    head = people.loc[numeric(people, "y_kalkali").eq(1), keys + ["age_group"]].copy()
    if head.duplicated(keys).any():
        raise ValueError("economic head is not unique")
    head["age_band"] = age_band(head.age_group)

    keep_mb = keys + [c for c in [
        "total_net", "net", "c3kaspit", "i11", "i13", "i14", "i141", "i142",
        "i121012", "t21", "t22",
    ] if c in mb]
    frame = weights[required_weights].merge(mb[keep_mb], on=keys, validate="one_to_one")
    income_columns = [
        "i122", "i123", "i124", "s411017", "s411025", "s411033",
        "s411041", "s411058", "s411066",
    ]
    frame = frame.merge(income[keys + [c for c in income_columns if c in income]],
                        on=keys, how="left", validate="one_to_one")
    frame = frame.merge(house[keys + [c for c in ["c322", "c323"] if c in house]],
                        on=keys, how="left", validate="one_to_one")
    frame = frame.merge(head[keys + ["age_band"]], on=keys, how="left", validate="one_to_one")

    product = product.loc[product.prodcode.notna(), keys + ["prodcode", "schum"]].copy()
    product["code"] = pd.to_numeric(product.prodcode, errors="coerce").astype("Int64").astype(str)
    product = product.loc[product.code.ne("<NA>")]
    product = product.groupby(keys + ["code"], as_index=False).schum.sum()
    # Only health and education are needed. Remove aggregate product codes when
    # a more detailed descendant exists, using a vectorized ancestor lookup.
    product = product.loc[product.code.str.startswith(("36", "371"))].copy()
    ancestors = []
    for length in (1, 2, 3, 4):
        child = product.loc[product.code.str.len().gt(length), keys + ["code"]].copy()
        child["code"] = child.code.str[:length]
        ancestors.append(child)
    parent_keys = pd.concat(ancestors, ignore_index=True).drop_duplicates()
    parent_keys["has_child"] = True
    product = product.merge(parent_keys, on=keys + ["code"], how="left")
    product = product.loc[product.has_child.ne(True)].drop(columns="has_child")
    product["category"] = product.code.map(expense_category)
    categories = product.groupby(keys + ["category"], as_index=False).schum.sum().pivot(
        index=keys, columns="category", values="schum"
    ).fillna(0.0).reset_index()
    frame = frame.merge(categories, on=keys, how="left", validate="one_to_one")

    shares = pd.read_csv(mortgage_shares)
    shares = shares.loc[shares.breakdown_dimension.eq("Age group"), ["column_label", "mortgage_interest_share"]]
    shares = shares.rename(columns={"column_label": "age_band"})
    frame = frame.merge(shares, on="age_band", how="left", validate="many_to_one")
    if frame.mortgage_interest_share.isna().any():
        raise ValueError("mortgage interest share is missing for an age band")

    out = pd.DataFrame({
        "year": frame.year.astype(int),
        "s_seker": frame.s_seker.astype(int),
        "misparmb": frame.misparmb.astype(int),
        "survey_year": frame.year.astype(int),
        "household_id": frame.misparmb.astype(int),
        "weight": numeric(frame, "weight_adj"),
        "household_size": numeric(frame, "nefashot").clip(lower=1).round().astype(int),
        "demographic_group": frame.sector.map({1: "Haredi", 2: "Arab", 3: "Jewish non-Haredi"}).fillna("Unknown").astype(object),
    })
    out["labor_income"] = numeric(frame, "i11") + numeric(frame, "i122")
    out["pension_income"] = numeric(frame, "i13")
    out["rental_property_income"] = numeric(frame, "i123")
    out["reported_interest_dividend_income"] = numeric(frame, "i124")
    out["imputed_owner_rent"] = numeric(frame, "i121012")
    out["transfers_income"] = numeric(frame, "i14")
    out["government_cash_benefits_reported"] = numeric(frame, "i141") + numeric(frame, "i142")
    out["private_transfer_income"] = out.transfers_income - out.government_cash_benefits_reported
    explained = (
        out.labor_income + out.pension_income + out.rental_property_income
        + out.reported_interest_dividend_income + out.transfers_income + out.imputed_owner_rent
    )
    out["other_income"] = numeric(frame, "total_net") + numeric(frame, "t21") - explained
    out["cash_consumption"] = numeric(frame, "c3kaspit")
    out["private_health_expense"] = numeric(frame, "Health")
    out["private_education_expense"] = numeric(frame, "Education") + numeric(frame, "Childcare")
    out["rent_paid"] = numeric(frame, "c322")
    out["imputed_housing_consumption"] = numeric(frame, "c323")
    out["other_cash_consumption"] = (
        out.cash_consumption - out.private_health_expense - out.private_education_expense - out.rent_paid
    )
    out["private_transfers_paid"] = numeric(frame, "t22")
    mortgage_payment = numeric(frame, "s411033")
    out["mortgage_interest"] = mortgage_payment * numeric(frame, "mortgage_interest_share").clip(0, 1)
    out["mortgage_principal"] = mortgage_payment - out.mortgage_interest
    out["pension_contributions"] = numeric(frame, "s411066")
    out["training_fund_contributions"] = numeric(frame, "s411041")
    out["provident_fund_contributions"] = numeric(frame, "s411025")
    out["life_exec_insurance_contributions"] = numeric(frame, "s411017") + numeric(frame, "s411058")
    out.columns = pd.Index(list(out.columns), dtype=object)
    if out.duplicated(keys).any():
        raise ValueError("market export contains duplicate household keys")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hes-root", type=Path, required=True)
    parser.add_argument("--pooled-weights", type=Path, required=True)
    parser.add_argument("--mortgage-shares", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.hes_root, args.pooled_weights, args.mortgage_shares)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_pickle(args.output)
    print(f"wrote {len(output):,} household rows to {args.output.name}")


if __name__ == "__main__":
    main()
