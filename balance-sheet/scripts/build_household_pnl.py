"""Build an equal-year 2021-2023 household P&L and 2023 wealth overlay.

The script deliberately does not household-join the HES and longitudinal files.
HES supplies flows; the 2023 longitudinal cross-section supplies age-band wealth
stocks. All published flow results are weighted monthly household means (NIS).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import build_household_frame


AGE_LABELS = ["<30", "30-39", "40-49", "50-64", "65+"]
AGE_ORDER = {label: i for i, label in enumerate(AGE_LABELS)}
GROUP_ORDER = ["Haredi", "Dati (Religious)", "Masorti", "Hiloni (Secular)", "Arab"]
SES_LABELS = ["1-2", "3-4", "5-6", "7-8", "9-10"]
DIRECT_CHILD_CODES = {309203,309369,334011,334037,341214,341222,341263,341271,341297,341305,353011,353029,353037,353045,353052,353060,353078,353086,353094,353102,353193,354076,354084,354126,355032,358085,358093,358101,358119,358168,371013,371021,371039,371047,371054,371062,371070,371088,371096,371146,371161,371179,371237,371302,371310,371328,371336,374066,392035,396069,396077}
CHILDCARE_CODES = {371013, 371021, 371039, 371047, 371161}
CAR_PURCHASE_CODES = {383026, 383265, 383406, 383430}
EXPENSE_CATEGORIES = ["Food", "Housing service", "Restaurants", "Travel & vacations", "Tobacco & smoking", "Car purchases", "Other durables", "Health", "Childcare", "Education", "Transport excluding cars", "Leisure & luxury", "Clothing & footwear", "Personal care & miscellaneous", "Child personal care & baby products", "Fees & professional services", "Donations", "Other consumption"]
# Only categories with a defensible per-person relationship are split between
# adults and children. Shared/fixed household purchases stay unallocated rather
# than being mechanically assigned using household headcount.
PERSON_VARIABLE_CATEGORIES = {
    "Food", "Restaurants", "Travel & vacations", "Health", "Childcare",
    "Education", "Leisure & luxury",
}

AGE_GROUP_MIDPOINT = {
    1: 2, 2: 7, 3: 12, 4: 16, 5: 21, 6: 27, 7: 32, 8: 37,
    9: 42, 10: 47, 11: 52, 12: 57, 13: 60.5, 14: 63,
    15: 65.5, 16: 68.5, 17: 72.5, 18: 77, 19: 82, 20: 87,
}


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age, [-np.inf, 29.999, 39.999, 49.999, 64.999, np.inf],
        labels=AGE_LABELS, ordered=True,
    ).astype("string")


def ses_band(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    return pd.cut(v, [0, 2, 4, 6, 8, 10], labels=SES_LABELS, ordered=True).astype("string")


def iter_breakdowns(df: pd.DataFrame):
    for order, label in enumerate(AGE_LABELS):
        yield "Age group", label, order, df["age_band"].eq(label)
    for order, label in enumerate(GROUP_ORDER):
        yield "Religious group", label, order, df["demographic_group"].eq(label)
    for order, label in enumerate(SES_LABELS):
        yield "Socioeconomic cluster", label, order, df["ses_band"].eq(label)

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="cp1255", low_memory=False)


def normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    canonical = {name.lower(): name for name in [
        "S_Seker", "MisparMb", "Weight", "Y_Kalkali", "Age_Group",
        "ProdCode", "Schum", "Nefashot", "NefashotAd18", "Nefashot18Up",
        "Baalut", "Rented_D", "Total_Net", "Net", "Nationality",
        "RamatDatiyut", "Cluster",
    ]}
    rename = {c: canonical[c.lower()] for c in df.columns if c.lower() in canonical}
    df = df.rename(columns=rename)
    return df.rename(columns={"S_Seker": "survey_year", "MisparMb": "household_id"})


GDP_PC_NOMINAL = {2021: 167000.0, 2022: 183000.0, 2023: 191000.0}
GDP_PC_BASE_YEAR = 2023


def gdp_pc_factor(years: pd.Series) -> pd.Series:
    base = GDP_PC_NOMINAL[GDP_PC_BASE_YEAR]
    mapped = pd.to_numeric(years, errors="coerce").map(GDP_PC_NOMINAL)
    if mapped.isna().any():
        raise ValueError("Missing nominal GDP-per-capita factor for a survey year")
    return base / mapped


def is_monetary_flow_column(name: str) -> bool:
    lower = name.lower()
    if lower in {"survey_year", "household_id", "weight"}:
        return False
    return lower in {"total_net", "net"} or lower.startswith(("i", "t", "c", "s"))


def load_hes_year(hes_root: Path, year: int) -> tuple[pd.DataFrame, ...]:
    release = hes_root / f"H{year}1021"
    prefix = f"H{year}1021data"
    frames = []
    for suffix in ["mb", "prat", "incmissim", "house", "prod"]:
        frame = normalise_keys(read_csv(release / f"{prefix}{suffix}.csv"))
        frame["survey_year"] = year
        frames.append(frame)
    return tuple(frames)


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def wmean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def expense_category(code: str) -> str:
    n = int(code)
    if n in CHILDCARE_CODES: return "Childcare"
    if code.startswith("308"): return "Restaurants"
    if code.startswith(("374", "382")): return "Travel & vacations"
    if n in CAR_PURCHASE_CODES: return "Car purchases"
    if code.startswith("391"): return "Tobacco & smoking"
    if n in {392035, 396069, 396077}: return "Child personal care & baby products"
    if n == 397067: return "Donations"
    if code.startswith(("394", "397")): return "Fees & professional services"
    if code.startswith(("30", "31")): return "Food"
    if code.startswith(("32", "33")): return "Housing service"
    if code.startswith("34") or (code.startswith("375") and not code.startswith("3751")): return "Other durables"
    if code.startswith("36"): return "Health"
    if code.startswith("371"): return "Education"
    if code.startswith("38"): return "Transport excluding cars"
    if code.startswith(("372", "373", "376", "377", "378", "395")) or code.startswith("3751"): return "Leisure & luxury"
    if code.startswith("35"): return "Clothing & footwear"
    if code.startswith("39"): return "Personal care & miscellaneous"
    return "Other consumption"


def aggregate_expense_categories(prod: pd.DataFrame, hh: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    p = prod.loc[prod["ProdCode"].notna(), keys + ["ProdCode", "Schum"]].copy()
    p["code"] = p["ProdCode"].astype("int64").astype(str)
    p = p.groupby(keys + ["code"], as_index=False)["Schum"].sum()

    def keep_terminal(g: pd.DataFrame) -> pd.Series:
        codes = g["code"].tolist()
        return pd.Series([not any(len(o) > len(c) and o.startswith(c) for o in codes) for c in codes], index=g.index)

    terminal = p.groupby(keys, group_keys=False).apply(keep_terminal)
    p = p.loc[terminal.astype(bool)].copy()
    p["category"] = p["code"].map(expense_category)
    fields = keys + ["age_band", "demographic_group", "ses_band", "weight", "household_size", "children", "c3"]
    detail = p.merge(hh[fields], on=keys, how="inner", validate="m:1")
    share = (detail["children"] / detail["household_size"]).clip(0, 1)
    is_child = detail["code"].astype("int64").isin(DIRECT_CHILD_CODES)
    is_variable = detail["category"].isin(PERSON_VARIABLE_CATEGORIES)
    detail["child_expense"] = np.where(is_variable, np.where(is_child, detail["Schum"], detail["Schum"] * share), 0.0)
    detail["parent_expense"] = np.where(is_variable, detail["Schum"] - detail["child_expense"], 0.0)
    detail["household_fixed_expense"] = np.where(is_variable, 0.0, detail["Schum"])
    detail["value_nis"] = detail["Schum"]
    measures = ["value_nis", "parent_expense", "child_expense", "household_fixed_expense"]
    by_hh = detail.groupby(keys + ["category"], as_index=False)[measures].sum()
    leaf = detail.groupby(keys, as_index=False)["Schum"].sum().rename(columns={"Schum": "leaf_total"})
    residual = hh[fields].merge(leaf, on=keys, how="left")
    residual["leaf_total"] = residual["leaf_total"].fillna(0.0)
    residual["value_nis"] = num(residual, "c3") - residual["leaf_total"]
    residual["child_expense"] = 0.0; residual["parent_expense"] = 0.0
    residual["household_fixed_expense"] = residual["value_nis"]
    residual["category"] = "Other consumption"
    by_hh = pd.concat([by_hh, residual[keys + ["category"] + measures]], ignore_index=True)
    by_hh = by_hh.groupby(keys + ["category"], as_index=False)[measures].sum()
    wide = by_hh.pivot(index=keys, columns="category", values=measures).fillna(0.0)
    wide.columns = [f"{m}__{c}" for m, c in wide.columns]
    base = hh[fields].merge(wide.reset_index(), on=keys, how="left").fillna(0.0)
    rows = []
    for dimension, label, column_order, mask in iter_breakdowns(base):
        g = base.loc[mask]
        for category_order, category in enumerate(EXPENSE_CATEGORIES):
            row = {"breakdown_dimension": dimension, "column_label": label, "column_order": column_order,
                   "category": category, "category_order": category_order,
                   "sample_households": int(len(g)), "weighted_households": float(g["weight"].sum())}
            for measure in measures:
                cell_w = g["weight"]
                row[measure] = wmean(g.get(f"{measure}__{category}", pd.Series(0.0, index=g.index)), cell_w)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_flows(hh: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "head_age_proxy", "household_size", "children", "has_children", "owner",
        "labor_income", "pension_income", "rental_property_income", "reported_interest_dividend_income", "other_asset_income", "transfers_income",
        "other_income", "imputed_owner_rent", "gross_economic_income", "direct_taxes",
        "disposable_economic_income", "cash_disposable_income", "cash_consumption",
        "private_transfers_paid", "cash_residual_saving", "cbs_economic_saving_before_private_transfers",
        "parent_consumption", "child_consumption", "rent_paid", "imputed_housing_consumption",
        "mortgage_payment", "has_mortgage_payment", "pension_contributions", "modeled_employer_pension_contributions", "training_fund_contributions",
        "provident_fund_contributions", "life_exec_insurance_contributions", "other_observed_financial_saving",
        "cbs_total_cash_saving", "contractual_financial_saving", "real_estate_investment",
        "home_purchase_net", "other_property_purchase", "home_capital_improvements",
        "other_saving_and_debt_flows", "other_debt_net_repayment", "housing_debt_movement", "unclassified_debt_movement", "unclassified_s44_movement", "other_housing_loan_repayment",
        "household_asset_sales_net", "household_loans_extended",
        "adult_consumption_per_adult", "child_consumption_per_child", "child_consumption_equivalence_scale",
    ]
    rows = []
    for dimension, label, column_order, mask in iter_breakdowns(hh):
        g = hh.loc[mask]
        row = {"breakdown_dimension": dimension, "column_label": label, "column_order": column_order,
               "sample_households": int(len(g)), "weighted_households": float(g["weight"].sum())}
        cell_w = g["weight"]
        for c in mean_cols: row[c] = wmean(g[c], cell_w)
        row["years_in_cell"] = int(g.loc[cell_w > 0, "survey_year"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_wealth() -> tuple[pd.DataFrame, dict]:
    p, _ = build_household_frame()
    p = p.loc[p["year"] == 2023].copy()
    p["age_band"] = age_band(pd.to_numeric(p["age"], errors="coerce"))
    p["ses_band"] = ses_band(p["ses10"])
    p["w"] = pd.to_numeric(p["weight_hh"], errors="coerce")
    p["demographic_group"] = p["grp"].map({1: "Haredi", 2: "Dati (Religious)", 3: "Masorti", 4: "Hiloni (Secular)", 5: "Arab"})
    specs = {"mortgage_balance_overlay": "mortg", "financial_assets_overlay": "tot_fin",
             "deposits_savings_overlay": "dep_sav", "nonbank_investments_overlay": "invest",
             "pension_wealth_overlay": "pension", "primary_home_assets_overlay": "home",
             "additional_property_assets_overlay": "addprop", "land_assets_overlay": "land",
             "housing_assets_overlay": "tot_re_full", "total_assets_overlay": "tot_a",
             "total_debt_overlay": "tot_l", "net_worth_overlay": "nw"}
    rows = []
    for dimension, label, column_order, mask in iter_breakdowns(p):
        g = p.loc[mask]
        row = {"breakdown_dimension": dimension, "column_label": label, "column_order": column_order}
        for out, src in specs.items(): row[out] = wmean(g[src], g["w"])
        holders = g.loc[pd.to_numeric(g["mortg"], errors="coerce") > 0]
        row["mortgage_balance_mortgage_hh"] = wmean(holders["mortg"], holders["w"])
        row["home_value_mortgage_hh"] = wmean(holders["home"], holders["w"])
        row["mortgage_prevalence_wealth"] = float(holders["w"].sum() / g["w"].sum()) if g["w"].sum() else float("nan")
        rows.append(row)
    coverage = {c: float(p.loc[p[c].notna(), "w"].sum() / p["w"].sum()) for c in specs.values()}
    return pd.DataFrame(rows), coverage

def build(project: Path, mortgage_rate: float, hes_root: Path, deposit_return: float = 0.01,
          investment_return: float = 0.04, pension_return: float = 0.04) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    releases = [load_hes_year(hes_root, year) for year in (2021, 2022, 2023)]
    mb, people, inc, house, prod = [
        pd.concat([release[i] for release in releases], ignore_index=True, sort=False)
        for i in range(5)
    ]
    # Replicate the Karlinsky 2023 preparation: express every monetary flow in
    # 2023 nominal-GDP-per-capita units before constructing accounting identities.
    for frame in (mb, inc, house):
        factor = gdp_pc_factor(frame["survey_year"])
        for column in [c for c in frame.columns if is_monetary_flow_column(c)]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
    prod["Schum"] = pd.to_numeric(prod["Schum"], errors="coerce") * gdp_pc_factor(prod["survey_year"])

    keys = ["survey_year", "household_id"]
    heads = people.loc[num(people, "Y_Kalkali") == 1, keys + ["Age_Group"]].copy()
    if heads.duplicated(keys).any():
        raise ValueError("Economic head is not unique within household")
    heads["head_age_proxy"] = num(heads, "Age_Group").map(AGE_GROUP_MIDPOINT)

    keep_mb = keys + [
        "Weight", "Nefashot", "NefashotAd18", "Nefashot18Up", "Baalut", "Rented_D",
        "i1", "i11", "i12", "i13", "i14", "t2", "t21", "t22", "c3",
        "Total_Net", "Net", "i1Kaspit", "c3Kaspit", "Nationality", "RamatDatiyut", "Cluster",
    ]
    # Some totals are absent from the household file but present in incmissim.
    hh = mb[[c for c in keep_mb if c in mb]].merge(heads, on=keys, validate="1:1")
    hh = hh.merge(inc, on=keys, how="left", validate="1:1", suffixes=("", "_inc"))
    hh = hh.merge(house[keys + [c for c in ["c322", "c323"] if c in house]],
                  on=keys, how="left", validate="1:1")
    prod["ProdCode"] = pd.to_numeric(prod["ProdCode"], errors="coerce")
    prod["Schum"] = pd.to_numeric(prod["Schum"], errors="coerce").fillna(0.0)
    child = (prod.loc[prod["ProdCode"].isin(DIRECT_CHILD_CODES)]
             .groupby(keys, as_index=False)["Schum"].sum()
             .rename(columns={"Schum": "direct_child_consumption"}))
    hh = hh.merge(child, on=keys, how="left", validate="1:1")

    hh["weight_original"] = num(hh, "Weight")
    hh["gdp_pc_nominal"] = pd.to_numeric(hh["survey_year"], errors="coerce").map(GDP_PC_NOMINAL)
    hh["infl_to_2023"] = gdp_pc_factor(hh["survey_year"])
    hh["weight_pool_pre_norm"] = hh["weight_original"] * hh["infl_to_2023"] / 3.0
    target_weight = hh.loc[hh["survey_year"] == GDP_PC_BASE_YEAR, "weight_original"].sum()
    weight_scale = target_weight / hh["weight_pool_pre_norm"].sum()
    hh["weight_adj"] = hh["weight_pool_pre_norm"] * weight_scale
    hh["weight"] = hh["weight_adj"]
    hh["household_size"] = num(hh, "Nefashot").clip(lower=1)
    hh["children"] = num(hh, "NefashotAd18").clip(lower=0)
    hh["adults"] = (hh["household_size"] - hh["children"]).clip(lower=1)
    hh["has_children"] = (hh["children"] > 0).astype(float)
    hh["owner"] = (num(hh, "i121012") > 0).astype(float)
    hh["age_band"] = age_band(hh["head_age_proxy"])
    hh["ses_band"] = num(hh, "Cluster").map({1: "1-2", 2: "3-4", 3: "5-6", 4: "7-8", 5: "9-10"}).astype("string")
    nationality = num(hh, "Nationality")
    religiosity = num(hh, "RamatDatiyut")
    hh["demographic_group"] = "Hiloni (Secular)"
    hh.loc[(nationality == 1) & (religiosity == 2), "demographic_group"] = "Masorti"
    hh.loc[(nationality == 1) & (religiosity == 3), "demographic_group"] = "Dati (Religious)"
    hh.loc[(nationality == 1) & (religiosity == 4), "demographic_group"] = "Haredi"
    hh.loc[nationality == 2, "demographic_group"] = "Arab"

    # Employer-provided car income (i122) is labor compensation in kind.
    hh["labor_income"] = num(hh, "i11") + num(hh, "i122")
    hh["pension_income"] = num(hh, "i13")
    hh["rental_property_income"] = num(hh, "i123")
    hh["reported_interest_dividend_income"] = num(hh, "i124")
    hh["other_asset_income"] = hh["rental_property_income"] + hh["reported_interest_dividend_income"]
    hh["transfers_income"] = num(hh, "i14")
    hh["imputed_owner_rent"] = num(hh, "i121012")
    reported_gross = num(hh, "Total_Net") + num(hh, "t21")
    explained_gross = (hh["labor_income"] + hh["pension_income"]
                       + hh["other_asset_income"] + hh["transfers_income"]
                       + hh["imputed_owner_rent"])
    hh["other_income"] = reported_gross - explained_gross
    hh["gross_economic_income"] = reported_gross
    hh["direct_taxes"] = num(hh, "t21")
    hh["disposable_economic_income"] = num(hh, "Total_Net")
    hh["cash_disposable_income"] = num(hh, "Net")
    hh["private_transfers_paid"] = num(hh, "t22")
    hh["cash_consumption"] = num(hh, "c3Kaspit")
    hh["cash_residual_saving"] = hh["cash_disposable_income"] - hh["cash_consumption"] - hh["private_transfers_paid"]
    hh["cbs_economic_saving_before_private_transfers"] = hh["disposable_economic_income"] - num(hh, "c3")

    hh["rent_paid"] = num(hh, "c322")
    hh["imputed_housing_consumption"] = num(hh, "c323")
    direct_child = np.minimum(num(hh, "direct_child_consumption"), num(hh, "c3"))
    shared_nonhousing = (num(hh, "c3") - hh["rent_paid"]
                         - hh["imputed_housing_consumption"] - direct_child).clip(lower=0)
    child_share = (hh["children"] / hh["household_size"]).clip(0, 1)
    eq_child_share = ((0.3 * hh["children"]) /
                      (1 + 0.5 * (hh["adults"] - 1) + 0.3 * hh["children"])).clip(0, 1)
    hh["child_consumption"] = direct_child + shared_nonhousing * child_share
    hh["parent_consumption"] = shared_nonhousing * (1 - child_share)
    hh["child_consumption_equivalence_scale"] = direct_child + shared_nonhousing * eq_child_share
    hh["adult_consumption_per_adult"] = hh["parent_consumption"] / hh["adults"]
    hh["child_consumption_per_child"] = np.where(
        hh["children"] > 0, hh["child_consumption"] / hh["children"], 0.0
    )

    hh["mortgage_payment"] = num(hh, "s411033")
    hh["has_mortgage_payment"] = (hh["mortgage_payment"] > 0).astype(float)
    hh["pension_contributions"] = num(hh, "s411066")
    hh["modeled_employer_pension_contributions"] = hh["pension_contributions"] * (0.125 / 0.06)
    hh["training_fund_contributions"] = num(hh, "s411041")
    hh["provident_fund_contributions"] = num(hh, "s411025")
    hh["life_exec_insurance_contributions"] = num(hh, "s411017") + num(hh, "s411058")
    # CBS released saving control: s4 = s41 + s42 + s44 (within rounding).
    hh["cbs_total_cash_saving"] = num(hh, "s4")
    hh["contractual_financial_saving"] = num(hh, "s41")
    known_contractual = (hh["mortgage_payment"] + hh["pension_contributions"]
                         + hh["training_fund_contributions"] + hh["provident_fund_contributions"]
                         + hh["life_exec_insurance_contributions"])
    hh["other_observed_financial_saving"] = hh["contractual_financial_saving"] - known_contractual
    hh["real_estate_investment"] = num(hh, "s42")
    hh["home_purchase_net"] = num(hh, "s421")
    hh["other_property_purchase"] = num(hh, "s422")
    hh["home_capital_improvements"] = num(hh, "s423")
    hh["other_saving_and_debt_flows"] = num(hh, "s44")
    hh["other_debt_net_repayment"] = num(hh, "s442")
    hh["housing_debt_movement"] = num(hh, "s442012")
    hh["other_housing_loan_repayment"] = num(hh, "s442038")
    # Keep unclassified debt visible for audit; do not call it net-worth saving.
    hh["unclassified_debt_movement"] = (
        hh["other_debt_net_repayment"] - hh["housing_debt_movement"]
        - hh["other_housing_loan_repayment"]
    )
    hh["household_asset_sales_net"] = num(hh, "s443")
    hh["household_loans_extended"] = num(hh, "s444")
    hh["unclassified_s44_movement"] = (
        hh["other_saving_and_debt_flows"] - hh["other_debt_net_repayment"]
        - hh["household_asset_sales_net"] - hh["household_loans_extended"]
    )
    expense_categories = aggregate_expense_categories(prod, hh, keys)

    result = aggregate_flows(hh)
    gemach = (expense_categories.loc[expense_categories["category"].eq("Donations"),
                 ["breakdown_dimension", "column_label", "column_order", "value_nis"]]
              .rename(columns={"value_nis": "community_gemach_saving"}))
    result = result.merge(gemach, on=["breakdown_dimension", "column_label", "column_order"], how="left", validate="1:1")
    result["community_gemach_saving"] = result["community_gemach_saving"].fillna(0.0)
    wealth, coverage = aggregate_wealth()
    result = result.merge(wealth, on=["breakdown_dimension", "column_label", "column_order"], validate="1:1")
    result["other_financial_assets_overlay"] = (
        result["financial_assets_overlay"] - result["pension_wealth_overlay"]
    ).clip(lower=0)
    result["other_debt_overlay"] = (
        result["total_debt_overlay"] - result["mortgage_balance_overlay"]
    ).clip(lower=0)
    result["mortgage_payment_prevalence"] = result["has_mortgage_payment"].clip(lower=0, upper=1)
    result["mortgage_payment_mortgage_hh"] = np.where(
        result["mortgage_payment_prevalence"] > 0,
        result["mortgage_payment"] / result["mortgage_payment_prevalence"], 0.0
    )
    result["mortgage_interest_mortgage_hh"] = (
        result["mortgage_balance_mortgage_hh"] * mortgage_rate / 12
    )
    result["mortgage_interest_share"] = np.where(
        result["mortgage_payment_mortgage_hh"] > 0,
        np.minimum(result["mortgage_interest_mortgage_hh"] / result["mortgage_payment_mortgage_hh"], 1.0), 0.0
    )
    result["mortgage_interest"] = result["mortgage_payment"] * result["mortgage_interest_share"]
    result["mortgage_principal"] = (result["mortgage_payment"] - result["mortgage_interest"]).clip(lower=0)
    result["housing_consumption"] = result["rent_paid"] + result["imputed_housing_consumption"]
    result["gross_imputed_housing_income"] = result["imputed_housing_consumption"]
    result["imputed_housing_asset_income"] = result["gross_imputed_housing_income"] - result["mortgage_interest"]
    result["equity_housing_service"] = result["imputed_housing_asset_income"]
    # Investment-property income remains the cash amount reported in the survey; no property-yield imputation.
    result["imputed_other_real_estate_income"] = 0.0
    result["imputed_deposit_savings_income"] = result["deposits_savings_overlay"] * deposit_return / 12
    result["imputed_nonbank_investment_income"] = result["nonbank_investments_overlay"] * investment_return / 12
    result["imputed_financial_asset_income"] = result["imputed_deposit_savings_income"] + result["imputed_nonbank_investment_income"]
    result["imputed_pension_asset_income"] = result["pension_wealth_overlay"] * pension_return / 12
    result["pension_income_adjustment"] = result["imputed_pension_asset_income"] - result["pension_income"]
    result["pension_economic_income"] = result["pension_income"] + result["pension_income_adjustment"]
    result["total_imputed_asset_income"] = result["imputed_housing_asset_income"] + result["imputed_financial_asset_income"] + result["imputed_pension_asset_income"]
    replacement = (-result["imputed_owner_rent"] - result["reported_interest_dividend_income"]
                   - result["pension_income"] + result["total_imputed_asset_income"])
    result["labor_economic_income"] = result["labor_income"] + result["modeled_employer_pension_contributions"]
    result["adjusted_gross_economic_income"] = result["gross_economic_income"] + replacement + result["modeled_employer_pension_contributions"]
    result["adjusted_disposable_economic_income"] = result["disposable_economic_income"] + replacement + result["modeled_employer_pension_contributions"]
    displayed_income = (result["labor_economic_income"] + result["transfers_income"] + result["pension_economic_income"]
                        + result["rental_property_income"] + result["imputed_housing_asset_income"]
                        + result["imputed_financial_asset_income"] + result["other_income"])
    result["income_reconciliation_gap"] = result["adjusted_gross_economic_income"] - displayed_income
    result["total_economic_expenses"] = (result["parent_consumption"] + result["child_consumption"] + result["housing_consumption"] + result["private_transfers_paid"] - result["community_gemach_saving"])
    result["total_saving"] = result["adjusted_disposable_economic_income"] - result["total_economic_expenses"]
    funded = (result["mortgage_principal"] + result["pension_contributions"]
              + result["training_fund_contributions"] + result["provident_fund_contributions"]
              + result["life_exec_insurance_contributions"]
              + result["other_observed_financial_saving"] + result["real_estate_investment"]
              + result["other_saving_and_debt_flows"] + result["community_gemach_saving"])
    result["cash_income"] = result["cash_disposable_income"]
    result["cash_living_outflows"] = (result["cash_consumption"] - result["community_gemach_saving"] + result["private_transfers_paid"]
                                      + result["mortgage_interest"])
    result["cash_flow_before_funded_saving"] = result["cash_income"] - result["cash_living_outflows"]
    result["noncash_income_bridge"] = result["adjusted_disposable_economic_income"] - result["cash_income"]
    result["noncash_expense_bridge"] = result["total_economic_expenses"] - result["cash_living_outflows"]
    result["noncash_saving_bridge"] = result["total_saving"] - result["cash_flow_before_funded_saving"]
    result["pension_training_provident_life_contributions"] = (
        result["pension_contributions"] + result["training_fund_contributions"]
        + result["provident_fund_contributions"] + result["life_exec_insurance_contributions"]
    )
    result["total_funded_cash_saving"] = funded
    result["free_cash_flow"] = result["cash_flow_before_funded_saving"] - result["total_funded_cash_saving"]
    result["cash_flow_identity_gap"] = result["cash_income"] - (
        result["cash_living_outflows"] + result["total_funded_cash_saving"] + result["free_cash_flow"]
    )
    result["survey_net_cash_surplus"] = (result["cash_disposable_income"] - result["cash_consumption"]
        - result["private_transfers_paid"] - result["mortgage_payment"]
        - result["pension_contributions"] - result["training_fund_contributions"]
        - result["provident_fund_contributions"] - result["life_exec_insurance_contributions"]
        - result["other_observed_financial_saving"] - result["real_estate_investment"]
        - result["other_saving_and_debt_flows"])
    result["survey_cash_reconciliation_gap"] = result["free_cash_flow"] - result["survey_net_cash_surplus"]
    # Indirect household cash-flow classification. Pension and long-term saving
    # are investing flows; mortgage principal is grouped with real-estate equity
    # investment together with community saving; mortgage principal and signed
    # housing-debt movements are financing.
    result["operating_cash_receipts"] = result["cash_income"] - result["pension_income"]
    result["operating_cash_expenses"] = -(
        result["cash_consumption"] - result["community_gemach_saving"]
        + result["private_transfers_paid"] + result["mortgage_interest"]
    )
    result["remove_noncash_income_cash_flow"] = -result["noncash_income_bridge"]
    result["remove_pension_withdrawals_from_operations"] = -result["pension_income"]
    result["operating_income_cash_adjustment"] = (
        result["remove_noncash_income_cash_flow"]
        + result["remove_pension_withdrawals_from_operations"]
    )
    result["remove_noncash_expenses_cash_flow"] = result["noncash_expense_bridge"]
    result["operating_expense_cash_adjustment"] = result["remove_noncash_expenses_cash_flow"]
    result["cash_flow_operating_activities"] = (
        result["operating_cash_receipts"] + result["operating_cash_expenses"]
    )
    result["property_investment_cash_flow"] = -result["real_estate_investment"]
    result["financial_investment_cash_flow"] = -result["other_observed_financial_saving"]
    result["household_asset_sales_cash_flow"] = -result["household_asset_sales_net"]
    result["household_loans_extended_cash_flow"] = -result["household_loans_extended"]
    result["other_investing_cash_flow"] = (
        result["financial_investment_cash_flow"] + result["household_asset_sales_cash_flow"]
        + result["household_loans_extended_cash_flow"]
    )
    result["pension_withdrawals_cash_flow"] = result["pension_income"]
    result["long_term_savings_contributions_cash_flow"] = -(
        result["pension_contributions"] + result["training_fund_contributions"]
        + result["provident_fund_contributions"] + result["life_exec_insurance_contributions"]
    )
    result["mortgage_principal_cash_flow"] = -result["mortgage_principal"]
    result["housing_debt_financing_cash_flow"] = -result["other_debt_net_repayment"]
    result["apartment_debt_movement_cash_flow"] = -result["housing_debt_movement"]
    result["other_housing_loan_cash_flow"] = -result["other_housing_loan_repayment"]
    result["unclassified_debt_cash_flow"] = -result["unclassified_debt_movement"]
    result["pension_investing_cash_flow"] = (
        result["pension_withdrawals_cash_flow"] - result["pension_contributions"]
    )
    result["other_financial_savings_investing_cash_flow"] = -(
        result["training_fund_contributions"] + result["provident_fund_contributions"]
        + result["life_exec_insurance_contributions"] + result["other_observed_financial_saving"]
    )
    result["other_asset_investing_cash_flow"] = (
        result["household_asset_sales_cash_flow"] + result["household_loans_extended_cash_flow"]
    )
    result["community_real_estate_cash_flow"] = -result["community_gemach_saving"]
    result["real_estate_investing_cash_flow"] = (
        result["property_investment_cash_flow"] + result["community_real_estate_cash_flow"]
    )
    result["cash_flow_investing_activities"] = (
        result["real_estate_investing_cash_flow"] + result["pension_investing_cash_flow"]
        + result["other_financial_savings_investing_cash_flow"]
        + result["other_asset_investing_cash_flow"]
    )
    result["cash_flow_financing_activities"] = (
        result["mortgage_principal_cash_flow"] + result["housing_debt_financing_cash_flow"]
    )
    result["net_change_in_unallocated_cash"] = (
        result["cash_flow_operating_activities"] + result["cash_flow_investing_activities"]
        + result["cash_flow_financing_activities"]
    )
    result["company_cash_flow_reconciliation_gap"] = (
        result["net_change_in_unallocated_cash"] - result["survey_net_cash_surplus"]
    )
    result["pension_balance_change"] = (
        result["pension_contributions"] + result["modeled_employer_pension_contributions"]
        + result["imputed_pension_asset_income"] - result["pension_income"]
    )
    # Acquisition and apartment-debt drawdown are netted once. Scheduled
    # mortgage principal is a separate recurring liability reduction. Aggregate
    # s44/s442 and s442038 stay memoranda because their economic split is unknown.
    result["net_property_acquisition_equity"] = (
        result["real_estate_investment"] + result["housing_debt_movement"]
    )
    result["net_property_equity_investment"] = result["net_property_acquisition_equity"]
    result["real_estate_transaction_allocation"] = (
        result["mortgage_principal"] + result["net_property_acquisition_equity"]
        + result["other_housing_loan_repayment"] + result["unclassified_debt_movement"]
        + result["community_gemach_saving"]
    )
    result["real_estate_and_housing_financing"] = result["real_estate_transaction_allocation"]
    result["financial_asset_saving_allocation"] = (
        result["pension_balance_change"] + result["training_fund_contributions"]
        + result["provident_fund_contributions"] + result["life_exec_insurance_contributions"]
        + result["other_observed_financial_saving"]
    )
    result["other_asset_and_lending_flows"] = (
        result["household_asset_sales_net"] + result["household_loans_extended"]
    )
    result["other_financial_asset_saving_allocation"] = (
        result["training_fund_contributions"] + result["provident_fund_contributions"]
        + result["life_exec_insurance_contributions"] + result["other_observed_financial_saving"]
    )
    result["real_estate_saving_allocation"] = result["real_estate_transaction_allocation"]
    result["transactional_accrual_saving"] = result["total_saving"]
    result["identified_transaction_allocation"] = (
        result["pension_balance_change"] + result["other_financial_asset_saving_allocation"]
        + result["real_estate_transaction_allocation"]
    )
    # Donations/community payments and ambiguous s44 movements stay memoranda
    # until a recoverable asset or liability movement can be evidenced.
    result["total_net_worth_saving_allocation"] = result["identified_transaction_allocation"]
    result["unallocated_transactional_saving"] = (
        result["transactional_accrual_saving"] - result["identified_transaction_allocation"]
    )
    result["residual_saving"] = result["unallocated_transactional_saving"]
    result["savings_rate_pct"] = 100 * result["total_saving"] / result["adjusted_disposable_economic_income"]
    result["pnl_identity_gap"] = result["adjusted_disposable_economic_income"] - (
        result["total_economic_expenses"] + result["total_saving"]
    )
    result["saving_decomposition_gap"] = result["total_saving"] - (result["total_net_worth_saving_allocation"] + result["residual_saving"])

    output_cols = [
        "breakdown_dimension", "column_label", "column_order", "sample_households", "years_in_cell", "weighted_households",
        "head_age_proxy", "household_size", "children", "has_children", "owner",
        "labor_income", "labor_economic_income", "pension_income", "pension_economic_income", "pension_income_adjustment", "rental_property_income", "reported_interest_dividend_income", "other_asset_income", "transfers_income",
        "other_income", "imputed_owner_rent", "gross_imputed_housing_income", "imputed_housing_asset_income", "equity_housing_service",
        "imputed_other_real_estate_income", "imputed_deposit_savings_income", "imputed_nonbank_investment_income", "imputed_financial_asset_income", "imputed_pension_asset_income", "total_imputed_asset_income", "income_reconciliation_gap",
        "gross_economic_income", "adjusted_gross_economic_income", "direct_taxes", "adjusted_disposable_economic_income",
        "disposable_economic_income", "cash_disposable_income", "cash_consumption",
        "private_transfers_paid", "cash_residual_saving",
        "cbs_economic_saving_before_private_transfers", "parent_consumption", "child_consumption",
        "rent_paid", "imputed_housing_consumption", "housing_consumption", "mortgage_interest", "mortgage_interest_share",
        "total_economic_expenses", "total_saving", "mortgage_principal",
        "pension_contributions", "modeled_employer_pension_contributions", "training_fund_contributions", "provident_fund_contributions",
        "life_exec_insurance_contributions", "other_observed_financial_saving", "community_gemach_saving",
        "cbs_total_cash_saving", "contractual_financial_saving", "real_estate_investment",
        "home_purchase_net", "other_property_purchase", "home_capital_improvements",
        "other_saving_and_debt_flows", "other_debt_net_repayment", "housing_debt_movement", "unclassified_debt_movement", "unclassified_s44_movement", "other_housing_loan_repayment",
        "household_asset_sales_net", "household_loans_extended",
        "real_estate_and_housing_financing", "net_property_acquisition_equity", "net_property_equity_investment", "real_estate_transaction_allocation",
        "financial_asset_saving_allocation", "other_financial_asset_saving_allocation",
        "real_estate_saving_allocation", "other_asset_and_lending_flows",
        "cash_income", "cash_living_outflows", "cash_flow_before_funded_saving", "noncash_income_bridge", "noncash_expense_bridge", "noncash_saving_bridge",
        "pension_training_provident_life_contributions", "total_funded_cash_saving",
        "free_cash_flow", "cash_flow_identity_gap", "survey_net_cash_surplus", "survey_cash_reconciliation_gap", "pension_balance_change", "transactional_accrual_saving", "identified_transaction_allocation", "total_net_worth_saving_allocation", "unallocated_transactional_saving", "residual_saving",
        "operating_cash_receipts", "operating_cash_expenses", "cash_flow_operating_activities",
        "remove_noncash_income_cash_flow", "remove_pension_withdrawals_from_operations",
        "operating_income_cash_adjustment", "remove_noncash_expenses_cash_flow",
        "operating_expense_cash_adjustment",
        "property_investment_cash_flow", "real_estate_investing_cash_flow", "financial_investment_cash_flow",
        "household_asset_sales_cash_flow", "household_loans_extended_cash_flow",
        "other_investing_cash_flow", "cash_flow_investing_activities",
        "pension_withdrawals_cash_flow", "long_term_savings_contributions_cash_flow",
        "pension_investing_cash_flow", "other_financial_savings_investing_cash_flow", "other_asset_investing_cash_flow",
        "mortgage_principal_cash_flow", "housing_debt_financing_cash_flow",
        "apartment_debt_movement_cash_flow", "other_housing_loan_cash_flow", "unclassified_debt_cash_flow",
        "community_real_estate_cash_flow", "cash_flow_financing_activities",
        "net_change_in_unallocated_cash", "company_cash_flow_reconciliation_gap",
        "savings_rate_pct", "adult_consumption_per_adult", "child_consumption_per_child",
        "mortgage_payment", "mortgage_payment_prevalence", "mortgage_payment_mortgage_hh", "mortgage_interest_mortgage_hh", "mortgage_balance_mortgage_hh", "home_value_mortgage_hh", "mortgage_prevalence_wealth", "mortgage_balance_overlay", "financial_assets_overlay",
        "other_financial_assets_overlay", "deposits_savings_overlay", "nonbank_investments_overlay", "pension_wealth_overlay", "primary_home_assets_overlay", "additional_property_assets_overlay", "land_assets_overlay", "housing_assets_overlay",
        "total_assets_overlay", "other_debt_overlay", "total_debt_overlay", "net_worth_overlay",
        "pnl_identity_gap", "saving_decomposition_gap",
    ]
    result = result.rename(columns={
        "head_age_proxy": "mean_head_age_proxy", "household_size": "mean_household_size",
        "children": "mean_children", "has_children": "share_with_children", "owner": "share_owner",
    })
    output_cols = [
        {"head_age_proxy": "mean_head_age_proxy", "household_size": "mean_household_size",
         "children": "mean_children", "has_children": "share_with_children", "owner": "share_owner"}.get(c, c)
        for c in output_cols
    ]
    result = result[output_cols]
    all_rows = result.loc[result["breakdown_dimension"] == "Age group"]
    sensitivity = all_rows[["column_label", "child_consumption"]].merge(
        aggregate_flows(hh).loc[lambda x: x["breakdown_dimension"] == "Age group",
                                ["column_label", "child_consumption_equivalence_scale"]],
        on="column_label"
    ).rename(columns={"column_label": "age_band", "child_consumption": "child_consumption_per_capita_allocation"})
    metadata = {
        "flow_source": "CBS Household Expenditure Survey 2021-2023 PUF, GDP-per-capita normalized and pooled",
        "flow_source_years": [2021, 2022, 2023],
        "monetary_basis": "2023-normalized using nominal GDP per capita",
        "gdp_per_capita_nominal_nis": {"2021": 167000, "2022": 183000, "2023": 191000},
        "wealth_source": "CBS longitudinal panel, 2023 cross-section (statistical age-band overlay)",
        "mortgage_annual_interest_rate_assumption": mortgage_rate,
        "annual_imputed_return_on_deposits_and_savings": deposit_return,
        "annual_imputed_return_on_nonbank_investments": investment_return,
        "annual_imputed_return_on_pension_assets": pension_return,
        "wealth_field_weighted_coverage": coverage,
        "households_in_flow_sample": int(len(hh)),
        "person_variable_expense_categories": sorted(PERSON_VARIABLE_CATEGORIES),
        "household_fixed_expense_categories": [c for c in EXPENSE_CATEGORIES if c not in PERSON_VARIABLE_CATEGORIES],
        "notes": [
            "Age is the midpoint of the CBS economic-head age group.",
            "2021-2023 monetary flows and survey weights are normalized using the 2023-to-year nominal GDP-per-capita ratio before pooling.",
            "Only clearly person-variable categories are split between adults and children; fixed/shared household costs and unclassified residual spending remain unallocated.",
            "Rent and owner imputed rent are combined as housing consumption; owner imputed rent is offset exactly by imputed housing-asset income.",
            "Mortgage interest is an expense and only mortgage principal is classified as funded saving; cars and durable goods remain expenses.",
            "Pension cash receipts remain in cash income; the economic P&L replaces them with the modeled pension return, and pension balance change equals contributions plus return less cash receipts.",
            "Modeled employer pension funding equals 12.5/6 times the observed employee pension deposit; it is added to non-cash labor compensation and pension accumulation, never to household cash flow.",
            "In the indirect cash-flow statement, pension and other financial saving are separate investing flows; community saving remains with real-estate investing; mortgage principal and signed housing-debt movements are financing flows.",
            "Reported rental/property income is retained without a modeled property yield; reported interest/dividends are replaced by modeled returns of 1% on deposits/savings and 4% on non-bank investments.",
            "CBS donations are presented as donations and community saving by user assumption; the public survey does not identify which payments, if any, are recoverable community/Gemach deposits.",
            "Mortgage interest uses the mortgage-holder conditional balance, the HES mortgage-payment prevalence, and the stated annual rate; net imputed housing income equals imputed rent less mortgage interest.",
            "Wealth stocks are not household-level joined to HES flows; missing 2023 stock fields are zero in unconditional means.",
        ],
    }
    return result, sensitivity, expense_categories, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mortgage-rate", type=float, default=0.045)
    parser.add_argument("--hes-root", type=Path, required=True, help="Directory containing downloaded CBS HES folders H20211021, H20221021 and H20231021.")
    parser.add_argument("--deposit-return", type=float, default=0.01)
    parser.add_argument("--investment-return", type=float, default=0.04)
    parser.add_argument("--pension-return", type=float, default=0.04)
    args = parser.parse_args()
    result, sensitivity, expense_categories, metadata = build(args.project, args.mortgage_rate, args.hes_root, args.deposit_return, args.investment_return, args.pension_return)
    figures = args.project / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result.to_csv(figures / "fig_household_pnl_by_age.csv", index=False, float_format="%.3f")
    sensitivity.to_csv(figures / "fig_household_pnl_child_allocation_sensitivity.csv", index=False, float_format="%.3f")
    expense_categories.to_csv(figures / "fig_household_expenses_by_age.csv", index=False, float_format="%.3f")
    (figures / "fig_household_pnl_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checks = {
        "max_abs_pnl_identity_gap": float(result["pnl_identity_gap"].abs().max()),
        "max_abs_saving_decomposition_gap": float(result["saving_decomposition_gap"].abs().max()),
        "breakdown_dimensions": int(result["breakdown_dimension"].nunique()),
        "columns_per_breakdown": int(result.groupby("breakdown_dimension")["column_label"].nunique().max()),
        "statement_columns": int(len(result)),
        "missing_cells": int(result.isna().sum().sum()),
        "max_abs_cash_flow_identity_gap": float(result["cash_flow_identity_gap"].abs().max()),
        "max_abs_survey_cash_reconciliation_gap": float(result["survey_cash_reconciliation_gap"].abs().max()),
        "max_abs_company_cash_flow_reconciliation_gap": float(result["company_cash_flow_reconciliation_gap"].abs().max()),
        "max_abs_expense_allocation_gap": float((expense_categories["value_nis"] - expense_categories["parent_expense"] - expense_categories["child_expense"] - expense_categories["household_fixed_expense"]).abs().max()),
    }
    (figures / "fig_household_pnl_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    if checks["max_abs_pnl_identity_gap"] > 1e-7 or checks["max_abs_saving_decomposition_gap"] > 1e-7:
        raise AssertionError(f"Accounting identity failed: {checks}")
    print(result.to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()












