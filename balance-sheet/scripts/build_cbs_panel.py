#!/usr/bin/env python3
"""
CBS Israel Longitudinal Survey — Panel Builder
===============================================
Builds cbs_longitudinal_panel_full.csv from 10 waves (2012-2023).
Each row = one household × one wave (economic head as individual representative).

Output files saved to the same folder as this script:
  - cbs_longitudinal_panel_full.csv
  - cbs_longitudinal_coverage.csv
"""

import pandas as pd
import numpy as np
import warnings
import sys
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
# Force stdout to UTF-8 so box-drawing chars print on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

# ─── PATHS ────────────────────────────────────────────────────────────────────

PARSER = argparse.ArgumentParser(description="Build the harmonized CBS longitudinal household panel.")
PARSER.add_argument("--data-root", type=Path, required=True,
                    help="Directory containing the downloaded CBS longitudinal release folders.")
PARSER.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "processed",
                    help="Output directory (default: PROJECT/processed).")
ARGS = PARSER.parse_args()
DATA_ROOT = ARGS.data_root
OUTPUT_DIR = ARGS.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODING = "cp1255"


def load_csv(path):
    """Load CSV; auto-detect UTF-8 BOM and use utf-8-sig instead of cp1255."""
    with open(path, "rb") as f:
        bom = f.read(3)
    enc = "utf-8-sig" if bom == b"\xef\xbb\xbf" else ENCODING
    return pd.read_csv(path, encoding=enc, low_memory=False)

# ─── WAVE DEFINITIONS ─────────────────────────────────────────────────────────

WAVES = [
    dict(num=1,  year="2012",      folder="H20121284",
         mb="H20121284DataMb.csv",         prat="H20121284dataprat.csv"),
    dict(num=2,  year="2013",      folder="H20131282",
         mb="H20131282datamb.csv",         prat="H20131282dataprat.csv"),
    dict(num=3,  year="2014-2015", folder="H201420151283",
         mb="H201420151283datamb.csv",     prat="H201420151283dataPrat.csv"),
    dict(num=4,  year="2016",      folder="H20161281",
         mb="h20161281datamb.csv",         prat="h20161281dataprat.csv"),
    dict(num=5,  year="2017",      folder="H20171281",
         mb="h20171281datamb.csv",         prat="h20171281dataprat.csv"),
    dict(num=6,  year="2018",      folder="H20181281",
         mb="h20181281datamb.csv",         prat="h20181281dataprat.csv",
         pkida="h20181281datapkida.csv"),
    dict(num=7,  year="2019",      folder="H20191281",
         mb="H20191281DATAMB.csv",         prat="H20191281DATAPRAT.csv",
         pkida="H20191281DATAPKIDA.csv"),
    dict(num=8,  year="2020",      folder="H20201281",
         mb="H20201281DataMb.csv",         prat="H20201281DataPrat.csv",
         pkida="H20201281datapkida.csv"),
    dict(num=9,  year="2021-2022", folder="H202120221281",
         mb="h202120221281datamb.csv",     prat="H202120221281dataprat.csv",
         pkida="H202120221281datapkida.csv"),
    dict(num=10, year="2023",      folder="H20231281",
         mb="H20231281DataMb.csv",         prat="H20231281dataprat.csv",
         pkida="H20231281datapkida.csv"),
]

EXPECTED_HH_COUNTS = {
    1: 5007, 2: 4621, 3: 4446, 4: 4180, 5: 4232,
    6: 4014, 7: 3983, 8: 4436, 9: 4075, 10: 3575,
}

# ─── MISSING VALUE CODES ──────────────────────────────────────────────────────

# Applied to all numeric columns
MISSING_ALL_NUMERIC = {99999999, 99999998}

# Applied additionally to continuous / amount fields
MISSING_AMOUNT_EXTRA = {9999999, 999999, 9999}

# Columns that are continuous amounts (receive both rounds of cleaning)
AMOUNT_FIELDS = {
    "income_total_annual", "income_work_annual", "income_pension_annual",
    "income_training_fund", "income_provident_fund", "income_scholarship",
    "income_interest", "income_capital_gains", "income_rental",
    "income_business", "income_govt_transfers", "income_abroad",
    "income_private_transfers",
    "amount_deposits", "amount_savings", "amount_mutual_funds",
    "amount_securities", "mortgage_balance", "non_housing_loan_balance",
    "credit_line_total", "apt_purchase_price", "apt_value_subjective",
    "apt_value_estimated", "value_additional_apts", "total_land_value",
    "monthly_rent", "apt_size_sqm",
}

# ─── FIELD CANDIDATE MAPS ─────────────────────────────────────────────────────
# Maps output column name → [candidate source column names] (case-insensitive).
# First match wins.

MB_FIELDS = {
    # Weights
    "weight_hh":                ["MishkalMb", "MishkalMB"],
    # Household composition
    "hh_size":                  ["SachNefashot"],
    "hh_children":              ["Nefashot0_17"],
    "hh_adults":                ["Nefashot18up"],
    # Income — annual, household level
    "income_total_annual":      ["SacShnatikolel_Lembnew"],
    "income_pension_annual":    ["sumHPensiaShnatit",    "PensiaAvoda"],
    "income_training_fund":     ["sumHHishShnatit",      "Hishtalmut"],
    "income_provident_fund":    ["sumHGemelShnatit",     "KupotGemel"],
    "income_scholarship":       ["sumHMilgaShnatit",     "MilgatLimudim"],
    "income_interest":          ["HRibitShnatit",        "Ribit"],
    "income_capital_gains":     ["HIhudHashkaotShnatit", "HHashkaotShnatit", "Hashkaot"],
    "income_rental":            ["HSchirutShnatit",      "Schirut"],
    "income_business":          ["HEsekShnatit",         "Esek"],
    "income_govt_transfers":    ["sumHMosdotShnatitnew", "Mosdotnew"],
    "income_abroad":            ["sumHHulShnatit",       "MosdotHul"],
    "income_private_transfers": ["sumHAnashimShnatit",   "HAnashimShnatit", "Anashim"],
    # Apartment value
    "apt_value_subjective":     ["M_ShuviDira", "m_ShuviDira", "ShuviDira"],
    "apt_value_estimated":      ["omdanShuviDira", "OmdanShuviDira"],
    "apt_purchase_price":       ["MechirKnuya1"],
    # Apartment characteristics
    "apt_size_sqm":             ["SetachDira1"],
    "apt_rooms":                ["K_MispHadarim1", "K_MispChadarim1", "k_MispChadarim"],
    "apt_floor":                ["MisparKoma1"],
    "apt_year_built":           ["ShanaBniya1"],
    "apt_type":                 ["SugDira1"],
    # Tenure / housing
    "owns_primary_apt":         ["k_baalut1", "k_baalut"],
    "tenure_type":              ["k_baalut1", "k_baalut"],
    "monthly_rent":             ["m_DmeSchirutChodeshi"],
    "has_elevator":             ["Maaliyot1"],
    "has_parking":              ["ChanayaPratit1"],
    "has_balcony":              ["MirpesetChitzonit1"],
    "has_garden":               ["GinaPratit1"],
    "ac_type":                  ["k_mizug1", "k_mizug"],
    "view_sea":                 ["NofYam1"],
    "view_green":               ["NofSetachYarok1"],
    "neighborhood_noise_road":  ["BaayaKvish1"],
    "neighborhood_trash":       ["BaayaZevel1"],
    "neighborhood_pollution":   ["BaayaZihum1"],
    # Additional property
    "owns_additional_apt":      ["BaalutDiraNosefet1"],
    "num_additional_apts":      ["BaalutDiraNosefetKama1"],
    "value_additional_apts":    ["ShuviDirotLoMegurim1"],
    "owns_agricultural_land":   ["KarkaChaklait1"],
    "owns_other_land":          ["KarkaLoChaklait1"],
    "owns_other_property":      ["KarkaAcheret1"],
    "total_land_value":         ["m_Mechirkarka"],
    # Financial assets — indicators
    "has_pension_fund":         ["kPensiaMB1",     "KPensiaMB1"],
    "has_exec_insurance":       ["BituachMenahalimMb1"],
    "has_provident_fund":       ["GemelVPizuyimMB1"],
    "has_training_fund":        ["kHishMB1",        "KHishMB1"],
    "has_mutual_funds":         ["kNeemanutMB1",    "KNeemanutMB1"],
    "has_stocks":               ["MenayotMB1"],
    "has_deposits":             ["PikdonotMB1"],
    "has_savings_plan":         ["TChisachonMB1"],
    # Financial assets — amounts
    "amount_deposits":          ["HaskaPikdon1"],
    "amount_savings":           ["HaskaChisachon1"],
    "amount_mutual_funds":      ["HaskaNeemanut1"],
    "amount_securities":        ["HaskaLoNeemanut1"],
    "cash_balance_category":    ["m_KesefMB1"],
    "has_foreign_currency":     ["MatbeaHutz1"],
    # Liabilities
    "has_mortgage":             ["m_HalvaaDiyur"],
    "mortgage_balance":         ["m_SchumHalvaaDiyur1"],
    "has_non_housing_loan":     ["m_HalvaaLoDiyur"],
    "non_housing_loan_balance": ["SchumHalvaaLoDiyur1"],
    "loan_source_bank":         ["HalvaaLoDiyurBank1"],
    "loan_source_credit":       ["HalvaaLoDiyurAshrai1"],
    "loan_source_employer":     ["HalvaaLoDiyurAvoda1"],
    "credit_line_total":        ["m_MisgeretAshrai"],
    "had_overdraft_12m":        ["Minus1"],
    "months_overdraft":         ["M_ChodashimMinus1"],
    "account_frozen":           ["CheshbonChasum1"],
    "credit_card_delayed":      ["ADechiyaMoedTashlum1"],
    "credit_card_revolving":    ["AKredit1"],
    # Vehicles
    "has_car":                  ["Meconit1", "Meconit"],
    "num_cars":                 ["k_MispMeconiyot1", "k_MispMeconiyot"],
    # Socioeconomic — municipality-level cluster, base year varies by wave
    # Waves 6-7: Cluster2015, Wave 8: CLUSTER2017, Wave 9: Cluster2019, Wave 10: CLUSTER2021
    "socioeconomic_cluster":    ["CLUSTER2021", "Cluster2021",
                                 "CLUSTER2017", "Cluster2019", "Cluster2015"],
    # Pension accumulation indicator (waves 6-10 only)
    # Binary 0/1: 1 = household has accrued pension rights beyond the flow income captured
    "zkifa_pensia_lemb":        ["zkifa_PensiaLemb"],
    # Subjective / forward-looking (stored in datamb, not dataprat)
    "financial_situation_past":   ["MazavKAvar1",    "MatzavKAvar",  "MazavKAvar"],
    "financial_situation_future": ["MazavKAtid1",    "MatzavKAtid",  "MazavKAtid"],
    "likely_to_save_future":      ["ChisachonAtid1", "ChisachonAtid"],
    "current_saving_status":      ["ChisachonHayom1","ChisachonHayom"],
}

PRAT_FIELDS = {
    # Panel identifier
    "mezaheprat":          ["MezahePrat"],
    # Weights (individual-level, from dataprat)
    "weight_individual":   ["MishkalPrat"],
    "weight_longitudinal": ["MishkalOrech"],
    # Demographics
    "nationality":         ["Leom", "leom"],
    "religion_jewish":     ["DatiutYehudi1",   "DatiutYehudi",  "datiutyehudi"],
    "religion_nonjewish":  ["DatiutLoYehudi1", "DatiutLoYehudi"],
    # Head characteristics — overridden by pkida for waves 6+
    "gender_head":         ["Min1", "Min"],
    "age_head":            ["gil1", "gilm"],
    "marital_status":      ["Mazav1", "Mazav"],
    "education_years":     ["K_ShnLimKlali"],
    "education_degree":    ["TeudaGvoha1", "TeudaGvoha"],
    # Labor
    "employment_status":   ["TchunatAvodaShnatit"],
    "employment_type":     ["MamadAvoda1", "MamadAvoda"],
    "sector":              ["SemelAnaf1",      "SemelAnaf"],
    "occupation":          ["SemelMishlach1",  "SemelMishlach"],
    # Origin — overridden by pkida for waves 6+
    "birth_continent":     ["M_eretz", "m_eretz"],
    "immigration_year":    ["K_ShnatAliya", "k_ShnatAliya"],
    # Other individual
    "language_primary":    ["K_SafaIkarit"],
    "income_work_annual":  ["SacShnati_LePratMeAvoda", "SacShnati_LePrat",
                            "SacShnati_LePratnew"],
    "health_status":       ["MatzavBriut1", "MatzavBriut"],
    "thought_of_poverty":  ["HashavAni1",   "HashavAni"],
}

# pkida provides these for waves 6+ (override dataprat values)
PKIDA_OVERRIDE = {
    "gender_head":      ["Min"],
    "birth_continent":  ["M_eretz"],
    "immigration_year": ["K_ShnatAliya", "k_ShnatAliya"],
}

# Ordered output columns (derived metrics appended at end)
OUTPUT_COLS_ORDER = [
    "mezaheprat", "siduri", "wave", "survey_year",
    "weight_hh", "weight_individual", "weight_longitudinal",
    "nationality", "religion_jewish", "religion_nonjewish",
    "gender_head", "age_head", "marital_status",
    "education_years", "education_degree",
    "employment_status", "employment_type", "sector", "occupation",
    "birth_continent", "immigration_year",
    "hh_size", "hh_children", "hh_adults",
    "socioeconomic_cluster", "language_primary",
    "income_total_annual", "income_work_annual", "income_pension_annual",
    "income_training_fund", "income_provident_fund", "income_scholarship",
    "income_interest", "income_capital_gains", "income_rental",
    "income_business", "income_govt_transfers", "income_abroad",
    "income_private_transfers",
    "has_pension_fund", "has_exec_insurance", "has_provident_fund",
    "has_training_fund", "has_mutual_funds", "has_stocks",
    "has_deposits", "has_savings_plan", "zkifa_pensia_lemb",
    "amount_deposits", "amount_savings", "amount_mutual_funds",
    "amount_securities", "cash_balance_category", "has_foreign_currency",
    "owns_primary_apt", "apt_purchase_price", "apt_value_subjective",
    "apt_value_estimated", "owns_additional_apt", "num_additional_apts",
    "value_additional_apts", "owns_agricultural_land", "owns_other_land",
    "owns_other_property", "total_land_value",
    "apt_size_sqm", "apt_rooms", "apt_floor", "apt_year_built",
    "has_car", "num_cars",
    "has_mortgage", "mortgage_balance", "has_non_housing_loan",
    "non_housing_loan_balance", "loan_source_bank", "loan_source_credit",
    "loan_source_employer", "credit_line_total", "had_overdraft_12m",
    "months_overdraft", "account_frozen", "credit_card_delayed",
    "credit_card_revolving",
    "tenure_type", "monthly_rent", "apt_type",
    "has_elevator", "has_parking", "has_balcony", "has_garden",
    "ac_type", "view_sea", "view_green",
    "neighborhood_noise_road", "neighborhood_trash", "neighborhood_pollution",
    "financial_situation_past", "financial_situation_future",
    "likely_to_save_future", "current_saving_status",
    "health_status", "thought_of_poverty",
    # Derived
    "net_worth_proxy", "financial_assets_total", "total_debt",
    "leverage_ratio", "income_per_capita", "work_income_share",
    "pension_income_share", "transfer_income_share",
]

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def find_col(df, candidates):
    """Case-insensitive lookup. Returns actual column name or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        actual = cols_lower.get(cand.lower())
        if actual is not None:
            return actual
    return None


def extract_fields(df, field_map, wave_num, source, missing_log):
    """
    Extract and standardise fields from df using candidate map.
    Returns {output_name: array} dict.
    Logs missing fields to missing_log.
    """
    result = {}
    for out_name, candidates in field_map.items():
        col = find_col(df, candidates)
        if col is None:
            missing_log.setdefault(out_name, []).append(wave_num)
            result[out_name] = np.full(len(df), np.nan, dtype=object)
        else:
            result[out_name] = df[col].values
    return result


def select_head(prat_df, wave_num):
    """
    Filter dataprat to one economic head per household.

    Priority order:
      1. YachasKirvaKalkali == 1   (waves 4-10)
      2. yachas1 / Yachas1 == 1   (waves 2-3; first-sampled person = economic head)
      3. Yachas == 1               (wave 1)

    In CBS survey design the first-listed person IS the economic head
    for waves 1-3 where no explicit economic-head flag exists.
    """
    head_col = find_col(prat_df, ["YachasKirvaKalkali"])
    if head_col:
        vals = pd.to_numeric(prat_df[head_col], errors="coerce")
        head = prat_df[vals == 1].copy()
        if len(head) > 0:
            print(f"    Head filter: {head_col}==1  → {len(head)} rows")
        else:
            warnings.warn(f"Wave {wave_num}: {head_col}==1 returned 0 rows")
            head_col = None

    if head_col is None or len(head) == 0:
        fallback_col = find_col(prat_df, ["yachas1", "Yachas1", "Yachas"])
        if fallback_col:
            vals = pd.to_numeric(prat_df[fallback_col], errors="coerce")
            head = prat_df[vals == 1].copy()
            print(f"    Head filter: {fallback_col}==1  → {len(head)} rows")
        else:
            warnings.warn(f"Wave {wave_num}: No head-filter column found — using all rows")
            head = prat_df.copy()

    # Safety dedup: one row per Siduri
    siduri_col = find_col(head, ["Siduri"])
    if siduri_col:
        n_before = len(head)
        head = head.drop_duplicates(subset=[siduri_col], keep="first")
        n_after = len(head)
        if n_before != n_after:
            warnings.warn(
                f"Wave {wave_num}: Deduplicated {n_before - n_after} extra head rows "
                f"(same Siduri) — kept first occurrence"
            )
    return head


def nan_safe_add(panel, cols):
    """
    Sum listed columns NaN-safely:
    - If at least one column is non-NaN, sum the non-NaN values.
    - If ALL columns are NaN for a row, result is NaN.
    """
    arrays = [pd.to_numeric(panel[c], errors="coerce") for c in cols if c in panel.columns]
    if not arrays:
        return pd.Series(np.nan, index=panel.index)
    stacked = pd.concat(arrays, axis=1)
    all_nan = stacked.isna().all(axis=1)
    result = stacked.fillna(0).sum(axis=1)
    result[all_nan] = np.nan
    return result


# ─── MAIN PROCESSING ──────────────────────────────────────────────────────────

missing_field_log = {}   # {output_field: [wave_nums where not found]}
wave_frames = []

print("=" * 70)
print("CBS Longitudinal Survey — Panel Builder")
print("=" * 70)

for wv in WAVES:
    wnum    = wv["num"]
    wyear   = wv["year"]
    folder  = DATA_ROOT / wv["folder"]
    print(f"\n{'─'*60}")
    print(f"Wave {wnum} ({wyear})  [{wv['folder']}]")

    # ── 1. Load datamb ──────────────────────────────────────────────────
    mb_path = folder / wv["mb"]
    mb = load_csv(mb_path)
    print(f"  datamb:  {mb.shape[0]:5d} rows × {mb.shape[1]:3d} cols")

    # ── 2. Load dataprat ────────────────────────────────────────────────
    prat_path = folder / wv["prat"]
    prat = load_csv(prat_path)
    print(f"  dataprat:{prat.shape[0]:5d} rows × {prat.shape[1]:3d} cols")

    # ── 3. Filter to household economic head ────────────────────────────
    head = select_head(prat, wnum)

    # ── 4. Load and join datapkida (waves 6-10) ─────────────────────────
    pkida_extra = {}   # will hold overridden field values keyed by out_name
    if "pkida" in wv:
        pkida_path = folder / wv["pkida"]
        pkida = load_csv(pkida_path)
        print(f"  datapkida:{pkida.shape[0]:5d} rows × {pkida.shape[1]:3d} cols")

        mzp_col_pk   = find_col(pkida, ["MezahePrat"])
        mzp_col_head = find_col(head,  ["MezahePrat"])

        if mzp_col_pk and mzp_col_head:
            # Deduplicate pkida on MezahePrat (keep first)
            pkida_dedup = pkida.drop_duplicates(subset=[mzp_col_pk], keep="first")

            # Pull override columns from pkida
            for out_name, cands in PKIDA_OVERRIDE.items():
                col = find_col(pkida_dedup, cands)
                if col:
                    lkp = pkida_dedup.set_index(mzp_col_pk)[col]
                    head_mzp = head[mzp_col_head]
                    pkida_extra[out_name] = head_mzp.map(lkp).values
                    print(f"    pkida override: {out_name} ← {col}  "
                          f"({(~pd.isna(pkida_extra[out_name])).sum()} matched)")
                else:
                    warnings.warn(f"Wave {wnum}: pkida override '{out_name}' not found")
        else:
            warnings.warn(f"Wave {wnum}: MezahePrat missing — skipping pkida merge")

    # ── 5. Extract fields ───────────────────────────────────────────────
    # From datamb
    mb_dict = extract_fields(mb, MB_FIELDS, wnum, "datamb", missing_field_log)

    siduri_col_mb = find_col(mb, ["Siduri"])
    wave_col_mb   = find_col(mb, ["wave"])
    sseker_col_mb = find_col(mb, ["s_seker"])

    mb_dict["siduri"]      = mb[siduri_col_mb].values if siduri_col_mb else np.full(len(mb), np.nan)
    mb_dict["wave"]        = mb[wave_col_mb].values   if wave_col_mb   else np.full(len(mb), wnum)
    if sseker_col_mb:
        year_vals = pd.to_numeric(mb[sseker_col_mb], errors="coerce")
        if year_vals.isna().all():
            # s_seker is a string range like "2014-2015"; take first 4 chars
            raw = mb[sseker_col_mb].dropna().iat[0] if mb[sseker_col_mb].notna().any() else wyear
            mb_dict["survey_year"] = np.full(len(mb), int(str(raw)[:4]))
        else:
            mb_dict["survey_year"] = year_vals.values
    else:
        mb_dict["survey_year"] = np.full(len(mb), int(wyear[:4]))

    mb_result = pd.DataFrame(mb_dict)

    # From dataprat head
    prat_dict = extract_fields(head, PRAT_FIELDS, wnum, "dataprat", missing_field_log)

    # Apply pkida overrides
    for out_name, arr in pkida_extra.items():
        prat_dict[out_name] = arr

    siduri_col_head = find_col(head, ["Siduri"])
    prat_dict["siduri"] = head[siduri_col_head].values if siduri_col_head else np.full(len(head), np.nan)

    prat_result = pd.DataFrame(prat_dict)

    # ── 6. Merge head into datamb on siduri ─────────────────────────────
    # Drop wave/survey_year from prat to avoid duplicates (already in mb)
    prat_result = prat_result.drop(
        columns=[c for c in ["wave", "survey_year"] if c in prat_result.columns]
    )

    # Convert siduri to same type for merge
    for df in [mb_result, prat_result]:
        df["siduri"] = pd.to_numeric(df["siduri"], errors="coerce")

    wave_df = mb_result.merge(prat_result, on="siduri", how="left", suffixes=("", "_prat"))
    print(f"  Merged wave: {wave_df.shape[0]:5d} rows × {wave_df.shape[1]:3d} cols")

    # Check for unexpected duplicate columns from merge
    dup_cols = [c for c in wave_df.columns if c.endswith("_prat")]
    if dup_cols:
        print(f"  WARNING: Duplicate columns from merge: {dup_cols[:5]}")

    # ── 7. Row count check ──────────────────────────────────────────────
    expected = EXPECTED_HH_COUNTS.get(wnum)
    actual   = len(wave_df)
    status   = "OK" if actual == expected else f"MISMATCH (expected {expected})"
    print(f"  Row count: {actual}  [{status}]")

    wave_frames.append(wave_df)

# ─── STACK ALL WAVES ──────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("Stacking waves...")
panel = pd.concat(wave_frames, ignore_index=True)
print(f"Panel shape (pre-clean): {panel.shape}")

# ─── MISSING VALUE HANDLING ───────────────────────────────────────────────────

print("Cleaning missing value codes...")

# Step 0: Coerce all output fields to numeric first (except string identifiers)
STRING_ID_FIELDS = {"mezaheprat"}   # these are hyphenated string IDs, not numbers
all_out_fields = (
    list(MB_FIELDS.keys()) + list(PRAT_FIELDS.keys()) +
    ["siduri", "wave", "survey_year"]
)
for col in all_out_fields:
    if col in panel.columns and col not in STRING_ID_FIELDS and panel[col].dtype == object:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

# Pass 1: large sentinel codes in all numeric columns
num_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    mask = panel[col].isin(MISSING_ALL_NUMERIC)
    if mask.any():
        panel.loc[mask, col] = np.nan

# Pass 2: medium sentinel codes in amount columns only
for out_name in AMOUNT_FIELDS:
    if out_name in panel.columns:
        mask = panel[out_name].isin(MISSING_AMOUNT_EXTRA)
        if mask.any():
            panel.loc[mask, out_name] = np.nan

# ─── MISSING FIELD LOG ────────────────────────────────────────────────────────

print("\n" + "="*70)
print("MISSING FIELDS BY WAVE (fields not found in source data)")
print("="*70)
if missing_field_log:
    for field, waves in sorted(missing_field_log.items()):
        print(f"  {field:<35s} missing in waves: {waves}")
else:
    print("  (none)")

# ─── MISSING VALUES PER COLUMN/WAVE SUMMARY ───────────────────────────────────

print("\n--- Missing value counts per column (sample of income/asset fields) ---")
sample_fields = [
    "income_total_annual", "income_work_annual", "apt_value_subjective",
    "mortgage_balance", "gender_head", "age_head", "nationality",
]
for f in sample_fields:
    if f in panel.columns:
        by_wave = panel.groupby("wave")[f].apply(lambda s: s.isna().sum())
        print(f"  {f:<35s}: {dict(by_wave)}")

# ─── DERIVED METRICS ──────────────────────────────────────────────────────────

print("\nComputing derived metrics...")

def col(name):
    return pd.to_numeric(panel[name], errors="coerce") if name in panel.columns \
           else pd.Series(np.nan, index=panel.index)

panel["financial_assets_total"] = nan_safe_add(panel, [
    "amount_deposits", "amount_savings", "amount_mutual_funds", "amount_securities"
])

panel["total_debt"] = nan_safe_add(panel, [
    "mortgage_balance", "non_housing_loan_balance"
])

# net_worth_proxy = apt_value + financial_assets - total_debt
_apt    = col("apt_value_subjective")
_fin    = panel["financial_assets_total"]
_debt   = panel["total_debt"]
_all_nan = _apt.isna() & _fin.isna() & _debt.isna()
panel["net_worth_proxy"] = (
    _apt.fillna(0) + _fin.fillna(0) - _debt.fillna(0)
)
panel.loc[_all_nan, "net_worth_proxy"] = np.nan

# leverage_ratio = total_debt / apt_value_subjective
_apt_val = col("apt_value_subjective")
panel["leverage_ratio"] = np.where(
    _apt_val.isna() | (_apt_val == 0),
    np.nan,
    panel["total_debt"] / _apt_val
)

# income_per_capita = income_total_annual / hh_size
_inc  = col("income_total_annual")
_size = col("hh_size")
panel["income_per_capita"] = np.where(
    _inc.isna() | _size.isna() | (_size == 0),
    np.nan,
    _inc / _size
)

# Work / pension / transfer income shares
_inc_safe = _inc.copy()
_inc_safe[_inc_safe == 0] = np.nan  # avoid div-by-zero

panel["work_income_share"]     = col("income_work_annual")     / _inc_safe
panel["pension_income_share"]  = col("income_pension_annual")  / _inc_safe
panel["transfer_income_share"] = col("income_govt_transfers")  / _inc_safe

# ─── REORDER COLUMNS ──────────────────────────────────────────────────────────

present_ordered = [c for c in OUTPUT_COLS_ORDER if c in panel.columns]
extra_cols      = [c for c in panel.columns if c not in present_ordered]
panel = panel[present_ordered + extra_cols]

# ─── VALIDATION CHECKS ────────────────────────────────────────────────────────

print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

# Check 1 — Row counts per wave
print("\n[1] Row counts per wave:")
counts = panel.groupby("wave").size()
all_ok = True
for wnum, expected in EXPECTED_HH_COUNTS.items():
    actual = counts.get(wnum, 0)
    ok = actual == expected
    all_ok = all_ok and ok
    mark = "✓" if ok else "✗"
    print(f"  Wave {wnum:2d}: {actual:5d} rows  (expected {expected:5d})  {mark}")
print(f"  Overall: {'ALL OK' if all_ok else 'ISSUES FOUND'}")

# Check 2 — Wave 10 income_total_annual completeness
print("\n[2] Wave 10 income_total_annual non-null rate:")
w10 = panel[panel["wave"] == 10]
if "income_total_annual" in w10.columns:
    pct = w10["income_total_annual"].notna().mean() * 100
    mark = "✓" if pct > 80 else "✗ (< 80%)"
    print(f"  {pct:.1f}% non-null  {mark}")
else:
    print("  Field not found")

# Check 3 — Mortgage sanity (has_mortgage==2 should have zero/NaN balance)
print("\n[3] Mortgage balance sanity (has_mortgage==2 → should be NaN/0):")
if "has_mortgage" in panel.columns and "mortgage_balance" in panel.columns:
    no_mort  = panel[pd.to_numeric(panel["has_mortgage"], errors="coerce") == 2]
    bal_ser  = pd.to_numeric(no_mort["mortgage_balance"], errors="coerce")
    bad_rows = ((bal_ser.notna()) & (bal_ser != 0)).sum()
    total    = len(no_mort)
    print(f"  Rows with has_mortgage==2: {total:,}")
    print(f"  Of those with non-null/non-zero balance: {bad_rows}")
    print(f"  {'✓' if bad_rows == 0 else '⚠ CHECK THESE ROWS'}")

# Check 4 — apt_value_subjective distribution for owner-occupiers
print("\n[4] apt_value_subjective for owner-occupiers (tenure_type==1):")
if "tenure_type" in panel.columns and "apt_value_subjective" in panel.columns:
    owners = panel[pd.to_numeric(panel["tenure_type"], errors="coerce") == 1]
    vals   = pd.to_numeric(owners["apt_value_subjective"], errors="coerce").dropna()
    if len(vals) > 0:
        med = vals.median()
        p10 = vals.quantile(0.10)
        p90 = vals.quantile(0.90)
        print(f"  n={len(vals):,}, median={med:,.0f} NIS,  P10={p10:,.0f},  P90={p90:,.0f}")
        mark = "✓" if 500_000 <= med <= 5_000_000 else "✗ (outside 500K–5M range)"
        print(f"  Median range check: {mark}")
    else:
        print("  No owner-occupier rows with apt_value data")

# Check 5 — MezahePrat uniqueness (waves 6-10)
print("\n[5] MezahePrat uniqueness per wave (waves 6-10):")
if "mezaheprat" in panel.columns:
    for wnum in range(6, 11):
        wdf  = panel[(panel["wave"] == wnum) & panel["mezaheprat"].notna()]
        dups = wdf["mezaheprat"].duplicated().sum()
        mark = "✓" if dups == 0 else f"✗ ({dups} duplicates)"
        print(f"  Wave {wnum}: {len(wdf):,} with MezahePrat,  {dups} duplicates  {mark}")

# ─── COVERAGE TABLE ───────────────────────────────────────────────────────────

print("\nBuilding coverage table...")
coverage_rows = []
fields_for_coverage = [c for c in OUTPUT_COLS_ORDER if c in panel.columns]
for field in fields_for_coverage:
    row = {"field": field}
    for wnum in range(1, 11):
        wdf = panel[panel["wave"] == wnum]
        if len(wdf) == 0:
            row[f"w{wnum}"] = None
        else:
            pct = wdf[field].notna().mean() * 100
            row[f"w{wnum}"] = round(pct, 1)
    coverage_rows.append(row)

coverage_df = pd.DataFrame(coverage_rows).set_index("field")

print("\n--- Coverage (% non-null) by field × wave ---")
print(coverage_df.to_string())

# ─── SAVE OUTPUTS ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("Saving output files...")

out_full = OUTPUT_DIR / "cbs_longitudinal_panel_full.csv"
out_cov  = OUTPUT_DIR / "cbs_longitudinal_coverage.csv"

panel.to_csv(out_full, index=False, encoding="utf-8-sig")
coverage_df.to_csv(out_cov, encoding="utf-8-sig")

print(f"  Full panel : {out_full}")
print(f"  Shape      : {panel.shape}")
print(f"  Coverage   : {out_cov}")

print("\n" + "="*70)
print("Done!")
print("="*70)
