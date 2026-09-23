"""
pipeline.py -- ONE script: panel CSV -> models -> per-household display columns ->
tidy aggregated CSV tables in ../figures/.

This is the ONLY place calculation happens. build_dashboard.py just reads the
figures/*.csv files this script writes and embeds them into output/index.html;
the browser only formats/plots, it does not aggregate or model anything.

Model functions (winsorize, bake_amount, hedonic_home, pension_wealth, grp_code,
remap_*, iv/amt) are copied VERBATIM from build_balance_sheet_data.py -- same
math, same coefficients, same winsor percentiles, same calibration. Do not
change them here; if build_balance_sheet_data.py's models change, resync.

Weighting: ALL weighted statistics use weight_hh (the CBS household design
weight). Weighted quantiles (median, p10/p25/p75/p90) use a weighted-quantile
helper (see weighted_quantile() below) -- NOT unweighted np.median/np.quantile.
Weighted-quantile method: sort by value, cumulative weight minus half the own
weight (Hazen / midpoint plotting-position convention), normalize to [0,1],
linear-interpolate the target quantile against value. This is the same
convention commonly used for survey-weighted percentiles (e.g. Stata's
[pweight] with the default definition) -- deterministic, no randomness.

Odd-wave financial rotation: deposits/savings, investments, mortgage, consumer
loans, additional-property value and land value are NOT collected in odd waves
(2012, 2014, 2017, 2019, 2021). In those waves the corresponding household
columns are NaN (not zero), and any aggregate that depends on them (Total
Financial Assets, Total Assets, Total Liabilities, Net Worth, full Total Real
Estate) is emitted as null with a note -- never fabricated or zero-filled.
"""

import csv
import hashlib
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(HERE, "..", "processed")
FIGURES_DIR = os.path.join(HERE, "..", "figures")
CSV_IN = os.path.join(PROCESSED_DIR, "cbs_longitudinal_panel_full.csv")

LATEST_WAVE = 10

WAVE_YEAR = {1: 2012, 2: 2013, 3: 2014, 4: 2016, 5: 2017,
             6: 2018, 7: 2019, 8: 2020, 9: 2021, 10: 2023}
EVEN_WAVES = {2, 4, 6, 8, 10}   # financial assets & liabilities collected only in these

# ======================================================================
# ---- Admin/macro benchmark scaling (added 2026-07-18) ----
# ======================================================================
# Pressure-tested against official end-2023 administrative aggregates
# (see docs/admin_benchmark_pressure_test.md). Weighted household population,
# wave 10 (2023) = 3,049,550. Survey total (Rs B) = per-HH weighted mean x
# 3,049,550. Factor = admin_total / survey_total, computed from the ACTUAL
# wave-10 weighted means produced by this pipeline (92,709 / 59,282 / 171,317
# / 23,045 for dep_sav / invest / mortg / consumer respectively, matching
# figures/fig_national_balance_sheet.csv pre-scaling baseline):
#
#   dep_sav  (Bank Deposits & Savings): survey total ~R282.72B; admin R614B
#            (Bank of Israel, "Short-Term Household Investments in Solid
#            Vehicles," Statistical Bulletin 2024 -- household-only deposits +
#            current accounts, end-2023) -> factor = 614 / 282.72074 = 2.1718.
#            Genuine under-capture (liquid-asset under-reporting / missing
#            top-wealth concentration), not a scope mismatch -- scaled.
#   mortg    (Mortgage): survey total ~R522.44B; admin R568B (Bank of Israel,
#            "Debt Developments in the Nonfinancial Private Sector, Q4 2023"
#            press release, 26-Mar-2024 -- household housing debt, end-2023)
#            -> factor = 568 / 522.43976 = 1.0872. Clean household-sector
#            match; small gap, scaled for consistency with the other two debt
#            lines from the same release.
#   consumer (Consumer Loans / non-housing debt): survey total ~R70.28B; admin
#            R224B (same BoI Q4-2023 release -- household non-housing debt,
#            end-2023) -> factor = 224 / 70.27688 = 3.1874. Investigated: the
#            large gap is the well-documented household-survey under-report of
#            revolving/credit-card/overdraft/informal consumer debt (small,
#            non-salient balances vs. the large, well-remembered mortgage line
#            from the SAME source/scope) -- not a scope mismatch (same BoI
#            household-debt series as mortgage) -- so no disqualifying reason
#            to withhold scaling. Scaled.
#   invest   (Investment Portfolio: mutual funds + directly-held securities):
#            NOT SCALED (factor 1.0). Researched BoI's "Public's Financial-
#            Assets Portfolio" series and the household-investments bulletin;
#            the only clean household-only pieces found (NIS money funds ~R70B
#            + Makam ~R14B) are already-partial building blocks, and the
#            headline mutual-fund/securities totals available (~R456B mutual
#            funds end-2023 per BoI) are for "the public" = households +
#            business sector combined, not households alone -- folding in a
#            households+business total would overstate this household-only
#            line. No defensible household-only mutual-fund + tradable-
#            securities total was found (ratio band ~0.7-1.2x, too wide to
#            call a genuine miss) -- left unscaled per the pressure-test doc's
#            recommendation, pending a dedicated BoI household-securities data
#            pull.
#            2026-07-18 DEEP-DIVE (still not scaled): pulled BoI's "Public's
#            Financial-Assets Portfolio" holder-level tables (Fig 1.9/1.10,
#            Statistical Bulletin 2024), which DO split a "Households*"
#            holder (households + NPISH) out from institutional investors /
#            mutual funds / corporate for two instruments: tradable corporate
#            bonds (households end-2023 = R49B, confirms the existing figure)
#            and Makam (households end-2023 = R11B, corroborating the ~R14B
#            household-investments-bulletin estimate). BUT the same source
#            does NOT split "equities in Israel" or mutual-fund-UNIT holdings
#            (the two largest pieces of this line, ~R685B and ~R456B "the
#            public" totals end-2023 respectively) by households vs. business
#            -- only a "public directly" vs. "institutional investors" cut is
#            published for those two instruments. No CBS/BoI "Financial
#            Accounts" (חשבונות פיננסיים) household-sector balance-sheet
#            publication distinct from "the public's" portfolio could be
#            located either. Still no defensible household-only total for the
#            full invest line -- remains at factor 1.0. See docs/admin_
#            benchmark_pressure_test.md Section 6 ("Investment Portfolio
#            deep-dive") for full detail and what was/wasn't found.
# CRITICAL: pension_wealth / the pension column is a SEPARATE modelled line
# owned by a different workstream -- it is NOT scaled here and must not be.
ADMIN_SCALE = {
    "dep_sav": 2.1718,   # -> admin R614B (BoI household deposits+current a/c, end-2023)
    "invest": 3.109,     # -> household est. ~R562B, end-2023 (see derivation below)
    #   No official household-only total exists (BoI publishes equities/mutual funds only
    #   for "the public"=HH+business). Best-estimate household total, differentiated shares:
    #   mutual funds 0.65 x R456B ("public") = R296B (retail-dominated product);
    #   direct equities 0.30 x R685B ("public") = R206B (households hold a minority --
    #   much is business cross-holdings/controlling blocks); + household bonds R49B +
    #   Makam R11B (both actual BoI household figures) = ~R562B. factor = 562/180.8.
    #   FLAGGED best-estimate: the two share assumptions are judgmental, not sourced.
    "mortg": 1.0872,     # -> admin R568B (BoI household housing debt, end-2023)
    "consumer": 3.1874,  # -> admin R224B (BoI household non-housing debt, end-2023)
}

GROUP_LABELS = {1: "Haredi", 2: "Dati (Religious)", 3: "Masorti",
                4: "Hiloni (Secular)", 5: "Arab"}
TENURE_LABELS = {1: "Owner", 2: "Renter", 3: "Other"}
SIZE_LABELS = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5+"}
MARITAL_LABELS = {1: "Married", 2: "Single", 3: "Divorced/Separated", 4: "Widowed"}
KIDS_LABELS = {0: "No children", 1: "With children"}
EDU_LABELS = {1: "No diploma", 2: "Secondary / Matriculation",
              3: "Post-secondary (non-academic)", 4: "Academic degree"}
OCC_LABELS = {1: "Managers & Professionals", 2: "Technicians & Clerical",
              3: "Sales & Service", 4: "Skilled Manual & Agriculture",
              5: "Operators & Unskilled"}
SES_LABELS = {1: "Low cluster (1-3)", 2: "Mid cluster (4-7)", 3: "High cluster (8-10)"}
AGE_BANDS = [("Under 35", 0, 35), ("35-44", 35, 45), ("45-54", 45, 55),
             ("55-64", 55, 65), ("65-74", 65, 75), ("75+", 75, 200)]

# ---- Cross-tab explorer (fig_metric_by_breakdown) ------------------------
# Industry / economic sector of employment. The panel's `sector` field
# (SemelAnaf) is the ISIC Rev.4 section LETTER of the economic head's industry
# of employment (NOT geography). Wave-10 coverage ~73% (2,603 / 3,575 heads);
# the ~27% with no code are heads not in employment (retired / unemployed /
# not-in-labour-force). The 22 ISIC sections are collapsed into 13 readable
# industry groups below; heads with no sector code (NaN) and the residual
# "unknown" code X (n~25, employed but industry not classified) both fold into
# "Not employed / other". See docs/cbs_survey_mapping.md.
SECTOR_GROUPS = [
    ("Agriculture, forestry & fishing", ["A"]),
    ("Manufacturing & utilities", ["B", "C", "D", "E"]),
    ("Construction", ["F"]),
    ("Wholesale & retail trade", ["G"]),
    ("Transport & storage", ["H"]),
    ("Hospitality & food service", ["I"]),
    ("Information & communication", ["J"]),
    ("Finance & insurance", ["K"]),
    ("Professional & business services", ["L", "M", "N"]),
    ("Public administration & defence", ["O"]),
    ("Education & teaching", ["P"]),
    ("Health & social work", ["Q"]),
    ("Arts & other services", ["R", "S", "T", "U"]),
]
NOT_EMPLOYED_LABEL = "Not employed / other"

# Metrics offered by the explorer: (csv column key, hh-frame column). Each cell
# is the weighted MEAN of the hh-frame column over households in that
# (breakdown, category), wave 10.
CROSS_TAB_METRICS = [
    ("net_worth", "nw"), ("total_assets", "tot_a"), ("total_real_estate", "tot_re"),
    ("primary_residence", "home"), ("total_financial", "tot_fin"), ("pension", "pension"),
    ("deposits_savings", "dep_sav"), ("investment", "invest"),
    ("total_liabilities", "tot_l"), ("mortgage", "mortg"), ("consumer_loans", "consumer"),
    ("income_annual", "inc_total"),
]
CROSS_TAB_METRIC_LABELS = {
    "net_worth": "Net worth", "total_assets": "Total assets",
    "total_real_estate": "Total real estate", "primary_residence": "Primary residence",
    "total_financial": "Total financial assets", "pension": "Pension",
    "deposits_savings": "Deposits & savings", "investment": "Investment portfolio",
    "total_liabilities": "Total liabilities", "mortgage": "Mortgage",
    "consumer_loans": "Consumer loans", "income_annual": "Annual income",
}
# Breakdowns: (key, label, ordinal?, ordered category labels). ordinal=True ->
# front-end keeps this natural order; ordinal=False -> front-end sorts bars
# descending by the chosen metric value.
CROSS_TAB_BREAKDOWNS = [
    ("group", "Social group", False, [GROUP_LABELS[c] for c in (1, 2, 3, 4, 5)]),
    ("income_decile", "Income decile", True, [str(d) for d in range(1, 11)]),
    # SES is now the FULL 10-level locality socioeconomic cluster (1-10, from the
    # raw socioeconomic_cluster field), not the old 3-bucket Low/Mid/High. Ordinal
    # natural order 1->10. (The 3-bucket remap_ses / SES_LABELS is still used by
    # fig_networth_by_dimension -- unchanged there.)
    ("ses", "Locality SES cluster (1–10)", True, [str(c) for c in range(1, 11)]),
    ("occ", "Occupation", False, [OCC_LABELS[c] for c in (1, 2, 3, 4, 5)] + [NOT_EMPLOYED_LABEL]),
    ("edu", "Education", False, [EDU_LABELS[c] for c in (1, 2, 3, 4)]),
    ("age_band", "Age band", True, [lab for lab, _, _ in AGE_BANDS]),
    ("tenure", "Tenure", False, [TENURE_LABELS[c] for c in (1, 2, 3)]),
    ("hh_size", "Household size", True, [SIZE_LABELS[c] for c in (1, 2, 3, 4, 5)]),
    ("marital", "Marital status", False, [MARITAL_LABELS[c] for c in (1, 2, 3, 4)]),
    ("children", "Children", False, [KIDS_LABELS[c] for c in (0, 1)]),
    ("sector", "Industry (sector of employment)", False,
     [lab for lab, _ in SECTOR_GROUPS] + [NOT_EMPLOYED_LABEL]),
]

AMOUNT_FIELDS = [
    "apt_value_subjective", "value_additional_apts", "total_land_value",
    "amount_deposits", "amount_savings", "amount_mutual_funds", "amount_securities",
    "mortgage_balance", "non_housing_loan_balance",
]
INCOME_FIELDS = [
    "income_total_annual", "income_work_annual", "income_pension_annual",
    "income_govt_transfers", "income_rental", "income_interest",
    "income_capital_gains", "income_per_capita",
    # additional household-level NON-LABOR income lines summing into
    # income_total_annual -- needed for the household-labor residual (see
    # build_household_frame's inc_work). income_business is LABOR (self-
    # employment) and is intentionally NOT subtracted / not listed here.
    "income_training_fund", "income_provident_fund", "income_scholarship",
    "income_abroad", "income_private_transfers",
]
NUMERIC = AMOUNT_FIELDS + INCOME_FIELDS + [
    "apt_value_estimated", "age_head", "socioeconomic_cluster",
    "weight_hh", "nationality", "religion_jewish", "tenure_type", "hh_size",
    "marital_status", "hh_children", "education_degree", "occupation",
    "owns_primary_apt", "owns_additional_apt", "owns_agricultural_land",
    "owns_other_land", "owns_other_property", "has_deposits", "has_savings_plan",
    "has_mutual_funds", "has_stocks", "has_mortgage", "has_non_housing_loan",
    "has_pension_fund",
]

# ======================================================================
# ---- Model functions copied VERBATIM from build_balance_sheet_data.py ----
# (math, coefficients, winsor percentiles, calibration all unchanged)
# ======================================================================


def winsorize(s, p):
    """Clip to [0, p-th percentile] over populated values."""
    cap = s.quantile(p)
    return s.clip(lower=0, upper=cap)


def collected_by_wave(df, col):
    """Boolean Series: True for rows whose wave has ANY populated value of col."""
    return df.groupby("wave")[col].transform(lambda x: bool(x.notna().any()))


def bake_amount(df, amount_col, holder_mask, p=0.99, seed=None):
    """Population display column for one amount:
       holder & reported  -> winsorized value
       holder & missing   -> STOCHASTIC REGRESSION IMPUTATION: log1p(amount) is
                             regressed on log(income) + group dummies + wave
                             dummies, fit on reporters only; each non-reporter's
                             predicted value is combined with a residual drawn
                             (with replacement, seeded via a deterministic hash
                             of the column name so results are reproducible run
                             to run) from the reporters' own empirical residual
                             distribution, then back-transformed and capped at
                             the same winsor percentile as reporters. This gives
                             non-reporters realistic within-{wave x group}
                             dispersion instead of one identical value.
                             Falls back to the OLD flat {wave x nationality}
                             median (with {wave}-only fallback) only when there
                             are fewer than 30 reporters to fit a stable model.
       non-holder          -> 0
       wave not collected  -> NaN

    AUDIT FINDING (fixed 2026-07-18): the prior version assigned every
    non-reporting holder in a {wave x nationality} cell the IDENTICAL median
    value. Because non-reporters are usually the majority of holders for these
    fields, this manufactured artificial within-group uniformity -- most
    visibly for Arab households, who have only one nationality cell (no
    religiosity split the way Jewish households do), so their bake_amount
    columns came out far tighter than any Jewish subgroup (net-worth Gini 0.44
    vs 0.55-0.66 elsewhere; p25 anomalously high at ~608k) purely as an
    imputation artifact, not a real feature of the underlying distribution.
    This directly inflated the compressed-inequality result under audit.
    """
    base = winsorize(df[amount_col], p)
    reported = base.where(holder_mask)  # NaN where not a holder or not reported
    coll = collected_by_wave(df, amount_col)

    out = pd.Series(np.nan, index=df.index)
    out[~holder_mask] = 0.0
    out[holder_mask] = base[holder_mask]
    need = holder_mask & base.isna() & coll

    if need.any():
        rep_mask = (holder_mask & reported.notna() & coll).to_numpy()
        if rep_mask.sum() >= 30:
            grp = _grp_series(df)
            loginc = np.log(pd.to_numeric(df["income_total_annual"], errors="coerce").clip(lower=1000))
            loginc = loginc.fillna(loginc.median())
            Xdf = pd.DataFrame({"loginc": loginc}, index=df.index)
            for w in sorted(df["wave"].dropna().unique())[1:]:
                Xdf[f"w{int(w)}"] = (df["wave"] == w).astype(float)
            for g in sorted(grp.unique())[1:]:
                Xdf[f"g{int(g)}"] = (grp == g).astype(float)
            feat = list(Xdf.columns)
            Xall = np.column_stack([np.ones(len(df))] + [Xdf[c].to_numpy() for c in feat])

            y = np.log1p(reported.fillna(0.0).to_numpy())
            Xr, yr = Xall[rep_mask], y[rep_mask]
            beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
            resid = yr - Xr @ beta

            if seed is None:
                seed = int(hashlib.md5(amount_col.encode()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            need_idx = np.where(need.to_numpy())[0]
            draws = rng.choice(resid, size=len(need_idx), replace=True)
            pred_log = Xall[need_idx] @ beta + draws
            cap = df[amount_col].quantile(p)
            pred = np.clip(np.expm1(pred_log), 0, cap)

            out_vals = out.to_numpy(copy=True)
            out_vals[need_idx] = pred
            out = pd.Series(out_vals, index=df.index)
        else:
            med_wn = reported.groupby([df["wave"], df["nationality"]]).transform("median")
            med_w = reported.groupby(df["wave"]).transform("median")
            impute = med_wn.fillna(med_w)
            out[need] = impute[need]

    out[~coll] = np.nan
    return out


def grp_code(nat, rj):
    """Simplified 5-group scheme:
       1 Haredi, 2 Dati, 3 Masorti (dati + traditional combined),
       4 Hiloni (secular, PLUS all residual: religiosity-NA Jews, 'Other'
         nationality, no-religion and unknown-nationality rows), 5 Arab.
    """
    if nat == 2:
        return 5  # Arab
    if nat == 1:                # Jewish
        if rj == 1:
            return 1            # Haredi
        if rj == 2:
            return 2            # Dati
        if rj in (3, 4):
            return 3            # Masorti (masorti-dati + masorti-traditional)
        return 4                # Hiloni (incl. Jewish religiosity not reported)
    return 4                    # 'Other' nationality / unknown -> folded into Hiloni


def _grp_series(df):
    return pd.Series([grp_code(df["nationality"][i], df["religion_jewish"][i])
                      for i in range(len(df))], index=df.index)


def hedonic_home(df, p=0.99, seed=0):
    """Hedonic imputation of primary-residence value.

    Owners who REPORTED a value keep their (winsorized) reported value. Owners who
    did NOT report are predicted from a log-linear hedonic model fitted on reporters:
        log(value) ~ rooms + sqm + log(sqm) + log(income) + dwelling-type
                     + crowding + parking + AC + socioeconomic-cluster DUMMIES
                     + year_built x locality-tier interaction + group + year dummies
    (Duan smearing applied when back-transforming). This makes the imputed value
    respond to house size and the household's socio-economic position, instead of a
    single flat {year x nationality} median. Non-owners contribute 0; the field is
    null in waves where it was not collected (wave 1).

    Returns (home_series, diagnostics_dict).
    """
    # WATERFALL value source: owner's declared value preferred; else CBS's own
    # estimate (omdanShuviDira, 2018+); the hedonic model fills only what remains.
    sub = winsorize(df["apt_value_subjective"], p)
    est = winsorize(df["apt_value_estimated"], p) if "apt_value_estimated" in df.columns \
        else pd.Series(np.nan, index=df.index)
    hv = sub.combine_first(est)                       # subjective if present, else CBS estimate
    n_sub = int(sub.notna().sum()); n_est_fill = int((sub.isna() & est.notna()).sum())
    cap = df["apt_value_subjective"].quantile(p)
    owner = (df["tenure_type"] == 1) | (df["owns_primary_apt"] == 1)
    coll = collected_by_wave(df, "apt_value_subjective")
    grp = _grp_series(df)

    # predictors (fill gaps so every owner gets a prediction)
    rooms = pd.to_numeric(df["apt_rooms"], errors="coerce")
    rooms = rooms.fillna(rooms.groupby([df["wave"], grp]).transform("median")).fillna(rooms.median())
    sqm = pd.to_numeric(df["apt_size_sqm"], errors="coerce")
    sqm = sqm.fillna(sqm.groupby(rooms.round()).transform("median")).fillna(sqm.median())
    # log(size): captures diminishing returns to floor area (a marginal m2 is worth
    # less on a big dwelling). Adding log(sqm) lifts held-out R2 ~+0.6pp over the
    # linear term alone; the linear `sqm` is KEPT (its coef collapses to ~0 once
    # log(sqm) is present, but it is retained as a core size predictor).
    logsqm = np.log(sqm.clip(lower=5))
    loginc = np.log(pd.to_numeric(df["income_total_annual"], errors="coerce").clip(lower=1000))
    loginc = loginc.fillna(loginc.median())
    # locality socioeconomic cluster (1-10): best available location/price proxy
    # (CBS suppresses true geography -- see below). Present 2018+ only; median-fill
    # NOT used as a level anymore. Instead entered as 10-level DUMMIES: this replaces
    # the old LINEAR cluster term and lifts held-out R2 from ~0.38 to ~0.46 (the SES
    # -> value gradient is strongly non-linear). Rows with a real cluster (2018+) get
    # one dummy per level (lowest level = dropped reference); rows where cluster is
    # MISSING (pre-2018) fall into the reference bucket (all dummies 0) -- clean,
    # because missing<=>pre-2018 is already fully absorbed by the wave dummies, so a
    # separate missing indicator would be collinear with them.
    clu_raw = pd.to_numeric(df["socioeconomic_cluster"], errors="coerce") \
        if "socioeconomic_cluster" in df.columns else pd.Series(np.nan, index=df.index)
    clu_present = clu_raw.notna()
    clu_int = clu_raw.round()
    clu_levels = sorted(int(x) for x in clu_int.dropna().unique())
    clu_ref = clu_levels[0] if clu_levels else None      # drop lowest as reference
    cluster_dummies = {f"clu{lv}": ((clu_int == lv) & clu_present).astype(float)
                       for lv in clu_levels if lv != clu_ref}

    # --- additional dwelling-quality predictors (field-scan additions) ---
    # apt_year_built (10-band vintage ordinal, higher=newer) is PERVERSE as a main
    # effect (newer -> lower value; the classic old-central-stock micro-location
    # confound). It is therefore NOT entered on its own. It IS entered in INTERACTED
    # form: its value slope differs by locality tier. In the top SES localities
    # (cluster>=8, a proxy for prime/central areas -- NOTE: this is AFFLUENCE, not
    # true geography, which CBS suppresses) older stock carries a preservation
    # premium (slope strongly negative in newer -> i.e. old = pricier); in low-SES
    # localities (cluster<=4, peripheral proxy) the slope is near-flat. This
    # interaction is correctly-signed on the central leg and adds ~+0.3pp held-out
    # R2 (robust across 20 seeds). CAVEAT: SES tier is a weak affluence proxy for
    # center/periphery; there is genuinely no district/peripherality field in this
    # survey (only present in the separate Social Survey). See
    # docs/hedonic_improvement_round2.md.
    yb_raw = pd.to_numeric(df["apt_year_built"], errors="coerce") \
        if "apt_year_built" in df.columns else pd.Series(np.nan, index=df.index)
    yb = yb_raw.fillna(yb_raw.median())
    yb_c = yb - yb.mean()                                 # centered ordinal band
    central = ((clu_raw >= 8) & clu_present).astype(float)   # top-SES / prime proxy
    periph = ((clu_raw <= 4) & clu_present).astype(float)    # low-SES / peripheral proxy
    yb_central = yb_c * central
    yb_periph = yb_c * periph

    # apt_type: 8-cat dwelling type, collapse thin cells into house-vs-apartment.
    # Raw codes: 1 roof apt, 2 garden apt, 3 duplex, 4 ordinary apt (base),
    # 5 two-family cottage, 6 row cottage, 7 single-family cottage, 8 detached house.
    apt_type_raw = pd.to_numeric(df["apt_type"], errors="coerce") if "apt_type" in df.columns \
        else pd.Series(np.nan, index=df.index)
    mode_type = apt_type_raw.mode(dropna=True)
    apt_type_raw = apt_type_raw.fillna(mode_type.iloc[0] if len(mode_type) else 4)
    is_house = apt_type_raw.isin([5, 6, 7, 8]).astype(float)          # cottage/detached
    is_special_apt = apt_type_raw.isin([1, 2, 3]).astype(float)       # roof/garden/duplex apt
    # (ordinary apt, code 4, is the omitted base category)

    # crowding = hh_size / apt_rooms, guarded against div-by-zero / missing rooms
    hh_size_n = pd.to_numeric(df["hh_size"], errors="coerce") if "hh_size" in df.columns \
        else pd.Series(np.nan, index=df.index)
    rooms_safe = rooms.replace(0, np.nan)
    crowd = hh_size_n / rooms_safe
    crowd = crowd.replace([np.inf, -np.inf], np.nan)
    crowd = crowd.fillna(crowd.median())

    # has_parking: binary 1/2 raw -> 1=has, else 0 (incl. NaN -> no parking)
    parking_raw = pd.to_numeric(df["has_parking"], errors="coerce") if "has_parking" in df.columns \
        else pd.Series(np.nan, index=df.index)
    has_parking = (parking_raw == 1).astype(float)

    # ac_type: 1=central, 2=unit(s), 3=none -> ordinal quality scale, higher=better
    # (3 - raw code) so central=2, units=1, none=0; NaN -> none (0)
    ac_raw = pd.to_numeric(df["ac_type"], errors="coerce") if "ac_type" in df.columns \
        else pd.Series(np.nan, index=df.index)
    ac_quality = (3 - ac_raw).clip(lower=0)
    ac_quality = ac_quality.fillna(0.0)

    Xdf = pd.DataFrame({"rooms": rooms, "sqm": sqm, "logsqm": logsqm, "loginc": loginc,
                        "is_house": is_house, "is_special_apt": is_special_apt,
                        "crowd": crowd, "has_parking": has_parking, "ac_quality": ac_quality,
                        "yb": yb_c, "yb_central": yb_central, "yb_periph": yb_periph},
                       index=df.index)
    for cname, cvals in cluster_dummies.items():          # 10-level cluster dummies (Stage 1)
        Xdf[cname] = cvals
    for w in sorted(df["wave"].dropna().unique())[1:]:
        Xdf[f"w{int(w)}"] = (df["wave"] == w).astype(float)
    for g in sorted(grp.unique())[1:]:
        Xdf[f"g{int(g)}"] = (grp == g).astype(float)
    feat = list(Xdf.columns)

    rep = (owner & hv.notna() & coll).to_numpy()
    Xall = np.column_stack([np.ones(len(df))] + [Xdf[c].to_numpy() for c in feat])
    y = np.log(hv.clip(lower=1000).to_numpy())

    Xr, yr = Xall[rep], y[rep]
    beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    resid = yr - Xr @ beta
    smear = float(np.mean(np.exp(resid)))
    r2 = float(1 - (resid ** 2).sum() / ((yr - yr.mean()) ** 2).sum())

    # held-out validation (80/20)
    rng = np.random.default_rng(seed)
    idx = np.where(rep)[0]; rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    b2, *_ = np.linalg.lstsq(Xall[tr], y[tr], rcond=None)
    pte = Xall[te] @ b2
    r2_oos = float(1 - ((y[te] - pte) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())

    pred = pd.Series(np.clip(np.exp(Xall @ beta) * smear, 0, cap), index=df.index)

    home = pd.Series(np.nan, index=df.index)
    home[~owner] = 0.0
    home[owner & hv.notna()] = hv[owner & hv.notna()]
    impute_mask = owner & hv.isna()
    home[impute_mask] = pred[impute_mask]
    # safety fallback if any prediction is NaN
    if home[impute_mask].isna().any():
        med = hv.where(owner).groupby([df["wave"], df["nationality"]]).transform("median")
        med = med.fillna(hv.where(owner).groupby(df["wave"]).transform("median"))
        still = impute_mask & home.isna()
        home[still] = med[still]
    home[~coll] = np.nan

    diag = {"r2_in": r2, "r2_oos": r2_oos, "smear": smear, "cap": float(cap),
            "n_reporters": int(rep.sum()), "n_imputed": int(impute_mask.sum()),
            "n_subjective": n_sub, "n_cbs_est_fill": n_est_fill,
            "coef": {f: float(b) for f, b in zip(["const"] + feat, beta)}}
    return home, diag


def pension_wealth(df, r=0.035, entry=27, ret_age=67, T=20, p=0.99, target_mean=830000.0):
    """Unified lifecycle pension / long-term-savings wealth per household.

    Replaces the earlier two-regime "accumulate vs. capitalize-at-M" stitch,
    which had (a) a ~59% non-economic jump at the accumulation->capitalization
    seam at age 67, and (b) no drawdown mechanism (retiree "wealth" merely
    rescaled observed pension income by age, so it could not decline in
    retirement). See docs/pension_model_pressure_test.md.

    ONE lifecycle curve:
      * Working years (age < 67): accumulate on (capped) HOUSEHOLD labor income
          (household total minus non-labor components; see body), with a per-
          earner ceiling (2x for married/partnered households, 1x otherwise),
          S(age) = c * min(hh_labor, ceiling*earners) * [((1+r)^Y - 1)/r],  Y = min(age,67)-27.
      * Retirement (age >= 67, or already drawing pension with little work income):
        the stock AT retirement is proxied from the household's own observed
        pension income (accumulated balances are not in the CBS survey -- only
        income flows), S67 = c * k_rel * pension_income, and is then DECUMULATED
        with an actuarial amortization schedule over T years at the same real
        return r:
            decum(t) = ((1+r)^T - (1+r)^t) / ((1+r)^T - 1),  t = age-67, decum(0)=1.
        So wealth PEAKS at retirement and DECLINES through retirement, instead of
        tracking the income flow.

    Two constants are solved (not assumed):
      * k_rel makes the mean retiree stock-at-retirement (ages 67-70) equal the
        mean worker stock at the boundary (ages 63-66), so the two legs agree at
        the seam BY CONSTRUCTION -> no discontinuity (decum(0)=1 at the boundary).
      * c (a single global level scalar, multiplying both legs equally so seam
        continuity is preserved) is solved so the wave-10 population-weighted
        pre-winsorization mean hits target_mean. Anchor = ~NIS 830k/HH, the
        conservative CMA long-term-savings aggregate (~NIS 2.5T end-2023) divided
        by ~3.05M households; see docs/admin_benchmark_pressure_test.md.

    Winsorized at p (floor 0); NaN where age is missing. NOTE: age_head is
    top-coded at 80 in the CBS source, so the retiree tail (81+) is not
    observable. Returns (series, diagnostics).
    """
    age = pd.to_numeric(df["age_head"], errors="coerce")
    pens = pd.to_numeric(df["income_pension_annual"], errors="coerce").fillna(0)

    # --- career base = HOUSEHOLD labor, not head-only (fix 2026-07-20) ---
    # income_work_annual (SacShnati_LePratMeAvoda) is the economic HEAD's
    # individual (לפרט) labor income; every other income line is HOUSEHOLD-level.
    # Using head-only dropped all secondary earners, so the modelled pension was
    # mis-distributed across households. Reuse the SAME household-labor residual
    # as the income-composition figure (build_household_frame's inc_work):
    # household total minus every household-level NON-LABOR component that sums
    # into it (pension, govt transfers, rental, interest, capital gains, income
    # from abroad, private transfers, training fund, provident fund, scholarship).
    # income_business is self-employment LABOR and is intentionally not subtracted.
    def col(c):
        return pd.to_numeric(df[c], errors="coerce").fillna(0)
    nonlabor = (col("income_pension_annual") + col("income_govt_transfers")
                + col("income_rental") + col("income_interest")
                + col("income_capital_gains") + col("income_abroad")
                + col("income_private_transfers") + col("income_training_fund")
                + col("income_provident_fund") + col("income_scholarship"))
    hh_labor = (col("income_total_annual") - nonlabor).clip(lower=0)

    # Contribution ceiling is PER-EARNER (NIS 50,695/mo = 608,340/yr, one
    # person's pension-contribution cap). Empirically ~6.5% of wave-10 households
    # (weighted) have household labor above one earner's ceiling, and ~89% of
    # those are married/partnered -- so a single-person cap would over-cap
    # dual-earner couples (base mean -10% vs -3% under an earner-count cap).
    # Proxy earner count: 2 for married/partnered households, 1 otherwise;
    # cap = ceiling * earners. This leaves ~1.6% of households still bound by
    # the cap (genuine high earners) with no implausible top tail.
    ceil = 50695 * 12
    married = pd.to_numeric(df["marital_status"], errors="coerce") == 1
    earners = pd.Series(np.where(married, 2, 1), index=df.index)
    career = np.minimum(hh_labor, ceil * earners)
    Y = np.clip(np.minimum(age.fillna(entry), ret_age) - entry, 0, ret_age - entry)
    AF = ((1 + r) ** Y - 1) / r
    # retiree household: low HOUSEHOLD labor relative to pension income
    retired = (age >= ret_age) | ((pens > 0) & (hh_labor < 0.5 * np.maximum(pens, 1)))

    # actuarial decumulation fraction of the retirement stock, by years since 67
    t = np.clip(age.fillna(ret_age) - ret_age, 0, T)
    decum = ((1 + r) ** T - (1 + r) ** t) / ((1 + r) ** T - 1)

    Sw_unit = pd.Series(career * AF, index=df.index)   # worker stock shape (before c)
    S67_unit = pens                                    # retiree stock proxy (before c*k_rel), pre-drawdown

    w_pos = (df["weight_hh"] > 0)
    def wm(mask, s):
        m = mask & s.notna() & w_pos
        ww = df["weight_hh"][m]
        return float((s[m] * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan

    # seam continuity: match the ADJACENT boundary ages so the two legs meet at 67
    # (worker accumulation peaks at 66, so calibrate retiree-67 stock to worker-66 stock).
    bw = wm((age >= 65) & (age < ret_age) & (~retired), Sw_unit)
    br = wm(retired & (age >= ret_age) & (age <= 68), S67_unit)
    k_rel = (bw / br) if (br and np.isfinite(br) and br > 0) else 0.0

    def build(c):
        w = pd.Series(c * Sw_unit, index=df.index)
        w[retired] = (c * k_rel * S67_unit * decum)[retired]
        return w.clip(lower=0)

    c = 0.10
    for _ in range(25):                                # solve c so wave-10 pre-winsor mean == target
        m = wm(df["wave"] == 10, build(c))
        if not np.isfinite(m) or m <= 0:
            break
        if abs(m - target_mean) / target_mean < 0.003:
            break
        c = c * target_mean / m

    wealth_precap = build(c)
    wealth_precap[age.isna()] = np.nan
    mean_precap = wm(df["wave"] == 10, wealth_precap)
    cap99 = wealth_precap.quantile(p)
    wealth = wealth_precap.clip(lower=0, upper=cap99)
    wealth[age.isna()] = np.nan

    w10 = (df["wave"] == 10)
    denom = df["weight_hh"][w10 & w_pos].sum()
    share_above_ceil = (float((df["weight_hh"][w10 & w_pos & (hh_labor > ceil)]).sum() / denom)
                        if denom > 0 else np.nan)
    share_still_capped = (float((df["weight_hh"][w10 & w_pos & (hh_labor > ceil * earners)]).sum() / denom)
                          if denom > 0 else np.nan)
    diag = {"c": float(c), "r": r, "k_rel": float(k_rel), "T": float(T),
            "mean_2023": wm(df["wave"] == 10, wealth), "cap99": float(cap99),
            "mean_2023_precap": float(mean_precap),
            "target_mean": float(target_mean),
            "career_base": "hh_labor_residual",
            "share_hh_above_1x_ceiling": share_above_ceil,
            "share_hh_still_capped": share_still_capped,
            "retired_share": wm(df["wave"] == 10, retired.astype(float))}
    return wealth, diag


def remap_marital(v):
    return {1: 1, 5: 2, 3: 3, 2: 3, 4: 4}.get(v)  # married/single/divorced+sep/widowed


def remap_edu(v):
    if v == 7:
        return 1  # no diploma
    if v in (1, 2):
        return 2  # secondary / matriculation
    if v == 3:
        return 3  # post-secondary non-academic
    if v in (4, 5, 6):
        return 4  # academic degree
    return None


def remap_ses(v):
    """Locality socioeconomic cluster 1-10 -> Low(1-3)/Mid(4-7)/High(8-10). 2018+ only."""
    if v is None:
        return None
    if v <= 3:
        return 1
    if v <= 7:
        return 2
    return 3


def remap_occ(v):
    if v in (1, 2):
        return 1  # managers & professionals
    if v in (3, 4):
        return 2  # technicians & clerical
    if v == 5:
        return 3  # sales & service
    if v in (6, 7):
        return 4  # skilled manual & agriculture
    if v in (8, 9):
        return 5  # operators & unskilled
    return None


def iv(x):
    """JSON-safe int or None."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def amt(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(round(x))


# ======================================================================
# ---- Aggregation helpers (new -- weighting/quantile logic for figures) ----
# ======================================================================


def weighted_quantile(values, weights, qs):
    """Weighted quantiles via the Hazen/midpoint plotting-position method:
    sort by value; cumulative weight minus half its own weight, normalized to
    [0,1]; linearly interpolate the target quantile(s) against value.
    Returns a list of floats (NaN where no data)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return [float("nan")] * len(qs)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cw = np.cumsum(weights) - 0.5 * weights
    cw = cw / weights.sum()
    return list(np.interp(qs, cw, values))


def wmean(df, mask, col):
    m = mask & df[col].notna()
    w = df.loc[m, "weight_hh"]
    if w.sum() <= 0:
        return float("nan")
    return float((df.loc[m, col] * w).sum() / w.sum())


def wshare1(df, mask, col):
    """Weighted share of rows where col == 1, among rows where col is not null."""
    m = mask & df[col].notna()
    w = df.loc[m, "weight_hh"]
    if w.sum() <= 0:
        return float("nan")
    hit = df.loc[m, col] == 1
    return float((w[hit]).sum() / w.sum())


def gini_weighted(values, weights):
    """Weighted Gini coefficient (mean-difference formula), computed on the
    non-null subset. Documented caveat: with net worth this can be negative or
    exceed [0,1]-typical bounds when values are negative (some HH have debt >
    assets); reported as-is, not clipped, since net worth is not a nonnegative
    quantity like income."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) < 2 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cw = np.cumsum(weights)
    cwn = cw / weights.sum()
    cwn_prev = np.concatenate([[0.0], cwn[:-1]])
    # weighted Gini via Lorenz curve trapezoid formula
    cum_val = np.cumsum(values * weights)
    total_val = cum_val[-1]
    if total_val == 0:
        return float("nan")
    lorenz = cum_val / total_val
    lorenz_prev = np.concatenate([[0.0], lorenz[:-1]])
    area = np.sum((cwn - cwn_prev) * (lorenz + lorenz_prev) / 2.0)
    return float(1 - 2 * area)


def top_share(values, weights, pct=0.90):
    """Weighted share of total value held by the top (1-pct) of the weighted
    distribution (e.g. pct=0.90 -> share held by top decile)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & (weights > 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    thresh = weighted_quantile(values, weights, [pct])[0]
    total = (values * weights).sum()
    if total == 0:
        return float("nan")
    top_mask = values >= thresh
    return float((values[top_mask] * weights[top_mask]).sum() / total)


def build_household_frame(p=0.99):
    """Read the panel CSV, apply the models, and return a per-household display
    DataFrame with the same semantics as build_balance_sheet_data.py's `cols`
    (winsorized/imputed amounts, modelled home & pension, remapped
    demographics). This is the single source aggregation reads from.

    `p` is the winsorization percentile threaded into hedonic_home/bake_amount/
    pension_wealth (default 0.99, matching the original build). main() also
    builds a second frame at p=0.995 used ONLY for the net-worth distribution
    figure (Gini/top-share/percentiles) -- see fig_networth_distribution call
    site and rationale there. All other figures keep the default p=0.99."""
    df = pd.read_csv(CSV_IN, low_memory=False)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["weight_hh"] > 0].reset_index(drop=True)

    owner = (df["tenure_type"] == 1) | (df["owns_primary_apt"] == 1)
    land_holder = (df["owns_agricultural_land"] == 1) | (df["owns_other_land"] == 1) | (df["owns_other_property"] == 1)

    home, home_diag = hedonic_home(df, p=p)
    addprop = bake_amount(df, "value_additional_apts", df["owns_additional_apt"] == 1, p=p)
    land = bake_amount(df, "total_land_value", land_holder, p=p)
    dep = bake_amount(df, "amount_deposits", df["has_deposits"] == 1, p=p)
    sav = bake_amount(df, "amount_savings", df["has_savings_plan"] == 1, p=p)
    mut = bake_amount(df, "amount_mutual_funds", df["has_mutual_funds"] == 1, p=p)
    sec = bake_amount(df, "amount_securities", df["has_stocks"] == 1, p=p)
    mortg = bake_amount(df, "mortgage_balance", df["has_mortgage"] == 1, p=p)
    consumer = bake_amount(df, "non_housing_loan_balance", df["has_non_housing_loan"] == 1, p=p)
    deposits_sav = (dep + sav) * ADMIN_SCALE["dep_sav"]
    invest = (mut + sec) * ADMIN_SCALE["invest"]
    mortg = mortg * ADMIN_SCALE["mortg"]
    consumer = consumer * ADMIN_SCALE["consumer"]
    # pension_wealth is a separate workstream's modelled line -- NOT scaled here.
    pension, pension_diag = pension_wealth(df, p=p)

    inc_total = winsorize(df["income_total_annual"], 0.995)
    ipc = winsorize(df["income_per_capita"], 0.995)

    def monthly(col):
        return (winsorize(df[col], 0.995).fillna(0.0) / 12.0)

    inc_pension = monthly("income_pension_annual")
    inc_govt = monthly("income_govt_transfers")
    inc_rental = monthly("income_rental")
    inc_interest = monthly("income_interest")
    inc_capgains = monthly("income_capital_gains")
    # --- WORK component = HOUSEHOLD labor residual (fix 2026-07-20) ---
    # The panel's income_work_annual (SacShnati_LePratMeAvoda) is the economic
    # HEAD's individual (לפרט) labor income, whereas income_total_annual and every
    # other income line here are HOUSEHOLD-level (למ"ב). Using the head-only field
    # as the "work" component dropped all secondary earners (~40% of household
    # labor), biasing mean_work low for every group and pulling the component-sum
    # total ~29% below income_total_annual/12 (see docs/arab_income_diagnostic.md).
    #
    # Fix: define monthly work as the household labor RESIDUAL -- the household
    # total minus every household-level NON-LABOR component that sums into it:
    #   pension, govt transfers, rental, interest, capital gains, income from
    #   abroad, private transfers/support, training fund, provident fund,
    #   education scholarship. income_business (HEsekShnatit) is self-employment
    #   LABOR income and is deliberately NOT subtracted (it belongs in work; it is
    #   ~0 in this data anyway). Clipped at 0 (a few HH have non-labor components
    #   exceeding the winsorized total due to rounding/imputation, and genuinely
    #   zero-labor households -- retirees/transfer-dependent -- land at 0).
    # This makes the work line HOUSEHOLD-level like the others, so the six-
    # component sum (mean_total_monthly) ties to income_total_annual/12 (to within
    # the small unshown-non-labor remainder: abroad/private/training/provident/
    # scholarship, ~1%, which are not displayed as their own bars).
    inc_total_m = monthly("income_total_annual")
    inc_abroad = monthly("income_abroad")
    inc_private = monthly("income_private_transfers")
    inc_training = monthly("income_training_fund")
    inc_provident = monthly("income_provident_fund")
    inc_scholarship = monthly("income_scholarship")
    inc_work = (inc_total_m - (inc_pension + inc_govt + inc_rental + inc_interest
                               + inc_capgains + inc_abroad + inc_private
                               + inc_training + inc_provident + inc_scholarship)
                ).clip(lower=0)

    n = len(df)
    grp = pd.Series([grp_code(df["nationality"][i], df["religion_jewish"][i]) for i in range(n)])
    ten = pd.Series([iv(df["tenure_type"][i]) if df["tenure_type"][i] in (1, 2)
                     else (3 if not np.isnan(df["tenure_type"][i]) else None) for i in range(n)])
    sz = pd.Series([(5 if df["hh_size"][i] >= 5 else iv(df["hh_size"][i]))
                    if not np.isnan(df["hh_size"][i]) else None for i in range(n)])
    age = pd.Series([iv(df["age_head"][i]) for i in range(n)])
    mar = pd.Series([remap_marital(iv(df["marital_status"][i])) for i in range(n)])
    kids = pd.Series([(1 if df["hh_children"][i] > 0 else 0)
                      if not np.isnan(df["hh_children"][i]) else None for i in range(n)])
    edu = pd.Series([remap_edu(iv(df["education_degree"][i])) for i in range(n)])
    occ = pd.Series([remap_occ(iv(df["occupation"][i])) for i in range(n)])
    ses = pd.Series([remap_ses(iv(df["socioeconomic_cluster"][i])) for i in range(n)])
    # raw 10-level locality socioeconomic cluster (1-10, present 2018+), used by the
    # cross-tab explorer/matrix. iv() -> None where missing (odd/pre-2018 waves).
    ses10 = pd.Series([iv(df["socioeconomic_cluster"][i]) for i in range(n)])

    # available = financial rotation collected this wave (even waves only)
    available = df["wave"].isin(EVEN_WAVES)

    tot_re_full = home + addprop + land          # only meaningful where available
    tot_re = np.where(available, tot_re_full, home)  # odd waves: RE total = home only (matches dashboard)
    tot_fin = deposits_sav + invest + pension
    tot_a_full = tot_re_full + tot_fin
    tot_l = mortg + consumer
    nw_full = tot_a_full - tot_l

    hh = pd.DataFrame({
        "wave": df["wave"], "year": df["wave"].map(WAVE_YEAR),
        "grp": grp, "ten": ten, "sz": sz, "age": age, "mar": mar, "kids": kids,
        "edu": edu, "occ": occ, "ses": ses, "ses10": ses10,
        # raw ISIC economic-sector letter (used only by fig_metric_by_breakdown's
        # industry cut; no existing figure references it -- purely additive).
        "sector": df["sector"] if "sector" in df.columns else pd.Series(np.nan, index=df.index),
        "weight_hh": df["weight_hh"],
        "available": available,
        "home": home, "addprop": addprop, "land": land,
        "tot_re": pd.Series(tot_re, index=df.index),          # home-only in odd waves
        "tot_re_full": tot_re_full,                            # NaN in odd waves (addprop/land NaN)
        "dep_sav": deposits_sav, "invest": invest, "pension": pension,
        "tot_fin": tot_fin,                                     # NaN in odd waves
        "tot_a": tot_a_full,                                    # NaN in odd waves
        "mortg": mortg, "consumer": consumer,
        "tot_l": tot_l,                                         # NaN in odd waves
        "nw": nw_full,                                          # NaN in odd waves
        "inc_total": inc_total, "ipc": ipc,
        "m_work": inc_work, "m_pension": inc_pension, "m_govt": inc_govt,
        "m_rental": inc_rental, "m_interest": inc_interest, "m_capgains": inc_capgains,
        "i_mortg": df["has_mortgage"], "i_loan": df["has_non_housing_loan"],
        "i_addl": df["owns_additional_apt"], "i_pension": df["has_pension_fund"],
        "i_invest": ((df["has_mutual_funds"] == 1) | (df["has_stocks"] == 1)).astype(float).where(
            df["has_mutual_funds"].notna() | df["has_stocks"].notna()),
    })
    diags = {"home": home_diag, "pension": pension_diag}
    return hh, diags


# ======================================================================
# ---- Figure builders ----
# ======================================================================

MANIFEST = []


def write_csv(name, rows, fieldnames, description):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    MANIFEST.append({"file": name, "description": description})
    print(f"  wrote {name} ({len(rows)} rows)")


def fig_national_balance_sheet(hh):
    latest = hh[hh["wave"] == LATEST_WAVE]
    w = latest["weight_hh"]
    m_home = wmean(latest, latest.index == latest.index, "home")
    m_add = wmean(latest, latest.index == latest.index, "addprop")
    m_land = wmean(latest, latest.index == latest.index, "land")
    m_dep = wmean(latest, latest.index == latest.index, "dep_sav")
    m_inv = wmean(latest, latest.index == latest.index, "invest")
    m_pen = wmean(latest, latest.index == latest.index, "pension")
    m_mortg = wmean(latest, latest.index == latest.index, "mortg")
    m_cons = wmean(latest, latest.index == latest.index, "consumer")
    tot_re = m_home + m_add + m_land
    tot_fin = m_dep + m_inv + m_pen
    tot_a = tot_re + tot_fin
    tot_l = m_mortg + m_cons
    nw = tot_a - tot_l
    rows = [
        {"item": "Primary Residence", "category": "asset_re", "value_nis": round(m_home), "is_modeled": False},
        {"item": "Additional Properties", "category": "asset_re", "value_nis": round(m_add), "is_modeled": False},
        {"item": "Land & Other Real Estate", "category": "asset_re", "value_nis": round(m_land), "is_modeled": False},
        {"item": "Total Real Estate", "category": "subtotal", "value_nis": round(tot_re), "is_modeled": False},
        {"item": "Bank Deposits & Savings", "category": "asset_fin", "value_nis": round(m_dep), "is_modeled": False},
        {"item": "Investment Portfolio", "category": "asset_fin", "value_nis": round(m_inv), "is_modeled": False},
        {"item": "Pension & Retirement Savings", "category": "asset_fin", "value_nis": round(m_pen), "is_modeled": True},
        {"item": "Total Financial Assets", "category": "subtotal", "value_nis": round(tot_fin), "is_modeled": False},
        {"item": "Total Assets", "category": "total", "value_nis": round(tot_a), "is_modeled": False},
        {"item": "Mortgage", "category": "liability", "value_nis": round(m_mortg), "is_modeled": False},
        {"item": "Consumer Loans", "category": "liability", "value_nis": round(m_cons), "is_modeled": False},
        {"item": "Total Liabilities", "category": "total", "value_nis": round(tot_l), "is_modeled": False},
        {"item": "Net Worth", "category": "networth", "value_nis": round(nw), "is_modeled": False},
    ]
    write_csv("fig_national_balance_sheet.csv", rows,
              ["item", "category", "value_nis", "is_modeled"],
              "National weighted-average household balance sheet, wave 10 (2023). "
              "is_modeled=True only for Pension (matches dashboard's original modelled/◇ flag); "
              "home value also uses hedonic imputation for non-reporting owners but was not flagged "
              "in the original UI, so that convention is preserved here.")


def fig_networth_by_group(hh):
    latest = hh[hh["wave"] == LATEST_WAVE]
    rows = []
    for code in sorted(latest["grp"].dropna().unique()):
        sub = latest[latest["grp"] == code]
        if len(sub) == 0:
            continue
        n = int(len(sub))
        mean_nw = wmean(sub, sub.index == sub.index, "nw")
        med_nw = weighted_quantile(sub["nw"], sub["weight_hh"], [0.5])[0]
        mean_ta = wmean(sub, sub.index == sub.index, "tot_a")
        mean_tl = wmean(sub, sub.index == sub.index, "tot_l")
        mean_inc = wmean(sub, sub.index == sub.index, "inc_total")
        rows.append({"group": GROUP_LABELS.get(int(code), str(code)), "n_households": n,
                     "mean_networth": round(mean_nw) if not np.isnan(mean_nw) else "",
                     "median_networth": round(med_nw) if not np.isnan(med_nw) else "",
                     "mean_total_assets": round(mean_ta) if not np.isnan(mean_ta) else "",
                     "mean_total_liabilities": round(mean_tl) if not np.isnan(mean_tl) else "",
                     "mean_income_annual": round(mean_inc) if not np.isnan(mean_inc) else ""})
    write_csv("fig_networth_by_group.csv", rows,
              ["group", "n_households", "mean_networth", "median_networth",
               "mean_total_assets", "mean_total_liabilities", "mean_income_annual"],
              "Net worth & balance sheet aggregates by social group, wave 10 (2023). Weighted by weight_hh; "
              "median_networth uses the weighted-quantile helper. n_households is the unweighted sample count.")


def fig_composition_by_group(hh):
    latest = hh[hh["wave"] == LATEST_WAVE]
    classes = [("Real Estate", "tot_re"), ("Financial ex-pension", None), ("Pension", "pension")]
    rows = []
    for code in sorted(latest["grp"].dropna().unique()):
        sub = latest[latest["grp"] == code]
        if len(sub) == 0:
            continue
        lab = GROUP_LABELS.get(int(code), str(code))
        m_re = wmean(sub, sub.index == sub.index, "tot_re")
        m_finexp = wmean(sub, sub.index == sub.index, "dep_sav") + wmean(sub, sub.index == sub.index, "invest")
        m_pen = wmean(sub, sub.index == sub.index, "pension")
        tot_a = m_re + m_finexp + m_pen
        for name, val in [("Real Estate", m_re), ("Financial ex-pension", m_finexp), ("Pension", m_pen)]:
            share = val / tot_a if tot_a else float("nan")
            rows.append({"group": lab, "asset_class": name,
                         "share_of_assets": round(share, 4) if not np.isnan(share) else "",
                         "mean_value_nis": round(val) if not np.isnan(val) else ""})
    write_csv("fig_composition_by_group.csv", rows,
              ["group", "asset_class", "share_of_assets", "mean_value_nis"],
              "Asset composition shares by social group, wave 10 (2023): Real Estate / Financial "
              "ex-pension (deposits+savings+investments) / Pension, as shares of (RE+Fin ex-pension+Pension).")


def fig_networth_by_dimension(hh):
    latest = hh[hh["wave"] == LATEST_WAVE]
    rows = []

    def emit(dimension, cat_label, mask):
        sub = latest[mask]
        n = int(len(sub))
        if n == 0:
            rows.append({"dimension": dimension, "category": cat_label, "n_households": 0,
                         "median_networth": "", "mean_networth": ""})
            return
        mean_nw = wmean(sub, sub.index == sub.index, "nw")
        med_nw = weighted_quantile(sub["nw"], sub["weight_hh"], [0.5])[0]
        rows.append({"dimension": dimension, "category": cat_label, "n_households": n,
                     "median_networth": round(med_nw) if not np.isnan(med_nw) else "",
                     "mean_networth": round(mean_nw) if not np.isnan(mean_nw) else ""})

    for code, lab in EDU_LABELS.items():
        emit("Education", lab, latest["edu"] == code)
    for lab, lo, hi in AGE_BANDS:
        emit("Age band", lab, (latest["age"] >= lo) & (latest["age"] < hi))
    for code, lab in TENURE_LABELS.items():
        emit("Tenure", lab, latest["ten"] == code)
    for code, lab in SIZE_LABELS.items():
        emit("Household size", lab, latest["sz"] == code)
    for code, lab in OCC_LABELS.items():
        emit("Occupation", lab, latest["occ"] == code)
    # audit fix 2026-07-20: heads with no coded occupation (retired / unemployed /
    # not-in-labour-force) were silently dropped, leaving the Occupation base non-
    # exhaustive & not comparable to the headline. Add them as an explicit category.
    emit("Occupation", "Not employed / other", latest["occ"].isna())
    for code, lab in SES_LABELS.items():
        emit("Locality SES", lab, latest["ses"] == code)

    write_csv("fig_networth_by_dimension.csv", rows,
              ["dimension", "category", "n_households", "median_networth", "mean_networth"],
              "Net worth by socio-demographic cut, wave 10 (2023). Weighted mean & weighted-quantile median.")


def fig_networth_distribution(hh):
    """NOTE ON WINSORIZATION (audit fix 2026-07-18): this figure is now built
    from a household frame constructed at p=0.995 (see main(), hh_dist), not
    the default p=0.99 used for every other figure. Rationale: capping every
    component amount at its 99th percentile is appropriate for a HEADLINE
    average/balance-sheet display (guards against a handful of data-entry
    errors dominating a mean), but it also mechanically caps the genuine top
    tail of the distribution -- directly suppressing top10_share and Gini,
    which are point estimates of exactly that tail. p99.5 keeps the same
    data-entry-error guard (still discards the top 0.5%) while preserving
    roughly twice as much real top-tail dispersion for the inequality figures
    specifically. This is not tuned to hit any particular Gini/top-share
    number -- it is applied uniformly to the whole distribution figure, and
    the same 99.5th-percentile threshold is already used elsewhere in this
    file for income (see build_household_frame's income winsorization)."""
    latest = hh[hh["wave"] == LATEST_WAVE]
    rows = []

    def emit(scope, sub):
        vals, wts = sub["nw"].to_numpy(), sub["weight_hh"].to_numpy()
        qs = weighted_quantile(vals, wts, [0.10, 0.25, 0.50, 0.75, 0.90])
        mean_v = wmean(sub, sub.index == sub.index, "nw")
        g = gini_weighted(vals, wts)
        t10 = top_share(vals, wts, 0.90)
        rows.append({"scope": scope, "p10": round(qs[0]) if not np.isnan(qs[0]) else "",
                     "p25": round(qs[1]) if not np.isnan(qs[1]) else "",
                     "p50": round(qs[2]) if not np.isnan(qs[2]) else "",
                     "p75": round(qs[3]) if not np.isnan(qs[3]) else "",
                     "p90": round(qs[4]) if not np.isnan(qs[4]) else "",
                     "mean": round(mean_v) if not np.isnan(mean_v) else "",
                     "top10_share": round(t10, 4) if not np.isnan(t10) else "",
                     "gini": round(g, 4) if not np.isnan(g) else ""})

    emit("All", latest)
    for code in sorted(latest["grp"].dropna().unique()):
        sub = latest[latest["grp"] == code]
        if len(sub) > 0:
            emit(GROUP_LABELS.get(int(code), str(code)), sub)

    write_csv("fig_networth_distribution.csv", rows,
              ["scope", "p10", "p25", "p50", "p75", "p90", "mean", "top10_share", "gini"],
              "Net-worth distribution, wave 10 (2023): weighted percentiles (Hazen/midpoint method), "
              "weighted mean, top10_share (weighted share of total net worth held by top decile by net worth), "
              "and a weighted Gini coefficient computed on net worth (can be unusual since some households "
              "have negative net worth -- reported as-is, not clipped). Built from a p=0.995 winsorization "
              "cap (vs p=0.99 for every other figure) to avoid double-suppressing the top tail that these "
              "specific inequality statistics are trying to measure -- see function docstring.")


def fig_income_wealth(hh):
    latest = hh[hh["wave"] == LATEST_WAVE].copy()
    vals, wts = latest["inc_total"].to_numpy(), latest["weight_hh"].to_numpy()
    edges = weighted_quantile(vals, wts, [i / 10 for i in range(1, 10)])
    valid = latest["inc_total"].notna()
    decile = pd.Series(np.nan, index=latest.index)
    decile[valid] = np.searchsorted(edges, latest.loc[valid, "inc_total"], side="right") + 1
    latest["decile"] = decile
    rows = []
    for d in range(1, 11):
        sub = latest[latest["decile"] == d]
        n = int(len(sub))
        if n == 0:
            continue
        mean_inc = wmean(sub, sub.index == sub.index, "inc_total")
        mean_nw = wmean(sub, sub.index == sub.index, "nw")
        mean_ta = wmean(sub, sub.index == sub.index, "tot_a")
        re_v = wmean(sub, sub.index == sub.index, "tot_re")
        fin_v = wmean(sub, sub.index == sub.index, "dep_sav") + wmean(sub, sub.index == sub.index, "invest")
        pen_v = wmean(sub, sub.index == sub.index, "pension")
        cap_inc = (wmean(sub, sub.index == sub.index, "m_rental")
                   + wmean(sub, sub.index == sub.index, "m_interest")
                   + wmean(sub, sub.index == sub.index, "m_capgains"))
        m_work = wmean(sub, sub.index == sub.index, "m_work")
        m_pension = wmean(sub, sub.index == sub.index, "m_pension")
        m_govt = wmean(sub, sub.index == sub.index, "m_govt")
        tot_monthly = m_work + m_pension + m_govt + cap_inc
        re_share = re_v / mean_ta if mean_ta else float("nan")
        fin_share = fin_v / mean_ta if mean_ta else float("nan")
        pen_share = pen_v / mean_ta if mean_ta else float("nan")
        cap_share = cap_inc / tot_monthly if tot_monthly else float("nan")
        rows.append({"income_decile": d, "n_households": n,
                     "mean_income_annual": round(mean_inc) if not np.isnan(mean_inc) else "",
                     "mean_networth": round(mean_nw) if not np.isnan(mean_nw) else "",
                     "mean_total_assets": round(mean_ta) if not np.isnan(mean_ta) else "",
                     "real_estate_share": round(re_share, 4) if not np.isnan(re_share) else "",
                     "financial_share": round(fin_share, 4) if not np.isnan(fin_share) else "",
                     "pension_share": round(pen_share, 4) if not np.isnan(pen_share) else "",
                     "capital_income_share": round(cap_share, 4) if not np.isnan(cap_share) else ""})
    write_csv("fig_income_wealth.csv", rows,
              ["income_decile", "n_households", "mean_income_annual", "mean_networth", "mean_total_assets",
               "real_estate_share", "financial_share", "pension_share", "capital_income_share"],
              "Income-wealth relationship, wave 10 (2023): weighted deciles of income_total_annual (decile 10 "
              "= highest income), each decile's mean income/net worth/assets, asset-composition shares, and "
              "the share of monthly income coming from capital (rental+interest+capital gains).")


def fig_trajectory(hh):
    rows = []
    scopes = [("All", None)] + [(GROUP_LABELS.get(int(c), str(c)), int(c))
                                 for c in sorted(hh["grp"].dropna().unique()) if c != 0]
    for wave in sorted(hh["wave"].dropna().unique()):
        wsub = hh[hh["wave"] == wave]
        avail = bool(wsub["available"].iloc[0]) if len(wsub) else False
        for scope, code in scopes:
            sub = wsub if code is None else wsub[wsub["grp"] == code]
            n = len(sub)
            ipc_mean = wmean(sub, sub.index == sub.index, "ipc") if n else float("nan")
            if avail and n:
                mean_nw = wmean(sub, sub.index == sub.index, "nw")
                med_nw = weighted_quantile(sub["nw"], sub["weight_hh"], [0.5])[0]
                note = ""
            else:
                mean_nw = float("nan"); med_nw = float("nan")
                note = "financial assets/liabilities not collected this wave (odd-year rotation) -- net worth undefined"
            rows.append({"year": int(WAVE_YEAR[int(wave)]), "scope": scope,
                         "mean_networth": round(mean_nw) if not np.isnan(mean_nw) else "",
                         "median_networth": round(med_nw) if not np.isnan(med_nw) else "",
                         "income_per_capita": round(ipc_mean) if not np.isnan(ipc_mean) else "",
                         "note": note})
    write_csv("fig_trajectory.csv", rows,
              ["year", "scope", "mean_networth", "median_networth", "income_per_capita", "note"],
              "Net worth & income per capita over all 10 waves (2012-2023), All + each social group. "
              "mean_networth/median_networth are null (not fabricated) in odd waves where financial assets & "
              "liabilities were not collected -- see the note column.")


def fig_participation(hh):
    latest = hh[hh["wave"] == LATEST_WAVE]
    rows = []
    for code in sorted(latest["grp"].dropna().unique()):
        sub = latest[latest["grp"] == code]
        if len(sub) == 0:
            continue
        rows.append({"group": GROUP_LABELS.get(int(code), str(code)),
                     "pct_with_mortgage": round(wshare1(sub, sub.index == sub.index, "i_mortg"), 4),
                     "pct_with_pension": round(wshare1(sub, sub.index == sub.index, "i_pension"), 4),
                     "pct_owns_additional_property": round(wshare1(sub, sub.index == sub.index, "i_addl"), 4),
                     "pct_with_investment": round(wshare1(sub, sub.index == sub.index, "i_invest"), 4)})
    write_csv("fig_participation.csv", rows,
              ["group", "pct_with_mortgage", "pct_with_pension", "pct_owns_additional_property", "pct_with_investment"],
              "Participation rates by social group, wave 10 (2023): weighted share of households holding "
              "each item (share of ==1 among non-null responses).")


def fig_ratios_by_group(hh):
    """ADDITIONAL (beyond contract): the dashboard's 'ratios' panel (Leverage,
    Equity, Debt-to-Income) was computed in-browser and is not covered by any
    contract table -- reproduced here per-group so it isn't dropped."""
    latest = hh[hh["wave"] == LATEST_WAVE]
    rows = []
    for code in sorted(latest["grp"].dropna().unique()):
        sub = latest[latest["grp"] == code]
        if len(sub) == 0:
            continue
        home_m = wmean(sub, sub.index == sub.index, "home")
        totl_m = wmean(sub, sub.index == sub.index, "tot_l")
        nw_m = wmean(sub, sub.index == sub.index, "nw")
        totfin_m = wmean(sub, sub.index == sub.index, "tot_fin")
        inc_m = wmean(sub, sub.index == sub.index, "inc_total")
        tota_m = wmean(sub, sub.index == sub.index, "tot_a")
        lev = min(totl_m / home_m, 2) if home_m else float("nan")
        # equity ratio = net worth / TOTAL assets (audit fix 2026-07-20: denominator
        # was home+financial only, which excluded additional-property+land that ARE in
        # net worth -> inflated ratio that could exceed 1; now the full asset base).
        eq = nw_m / tota_m if tota_m else float("nan")
        dti = totl_m / inc_m if inc_m else float("nan")
        rows.append({"group": GROUP_LABELS.get(int(code), str(code)),
                     "leverage_ratio": round(lev, 3) if not np.isnan(lev) else "",
                     "equity_ratio": round(eq, 4) if not np.isnan(eq) else "",
                     "debt_to_income_ratio": round(dti, 3) if not np.isnan(dti) else ""})
    write_csv("fig_ratios_by_group.csv", rows,
              ["group", "leverage_ratio", "equity_ratio", "debt_to_income_ratio"],
              "ADDED beyond the figure contract: reproduces the dashboard's Leverage/Equity/DTI ratio "
              "panel (previously computed client-side) as weighted-mean-based ratios by group, wave 10.")


def fig_income_composition_by_group(hh):
    """ADDITIONAL (beyond contract): the dashboard's 'Monthly Income by Source'
    panel (work/pension/govt/rental/interest/capgains) was computed in-browser
    from raw columns and is not covered by any contract table."""
    latest = hh[hh["wave"] == LATEST_WAVE]
    rows = []
    scopes = [("All", None)] + [(GROUP_LABELS.get(int(c), str(c)), int(c))
                                 for c in sorted(latest["grp"].dropna().unique()) if c != 0]
    for lab, code in scopes:
        sub = latest if code is None else latest[latest["grp"] == code]
        if len(sub) == 0:
            continue
        vals = {k: wmean(sub, sub.index == sub.index, k)
                for k in ["m_work", "m_pension", "m_govt", "m_rental", "m_interest", "m_capgains"]}
        tot = sum(v for v in vals.values() if not np.isnan(v))
        rows.append({"group": lab,
                     "mean_work": round(vals["m_work"]), "mean_pension": round(vals["m_pension"]),
                     "mean_govt_transfers": round(vals["m_govt"]), "mean_rental": round(vals["m_rental"]),
                     "mean_interest": round(vals["m_interest"]), "mean_capital_gains": round(vals["m_capgains"]),
                     "mean_total_monthly": round(tot)})
    write_csv("fig_income_composition_by_group.csv", rows,
              ["group", "mean_work", "mean_pension", "mean_govt_transfers", "mean_rental",
               "mean_interest", "mean_capital_gains", "mean_total_monthly"],
              "ADDED beyond the figure contract: reproduces the dashboard's monthly income-by-source "
              "bar chart (All + each group), wave 10 (2023), all NIS/month.")


def crosstab_specs(latest):
    """Shared category definitions for BOTH the 1-D cross-tab explorer
    (fig_metric_by_breakdown) and the 2-D matrix (fig_metric_matrix), so the two
    figures use IDENTICAL breakdowns / category masks / labels / order.

    Returns a list of (breakdown_key, category_label, boolean_mask) tuples in
    canonical category order, computed on the passed wave-10 `latest` frame.
    Mutates `latest` by adding a `decile` column (income deciles). SES is the
    FULL 10-level locality cluster (ses10), categories "1".."10"."""
    # income deciles -- reuse fig_income_wealth's exact weighted-decile logic
    vals, wts = latest["inc_total"].to_numpy(), latest["weight_hh"].to_numpy()
    edges = weighted_quantile(vals, wts, [i / 10 for i in range(1, 10)])
    valid = latest["inc_total"].notna()
    decile = pd.Series(np.nan, index=latest.index)
    decile[valid] = np.searchsorted(edges, latest.loc[valid, "inc_total"], side="right") + 1
    latest["decile"] = decile

    # collapse raw ISIC sector letters into readable industry groups
    sec_map = {}
    for lab, letters in SECTOR_GROUPS:
        for L in letters:
            sec_map[L] = lab
    sec = latest["sector"].map(
        lambda s: sec_map.get(str(s).strip().upper(), NOT_EMPLOYED_LABEL)
        if pd.notna(s) else NOT_EMPLOYED_LABEL)

    # (breakdown_key, category_label, boolean mask) in canonical category order
    specs = []
    for c in (1, 2, 3, 4, 5):
        specs.append(("group", GROUP_LABELS[c], latest["grp"] == c))
    for d in range(1, 11):
        specs.append(("income_decile", str(d), latest["decile"] == d))
    for c in range(1, 11):     # 10-level locality socioeconomic cluster (was 3-bucket)
        specs.append(("ses", str(c), latest["ses10"] == c))
    for c in (1, 2, 3, 4, 5):
        specs.append(("occ", OCC_LABELS[c], latest["occ"] == c))
    specs.append(("occ", NOT_EMPLOYED_LABEL, latest["occ"].isna()))
    for c in (1, 2, 3, 4):
        specs.append(("edu", EDU_LABELS[c], latest["edu"] == c))
    for lab, lo, hi in AGE_BANDS:
        specs.append(("age_band", lab, (latest["age"] >= lo) & (latest["age"] < hi)))
    for c in (1, 2, 3):
        specs.append(("tenure", TENURE_LABELS[c], latest["ten"] == c))
    for c in (1, 2, 3, 4, 5):
        specs.append(("hh_size", SIZE_LABELS[c], latest["sz"] == c))
    for c in (1, 2, 3, 4):
        specs.append(("marital", MARITAL_LABELS[c], latest["mar"] == c))
    for c in (0, 1):
        specs.append(("children", KIDS_LABELS[c], latest["kids"] == c))
    for lab, _ in SECTOR_GROUPS:
        specs.append(("sector", lab, sec == lab))
    specs.append(("sector", NOT_EMPLOYED_LABEL, sec == NOT_EMPLOYED_LABEL))
    return specs


def fig_metric_by_breakdown(hh):
    """Cross-tab explorer source: for WAVE 10 only, the weighted MEAN of every
    metric (net worth, assets, liabilities, income, ...) within every category
    of every breakdown (social group, income decile, SES, occupation, ...), in
    WIDE format -- one row per (breakdown, category), one column per metric.

    The front-end explorer only filters these pre-computed rows to a chosen
    (metric, breakdown) and plots them; NO calculation happens in the browser.
    Every value uses the SAME wmean() and the SAME income-decile logic as the
    dedicated figures, so e.g. net_worth x group here == fig_networth_by_group
    mean_networth, income_annual x decile == fig_income_wealth mean_income_annual,
    and pension x group == fig_composition_by_group Pension mean_value_nis."""
    latest = hh[hh["wave"] == LATEST_WAVE].copy()
    specs = crosstab_specs(latest)

    rows = []
    for bkey, clabel, mask in specs:
        sub = latest[mask]
        n = int(len(sub))
        if n == 0:
            continue
        row = {"breakdown": bkey, "category": clabel, "n_households": n}
        for mkey, col in CROSS_TAB_METRICS:
            v = wmean(sub, sub.index == sub.index, col)
            row[mkey] = round(v) if not np.isnan(v) else ""
        rows.append(row)

    fieldnames = ["breakdown", "category", "n_households"] + [m[0] for m in CROSS_TAB_METRICS]
    write_csv("fig_metric_by_breakdown.csv", rows, fieldnames,
              "Cross-tab explorer: weighted-mean of every metric by every breakdown category, "
              "wave 10 (2023), wide format (one row per breakdown x category, one column per metric). "
              "n_households is the unweighted sample count per cell. Feeds the two-dropdown explorer "
              "panel; the front-end only filters + sorts + plots (no browser-side calculation). "
              "Breakdown/metric keys, labels and category order are in fig_meta.json -> cross_tab.")


def fig_metric_matrix(hh):
    """Two-way matrix source: for WAVE 10 only, the weighted MEAN of every metric
    within every NON-EMPTY (dim_a category x dim_b category) intersection, for
    every UNORDERED pair of breakdowns (dim_a listed before dim_b in the fixed
    key order of CROSS_TAB_BREAKDOWNS). LONG format -- one row per non-empty cell:
        dim_a | cat_a | dim_b | cat_b | n_households | <12 metric columns>

    Uses the SAME breakdowns / category definitions as fig_metric_by_breakdown
    (crosstab_specs), incl. the 10-level SES. Cells with n_households == 0 are
    SKIPPED (most intersections are empty). n_households is the unweighted count
    per cell so the front-end can grey n<30 cells. Money rounded to whole NIS.

    The front-end matrix panel only reads these cell values, computes the
    min->max for a sequential colour scale, formats, and lays out the grid (it
    may transpose A/B); NO weighted mean is computed in the browser."""
    latest = hh[hh["wave"] == LATEST_WAVE].copy()
    specs = crosstab_specs(latest)

    # group masks by breakdown key, preserving canonical category order
    key_order = [b[0] for b in CROSS_TAB_BREAKDOWNS]
    by_key = {k: [] for k in key_order}
    for bkey, clabel, mask in specs:
        by_key[bkey].append((clabel, mask))

    rows = []
    for ia in range(len(key_order)):
        for ib in range(ia + 1, len(key_order)):
            ka, kb = key_order[ia], key_order[ib]
            for ca, ma in by_key[ka]:
                for cb, mb in by_key[kb]:
                    sub = latest[ma & mb]
                    n = int(len(sub))
                    if n == 0:
                        continue
                    row = {"dim_a": ka, "cat_a": ca, "dim_b": kb, "cat_b": cb,
                           "n_households": n}
                    for mkey, col in CROSS_TAB_METRICS:
                        v = wmean(sub, sub.index == sub.index, col)
                        row[mkey] = round(v) if not np.isnan(v) else ""
                    rows.append(row)

    fieldnames = (["dim_a", "cat_a", "dim_b", "cat_b", "n_households"]
                  + [m[0] for m in CROSS_TAB_METRICS])
    write_csv("fig_metric_matrix.csv", rows, fieldnames,
              "Two-way matrix explorer: weighted-mean of every metric in every non-empty "
              "(dim_a category x dim_b category) cell, for every unordered pair of breakdowns, "
              "wave 10 (2023), long format (one row per non-empty cell). n_households is the "
              "unweighted sample count per cell (front-end greys n<30). Empty intersections are "
              "omitted. Feeds the two-way matrix heatmap panel; the front-end only reads cell "
              "values, colour-scales min->max, formats and lays out the grid (no browser-side "
              "calculation). Breakdown/metric keys, labels and category order are in "
              "fig_meta.json -> cross_tab (breakdowns/metrics) and -> matrix (defaults).")


def fig_meta(hh, diags):
    home_diag, pension_diag = diags["home"], diags["pension"]
    meta = {
        "source": "cbs_longitudinal_panel_full.csv",
        "unit": "household-wave, weighted by weight_hh",
        "n_rows": int(len(hh)),
        "latest_wave": LATEST_WAVE,
        "wave_year": {str(k): v for k, v in WAVE_YEAR.items()},
        "even_waves": sorted(EVEN_WAVES),
        "weighted_quantile_method": "Hazen/midpoint plotting position: sort by value, cumulative "
                                    "weight minus half its own weight normalized to [0,1], linear "
                                    "interpolation of the target quantile against value.",
        "winsorization": "Amounts winsorized at the 99th percentile (income at 99.5th), floored at 0, "
                         "for all figures EXCEPT fig_networth_distribution.csv, which is built from a "
                         "separate p=99.5th-percentile household frame to avoid double-suppressing the "
                         "top tail that Gini/top10_share are measuring (see that figure's docstring).",
        "odd_wave_caveat": "Financial assets & liabilities (deposits, investments, mortgage, consumer "
                           "loans, additional property, land) are not collected in odd waves (2012, "
                           "2014, 2017, 2019, 2021); those aggregates are null in odd waves, not zero.",
        "home_model": {"held_out_r2": round(home_diag["r2_oos"], 3),
                       "in_sample_r2": round(home_diag["r2_in"], 3),
                       "n_subjective_reported": home_diag["n_subjective"],
                       "n_cbs_estimate_fill": home_diag["n_cbs_est_fill"],
                       "n_model_imputed": home_diag["n_imputed"]},
        "pension_model": {"model": "unified lifecycle: accumulate to retirement, then actuarial drawdown",
                          "contribution_rate_c": round(pension_diag["c"], 4),
                          "real_return_r": pension_diag["r"],
                          "retiree_stock_scalar_k_rel": round(pension_diag["k_rel"], 2),
                          "drawdown_horizon_years_T": pension_diag["T"],
                          "mean_2023": round(pension_diag["mean_2023"]),
                          "mean_2023_before_p99_winsorization": round(pension_diag["mean_2023_precap"]),
                          "macro_benchmark_2023": round(pension_diag["target_mean"]),
                          "macro_benchmark_note": "Target NIS 830,000/household = conservative CMA "
                          "long-term-savings aggregate (~NIS 2.5T end-2023, pension+gemel+hishtalmut+"
                          "managers-insurance) over ~3.05M weighted households (docs/admin_benchmark_"
                          "pressure_test.md). 2026-07-18: replaced the earlier two-regime accumulate-vs-"
                          "capitalize-at-M model (which had a ~59% seam jump at age 67 and no drawdown) "
                          "with ONE lifecycle -- accumulate on capped HOUSEHOLD labour income to age 67, then "
                          "decumulate the retirement stock over T years at real return r. k_rel is solved "
                          "so retiree stock-at-retirement matches worker stock at the 63-66/67-70 boundary "
                          "(seam continuous by construction); c is the single level scalar solved so the "
                          "pre-winsorization wave-10 mean hits the NIS 830k anchor (down from the prior 980k "
                          "overshoot -- admin benchmark shows ~NIS 820-850k/HH). Accumulated balances are "
                          "not in the CBS survey; retiree stock is proxied from observed pension income."},
        "admin_scaling": {
            "note": "dep_sav, invest, mortg, consumer are scaled by a flat per-line factor "
                    "(admin_total / survey_total, survey_total = pre-scaling weighted mean x "
                    "3,049,550 weighted wave-10 households) applied at household level before "
                    "any aggregation, so every downstream figure (national balance sheet, by-group, "
                    "distribution/percentiles, income-wealth, ratios) reflects the scaled level "
                    "while within-line ratios between households/groups are unchanged (flat scale "
                    "preserves shape). pension_wealth is NOT scaled (separate workstream). See "
                    "docs/admin_benchmark_pressure_test.md and the ADMIN_SCALE constant in "
                    "pipeline.py for full source detail.",
            "dep_sav": {"factor": ADMIN_SCALE["dep_sav"], "admin_total_nis_b": 614,
                        "pre_scale_survey_total_nis_b": 282.72,
                        "source": "Bank of Israel, 'Short-Term Household Investments in Solid "
                                  "Vehicles,' Statistical Bulletin 2024 (Tayar & Zilberberg) -- "
                                  "household deposits + current accounts, end-2023."},
            "mortg": {"factor": ADMIN_SCALE["mortg"], "admin_total_nis_b": 568,
                      "pre_scale_survey_total_nis_b": 522.44,
                      "source": "Bank of Israel, 'Debt Developments in the Nonfinancial Private "
                                "Sector, Fourth Quarter of 2023' press release (26-Mar-2024) -- "
                                "household housing debt, end-2023."},
            "consumer": {"factor": ADMIN_SCALE["consumer"], "admin_total_nis_b": 224,
                         "pre_scale_survey_total_nis_b": 70.28,
                         "source": "Same BoI Q4-2023 debt release -- household non-housing debt, "
                                   "end-2023."},
            "invest": {"factor": ADMIN_SCALE["invest"], "admin_total_nis_b": 562,
                       "admin_total_is_estimate": True,
                       "pre_scale_survey_total_nis_b": 180.78,
                       "source": "BEST-ESTIMATE household total ~NIS 562B, end-2023 (no official "
                                 "household-only figure exists -- BoI publishes equities & mutual "
                                 "funds only for 'the public' = households + business). Built from "
                                 "differentiated household shares of the end-2023 'public' totals: "
                                 "mutual funds 0.65 x NIS 456B = NIS 296B (retail-dominated product); "
                                 "direct equities 0.30 x NIS 685B = NIS 206B (households hold a "
                                 "minority -- much of 'the public's' equity is business cross-holdings "
                                 "and controlling blocks held via corporate entities); + actual BoI "
                                 "household-only figures for tradable corporate bonds (NIS 49B) and "
                                 "Makam (NIS 11B). Total ~NIS 562B; factor = 562/180.8 = 3.109. The "
                                 "0.65 / 0.30 shares are judgmental (not sourced) -- FLAGGED best-"
                                 "estimate, verify. See docs/admin_benchmark_pressure_test.md Sec 6."},
        },
        "dating_audit_2026_07_18": {
            "note": "All admin benchmarks below were re-verified as END-2023 (or Q4-2023 / "
                    "2023-annual) dated -- no corrections were required; every factor and the "
                    "pension target_mean are unchanged from the prior pass. See docs/admin_"
                    "benchmark_pressure_test.md Section 5 for the line-by-line audit.",
            "dep_sav": "NIS 614B confirmed end-Dec-2023 (88% of NIS 698B total household solid "
                       "NIS investment, per BoI household-investments bulletin Figure 2).",
            "mortg_and_consumer": "NIS 568B / NIS 224B confirmed Q4-2023 (end-2023) -- BoI "
                                  "'Debt Developments in the Nonfinancial Private Sector, Fourth "
                                  "Quarter of 2023' press release, 26-Mar-2024.",
            "pension": "NIS 2.5T confirmed end-2023 -- CMA (Capital Market, Insurance and Savings "
                       "Authority) Comprehensive Report for 2023, reported via Globes "
                       "('Israelis' long-term savings up 10% in 2023'): total grew ~10% during "
                       "2023 to NIS 2.5T. The Dec-2024 institutional-investor mix (48%/30%/22%) "
                       "used only for the provident/hishtalmut/insurance SUB-split within that "
                       "total remains a documented approximation (see admin_scaling pension note) "
                       "-- the headline 2.5T anchor itself is end-2023, not end-2024.",
            "households": "3,049,550 confirmed as the wave-10 (2023) weighted household population."
        },
        "group_labels": GROUP_LABELS, "tenure_labels": TENURE_LABELS, "size_labels": SIZE_LABELS,
        "marital_labels": MARITAL_LABELS, "kids_labels": KIDS_LABELS, "edu_labels": EDU_LABELS,
        "occ_labels": OCC_LABELS, "ses_labels": SES_LABELS, "currency": "NIS",
        # Cross-tab explorer sync data: metric + breakdown keys/labels + category
        # order so the front-end dropdowns and per-breakdown sort stay in lockstep
        # with fig_metric_by_breakdown.csv (see fig_metric_by_breakdown()).
        "cross_tab": {
            "wave": LATEST_WAVE, "default_metric": "net_worth", "default_breakdown": "group",
            "metrics": [{"key": k, "label": CROSS_TAB_METRIC_LABELS[k]} for k, _ in CROSS_TAB_METRICS],
            "breakdowns": [{"key": bk, "label": bl, "ordinal": ordn, "categories": cats}
                           for bk, bl, ordn, cats in CROSS_TAB_BREAKDOWNS],
            "note": "Each cell = weighted mean of the metric over households in that (breakdown, "
                    "category), wave 10 (2023). ordinal=True breakdowns are shown in the listed "
                    "natural category order; ordinal=False are sorted descending by the chosen "
                    "metric value in the front-end. Industry uses the ISIC economic-sector letter "
                    "(sector of employment); heads not in employment fall into 'Not employed / other'.",
        },
        # Two-way matrix sync data. The matrix REUSES cross_tab.metrics and
        # cross_tab.breakdowns (same keys/labels/ordinal flags/category order);
        # this block only carries the matrix panel's defaults + notes so the
        # three dropdowns + axis ordering stay in lockstep with fig_metric_matrix.csv.
        "matrix": {
            "wave": LATEST_WAVE,
            "default_metric": "net_worth",
            "default_dim_a": "income_decile",
            "default_dim_b": "ses",
            "tiny_n": 30,
            "note": "Each cell = weighted mean of the chosen metric over households in that "
                    "(dim_a category INTERSECT dim_b category), wave 10 (2023). Rows/columns are "
                    "laid out in each dimension's cross_tab category order (ordinal dims in natural "
                    "order; nominal dims in their canonical listed order). fig_metric_matrix.csv "
                    "stores one row per unordered breakdown pair (dim_a before dim_b in cross_tab "
                    "order); the front-end transposes when the user picks them the other way. Empty "
                    "intersections are absent from the CSV (blank in the grid); cells with n<30 are "
                    "greyed and their value blanked (n shown on hover).",
        },
    }
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "fig_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    MANIFEST.append({"file": "fig_meta.json", "description":
                     "Methodology facts for footnotes: sample size, wave-year map, model diagnostics, "
                     "winsorization, odd-wave caveat, and label dictionaries."})
    print(f"  wrote fig_meta.json")
    return meta


def main():
    print("Reading panel + applying models ...")
    hh, diags = build_household_frame()
    print(f"  {len(hh)} household-wave rows")
    print("  building p=0.995 frame for the net-worth distribution figure only ...")
    hh_dist, _ = build_household_frame(p=0.995)

    print("Building figures ...")
    fig_national_balance_sheet(hh)
    fig_networth_by_group(hh)
    fig_composition_by_group(hh)
    fig_networth_by_dimension(hh)
    fig_networth_distribution(hh_dist)
    fig_income_wealth(hh)
    fig_trajectory(hh)
    fig_participation(hh)
    fig_ratios_by_group(hh)
    fig_income_composition_by_group(hh)
    fig_metric_by_breakdown(hh)
    fig_metric_matrix(hh)
    fig_meta(hh, diags)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    manifest_path = os.path.join(FIGURES_DIR, "_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "description"])
        w.writeheader()
        for m in MANIFEST:
            w.writerow(m)
    print(f"\nWrote {manifest_path} ({len(MANIFEST)} entries)")

    # ---- sanity: wave-10 ALL cross-check numbers (compare to build_balance_sheet_data.py stdout) ----
    latest = hh[hh["wave"] == LATEST_WAVE]
    mean_home = wmean(latest, latest.index == latest.index, "home")
    mean_nw = wmean(latest, latest.index == latest.index, "nw")
    print(f"\n[cross-check] wave 10 ALL: mean home value = {mean_home:,.0f}  mean net worth = {mean_nw:,.0f}")


if __name__ == "__main__":
    main()
