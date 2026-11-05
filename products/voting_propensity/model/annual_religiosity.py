"""Weighted retained-survey religiosity, with an explicit Census-anchored bridge.

HES measures household religiosity assigned to adult household members. Social
Survey2024 measures individual20+ identification and is never spliced into HES.
Licensed records remain external; only aggregate evidence is written locally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


HES_FOLDERS = {2016: "H20161024", 2017: "H20171023", 2018: "H20181023", **{y: f"H{y}1021" for y in range(2019, 2024)}}
RELIGIOSITY = {1: "secular", 2: "traditional", 3: "religious", 4: "haredi", 5: "mixed", 6: "other"}
CATEGORIES = list(RELIGIOSITY.values())+["unknown"]
HES_AGES = {5: "18_24", 6: "25_34", 7: "25_34", 8: "35_44", 9: "35_44",
            10: "45_54", 11: "45_54", 12: "55_64", 13: "55_64", 14: "55_64", **{i: "65_plus" for i in range(15, 21)}}
SS_AGES = {1: "20_24", 2: "25_34", 3: "25_34", 4: "35_44", 5: "35_44", 6: "45_54", 7: "45_54", 8: "55_64", 9: "55_64", 10: "65_plus", 11: "65_plus"}
SOCIAL_HISTORY = {"filename": "social_survey2002_2021_tableA.xlsx",
                  "source_url": "https://www.cbs.gov.il/he/publications/doclib/2024/1905/t_a.xlsx",
                  "sha256": "b99c94b8b77f349c8f65d1b6fc73051d0916aa3937ecea447f784f4e5459edcf"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_file(folder, suffix):
    paths = [p for p in folder.iterdir() if p.name.lower().endswith(suffix.lower())]
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix} in {folder.name}; found {len(paths)}")
    return paths[0]


def read_columns(path, required=None):
    options = {"usecols": lambda column: column.strip().lower() in required} if required else {}
    try:
        frame = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", **options)
        frame.attrs["encoding"] = "utf-8-sig"
    except UnicodeDecodeError:
        frame = pd.read_csv(path, low_memory=False, encoding="cp1255", **options)
        frame.attrs["encoding"] = "cp1255"
    frame.columns = frame.columns.str.strip().str.lower()
    if frame.columns.duplicated().any():
        raise ValueError("Case-normalized columns collide")
    return frame


def hes_adults(households, persons, year):
    keys = ["s_seker", "misparmb"]
    required_hh = keys+["ramatdatiyut", "religion", "nationality"]
    required_person = keys+["prat", "age_group", "min", "nationality", "weight"]
    if set(required_hh)-set(households) or set(required_person)-set(persons):
        raise ValueError("HES source lacks required variables")
    if households.duplicated(keys).any() or persons.duplicated(keys+["prat"]).any():
        raise ValueError("HES household/person keys are not unique")
    if not pd.to_numeric(persons.s_seker, errors="raise").eq(year).all():
        raise ValueError("HES person source contains another survey year")
    hh = households[required_hh].rename(columns={"nationality": "household_head_nationality", "religion": "household_head_religion"})
    merged = persons[required_person].merge(hh, on=keys, how="left", validate="many_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("HES person has no household record")
    age = pd.to_numeric(merged.age_group, errors="coerce")
    weight = pd.to_numeric(merged.weight, errors="coerce")
    if (weight.dropna() < 0).any():
        raise ValueError("Negative HES design weights")
    adult = age.isin(HES_AGES) & weight.gt(0)
    selected = merged.loc[adult].copy()
    out = pd.DataFrame(index=selected.index)
    out["household"] = selected.s_seker.astype(str)+"|"+selected.misparmb.astype(str)
    out["age_band"] = age.loc[adult].map(HES_AGES)
    out["sex"] = pd.to_numeric(selected["min"], errors="coerce").map({1: "male", 2: "female"}).fillna("unknown")
    out["population_group"] = pd.to_numeric(selected.nationality, errors="coerce").map({1: "jewish", 2: "arab", 3: "other"}).fillna("unknown")
    out["religiosity"] = pd.to_numeric(selected.ramatdatiyut, errors="coerce").map(RELIGIOSITY).fillna("unknown")
    out["weight"] = weight.loc[adult]
    return out, {"person_records": len(persons), "adult_positive_weight_records": len(out),
                 "excluded_records": len(persons)-len(out), "weighted_adults": float(out.weight.sum()),
                 "nationality_basis": "household_head_before2019" if year < 2019 else "individual_age15plus",
                 "citizenship": "Not observed; Nationality is population/ethnic grouping, not citizen eligibility",
                 "religion_basis": "HES religion is household-head religion; no individual Druze inference"}


def weighted_distribution(frame):
    """Household-cluster linearization; strata/PSU design not available here."""
    if frame.empty or not np.isfinite(frame.weight).all() or frame.weight.le(0).any():
        raise ValueError("Distribution requires finite positive design weights")
    total = frame.weight.sum()
    clusters = frame.groupby("household").weight.sum()
    n_cluster = len(clusters)
    effective_hh = total**2/(clusters**2).sum()
    effective_person = total**2/(frame.weight**2).sum()
    rows = []
    for category in CATEGORIES:
        selected = frame.religiosity.eq(category)
        numerator = frame.loc[selected, "weight"].sum()
        share = float(numerator/total)
        group_cluster = frame.loc[selected].groupby("household").weight.sum().reindex(clusters.index, fill_value=0.)
        contrast = group_cluster-share*clusters
        variance = float((contrast**2).sum()/total**2*n_cluster/(n_cluster-1)) if n_cluster > 1 else np.nan
        se = float(np.sqrt(variance))
        rows.append({"religiosity": category, "weighted_adults": float(numerator), "denominator_weighted_adults": float(total),
                     "share": share, "person_records": int(selected.sum()), "denominator_person_records": len(frame),
                     "household_clusters": n_cluster, "kish_effective_household_n": float(effective_hh),
                     "kish_effective_person_n": float(effective_person), "cluster_standard_error": se,
                     "approx95_lower": max(0., share-1.96*se), "approx95_upper": min(1., share+1.96*se)})
    return rows


def summarize(frame, year, source_id, measurement_basis):
    rows = []
    subsets = [("all", "all", group, z) for group, z in frame.groupby("population_group")]
    subsets += [(age, sex, group, z) for (age, sex, group), z in frame.groupby(["age_band", "sex", "population_group"])]
    for age, sex, group, subset in subsets:
        for row in weighted_distribution(subset):
            rows.append({"year": year, "age_band": age, "sex": sex, "population_group": group,
                         "source_id": source_id, "measurement_basis": measurement_basis,
                         "status": "weighted_survey_estimate", **row})
    return rows


def social2024(path):
    needed = ["serialnumber", "gil", "minn", "dat", "datiutyehudi", "pop_group", "nn"]
    frame = read_columns(path, needed)
    if set(needed)-set(frame):
        raise ValueError("Social Survey2024 lacks required variables")
    weight = pd.to_numeric(frame.nn, errors="coerce")
    if (weight.dropna() < 0).any():
        raise ValueError("Negative Social Survey design weights")
    # This is self-reported Jewish RELIGION, not an inferred Jewish ethnicity.
    keep = pd.to_numeric(frame.dat, errors="coerce").eq(1) & weight.gt(0)
    selected = frame.loc[keep]
    out = pd.DataFrame(index=selected.index)
    out["household"] = selected.serialnumber.astype(str)
    out["age_band"] = pd.to_numeric(selected.gil, errors="coerce").map(SS_AGES).fillna("unknown")
    out["sex"] = pd.to_numeric(selected.minn, errors="coerce").map({1: "male", 2: "female"}).fillna("unknown")
    out["population_group"] = "jewish_self_reported_religion"
    out["religiosity"] = pd.to_numeric(selected.datiutyehudi, errors="coerce").map({1: "haredi", 2: "religious", 3: "traditional", 4: "traditional", 5: "secular"}).fillna("unknown")
    out["weight"] = weight.loc[keep]
    return out


def fit_relative_trends(observed, years=range(2016, 2027), shrinkage_households=1000.):
    """Smooth age/sex trends shrink to national household-definition trends.

    Half an effective-household pseudo-count stabilizes model log-shares only;
    observed estimates remain untouched. Other/unknown factors stay1: residual
    categories/response are not interpreted as demographic change across the
    nationality-definition break. All2022 factors are exactly1.
    """
    source = observed.loc[observed.population_group.eq("jewish")].copy()
    if source.year.nunique() < 3:
        raise ValueError("Relative trends require at least three survey years")
    def slope(frame):
        n = frame.kish_effective_household_n.to_numpy(float)
        smooth = (frame.share.to_numpy(float)*n+.5)/(n+.5*len(CATEGORIES))
        time = frame.year.to_numpy(float)
        dt = time-np.average(time, weights=n)
        # Estimate a free intercept; center relative factors on the Census2022
        # anchor afterwards, rather than force a noisy survey point as intercept.
        beta = float(np.sum(n*dt*np.log(smooth))/np.sum(n*dt**2))
        information = float(np.sum(n*frame.share.to_numpy(float)*(1-frame.share.to_numpy(float))))
        intercept = float(np.average(np.log(smooth), weights=n)-beta*(np.average(time, weights=n)-2022))
        return beta, float(n.sum()), information, intercept
    national = {category: slope(z)[0] for category, z in source.loc[source.age_band.eq("all")].groupby("religiosity")}
    result = []
    observed_years = sorted(source.year.unique())
    for (age, sex, category), z in source.groupby(["age_band", "sex", "religiosity"]):
        if len(z) < 3:
            continue
        local, effective, information, intercept = slope(z)
        pooling = 1. if age == "all" else information/(information+shrinkage_households)
        beta = pooling*local+(1-pooling)*national[category]
        if category in ["other", "unknown"]:
            beta = 0.
        for year in years:
            result.append({"year": year, "age_band": age, "sex": sex, "population_group": "jewish", "religiosity": category,
                           "relative_factor": float(np.exp(beta*(year-2022))), "annual_log_slope": beta,
                           "fitted_log_intercept_2022": intercept,
                           "cell_weight_in_pooled_slope": pooling, "summed_effective_household_n": effective,
                           "trend_information_households": information,
                           "anchor_year": 2022, "observed_source_years": ";".join(map(str, observed_years)),
                           "measurement_basis": "adult18plus_household_religiosity_HES",
                           "status": "anchor" if year == 2022 else "backcast" if year < min(observed_years) else "projected" if year > max(observed_years) else "smoothed_survey_trend",
                           "method": "Free-intercept weighted log-share slope centered on2022 Census; age-sex shrinkage toward national using effective category information; residual other/unknown factor1"})
    return pd.DataFrame(result)


def survey_holdout_validation(observed):
    """Prediction of survey composition, not validation of unobserved local truth."""
    core = ["secular", "traditional", "religious", "haredi"]
    source = observed.loc[observed.population_group.eq("jewish") & observed.religiosity.isin(core)]
    frozen = source.loc[source.year.eq(2022)].set_index(["age_band", "sex", "religiosity"]).sort_index()
    rows = []
    for held_year in [2019, 2022, 2023]:
        fitted = fit_relative_trends(observed.loc[observed.year.ne(held_year)], years=[held_year])
        test = source.loc[source.year.eq(held_year)]
        for (age, sex), actual in test.groupby(["age_band", "sex"]):
            model = fitted.loc[fitted.age_band.eq(age) & fitted.sex.eq(sex) & fitted.religiosity.isin(core)].set_index("religiosity")
            actual = actual.set_index("religiosity").reindex(core)
            if len(model) != 4:
                continue
            prediction = np.exp(model.reindex(core).fitted_log_intercept_2022+model.reindex(core).annual_log_slope*(held_year-2022))
            prediction /= prediction.sum()
            truth = actual.share/actual.share.sum()
            baseline = frozen.loc[(age, sex)].reindex(core).share
            baseline /= baseline.sum()
            for category in core:
                rows.append({"held_year": held_year, "age_band": age, "sex": sex, "religiosity": category,
                             "observed_share": float(truth[category]), "pooled_trend_prediction": float(prediction[category]),
                             "frozen2022_prediction": None if held_year == 2022 else float(baseline[category]),
                             "pooled_absolute_error_pp": float(100*abs(prediction[category]-truth[category])),
                             "frozen_absolute_error_pp": None if held_year == 2022 else float(100*abs(baseline[category]-truth[category])),
                             "effective_household_n": float(actual.kish_effective_household_n.iloc[0]),
                             "status": "baseline_identity_excluded" if held_year == 2022 else "retrospective_leave_year_out",
                             "measurement_scope": "Known-four HES household18plus composition; pooled fitted levels. Does not validate Census-anchored local reconstruction"})
    return pd.DataFrame(rows)


def historical_social_controls(path):
    """Read actual annual20+ individual estimates from the pinned official table."""
    if sha256(path) != SOCIAL_HISTORY["sha256"]:
        raise ValueError("Historical Social Survey table differs from pinned source")
    table = pd.read_excel(path, header=None)
    clean = lambda value: re.sub(r"\s+", "", str(value)).replace("–", "-").replace("־", "-")
    headings = [i for i, value in table.iloc[:, 1].items() if clean(value) == "מידתדתיות-יהודים"]
    if len(headings) != 1:
        raise ValueError("Historical Jewish religiosity heading changed")
    labels = ["חרדים", "דתיים", "מסורתיים-דתיים", "מסורתיים-לאכלכךדתיים", "לאדתיים,חילונים"]
    start = headings[0]+1
    if [clean(value) for value in table.iloc[start:start+5, 1]] != labels:
        raise ValueError("Historical Jewish religiosity row definitions changed")
    years = {}
    for column, value in table.iloc[3].items():
        match = re.fullmatch(r"R?(20\d\d)(?:\.0)?", str(value))
        if match:
            years[int(match.group(1))] = column
    if set(years) != set(range(2002, 2022)):
        raise ValueError("Historical Social Survey year coverage changed")
    rows = []
    for year, column in sorted(years.items()):
        values = pd.to_numeric(table.iloc[start:start+5, column], errors="raise").to_numpy(float)/100
        if not np.isfinite(values).all() or min(values) < 0 or values.sum() > 1.001:
            raise ValueError("Historical Social Survey shares invalid")
        categories = {"haredi": values[0], "religious": values[1], "traditional": values[2]+values[3], "secular": values[4]}
        categories["unknown"] = max(0., 1-values.sum())
        for category, share in categories.items():
            rows.append({"year": year, "age_band": "all20plus", "sex": "all", "population_group": "jewish",
                         "religiosity": category, "share": float(share), "source_id": "cbs_social_survey2021_multiyear_tableA",
                         "measurement_basis": "individual20plus_Jewish_religiosity", "status": "published_survey_estimate" if category != "unknown" else "unreported_residual",
                         "precision": "Published point estimate; sampling variance not supplied in this table"})
    return pd.DataFrame(rows)


def bridge_historical_trends(recent_factors, observed, social):
    """Use observed early SS relative changes; never a13year HES extrapolation.

    Earliest compatible HES overlap aligns definitions by relative changes.
    Earlier age-sex differences and mixed factors stay at that overlap; SS has
    no matching historical household mixed/other/age-sex cells.
    """
    core = ["secular", "traditional", "religious", "haredi"]
    source = social.loc[social.religiosity.isin(core)].pivot(index="year", columns="religiosity", values="share")
    source = source.div(source.sum(axis=1), axis=0)
    edge_year = int(observed.year.min())
    if edge_year not in source.index:
        raise ValueError("Historical definition bridge requires overlapping SS year")
    edge = recent_factors.loc[recent_factors.year.eq(edge_year)]
    rows = []
    for year in source.index[source.index < edge_year]:
        for _, base in edge.iterrows():
            row = base.to_dict()
            category = row["religiosity"]
            ratio = source.loc[year, category]/source.loc[edge_year, category] if category in core else 1.
            row.update({"year": int(year), "relative_factor": float(base.relative_factor*ratio),
                        "annual_log_slope": None, "status": "historical_relative_measurement_bridge",
                        "measurement_bridge_edge_year": edge_year,
                        "observed_source_years": "2002-2021SocialSurvey;"+";".join(map(str, sorted(observed.year.unique())))+"HES",
                        "method": f"Actual SS national20+ known-four relative change versus{edge_year}, times household-compatible HES{edge_year} age-sex factor; mixed held at edge factor, other/unknown1"})
            rows.append(row)
    overlap = []
    national = observed.loc[observed.population_group.eq("jewish") & observed.age_band.eq("all") & observed.religiosity.isin(core)]
    household = national.pivot(index="year", columns="religiosity", values="share")
    household = household.div(household.sum(axis=1), axis=0)
    for year in household.index.intersection(source.index):
        for category in core:
            hes_change = household.loc[year, category]/household.loc[edge_year, category]
            ss_change = source.loc[year, category]/source.loc[edge_year, category]
            overlap.append({"year": int(year), "religiosity": category, "hes_household18plus_core_share": household.loc[year, category],
                            "ss_individual20plus_core_share": source.loc[year, category],
                            "measurement_bridge_edge_year": edge_year,
                            "hes_relative_to_edge_year": hes_change, "ss_relative_to_edge_year": ss_change,
                            "log_relative_difference": float(np.log(hes_change/ss_change)),
                            "interpretation": "Definition/age/sample disagreement; relative-trend transfer is an assumption, not observed household history"})
    return pd.concat([pd.DataFrame(rows), recent_factors], ignore_index=True), pd.DataFrame(overlap)


def apply_relative_factors(anchor_cells, factors, year):
    """Change Jewish household shares within locality/age/sex; retain residual mass."""
    cells = anchor_cells.copy()
    cells["source_weight"] = cells.person_weight
    trend = factors.loc[factors.year.eq(year) & factors.age_band.ne("all")]
    if trend.duplicated(["age_band", "sex", "religiosity"]).any():
        raise ValueError("Duplicate religiosity bridge factors")
    trend = trend[["age_band", "sex", "religiosity", "relative_factor"]].rename(columns={"religiosity": "household_religiosity"})
    cells = cells.merge(trend, on=["age_band", "sex", "household_religiosity"], how="left", validate="many_to_one")
    jewish = cells.population_group.eq("jewish") & ~cells.household_religiosity.isin(["other", "unknown"])
    if cells.loc[jewish, "relative_factor"].isna().any():
        raise ValueError("Missing age-sex religiosity factor; no automatic national fallback")
    keys = ["locality_code", "subquarter_code", "age_band", "sex", "population_group"]
    fixed_mass = cells.loc[jewish].groupby(keys).person_weight.transform("sum")
    changed = cells.loc[jewish, "person_weight"]*cells.loc[jewish, "relative_factor"]
    subtotal = changed.groupby([cells.loc[jewish, key] for key in keys]).transform("sum")
    cells.loc[jewish, "person_weight"] = changed*fixed_mass/subtotal
    return cells.drop(columns="relative_factor")


def build(useful_data, output, raw=None, download=False, anchor_cells=None):
    output.mkdir(parents=True, exist_ok=True)
    hes_root = useful_data/"סקרי למס"/"סקר הוצאות משקי בית 2016 - 2023"
    rows, evidence = [], []
    for year, folder in HES_FOLDERS.items():
        directory = hes_root/folder
        household_path, person_path = find_file(directory, "datamb.csv"), find_file(directory, "dataprat.csv")
        households = read_columns(household_path, ["s_seker", "misparmb", "ramatdatiyut", "religion", "nationality"])
        persons = read_columns(person_path, ["s_seker", "misparmb", "prat", "age_group", "min", "nationality", "weight"])
        adults, audit = hes_adults(households, persons, year)
        source_id = f"cbs_hes_{year}_licensed_puf"
        rows.extend(summarize(adults, year, source_id, "adult18plus_household_religiosity_HES"))
        files = [household_path, person_path, find_file(directory, "codebookmb.xls"), find_file(directory, "codebookprat.xls")]
        evidence.append({"source_id": source_id, "year": year, "publisher": "Israel Central Bureau of Statistics",
                         "files": [{"relative_to_useful_data": p.relative_to(useful_data).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in files], **audit})
    observed = pd.DataFrame(rows)
    observed.to_csv(output/"hes_observed_religiosity.csv", index=False, encoding="utf-8-sig")
    primary_observed = observed.loc[observed.year.ge(2019)]
    full_window = fit_relative_trends(observed)
    trends = fit_relative_trends(primary_observed, years=range(2019, 2027))
    survey_holdout_validation(primary_observed).to_csv(output/"hes_holdout_validation.csv", index=False, encoding="utf-8-sig")
    survey_holdout_validation(observed).to_csv(output/"hes_holdout_full_window_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity = full_window.loc[full_window.year.eq(2026)].merge(trends.loc[trends.year.eq(2026)],
        on=["year", "age_band", "sex", "population_group", "religiosity"], suffixes=("_all2016_2023", "_since2019"), validate="one_to_one")
    sensitivity.to_csv(output/"definition_break_sensitivity.csv", index=False, encoding="utf-8-sig")
    if raw is not None:
        history_path = raw/SOCIAL_HISTORY["filename"]
        if download:
            try:
                from .demographic_foundation import archive_url
            except ImportError:
                from demographic_foundation import archive_url
            archive_url(SOCIAL_HISTORY["source_url"], history_path, expected_sha256=SOCIAL_HISTORY["sha256"])
        social_history = historical_social_controls(history_path)
        social_history.to_csv(output/"historical_social_controls.csv", index=False, encoding="utf-8-sig")
        trends, overlap = bridge_historical_trends(trends, primary_observed, social_history)
        full_window, full_overlap = bridge_historical_trends(full_window, observed, social_history)
        overlap.to_csv(output/"measurement_bridge_overlap.csv", index=False, encoding="utf-8-sig")
        full_overlap.to_csv(output/"measurement_bridge_full_window_sensitivity.csv", index=False, encoding="utf-8-sig")
        evidence.append({"source_id": "cbs_social_survey2021_multiyear_tableA", **SOCIAL_HISTORY,
                         "years": list(range(2002, 2022)), "measurement_basis": "Published national individual20plus Jewish religiosity"})
    trends.to_csv(output/"religiosity_relative_factors.csv", index=False, encoding="utf-8-sig")
    full_window.to_csv(output/"religiosity_relative_factors_full_window_sensitivity.csv", index=False, encoding="utf-8-sig")
    if anchor_cells is not None:
        anchor = pd.read_csv(anchor_cells)
        comparisons = []
        for label, frame in [("anchor2022", anchor), ("primary2019_2023", apply_relative_factors(anchor, trends, 2026)),
                             ("sensitivity2016_2023", apply_relative_factors(anchor, full_window, 2026))]:
            groups = np.where(frame.population_group.eq("jewish"), "jewish_"+frame.household_religiosity, frame.population_group)
            grouped = frame.groupby(groups).person_weight.sum()
            for group, weight in grouped.items():
                comparisons.append({"variant": label, "group": group, "adult_weight": weight, "share": weight/grouped.sum(),
                                    "scope": "Isolated religiosity-trend effect on2022adult Census anchor; annual age/population controls not applied; no turnout rates"})
        pd.DataFrame(comparisons).to_csv(output/"trend_window2026_anchor_mix_comparison.csv", index=False, encoding="utf-8-sig")
    social_path = useful_data/"סקרי למס"/"הסקר החברתי 2020 - 2024"/"H20241501"/"H20241501data.csv"
    social = social2024(social_path)
    social_output = pd.DataFrame(summarize(social, 2024, "cbs_social_survey_2024_licensed_puf", "individual20plus_self_reported_Jewish_religion_not_HES_household_definition"))
    # Mixed/other HOUSEHOLD religiosity are absent response options here, not zeros.
    social_output.loc[~social_output.religiosity.isin(["mixed", "other"])].to_csv(output/"social2024_separate_measurement.csv", index=False, encoding="utf-8-sig")
    evidence.append({"source_id": "cbs_social_survey_2024_licensed_puf", "file": social_path.relative_to(useful_data).as_posix(),
                     "sha256": sha256(social_path), "person_records": len(social), "weight_variable": "nn",
                     "codebook_file": "H20241501codebook.xlsx", "codebook_sha256": sha256(social_path.parent/"H20241501codebook.xlsx"),
                     "age_scope": "20plus; cannot separately identify18-19", "measurement_basis": "Individual Jewish religion and individual religiosity; not spliced into HES"})
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "builder_sha256": sha256(__file__), "sources": evidence,
             "primary_hes_trend_years": list(range(2019, 2024)), "sensitivity_hes_trend_years": list(HES_FOLDERS),
             "primary_historical_measurement_bridge_edge": 2019, "sensitivity_historical_measurement_bridge_edge": 2016,
             "trend_shrinkage_households": 1000., "residual_categories_fixed_at_anchor": ["other", "unknown"],
             "anchor_cells": {"filename": anchor_cells.name, "sha256": sha256(anchor_cells)} if anchor_cells is not None else None,
             "hes_years": list(HES_FOLDERS), "source_category_mapping": RELIGIOSITY, "hes_age_code_mapping": HES_AGES,
             "precision": "Person design weights; household-cluster linearization SE and Kish effective-household n. No sampling strata/PSU/FPC adjustment; approximate intervals are descriptive.",
             "bridge": "Relative household-definition HES slopes centered on Census2022; sparse age-sex slopes shrink to national. Mixed remains explicit; residual other/unknown anchor mass unchanged. Locality-relative heterogeneity beyond2022 is not observed.",
             "limits": ["No retained pre2016 HES or Census2008 files found in catalog roots", "Source folder named SocialSurvey2020-2024 contains duplicated HES2020-2022; those are not independent surveys", "Before2019 person nationality was household-head nationality; from2019 individually reported for15+", "Survey private-household adults are not all Census residents/citizens or the electoral roll", "Primary before2019 uses actual national individual20plus SS relative changes versus2019, with household age-sex and mixed structure held at that bridge edge; full-window sensitivity uses2016edge", "No annual small-area religiosity is observed", "2024 Social Survey individual20+ evidence is a separate measurement; household mixed/other categories are unmeasured, not zeros", "No turnout or party outcomes enter extraction/trends"]}
    (output/"religiosity_source_evidence.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return observed, trends, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--useful-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--anchor-cells", type=Path)
    args = parser.parse_args()
    observed, trends, audit = build(args.useful_data, args.output, args.raw, args.download, args.anchor_cells)
    print(json.dumps({"hes_years": audit["hes_years"], "observed_rows": len(observed), "trend_rows": len(trends)}))


if __name__ == "__main__":
    main()
