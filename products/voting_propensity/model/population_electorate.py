"""Separate resident population, registered residents and registered electors.

Only CEC eligible denominators enter allocation; no turnout or party outcomes
enter population classification. Citizen-adjusted resident priors are estimates,
not a recovered electoral roll. Missing locality priors retain unallocated mass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

REGISTRY_RESOURCES = {"2026-11-01": "64edd0ee-3d5d-43ce-8562-c336c24dbc1f",
                      "2019-04-01": "a5e7080d-3c37-49c2-8cd0-cb2724809216",
                      "2019-09-01": "18e3c86d-5236-4322-aac0-0b313aa86a60",
                      "2020-02-02": "04a2d2da-d365-4dad-be00-5f815174cf3f"}
REGISTRY_CATALOG = "https://data.gov.il/api/3/action/package_show?id=residents_in_israel_by_communities_and_age_groups"
JERUSALEM_SOURCE = "https://fs.knesset.gov.il/globaldocs/MMM/8c56802e-d583-ef11-a851-005056aa1f91/2_8c56802e-d583-ef11-a851-005056aa1f91_11_20794.pdf"
ELIGIBILITY_SOURCE = "https://www.gov.il/en/pages/knesset25-elections-info?chapterIndex=12"
DRUZE_SOURCE = "https://www.cbs.gov.il/he/mediarelease/DocLib/2024/124/11_24_124b.pdf"
NATIONAL_CITIZEN_SOURCES = {2022: "https://www.cbs.gov.il/he/mediarelease/DocLib/2022/342/24_22_342b.pdf",
                          2026: "https://www.cbs.gov.il/he/mediarelease/DocLib/2026/332/24_26_332b.pdf"}
NATIONAL_RESIDENT_ELIGIBLE = {2022: {"jewish": 4794000., "arab": 1044000., "other": 351000.},
                            2026: {"jewish": 5017000., "arab": 1159000., "other": 346000.}}
NATIONAL_ELECTION_CITIZENS = {
    "K21": {"year":2019,"counts":{"jewish":5800000*.79,"arab":5800000*.16,"other":5800000*.05},
            "source":"https://www.cbs.gov.il/he/mediarelease/DocLib/2019/080/24_19_080b.pdf","precision":"5.8million headline and79/16/5 integer percentages; approximate derived group counts","total_rounding":50000},
    "K22": {"year":2019,"counts":{"jewish":5800000*.79,"arab":5800000*.16,"other":5800000*.05},
            "source":"https://www.cbs.gov.il/he/mediarelease/DocLib/2019/277/24_19_277b.pdf","precision":"5.8million headline and79/16/5 integer percentages; approximate derived group counts","total_rounding":50000},
    "K23": {"year":2020,"counts":{"jewish":5900000*.78,"arab":5900000*.17,"other":5900000*.05},
            "source":"https://www.cbs.gov.il/he/mediarelease/DocLib/2020/057/24_20_057b.pdf","precision":"5.9million headline and78/17/5 integer percentages; approximate derived group counts","total_rounding":50000},
    "K24": {"year":2021,"counts":{"jewish":4670000.,"arab":997000.,"other":320000.},
            "source":"https://www.cbs.gov.il/he/mediarelease/DocLib/2021/092/24_21_092b.pdf","precision":"Published group counts rounded to1000persons","total_rounding":1500},
    "K25": {"year":2022,"counts":NATIONAL_RESIDENT_ELIGIBLE[2022],"source":NATIONAL_CITIZEN_SOURCES[2022],
            "precision":"Published group counts rounded to1000persons","total_rounding":1500},
    "K26": {"year":2026,"counts":NATIONAL_RESIDENT_ELIGIBLE[2026],"source":NATIONAL_CITIZEN_SOURCES[2026],
            "precision":"Published group counts rounded to1000persons","total_rounding":1500}}
NATIONAL_REFERENCE_DATES={"K21":"2019-03-10","K22":"2019-09-10","K23":"2020-02-25",
                          "K24":"2021-03-15","K25":"2022-11-01","K26":"2026-09"}
# CBS Table B: rounded all-age religious population shares, not adult identities.
DRUZE_SHARES_2022 = {494: .97, 502: .99, 481: .57, 480: 1., 4201: 1., 534: .75,
    1296: .96, 1295: 1., 4001: 1., 485: 1., 496: .97, 8800: .14, 536: .78,
    525: 1., 473: .30, 4203: 1., 543: .31, 4502: 1., 546: .97}
# PIBA July 2024, reproduced in Knesset Research 26 December 2024, Table 1 p6.
# Every tuple is citizens male/female, permanent male/female, temporary male/female,
# status-under-review male/female. Ages 18+ are observed, not inferred from 19+.
JERUSALEM_STATUS = {"0_17": (7880, 7292, 64794, 60664, 484, 446, 469, 416),
                    "18_30": (3865, 3667, 41036, 39529, 86, 121, 7, 11),
                    "31_50": (3895, 3307, 36821, 37443, 223, 1220, 0, 0),
                    "51_70": (1843, 1418, 18536, 21936, 1124, 336, 1, 0),
                    "71_plus": (273, 301, 4720, 6088, 65, 12, 2, 1)}
ELECTION_DATES = {"K16": "2003-01-28", "K17": "2006-03-28", "K18": "2009-02-10", "K19": "2013-01-22", "K20": "2015-03-17",
                  "K21": "2019-04-09", "K22": "2019-09-17", "K23": "2020-03-02", "K24": "2021-03-23", "K25": "2022-11-01", "K26": "2026-10-27"}


def fetch_source(url, path):
    """Public raw bytes plus retained SHA; existing files must verify."""
    metadata_path = path.with_name(path.name+".metadata.json")
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        content = path.read_bytes()
        if metadata["source_url"] != url or metadata["sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError("Source cache identity/hash mismatch")
        return content
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    metadata_path.write_text(json.dumps({"source_url": url, "sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_utc": datetime.now(timezone.utc).isoformat()}, indent=2)+"\n", encoding="utf-8")
    return content


def normalize_registry(payload, reference_date, source_id):
    result = payload.get("result", {})
    records = result.get("records", [])
    if payload.get("success") is not True or len(records) != result.get("total") or result.get("resource_id") != source_id:
        raise ValueError("Incomplete or wrong-resource PIBA response")
    raw = pd.DataFrame(records)
    fields = {"גיל_19_45": "age19_45", "גיל_46_55": "age46_55", "גיל_56_64": "age56_64", "גיל_65_פלוס": "age65_plus",
              "גיל_6_18": "age6_18", "סהכ": "all_ages"}
    youngest = "גיל_0_5" if "גיל_0_5" in raw else "גיל_0_6"
    fields[youngest] = "youngest_source_bin"
    values = raw[list(fields)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any() or not np.equal(values, np.floor(values)).all().all():
        raise ValueError("PIBA counts must be finite nonnegative integers")
    codes = pd.to_numeric(raw["סמל_ישוב"], errors="raise").astype(int)
    if codes.duplicated().any() or not values.drop(columns="סהכ").sum(axis=1).eq(values["סהכ"]).all():
        raise ValueError("PIBA locality identity or age-mass check failed")
    base = pd.DataFrame({"year": int(reference_date[:4]), "locality_code": codes, "locality_name": raw["שם_ישוב"].str.strip(),
                         "subdistrict_code": pd.to_numeric(raw["סמל_נפה"]),
                         "regional_council_code": pd.to_numeric(raw["קוד_מועצה_אזורית"]),
                         "reference_date": reference_date, "source_id": source_id,
                         "status": "observed_population_register_including_abroad_and_non_citizens",
                         "dimension": "registry_age", "unit": "persons"})
    outputs = []
    for column, category in fields.items():
        part = base.copy();part["category"], part["value"] = category, values[column]
        outputs.append(part)
    part = base.copy();part["category"] = "age19_plus"
    part["value"] = values[["גיל_19_45", "גיל_46_55", "גיל_56_64", "גיל_65_פלוס"]].sum(axis=1)
    outputs.append(part)
    return pd.concat(outputs, ignore_index=True)


def verify_registry_source_date(resource,reference_date,source_id):
    """A rolling datastore ID must not silently acquire a different dated file."""
    date=datetime.strptime(reference_date,"%Y-%m-%d")
    url=str(resource.get("url",""))
    if resource.get("id")!=source_id or not (date.strftime("%Y_%m_%d") in url or date.strftime("%d%m%Y") in url):
        raise ValueError("PIBA resource no longer identifies the pinned reference date; select a new explicit snapshot")


def jerusalem_citizenship_controls():
    rows = []
    for age, counts in JERUSALEM_STATUS.items():
        for status_index, status in enumerate(["citizen", "permanent_non_citizen", "temporary", "status_under_review"]):
            for sex_index, sex in enumerate(["male", "female"]):
                rows.append({"year": 2024, "locality_code": 3000, "geographic_scope": "East Jerusalem Arab population in PIBA response",
                             "age_band": age, "sex": sex, "dimension": "citizenship", "category": status,
                             "value": counts[2*status_index+sex_index], "source_id": JERUSALEM_SOURCE,
                             "reference_date": "2024-07-11", "status": "official_PIBA_counts_published_by_Knesset_Research"})
    return pd.DataFrame(rows)


def registry_adult_controls(registry, national):
    """Observed19+/65+ plus estimated age18 using an observed national age shape.

    No uniform age18 assumption: exact national18/(6..18) ratios are explicit.
    Applying the national ratio to a locality remains an estimated age split.
    """
    rows = []
    national = national.copy()
    national["age"] = pd.to_numeric(national.age, errors="raise")
    for (date, code), frame in registry.groupby(["reference_date", "locality_code"]):
        year = int(date[:4])
        n = national.loc[national.year.eq(year)]
        if n.empty:
            continue
        population = pd.to_numeric(n.population, errors="raise")
        denominator = float(population.loc[n.age.between(6, 18)].sum())
        if denominator <= 0:
            raise ValueError("National child-age support is absent")
        ratio = float(population.loc[n.age.eq(18)].sum()/denominator)
        counts = frame.set_index("category").value
        age18 = float(counts["age6_18"]*ratio)
        rows.append({"year": year, "locality_code": code, "reference_date": date,
                     "registered_age19_plus_observed": float(counts["age19_plus"]),
                     "registered_age65_plus_observed": float(counts["age65_plus"]),
                     "registered_age18_estimated": age18,
                     "registered_age18_plus_estimated": float(counts["age19_plus"]+age18),
                     "national_age18_fraction_within6_18": ratio,
                     "status": "observed_registry19plus_with_national_singleage_estimated18",
                     "source_id": frame.source_id.iloc[0],
                     "age18_prior_source_ids": "|".join(sorted(set(n.source_id.astype(str)))),
                     "citizenship_restricted": False, "includes_overseas_registered_persons": True})
    return pd.DataFrame(rows)


def jerusalem_adult_rate():
    adults = [counts for age, counts in JERUSALEM_STATUS.items() if age != "0_17"]
    citizen = sum(sum(counts[:2]) for counts in adults)
    total = sum(sum(counts) for counts in adults)
    return citizen/total, citizen, total


def jerusalem_wholecity_scope_rate(wholecity_arab_adults_2024):
    """Alternative scope: known East counts plus citizen outside-East residual.

    This assumes compatible resident/registry universes and citizenship1 outside
    East Jerusalem; it is a sensitivity, not an identified city citizenship rate.
    """
    _, citizens, east_adults = jerusalem_adult_rate()
    whole = float(wholecity_arab_adults_2024)
    if not np.isfinite(whole) or whole < east_adults:
        raise ValueError("Whole-city prior is smaller than East registered-adult control; source universes are incompatible")
    outside = whole-east_adults
    noncitizen_or_unconfirmed = east_adults-citizens
    return (whole-noncitizen_or_unconfirmed)/whole, {"wholecity_Arab_adult_prior_2024": whole,
        "observed_East_Arab_registered_adults_2024": east_adults,
        "observed_East_Arab_citizen_adults_2024": citizens,
        "East_noncitizen_or_unconfirmed_adults": noncitizen_or_unconfirmed,
        "outside_East_residual_adults_assumed_citizens": outside,
        "scope_assumption": "Outside-East residual citizenship1; compatible resident/registry universes assumed, not observed"}


def national_citizen_controls():
    rows = [{"election":election,"year": spec["year"], "dimension": "resident_eligible_citizens", "category": group, "value": value,
             "unit": "persons", "source_precision":spec["precision"], "status": "official_CBS_report_rounded_control",
             "reference_date":NATIONAL_REFERENCE_DATES[election],"election_date":ELECTION_DATES[election],
             "reference_basis":"Pre-election CBS resident-eligible estimate; report/reference period retained separately from election date",
             "source_id":spec["source"], "includes_overseas": False}
            for election,spec in NATIONAL_ELECTION_CITIZENS.items() for group,value in spec["counts"].items()]
    # Haredi is a subset of Jewish, not a fourth exhaustive population group.
    for year, value in [(2022,512000.),(2026,684000.)]:
        rows.append({"year": year,"dimension":"resident_eligible_Haredi_validation_subset","category":"haredi","value":value,
                     "unit":"persons","reported_precision_persons":1000,"status":"official_CBS_model_estimate_definition_check_required",
                     "reference_date":"2022-11-01" if year==2022 else "2025-12-31","source_id":NATIONAL_CITIZEN_SOURCES[year],"includes_overseas":False})
    return pd.DataFrame(rows)


def rake_register_matrix(seed, local_totals, national_totals, tolerance=1e-7):
    """Four-column IPF: resident citizens by group plus unidentified roll gap."""
    matrix=np.asarray(seed,dtype=float).copy()
    rows=np.asarray(local_totals,dtype=float);columns=np.asarray(national_totals,dtype=float)
    if matrix.shape!=(len(rows),len(columns)) or not np.isfinite(matrix).all() or (matrix<0).any() or (rows<0).any() or (columns<0).any():
        raise ValueError("Invalid register calibration inputs")
    if not np.isclose(rows.sum(),columns.sum(),rtol=0,atol=1e-6):
        raise ValueError("National citizen and gap targets must equal official register")
    if any(columns[j]>rows[matrix[:,j]>0].sum()+tolerance for j in range(len(columns))):
        raise ValueError("National citizen target exceeds supported locality capacity")
    for iteration in range(5000):
        sums=matrix.sum(axis=0)
        if ((sums<=0)&(columns>0)).any():raise ValueError("Unsupported national citizenship group")
        matrix*=np.divide(columns,sums,out=np.zeros_like(columns),where=sums>0)
        sums=matrix.sum(axis=1)
        if ((sums<=0)&(rows>0)).any():raise ValueError("Unsupported electorate locality")
        matrix*=np.divide(rows,sums,out=np.zeros_like(rows),where=sums>0)[:,None]
        error=max(float(abs(matrix.sum(axis=0)-columns).max()),float(abs(matrix.sum(axis=1)-rows).max()))
        if error<=tolerance:break
    else:raise ValueError("Citizen/register calibration did not converge")
    return matrix,{"iterations":iteration+1,"max_mass_error_persons":error}


def calibrate_national_electorate(eligible,adult_priors,jerusalem_rate=None,specifications=None):
    """National resident-citizen constraints; local registered mixture scenarios.

    National broad-group totals apply only to resident citizens. A fourth column
    retains the register/resident difference. The main registered estimate assigns
    that column within each locality by its fitted resident-citizen composition.
    """
    groups=["jewish","arab","other"]
    rate=jerusalem_adult_rate()[0] if jerusalem_rate is None else float(jerusalem_rate)
    if not np.isfinite(rate) or not 0<=rate<=1:raise ValueError("Invalid citizenship prior factor")
    source=adult_priors.loc[adult_priors.locality_code.ne(9999)].copy()
    if "period_id" not in source:source["period_id"]=source.year.astype(str)
    source["period_id"]=source.period_id.astype(str)
    if not np.isfinite(source.adult_population).all() or (source.adult_population<0).any():raise ValueError("Invalid resident adult population")
    outputs=[];audits=[]
    specs=NATIONAL_ELECTION_CITIZENS if specifications is None else specifications
    for election,z in eligible.loc[eligible.election.isin(specs)].groupby("election"):
        z=z.sort_values("locality_code").reset_index(drop=True);year=int(z.year.iloc[0]);n=z.registered_eligible.to_numpy(float)
        spec=specs[election]
        target=np.array([spec["counts"][g] for g in groups]);gap=float(n.sum()-target.sum())
        if gap<0:raise ValueError("Resident eligible target exceeds complete CEC register; national/local coverage mismatch")
        period=election if source.period_id.eq(election).any() else str(year)
        selected=source.loc[source.period_id.eq(period)]
        if selected.empty:raise ValueError("No resident adult priors for electorate period "+period)
        p=selected.groupby(["locality_code","broad_group"]).adult_population.sum().unstack(fill_value=0).reindex(index=z.locality_code,columns=groups,fill_value=0).fillna(0)
        original=p.to_numpy(float);missing=original.sum(axis=1)<=0
        # Explicit national broad-group fallback never assigns a secular profile.
        initial=original.copy();initial[missing]=n[missing,None]*target[None,:]/target.sum()
        initial[z.locality_code.eq(3000).to_numpy(),groups.index("arab")]*=rate
        gap_prior=np.maximum(n-initial.sum(axis=1),n*(gap/max(1,n.sum()))*.01)
        matrix,audit=rake_register_matrix(np.column_stack([initial,gap_prior]),n,np.append(target,gap))
        for i,row in enumerate(z.itertuples()):
            resident=matrix[i,:3];resident_total=float(resident.sum());registered=resident/resident_total*row.registered_eligible if resident_total>0 else np.zeros(3)
            if resident_total>0:registered[int(np.argmax(registered))]+=row.registered_eligible-float(registered.sum())
            for j,g in enumerate(groups):
                outputs.append({"election":election,"year":year,"locality_code":row.locality_code,"broad_group":g,
                    "registered_eligible":row.registered_eligible,"resident_adult_prior":original[i,j],"population_prior_period":period,
                    "calibrated_resident_citizen_count":resident[j],"estimated_registered_group_count":registered[j],
                    "estimated_eligible":registered[j],"resident_citizen_estimate":resident[j],"unallocated_gap":matrix[i,3],
                    "unallocated_gap_is_repeated_locality_control_not_additive":True,
                    "population_to_registered_factor":registered[j]/original[i,j] if original[i,j]>0 else None,
                    "population_to_resident_citizen_factor":resident[j]/original[i,j] if original[i,j]>0 else None,
                    "conservative_unallocated_register_gap":matrix[i,3],
                    "prior_support":"national_broad_group_fallback_no_local_population_support" if missing[i] else "provided_locality_resident_population_prior",
                    "unsupported_locality_eligible_mass":row.registered_eligible if missing[i] else 0.,
                    "citizenship_prior":"carried_EastJerusalem2024_rate" if row.locality_code==3000 and g=="arab" else "factor1_prior_before_national_citizen_calibration",
                    "national_citizen_source":spec["source"],"national_source_precision":spec["precision"],
                    "national_citizen_reference_date":NATIONAL_REFERENCE_DATES.get(election,"synthetic"),
                    "register_gap_assumption":"Primary: same citizen-group composition within locality. Conservative: retain roll gap unallocated."})
        audits.append({"election":election,"year":year,**audit,"registered_total":float(n.sum()),"resident_citizen_targets":dict(zip(groups,target)),
            "register_minus_resident_gap":gap,"national_source_precision":spec["precision"],"total_rounding_uncertainty_persons":spec["total_rounding"],
            "unsupported_locality_eligible_mass":float(n[missing].sum()),"unsupported_localities":int(missing.sum()),
            "cec_locality_frame_resident_adult_priors":dict(zip(groups,original.sum(axis=0))),
            "resident_citizen_target_minus_cec_frame_adult_prior":dict(zip(groups,target-original.sum(axis=0))),
            "all_geography_resident_adult_priors":selected.groupby("broad_group").adult_population.sum().reindex(groups,fill_value=0).to_dict(),
            "resident_citizen_target_minus_all_geography_adult_prior":dict(zip(groups,target-selected.groupby("broad_group").adult_population.sum().reindex(groups,fill_value=0).to_numpy())),
            "resident_prior_outside_cec_locality_frame":selected.loc[~selected.locality_code.isin(z.locality_code)].groupby("broad_group").adult_population.sum().reindex(groups,fill_value=0).to_dict(),
            "resident_prior_locality_codes_outside_cec_frame":sorted(int(code) for code in set(selected.locality_code)-set(z.locality_code)),
            "national_target_geographic_frame_assumption":"All national resident-citizen controls allocated across CEC locality codes; resident priors outside this frame are disclosed, not separately identified electoral-roll populations.",
            "calibration_factors_not_individual_citizenship_probabilities":True,
            "locality_group_cells_citizen_estimate_above_resident_prior":int(((matrix[:,:3]>original+1e-7)&(original>0)).sum()),
            "national_broad_group_calibration_not_local_citizenship_observation":True})
    return pd.DataFrame(outputs),audits


def jerusalem_scope_comparison(eligible, priors):
    """Actual electorate allocations under both explicit citizenship scopes."""
    calendar=priors.loc[priors.period_id.astype(str).eq(priors.year.astype(str))] if "period_id" in priors else priors
    totals = calendar.loc[calendar.locality_code.eq(3000)&calendar.broad_group.eq("arab")].groupby("year").adult_population.sum()
    if 2024 in totals:
        anchor, method = float(totals.loc[2024]), "provided_2024_wholecity_Arab_adult_prior"
    elif {2022, 2026} <= set(totals.index):
        anchor = float((totals.loc[2022]+totals.loc[2026])/2)
        method = "2024_linear_midpoint_of_2022_and_2026_wholecity_Arab_adult_priors"
    else:
        return pd.DataFrame(), {"status": "scope_comparison_requires2024_or2022_and2026_Arab_adult_priors"}
    try:
        alternative, details = jerusalem_wholecity_scope_rate(anchor)
    except ValueError as error:
        return pd.DataFrame(), {"status": "incompatible_population_universes", "reason": str(error), "wholecity_prior": anchor}
    details.update({"status": "explicit_scope_sensitivity_not_identified_citizenship", "anchor_method": method,
                    "wholecity_scope_citizenship_factor": alternative})
    selected = eligible.loc[eligible.locality_code.eq(3000)&eligible.year.isin([2022,2026])]
    frames = []
    for label, rate in [("East_rate_transferred_to_whole_city", jerusalem_adult_rate()[0]),
                        ("East_noncitizen_mass_plus_outside_citizen1", alternative)]:
        result = allocate_electorate(selected,priors,jerusalem_rate=rate)
        result["citizenship_scope_scenario"] = label
        result["wholecity_2024_prior"] = anchor
        result["scope_factor"] = rate
        result["anchor_method"] = method
        frames.append(result)
    return pd.concat(frames,ignore_index=True), details


def religion_controls():
    return pd.DataFrame([{"year": 2022, "locality_code": code, "dimension": "religion_share",
        "category": "druze", "value": share, "unit": "fraction_of_all_age_locality_population",
        "status": "official_rounded_all_age_fraction_not_adult_person_identity", "source_id": DRUZE_SOURCE}
        for code, share in DRUZE_SHARES_2022.items()])


def legacy_religion_audit(project):
    path = project/"data/clean/model_input_v2.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path,usecols=["election","locality_code","share_druze","is_envelope"])
    frame = frame.loc[frame.election.eq("K25")&~frame.is_envelope.astype(str).str.lower().eq("true")]
    legacy = frame.groupby("locality_code").share_druze.agg(["min","max"])
    output = religion_controls()
    output["legacy_share_druze_min"] = output.locality_code.map(legacy["min"])
    output["legacy_share_druze_max"] = output.locality_code.map(legacy["max"])
    output["comparison_scope"] = "Legacy PUF-derived modeled fraction versus official rounded all-age religion fraction; not individual religion or exact adult denominator"
    return output


def audit_2022_denominators(project, eligible):
    source = project/"data/processed/demographic_foundation/adult2022_cells.csv"
    if not source.exists():
        return None, {"status": "adult2022_aggregate_not_available"}
    cells = pd.read_csv(source, usecols=["locality_code", "population_group", "person_weight"])
    grouped = cells.groupby("locality_code").person_weight.sum()
    masked = float(grouped.get(9999, 0))
    grouped = grouped.drop(index=9999, errors="ignore")
    frame = eligible.loc[eligible.election.eq("K25")].copy()
    frame["resident_adults_2022_PUF"] = frame.locality_code.map(grouped)
    frame["named_PUF_coverage"] = frame.resident_adults_2022_PUF.notna()
    frame["register_minus_resident_adults"] = frame.registered_eligible-frame.resident_adults_2022_PUF
    return frame, {"status": "resident_and_electorate_universes_compared_not_equated",
        "registered_eligible": float(frame.registered_eligible.sum()),
        "PUF_resident_adults": float(grouped.sum()+masked), "PUF_masked_locality_adults": masked,
        "eligible_outside_named_PUF": float(frame.loc[~frame.named_PUF_coverage, "registered_eligible"].sum()),
        "localities_outside_named_PUF": int((~frame.named_PUF_coverage).sum()),
        "Jerusalem": frame.loc[frame.locality_code.eq(3000), ["registered_eligible", "resident_adults_2022_PUF", "register_minus_resident_adults"]].to_dict("records")}


def eligible_denominators(project):
    """Read only official eligible controls, locality identities and envelope flags."""
    outputs = []
    for election, date in ELECTION_DATES.items():
        source = None
        if election == "K26":
            source = project/"data/processed/k26/current/ballots.csv"
            raw = pd.read_csv(source, usecols=["locality_code", "locality_name", "eligible_voters", "external_envelope"])
            raw = raw.loc[~raw.external_envelope.astype(str).str.lower().eq("true")]
        elif int(election[1:]) >= 21:
            source = project/"data/raw/ballots"/(election+"_kalpi_clean.csv")
            raw = pd.read_csv(source, usecols=["סמל ישוב", "שם ישוב", "בזב"]).rename(columns={"סמל ישוב": "locality_code", "שם ישוב": "locality_name", "בזב": "eligible_voters"})
        else:
            candidates = [project/"data/raw/ballots"/directory/(election+"_official_datastore.json") for directory in ["historical_model", "historical_turnout"]]
            source = next((path for path in candidates if path.exists()), None)
            if source is None:
                continue
            records = json.loads(source.read_text(encoding="utf-8-sig"))["result"]["records"]
            first = records[0]
            eligible_field = next((field for field in ["בוחרים", "בזב", "בז''ב"] if field in first), None)
            if eligible_field is None or "סמל ישוב" not in first:
                continue
            raw = pd.DataFrame([{"locality_code": row.get("סמל ישוב"), "locality_name": row.get("שם ישוב"), "eligible_voters": row.get(eligible_field)} for row in records])
        raw = raw.loc[~raw.locality_name.fillna("").astype(str).str.contains("מעטפות", regex=False)].copy()
        raw["eligible_voters"] = pd.to_numeric(raw.eligible_voters, errors="coerce")
        raw["locality_code"] = pd.to_numeric(raw.locality_code, errors="coerce")
        raw = raw.loc[raw.eligible_voters.notna() & raw.locality_code.notna()]
        if (raw.eligible_voters < 0).any():
            raise ValueError("Negative eligible count")
        raw["locality_code"] = raw.locality_code.astype(int)
        grouped = raw.groupby("locality_code", as_index=False).agg(registered_eligible=("eligible_voters", "sum"), locality_name=("locality_name", "first"))
        grouped["election"], grouped["year"], grouped["reference_date"] = election, int(date[:4]), date
        grouped["source_id"] = str(source.relative_to(project))
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def allocate_electorate(eligible, adult_priors, scenario="proportional_citizen_prior", jerusalem_rate=None):
    """Calibrate estimates to exact official N; retain missing priors as unknown.

    The alternative conservative gap scenario leaves positive register excess
    unallocated rather than assigning overseas/coverage differences to groups.
    """
    if scenario not in {"proportional_citizen_prior", "unallocated_register_excess"}:
        raise ValueError("Unknown electorate allocation scenario")
    keys = ["year", "locality_code"]
    if eligible.duplicated(["election", "locality_code"]).any():
        raise ValueError("Duplicate electorate identity")
    prior = adult_priors.copy()
    required = set(keys+["broad_group", "adult_population"])
    if required-set(prior):
        raise ValueError("Adult priors require year/locality/group/adult population")
    prior["adult_population"] = pd.to_numeric(prior.adult_population, errors="raise")
    if not np.isfinite(prior.adult_population).all() or (prior.adult_population < 0).any():
        raise ValueError("Invalid adult prior")
    # 9999 is a masked locality pool, including small urban and council localities.
    prior = prior.loc[prior.locality_code.ne(9999)]
    if "period_id" not in prior:prior["period_id"]=prior.year.astype(str)
    prior["period_id"]=prior.period_id.astype(str)
    periods=set(prior.period_id)
    priors = {key: group for key, group in prior.groupby(["period_id","locality_code"])}
    observed_rate, _, _ = jerusalem_adult_rate()
    rate = observed_rate if jerusalem_rate is None else float(jerusalem_rate)
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Citizenship factor must lie in [0,1]")
    rows = []
    for entry in eligible.itertuples():
        n = float(entry.registered_eligible)
        if not np.isfinite(n) or n < 0:
            raise ValueError("Invalid eligible total")
        period=entry.election if entry.election in periods else str(entry.year)
        frame = priors.get((period, entry.locality_code))
        base = {"election": entry.election, "year": entry.year, "locality_code": entry.locality_code,
                "registered_eligible": n, "scenario": scenario, "registered_total_source": entry.source_id}
        base["population_prior_period"]=period
        if frame is None or frame.adult_population.sum() <= 0:
            rows.append({**base, "broad_group": "unallocated", "estimated_eligible": n, "status": "no_supported_locality_population_prior",
                         "citizenship_source_year": None, "citizenship_factor": None, "unallocated_mass": n})
            continue
        weights = frame.groupby("broad_group").adult_population.sum()
        factors = pd.Series(1., index=weights.index)
        if "foreign" in factors:
            factors.loc["foreign"] = 0.
        jerusalem = entry.locality_code == 3000 and "arab" in weights.index
        if jerusalem:
            factors.loc["arab"] = rate
        citizen_prior = weights*factors
        total = float(citizen_prior.sum())
        if total <= 0:
            rows.append({**base, "broad_group": "unallocated", "estimated_eligible": n, "unallocated_mass": n,
                         "status": "no_supported_citizen_prior"})
            continue
        assigned_total = min(n, total) if scenario == "unallocated_register_excess" else n
        values = (citizen_prior/total*assigned_total).to_numpy(copy=True)
        values[-1] = assigned_total-float(values[:-1].sum())
        for group, value in zip(weights.index, values):
            rows.append({**base, "broad_group": group, "estimated_eligible": float(value), "unallocated_mass": 0.,
                         "resident_adult_prior": float(weights[group]), "citizen_resident_prior": float(citizen_prior[group]),
                         "citizenship_factor": float(factors[group]), "citizenship_source_year": 2024 if jerusalem and group == "arab" else None,
                         "status": "2024_EastJerusalem_citizenship_rate_carried_scenario" if jerusalem and group == "arab" else "citizenship_unobserved_factor1_assumption",
                         "registration_assumption": "Register composition follows adjusted resident priors" if scenario == "proportional_citizen_prior" else "Positive register excess remains unallocated"})
        if assigned_total < n:
            rows.append({**base, "broad_group": "unallocated_register_gap", "estimated_eligible": n-assigned_total,
                         "unallocated_mass": n-assigned_total, "status": "overseas_or_residence_coverage_gap_unidentified"})
    output = pd.DataFrame(rows)
    actual = output.groupby(["election", "locality_code"]).estimated_eligible.sum()
    expected = eligible.set_index(["election", "locality_code"]).registered_eligible
    if not np.allclose(actual.sort_index(), expected.sort_index(), rtol=0, atol=1e-8):
        raise ValueError("Electorate allocation failed exact official-total closure")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--adult-controls", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    output = args.project/"data/processed/population_reconstruction"
    sources = output/"sources";sources.mkdir(parents=True, exist_ok=True)
    registry = []
    for date, source_id in REGISTRY_RESOURCES.items():
        path = sources/("piba_registry_"+date.replace("-", "_")+".json")
        url = "https://data.gov.il/api/3/action/datastore_search?resource_id="+source_id+"&limit=10000"
        if not path.exists() and not args.download:
            continue
        if not path.exists():
            with urllib.request.urlopen("https://data.gov.il/api/3/action/resource_show?id="+source_id,timeout=60) as response:
                resource=json.loads(response.read())["result"]
            verify_registry_source_date(resource,date,source_id)
        content = fetch_source(url, path)
        registry.append(normalize_registry(json.loads(content), date, source_id))
    if registry:
        registry_frame = pd.concat(registry, ignore_index=True)
        registry_frame.to_csv(output/"registry_age_controls.csv", index=False, encoding="utf-8-sig")
        national_path = output/"national_age_sex_group_controls.csv"
        if national_path.exists():
            registry_adult_controls(registry_frame, pd.read_csv(national_path)).to_csv(output/"registry_adult_controls.csv", index=False, encoding="utf-8-sig")
    jerusalem_citizenship_controls().to_csv(output/"jerusalem_citizenship_controls.csv", index=False, encoding="utf-8-sig")
    national_citizen_controls().to_csv(output/"national_citizen_controls.csv",index=False,encoding="utf-8-sig")
    religion_controls().to_csv(output/"religion_controls2022.csv", index=False, encoding="utf-8-sig")
    legacy = legacy_religion_audit(args.project)
    if legacy is not None:
        legacy.to_csv(output/"legacy_druze_comparison2022.csv",index=False,encoding="utf-8-sig")
    if args.download:
        fetch_source(JERUSALEM_SOURCE, sources/"jerusalem_citizenship_2024.pdf")
        fetch_source(DRUZE_SOURCE, sources/"CBS_Druze2022_tableB.pdf")
        for election,spec in NATIONAL_ELECTION_CITIZENS.items():
            label=str(spec["year"]) if election in {"K25","K26"} else election
            fetch_source(spec["source"],sources/("CBS_resident_eligible_"+label+".pdf"))
    eligible = eligible_denominators(args.project)
    eligible.to_csv(output/"registered_eligible_controls.csv", index=False, encoding="utf-8-sig")
    comparison, coverage = audit_2022_denominators(args.project, eligible)
    if comparison is not None:
        comparison.to_csv(output/"resident_vs_register2022.csv", index=False, encoding="utf-8-sig")
    scope_audit = {"status":"adult_controls_not_supplied"}
    national_audit=[]
    population_input={}
    if args.adult_controls is not None:
        wanted={"period_id","year","locality_code","broad_group","adult_population","population"}
        priors = pd.read_csv(args.adult_controls,usecols=lambda name:name in wanted)
        if "adult_population" not in priors and "population" in priors:
            priors = priors.rename(columns={"population": "adult_population"})
        dimensions=[name for name in ["period_id","year","locality_code","broad_group"] if name in priors]
        priors=priors.groupby(dimensions,as_index=False).adult_population.sum()
        population_input={"path":str(args.adult_controls),"sha256":hashlib.sha256(args.adult_controls.read_bytes()).hexdigest(),
                          "period_ids":sorted(set(priors.period_id.astype(str))) if "period_id" in priors else sorted(int(year) for year in priors.year.unique())}
        for scenario in ["proportional_citizen_prior", "unallocated_register_excess"]:
            allocate_electorate(eligible, priors, scenario).to_csv(output/("electorate_"+scenario+".csv"), index=False, encoding="utf-8-sig")
        annual_rates = [sum(counts[:2])/sum(counts) for age, counts in JERUSALEM_STATUS.items() if age != "0_17"]
        sensitivity = []
        for label, rate in [("source_age_conditional_min", min(annual_rates)), ("pooled_adult_source_rate", jerusalem_adult_rate()[0]), ("source_age_conditional_max", max(annual_rates))]:
            result = allocate_electorate(eligible.loc[eligible.locality_code.eq(3000)], priors, jerusalem_rate=rate)
            result["citizenship_sensitivity"] = label
            sensitivity.append(result)
        pd.concat(sensitivity, ignore_index=True).to_csv(output/"jerusalem_citizenship_sensitivity.csv", index=False, encoding="utf-8-sig")
        scopes,scope_audit = jerusalem_scope_comparison(eligible,priors)
        if len(scopes):
            scopes.to_csv(output/"jerusalem_scope_scenarios.csv",index=False,encoding="utf-8-sig")
        # Country calibration requires priors for every modeled national year.
        if set(NATIONAL_RESIDENT_ELIGIBLE)<=set(priors.year):
            factors,national_audit=calibrate_national_electorate(eligible,priors)
            factors.to_csv(output/"citizen_electorate_factors.csv",index=False,encoding="utf-8-sig")
            factors[["election","year","locality_code","unallocated_gap"]].drop_duplicates().to_csv(output/"locality_roll_gap_controls.csv",index=False,encoding="utf-8-sig")
            if scope_audit.get("wholecity_scope_citizenship_factor") is not None:
                alternative,_=calibrate_national_electorate(eligible,priors,jerusalem_rate=scope_audit["wholecity_scope_citizenship_factor"])
                alternative.to_csv(output/"citizen_electorate_factors_scope_sensitivity.csv",index=False,encoding="utf-8-sig")
                base=factors.loc[factors.locality_code.eq(3000)&factors.year.isin([2022,2026])].copy();base["scope"]="East_rate_transferred_to_whole_city"
                alt=alternative.loc[alternative.locality_code.eq(3000)&alternative.year.isin([2022,2026])].copy();alt["scope"]="East_noncitizen_mass_plus_outside_citizen1"
                pd.concat([base,alt],ignore_index=True).to_csv(output/"jerusalem_nationally_calibrated_scope_scenarios.csv",index=False,encoding="utf-8-sig")
    rate, citizen, total = jerusalem_adult_rate()
    audit = {"electorate_definition_source": ELIGIBILITY_SOURCE, "jerusalem_citizenship_source": JERUSALEM_SOURCE,
             "builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "jerusalem_2024_Arab_adult_citizens": citizen, "jerusalem_2024_Arab_adult_registered_residents": total,
             "jerusalem_2024_Arab_adult_citizenship_rate": rate,
             "registry_sources": REGISTRY_RESOURCES, "outcome_data_used_for_population_classification": False,
             "resident_vs_register_2022": coverage,
             "missing_official_eligible_elections": sorted(set(ELECTION_DATES)-set(eligible.election)),
             "Druze_controls_source": DRUZE_SOURCE,
             "jerusalem_scope_comparison": scope_audit,
             "national_citizen_calibration":national_audit,
             "resident_population_input":population_input,
             "limitations": ["PIBA registry includes living registered persons abroad and non-citizens; it is neither CBS resident population nor electoral eligibility.",
                             "PIBA observed adult bin starts19; age18 belongs to6–18. No exact18+ count is fabricated.",
                             "Jerusalem citizenship anchor is EastJerusalem Arab population July2024; transfer to whole-city Arab adults or another year is a labeled sensitivity assumption.",
                             "Other group citizenship factors1 are assumptions, not observed citizen shares.",
                             "Exact total closure does not identify subgroup electorate composition or overseas voter composition.",
                             "CensusPUF9999 denotes a masked locality pool, including small urban and regional-council localities, not one locality or CEC envelope code. The primary national calibration uses a flagged national broad-group fallback where locality priors are absent; diagnostic allocation scenarios retain unsupported mass unallocated."]}
    (output/"electorate_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"eligible_locality_elections": len(eligible), "Jerusalem_citizen_adult_rate": rate, "registry_dates": list(REGISTRY_RESOURCES)}))


if __name__ == "__main__":
    main()
