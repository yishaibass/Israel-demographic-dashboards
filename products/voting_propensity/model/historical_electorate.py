"""Historical resident-citizen controls, kept separate from published defaults.

CBS headline totals and integer group percentages are rounded observations.
For 2003 only the total has been recovered; an optional demographic split is
explicitly estimated and must not be presented as an observed group count.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from population_electorate import (calibrate_national_electorate,
    eligible_denominators, fetch_source, NATIONAL_ELECTION_CITIZENS)


HISTORICAL_CITIZENS = {
    "K18": {"year": 2009, "total": 4800000., "shares": (.81, .14, .05),
        "reference_date": "2009-01-27", "release_year": 2009, "release_number": 16},
    "K19": {"year": 2013, "total": 5100000., "shares": (.81, .15, .04),
        "reference_date": "2012-12-26", "release_year": 2012, "release_number": 354},
    "K20": {"year": 2015, "total": 5300000., "shares": (.80, .15, .05),
        "reference_date": "2015-02-09", "release_year": 2015, "release_number": 29},
}
HISTORICAL_TOTALS_ONLY = {
    "K16": {"year": 2003, "total": 4300000., "reference_date": "2003 election",
        "release_year": 2009, "release_number": 24},
}
GROUPS = ("jewish", "arab", "other")


def source_url(spec):
    year, number = spec["release_year"], spec["release_number"]
    return f"https://www.cbs.gov.il/he/mediarelease/DocLib/{year}/{number:03d}/24_{str(year)[2:]}_{number:03d}b.pdf"


def period_group_totals(adult_priors, election, year):
    priors = adult_priors.copy()
    if "period_id" in priors and priors.period_id.astype(str).eq(election).any():
        priors = priors.loc[priors.period_id.astype(str).eq(election)]
    else:
        priors = priors.loc[priors.year.eq(year)]
        if "period_id" in priors:
            priors = priors.loc[priors.period_id.astype(str).eq(str(year))]
    priors = priors.loc[priors.broad_group.isin(GROUPS)]
    if priors.empty or not np.isfinite(priors.adult_population).all() or (priors.adult_population < 0).any():
        raise ValueError(f"{year} requires finite nonnegative population support")
    return priors.groupby("broad_group").adult_population.sum().reindex(GROUPS, fill_value=0)


def historical_specifications(adult_priors=None, *, include_estimated_2003_split=False,
                              split_2003="backcast_2009_group_ratios"):
    """Return calibrator specifications without modifying its current defaults."""
    result = {}
    for election, observation in HISTORICAL_CITIZENS.items():
        spec = deepcopy(observation)
        spec["counts"] = dict(zip(GROUPS, np.array(spec["shares"])*spec["total"]))
        spec.update(source=source_url(spec), total_rounding=50000,
            precision="Headline rounded to 0.1 million; group shares rounded to integer percentage points; derived group counts approximate",
            group_control_status="CBS_observed_rounded_total_and_group_percentages",
            group_share_rounding_percentage_points=.5)
        result[election] = spec
    if include_estimated_2003_split:
        if adult_priors is None:
            raise ValueError("2003 estimated split requires election-date adult priors")
        spec = deepcopy(HISTORICAL_TOTALS_ONLY["K16"])
        masses = period_group_totals(adult_priors, "K16", 2003)
        if split_2003 == "backcast_2009_group_ratios":
            adults2009 = period_group_totals(adult_priors, "K18", 2009)
            if (adults2009 <= 0).any():
                raise ValueError("2009 backcast needs positive support in every broad group")
            ratios = pd.Series(result["K18"]["counts"])/adults2009
            masses *= ratios
            assumption = "BACKCAST: CBS 2009 group citizen totals divided by dated 2009 resident adult group totals, applied to dated 2003 resident adult groups, then scaled to CBS 4.3 million total; ratios are aggregate calibration factors, not individual citizenship probabilities"
        elif split_2003 == "unadjusted_resident_mix":
            ratios = pd.Series(1., index=GROUPS)
            assumption = "SENSITIVITY: unadjusted dated 2003 resident adult group mix scaled to CBS 4.3 million total; group citizenship differences unobserved"
        else:
            raise ValueError("Unknown 2003 group split")
        if masses.sum() <= 0:
            raise ValueError("2003 population support has zero mass")
        spec["counts"] = (masses/masses.sum()*spec["total"]).to_dict()
        spec.update(source=source_url(spec), total_rounding=50000,
            precision="CBS 4.3 million total rounded to 0.1 million; no observed 2003 group percentages recovered",
            group_control_status="BACKCAST_2003_group_split_not_observed" if split_2003=="backcast_2009_group_ratios" else "SENSITIVITY_2003_resident_mix_not_observed",
            group_split_assumption=assumption,
            group_split_2009_citizen_adult_ratios=ratios.to_dict(),
            group_share_rounding_percentage_points=None)
        result["K16"] = spec
    return result


def historical_controls(eligible, adult_priors, *, include_estimated_2003_split=False,
                        jerusalem_rate=None, split_2003="backcast_2009_group_ratios", include_recent=False):
    specs = historical_specifications(adult_priors,
        include_estimated_2003_split=include_estimated_2003_split,
        split_2003=split_2003)
    if include_recent:
        from population_electorate import NATIONAL_REFERENCE_DATES
        recent = deepcopy(NATIONAL_ELECTION_CITIZENS)
        for election, spec in recent.items():
            spec.update(reference_date=NATIONAL_REFERENCE_DATES[election],
                group_control_status="CBS_observed_rounded_total_and_group_percentages" if election in {"K21","K22","K23"} else "CBS_observed_rounded_group_counts",
                group_share_rounding_percentage_points=.5 if election in {"K21","K22","K23"} else None)
        specs.update(recent)
    controls, audits = calibrate_national_electorate(eligible, adult_priors,
        jerusalem_rate=jerusalem_rate, specifications=specs)
    if not controls.empty:
        controls["national_citizen_reference_date"] = controls.election.map({e:s["reference_date"] for e,s in specs.items()})
        controls["national_group_control_status"] = controls.election.map({e:s["group_control_status"] for e,s in specs.items()})
    for audit in audits:
        spec = specs[audit["election"]]
        audit.update(national_citizen_source=spec["source"],
            national_citizen_reference_date=spec["reference_date"],
            national_group_control_status=spec["group_control_status"],
            group_share_rounding_percentage_points=spec["group_share_rounding_percentage_points"],
            group_split_assumption=spec.get("group_split_assumption"),
            group_split_2009_citizen_adult_ratios=spec.get("group_split_2009_citizen_adult_ratios"),
            source_page=1,
            source_total_reference="2003 row of retrospective table" if audit["election"]=="K16" else "Release headline and population-group paragraph")
    return controls, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--adult-controls", type=Path)
    parser.add_argument("--include-estimated-2003-split", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    output = args.project/"data/processed/population_reconstruction/historical"
    sources = output/"sources"
    observations = {**HISTORICAL_CITIZENS, **(HISTORICAL_TOTALS_ONLY if args.include_estimated_2003_split else {})}
    for spec in observations.values():
        path = sources/f"CBS_{spec['release_year']}_{spec['release_number']:03d}.pdf"
        if not path.exists() and not args.download:
            raise FileNotFoundError(f"Missing historical source {path.name}; rerun with --download")
        if not fetch_source(source_url(spec),path).startswith(b"%PDF"):
            raise ValueError("Historical source is not a PDF")
    priors = pd.read_csv(args.adult_controls or args.project/"data/processed/population_reconstruction/adult_controls.csv.gz", low_memory=False)
    eligible = eligible_denominators(args.project)
    controls, audits = historical_controls(eligible, priors,
        include_estimated_2003_split=args.include_estimated_2003_split, include_recent=True)
    output.mkdir(parents=True, exist_ok=True)
    controls.to_csv(output/"national_citizen_electorate_controls.csv.gz", index=False)
    (output/"national_citizen_electorate_audit.json").write_text(json.dumps(audits, indent=2)+"\n", encoding="utf-8")
    if args.include_estimated_2003_split:
        sensitivity, sensitivity_audits = historical_controls(eligible, priors,
            include_estimated_2003_split=True, include_recent=True, split_2003="unadjusted_resident_mix")
        sensitivity.to_csv(output/"national_citizen_electorate_controls_2003_resident_mix.csv.gz", index=False)
        (output/"national_citizen_electorate_audit_2003_resident_mix.json").write_text(json.dumps(sensitivity_audits, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"elections": sorted(controls.election.unique()), "rows": len(controls),
        "estimated_2003_group_split": args.include_estimated_2003_split}))


if __name__ == "__main__":
    main()
