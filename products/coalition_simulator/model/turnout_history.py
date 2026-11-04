"""Turnout-only historical extension with one fixed locality demographic design.

Official older CEC counts are archived separately from derived results. Published
simulator parameters are never overwritten. The 2022 Census/K25 locality shares
are carried backward and forward explicitly, rather than estimated as historical
population composition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from products.coalition_simulator.model.k26_comparison import GROUPS, SHARES, locality_composition, truth, weighted_r2

RESOURCES = {
    "K20": "929b50c6-f455-4be2-b438-ec6af01421f2",
    "K19": "c20cdcef-4d42-4241-a41b-6ca7ae51002d",
    "K18": "840edb33-90ac-4176-8ad9-4cdcb8e5caa5",
}
DATES = {"K18":"2009-02-10", "K19":"2013-01-22", "K20":"2015-03-17",
         "K21":"2019-04-09", "K22":"2019-09-17", "K23":"2020-03-02",
         "K24":"2021-03-23", "K25":"2022-11-01", "K26":"2026-10-27"}
WINDOWS = {"recent5":[f"K{k}" for k in range(21,26)],
           "extended7":[f"K{k}" for k in range(19,26)],
           "extended8":[f"K{k}" for k in range(18,26)]}
COUNT_COLUMNS = ["eligible_voters", "total_voted", "valid_votes", "invalid_votes"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fetch(url):
    request = urllib.request.Request(url,headers={"User-Agent":"IsraelVotingResearch/1.0"})
    with urllib.request.urlopen(request,timeout=30) as response:
        return response.read()


def archived_records(path,expected_resource=None):
    payload = Path(path).read_bytes()
    metadata = load_json(Path(path).with_suffix(".metadata.json"))
    if hashlib.sha256(payload).hexdigest()!=metadata["sha256"]:
        raise ValueError("Historical raw hash mismatch")
    document = json.loads(payload)
    if not document.get("success"):
        raise ValueError("Historical archive API success missing")
    result = document["result"]
    if expected_resource and (result.get("resource_id")!=expected_resource or metadata.get("resource_id")!=expected_resource):
        raise ValueError("Historical archive resource identity mismatch")
    if len(result["records"])!=result["total"]:
        raise ValueError("Historical archive is truncated")
    return result["records"]


def download_older(root):
    """Preserve the complete primary API response bytes, resource metadata and hash."""
    raw = root / "data/raw/ballots/historical_turnout"
    raw.mkdir(parents=True,exist_ok=True)
    for election, resource in RESOURCES.items():
        path = raw / f"{election}_official_datastore.json"
        metadata_path = path.with_suffix(".metadata.json")
        if path.exists():
            metadata = load_json(metadata_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("Historical raw hash mismatch: " + election)
            continue
        resource_url = "https://data.gov.il/api/3/action/resource_show?id=" + resource
        description = json.loads(fetch(resource_url))
        url = "https://data.gov.il/api/3/action/datastore_search?resource_id=" + resource + "&limit=20000"
        payload = fetch(url)
        document = json.loads(payload)
        if not document.get("success") or not description.get("success"):
            raise ValueError("Official API did not return success: " + election)
        result = document["result"]
        if len(result["records"]) != result["total"]:
            raise ValueError("Historical API response is truncated: " + election)
        _,quality = normalize_counts(pd.DataFrame(result["records"]), election,exclude_invalid_localities=True)
        metadata = {"election":election,"election_date":DATES[election],
                    "source_url":url,"resource_metadata_url":resource_url,
                    "resource_id":resource,"publisher":"Central Elections Committee via data.gov.il",
                    "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
                    "sha256":hashlib.sha256(payload).hexdigest(),"raw_format":"Official CKAN JSON response bytes",
                    "records":result["total"],"resource_metadata":description["result"],"source_quality":quality}
        path.write_bytes(payload)
        metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


def normalize_counts(frame, election,exclude_invalid_localities=False):
    """Reconcile every source row before aggregating eligible, regular localities."""
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    aliases = {"סמל ישוב":"locality_code","שם ישוב":"locality_name",
               "בזב":"eligible_voters","בז''ב":"eligible_voters",
               "מצביעים":"total_voted","כשרים":"valid_votes","פסולים":"invalid_votes"}
    frame = frame.rename(columns=aliases)
    required = ["locality_code","locality_name"] + COUNT_COLUMNS
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f"{election}: missing count fields {sorted(missing)}")
    for column in ["locality_code"] + COUNT_COLUMNS:
        value = pd.to_numeric(frame[column],errors="raise")
        if value.isna().any() or value.lt(0).any() or not np.equal(value,np.floor(value)).all():
            raise ValueError(f"{election}: noninteger/missing/negative {column}")
        frame[column] = value.astype("int64")
    if not (frame.valid_votes + frame.invalid_votes).eq(frame.total_voted).all():
        raise ValueError(f"{election}: valid + invalid != voted")
    external = (frame.locality_code.isin([0,9999]) | frame.locality_name.astype(str).str.contains("מעטפות",na=False)
                | frame.eligible_voters.eq(0))
    regular = frame.loc[~external,required].copy()
    invalid_rows = regular.loc[regular.total_voted.gt(regular.eligible_voters)]
    regular.locality_name = regular.locality_name.fillna("")
    grouped = regular.groupby("locality_code",as_index=False).agg(
        locality_name=("locality_name","first"),**{c:(c,"sum") for c in COUNT_COLUMNS})
    invalid = grouped.loc[grouped.total_voted.gt(grouped.eligible_voters)]
    if len(invalid) and not exclude_invalid_localities:
        raise ValueError(f"{election}: regular locality voters exceed eligible")
    invalid_codes = set(invalid.locality_code)
    excluded_invalid = regular.loc[regular.locality_code.isin(invalid_codes)]
    regular = regular.loc[~regular.locality_code.isin(invalid_codes)]
    grouped = grouped.loc[~grouped.locality_code.isin(invalid_codes)].copy()
    grouped.insert(0,"election",election)
    audit = {"election":election,"source_rows":len(frame),"regular_source_rows":len(regular),
             "excluded_envelope_or_zero_eligible_rows":int(external.sum()),
             "source_totals":{c:int(frame[c].sum()) for c in COUNT_COLUMNS},
             "regular_totals":{c:int(regular[c].sum()) for c in COUNT_COLUMNS},
             "excluded_invalid_locality_codes":sorted(int(c) for c in invalid_codes),
             "excluded_invalid_locality_rows":len(excluded_invalid),
             "incoherent_source_rows":invalid_rows.to_dict("records"),
             "incoherent_locality_aggregates":invalid.to_dict("records"),
             "excluded_invalid_locality_totals":{c:int(excluded_invalid[c].sum()) for c in COUNT_COLUMNS},
             "excluded_invalid_eligible_fraction_of_source":float(excluded_invalid.eligible_voters.sum()/frame.eligible_voters.sum()),
             "excluded_invalid_eligible_fraction_of_regular_source":float(excluded_invalid.eligible_voters.sum()/(regular.eligible_voters.sum()+excluded_invalid.eligible_voters.sum())),
             "regular_localities":len(grouped),"row_valid_invalid_reconciled":True}
    return grouped,audit


def build_panel(root):
    historical = pd.read_csv(root / "data/clean/model_input_v2.csv")
    composition,composition_audit = locality_composition(historical)
    frames,coverage = [],[]
    for k in range(18,27):
        election = f"K{k}"
        if election in RESOURCES:
            path = root / f"data/raw/ballots/historical_turnout/{election}_official_datastore.json"
            if not path.exists():
                continue
            raw = pd.DataFrame(archived_records(path,RESOURCES[election]))
        elif k < 26:
            path = root / f"data/raw/ballots/K{k} Kalpi.csv"
            raw = pd.read_csv(path,encoding="utf-8-sig" if k==25 else "cp1255")
        else:
            raw = pd.read_csv(root / "data/processed/k26/current/ballots.csv")
            # Canonical K26 counts retain the external-envelope flag explicitly.
            raw = raw.loc[~truth(raw.external_envelope)]
        counts,audit = normalize_counts(raw,election,exclude_invalid_localities=True)
        matched = counts.join(composition,on="locality_code",how="inner",validate="one_to_one")
        audit.update(matched_localities=len(matched),matched_eligible=int(matched.eligible_voters.sum()),
                     matched_eligible_fraction=float(matched.eligible_voters.sum()/counts.eligible_voters.sum()),
                     unmatched_locality_codes=sorted(int(c) for c in set(counts.locality_code)-set(matched.locality_code)))
        frames.append(matched)
        coverage.append(audit)
    panel = pd.concat(frames,ignore_index=True)
    if panel.duplicated(["election","locality_code"]).any():
        raise ValueError("Duplicate election/locality panel identities")
    return panel,{"composition":composition_audit,"elections":coverage}


def fit_turnout(frame):
    weights = frame.eligible_voters.to_numpy(float)
    if np.any(weights<=0) or frame.total_voted.gt(frame.eligible_voters).any():
        raise ValueError("Invalid turnout weights/counts")
    x = frame[SHARES].to_numpy(float)
    if len(frame)<len(GROUPS) or np.linalg.matrix_rank(x)!=len(GROUPS):
        raise ValueError("Six-group turnout design lacks rank")
    y = (frame.total_voted/frame.eligible_voters).to_numpy(float)
    sw = np.sqrt(weights/weights.sum())
    fit = lsq_linear(x*sw[:,None],y*sw,bounds=(.2,.95))
    if not fit.success:
        raise ValueError("Turnout least-squares failed")
    return fit.x,{"localities":len(frame),"eligible":int(weights.sum()),
                  "observed_matched_turnout":float(np.average(y,weights=weights)),
                  "weighted_r2":weighted_r2(y,x@fit.x,weights)}


def summarize_windows(coefficients,windows=WINDOWS):
    """Every baseline excludes2026; SD describes election-to-election dispersion."""
    output = []
    for name,elections in windows.items():
        if "K26" in elections:
            raise ValueError("2026 must be excluded from historical reference")
        if not set(elections).issubset(coefficients):
            continue
        history = np.array([coefficients[e] for e in elections])
        mean,sd = history.mean(axis=0),history.std(axis=0,ddof=0)
        current = np.asarray(coefficients["K26"])
        for i,group in enumerate(GROUPS):
            output.append({"history":name,"historical_elections":len(elections),
                           "distinct_calendar_years":len({DATES[e][:4] for e in elections}),
                           "group":group,"mean":float(mean[i]),"sd_ddof0":float(sd[i]),
                           "k26_estimate":float(current[i]),"k26_shift_pp":float(100*(current[i]-mean[i])),
                           "k26_shift_sd":float((current[i]-mean[i])/sd[i]) if sd[i]>=1e-6 else None})
    return output


def balanced_panel(panel):
    common = set.intersection(*(set(f.locality_code) for _,f in panel.groupby("election")))
    return panel.loc[panel.locality_code.isin(common)].copy()


def run(root,download=False):
    root = Path(root)
    if download:
        download_older(root)
    panel,audit = build_panel(root)
    out = root / "data/processed/k26/current/turnout_research/history"
    out.mkdir(parents=True,exist_ok=True)
    panel.to_csv(out / "locality_panel.csv",index=False,encoding="utf-8-sig")
    balanced = balanced_panel(panel)
    common = set(balanced.locality_code)
    rows, fits = [],[]
    for case,subset in [("all_matched",panel),("balanced_common",balanced)]:
        coefficients = {}
        for election,frame in subset.groupby("election",sort=True):
            coefficients[election],fit = fit_turnout(frame)
            fits.append({"sample":case,"election":election,**fit,
                         **{group:float(coefficients[election][i]) for i,group in enumerate(GROUPS)}})
        for row in summarize_windows(coefficients):
            rows.append({"sample":case,**row})
    pd.DataFrame(fits).to_csv(out / "election_coefficients.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out / "history_comparison.csv",index=False,encoding="utf-8-sig")
    input_paths = [root/"data/clean/model_input_v2.csv",root/"data/processed/k26/current/ballots.csv"]
    input_paths += [root/f"data/raw/ballots/K{k} Kalpi.csv" for k in range(21,26)]
    input_paths += [root/f"data/raw/ballots/historical_turnout/{e}_official_datastore.json" for e in RESOURCES
                    if (root/f"data/raw/ballots/historical_turnout/{e}_official_datastore.json").exists()]
    audit.update({"panel_rows":len(panel),"panel_elections":sorted(panel.election.unique()),
        "inputs":[{"path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in input_paths],
        "balanced_common_localities":len(common),"windows":WINDOWS,"historical_sd_ddof":0,
        "turnout_bounds":[.2,.95],"fit_weights":"eligible voters; square-root weights in least squares",
        "same_design":"All K18-K26 fits use full regular-locality counts and the same fixed K25 composition. Recent5 is reestimated on this design, not taken from published coefficients.",
        "reference_excludes_2026":True,"published_defaults_modified":False,
        "limitations":["2022 Census/K25 composition is carried backward to2009 and forward to2026; historical demographic change is not observed.",
          "Locality codes are exact joins; splits, mergers and within-locality boundary/composition changes are not repaired by a historical boundary crosswalk.",
          "Historical elections are not independent years: K21 and K22 both occurred in2019. More ballots do not add temporal replications.",
          "Group coefficients are ecological estimates; shifts in historical SD are descriptive, not confidence levels.",
          "External-envelope votes are excluded because they cannot be assigned to home-locality demographics.",
          "K25 composition uses only mapped K25 boxes and is assumed representative of each entire locality."]})
    (out / "readiness.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),"elections":audit["panel_elections"],"rows":len(panel),"balanced_localities":len(common)}))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    args = parser.parse_args()
    run(args.root,args.download)


if __name__=="__main__":
    main()
