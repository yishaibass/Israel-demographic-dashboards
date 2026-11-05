"""Official historical Knesset counts with explicit observed-family partitions.

Joint lists merge every observation touching their family components. Unreviewed
symbols remain visible and prevent training readiness. Historical locality IDs
are retained as observed; this module never projects 2022 composition backward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

FAMILIES = ["likud", "national_religious_right", "shas", "utj", "arab", "liberal_center", "zionist_left", "secular_national_right", "other"]
DATES = {"K16": "2003-01-28", "K17": "2006-03-28", "K18": "2009-02-10", "K19": "2013-01-22", "K20": "2015-03-17"}
RESOURCES = {"K16": "498b48e9-5af6-474d-b7a4-5ac1e21d3a08", "K17": "70f8bc93-8d98-4c20-ad7c-768af713f1c5",
             "K18": "840edb33-90ac-4176-8ad9-4cdcb8e5caa5", "K19": "c20cdcef-4d42-4241-a41b-6ca7ae51002d",
             "K20": "929b50c6-f455-4be2-b438-ec6af01421f2"}
SOURCE_URLS = {election: "https://data.gov.il/api/3/action/datastore_search?resource_id="+resource+
               "&limit="+("30000" if election in {"K16", "K17"} else "20000") for election, resource in RESOURCES.items()}
CROSSWALK = Path(__file__).with_name("historical_party_crosswalk_v1.csv")
METADATA = {"_id", "סמל ישוב", "שם ישוב", "סמל קלפי", "מספר קלפי", "כתובת", "בוחרים", "בזב", "בז''ב", "מצביעים", "כשרים", "פסולים", "ת. עדכון"}


def validate_archive(content, metadata, election):
    """Check pinned source identity, retained byte hash and complete CKAN result."""
    if election not in RESOURCES:
        raise ValueError("Unsupported official Knesset election")
    if (metadata.get("election") != election or metadata.get("election_date") != DATES[election]
            or metadata.get("resource_id") != RESOURCES[election]
            or metadata.get("source_url") != SOURCE_URLS[election]
            or metadata.get("ballot_type", "knesset") != "knesset"):
        raise ValueError("Official archive metadata source/election identity mismatch")
    if metadata.get("sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("Official archive hash mismatch against retained metadata")
    payload = json.loads(content.decode("utf-8-sig"))
    result = payload.get("result", {})
    records = result.get("records")
    if payload.get("success") is not True or result.get("resource_id") != RESOURCES[election]:
        raise ValueError("Official CKAN response resource identity mismatch")
    if (not isinstance(records, list) or not records or type(result.get("total")) is not int
            or len(records) != result["total"] or metadata.get("records") != len(records)
            or result.get("offset", 0) != 0 or result.get("total_was_estimated", False)):
        raise ValueError("Official archive is paginated, incomplete or has an estimated total")
    identifiers = [record.get("_id") for record in records if isinstance(record, dict)]
    if len(identifiers) != len(records) or any(type(value) is not int for value in identifiers) or len(set(identifiers)) != len(records):
        raise ValueError("Official archive lacks unique datastore row identities")
    return payload


def read_verified_archive(path, election):
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        raise ValueError("Official archive is missing its retained metadata sidecar")
    content = path.read_bytes()
    metadata = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    return validate_archive(content, metadata, election), metadata


def acquire_archive(project, election, download=False):
    """Reuse only verified caches; optionally acquire missing pinned public data."""
    if election not in RESOURCES:
        raise ValueError("Unsupported official Knesset election")
    candidates = [project/"data/raw/ballots"/folder/(election+"_official_datastore.json")
                  for folder in ["historical_model", "historical_turnout"]]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is not None:
        payload, metadata = read_verified_archive(existing, election)
        return existing, payload, metadata
    if not download:
        raise FileNotFoundError("No verified official Knesset archive for "+election+"; use --download")
    with urllib.request.urlopen(SOURCE_URLS[election], timeout=60) as response:
        content = response.read()
    decoded = json.loads(content.decode("utf-8-sig"))
    metadata = {"election": election, "election_date": DATES[election], "ballot_type": "knesset",
                "resource_id": RESOURCES[election], "source_url": SOURCE_URLS[election],
                "publisher": "Central Elections Committee via data.gov.il",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "records": len(decoded.get("result", {}).get("records", [])),
                "raw_format": "Official CKAN JSON response bytes"}
    payload = validate_archive(content, metadata, election)
    destination = candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write only after the complete response passes; invalid cached files are never replaced.
    destination.write_bytes(content)
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return destination, payload, metadata


def observation_partition(crosswalk):
    """Connected family components give disjoint, never double-counted targets."""
    parent = {family: family for family in FAMILIES}
    def find(value):
        while parent[value] != value:
            value = parent[value]
        return value
    for row in crosswalk.itertuples():
        members = str(row.families).split("|") if row.families else []
        if any(family not in FAMILIES for family in members):
            raise ValueError("Crosswalk references unknown family")
        if row.aggregation_rule == "review":
            continue
        if not members or row.aggregation_rule not in {"single", "sum"}:
            raise ValueError("Explicit single/sum/review aggregation rule required")
        for member in members[1:]:
            parent[find(member)] = find(members[0])
    groups = {}
    for family in FAMILIES:
        groups.setdefault(find(family), []).append(family)
    return list(groups.values())


def load_crosswalk(path, election, symbols):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"version", "election", "ballot_symbol", "label", "families", "aggregation_rule", "evidence_url", "confidence", "reason"}
    if required-set(table):
        raise ValueError("Historical crosswalk schema incomplete")
    table = table.loc[table.election.eq(election)].copy()
    if table.ballot_symbol.duplicated().any() or set(table.ballot_symbol) != set(symbols):
        raise ValueError("Crosswalk must cover each observed symbol exactly once")
    return table


def numeric(series):
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Counts must be finite nonnegative integers")
    return values.astype(np.int64)


def build_targets(raw, crosswalk, election, date):
    frame = raw.copy()
    symbols = [column for column in frame if column not in METADATA]
    if set(symbols) != set(crosswalk.ballot_symbol):
        raise ValueError("Every observed party column requires an explicit mapping")
    empty_footer = frame[symbols+["מצביעים", "כשרים"]].apply(pd.to_numeric, errors="coerce").isna().all(axis=1) & frame["שם ישוב"].fillna("").astype(str).str.strip().eq("")
    footer_rows = int(empty_footer.sum())
    frame = frame.loc[~empty_footer].copy()
    for column in symbols+["מצביעים", "כשרים"]:
        frame[column] = numeric(frame[column])
    if "פסולים" not in frame:
        frame["פסולים"] = frame["מצביעים"]-frame["כשרים"]
    else:
        frame["פסולים"] = numeric(frame["פסולים"])
    frame["eligible"] = np.nan
    for column in ["בזב", "בז''ב", "בוחרים"]:
        if column in frame:
            frame["eligible"] = numeric(frame[column])
            break
    frame["locality"] = pd.to_numeric(frame["סמל ישוב"], errors="raise").astype("Int64") if "סמל ישוב" in frame else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["locality_name"] = frame["שם ישוב"].fillna("").astype(str).str.strip()
    envelope = frame.locality_name.str.contains("מעטפות", regex=False) | frame.locality.eq(0).fillna(False)
    inconsistent = ~frame[symbols].sum(axis=1).eq(frame["כשרים"]) | ~frame["כשרים"].add(frame["פסולים"]).eq(frame["מצביעים"])
    inconsistent |= frame.eligible.notna() & frame.eligible.lt(frame["מצביעים"]) & ~envelope
    excluded = frame.loc[inconsistent & ~envelope].copy()
    kept = frame.loc[~envelope & ~inconsistent].copy()
    kept["locality_id"] = kept.locality.astype(str).where(kept.locality.notna(), "name:"+kept.locality_name)
    keys = ["locality_id", "locality", "locality_name"]
    values = ["eligible", "מצביעים", "כשרים", "פסולים"]+symbols
    aggregate = kept.groupby(keys, dropna=False, as_index=False)[values].sum(min_count=1)
    aggregate = aggregate.rename(columns={"מצביעים": "voted", "כשרים": "valid", "פסולים": "invalid"})
    aggregate["election"], aggregate["date"] = election, date
    aggregate["geography_ready"] = aggregate.locality.notna()
    aggregate["turnout_ready"] = aggregate.geography_ready & aggregate.eligible.notna() & aggregate.eligible.gt(0)
    totals = aggregate[["election", "date"]+keys+["eligible", "voted", "valid", "invalid", "geography_ready", "turnout_ready"]].copy()
    partition = observation_partition(crosswalk)
    by_symbol = crosswalk.set_index("ballot_symbol")
    review_symbols = [symbol for symbol in symbols if by_symbol.loc[symbol, "aggregation_rule"] == "review"]
    unresolved = [symbol for symbol in review_symbols if aggregate[symbol].sum() > 0]
    observations = []
    wide = totals.copy()
    definition = []
    for families in partition:
        assigned = [symbol for symbol in symbols if by_symbol.loc[symbol, "aggregation_rule"] != "review" and set(by_symbol.loc[symbol, "families"].split("|")).issubset(families)]
        identifier = "__".join(families)
        definition.append({"id": identifier, "families": families})
        votes = aggregate[assigned].sum(axis=1) if assigned else pd.Series(0, index=aggregate.index)
        wide[identifier] = votes
        part = totals.copy()
        part["observation_id"], part["families"] = identifier, json.dumps(families)
        part["votes"], part["source_symbols"] = votes, json.dumps(assigned, ensure_ascii=False)
        part["available"] = not unresolved
        observations.append(part)
    for symbol in review_symbols:
        part = totals.copy()
        part["observation_id"], part["families"] = "review_"+symbol, "[]"
        part["votes"], part["source_symbols"], part["available"] = aggregate[symbol], json.dumps([symbol], ensure_ascii=False), False
        observations.append(part)
        wide["review_"+symbol] = aggregate[symbol]
    long = pd.concat(observations, ignore_index=True)
    if not long.groupby("locality_id").votes.sum().eq(totals.set_index("locality_id").valid).all():
        raise ValueError("Observed partition does not preserve valid vote mass")
    review = crosswalk.loc[crosswalk.ballot_symbol.isin(review_symbols)].copy()
    review["regular_votes"] = [int(aggregate[symbol].sum()) for symbol in review.ballot_symbol]
    unresolved_mass = int(review.regular_votes.sum())
    audit = {"election": election, "date": date, "ballot_type": "knesset", "observed_partition": definition,
             "available_families": [f for members in partition for f in members] if not unresolved else [],
             "separately_identifiable_families": [f for members in partition if len(members) == 1 for f in members] if not unresolved else [],
             "training_ready": bool(not unresolved and totals.geography_ready.all() and totals.turnout_ready.all()),
             "training_ready_scope": "Count integrity, analytical party crosswalk and observed locality codes/eligible counts only; historical demographic and cross-year geographic support require separate verification",
             "demographic_training_support_verified": False,
             "unresolved_votes": unresolved_mass, "unresolved_share": unresolved_mass/max(1, int(totals.valid.sum())),
             "unreviewed_zero_vote_symbols": [symbol for symbol in review_symbols if symbol not in unresolved],
             "regular_valid_votes": int(totals.valid.sum()), "source_valid_votes": int(frame["כשרים"].sum()),
             "localities": len(totals), "envelope_rows_excluded": int(envelope.sum()),
             "empty_source_footer_rows_excluded": footer_rows,
             "inconsistent_regular_rows_excluded": len(excluded), "inconsistent_regular_valid_votes": int(excluded["כשרים"].sum()),
             "inconsistent_locality_codes": sorted(int(code) for code in excluded.locality.dropna().unique()),
             "locality_codes_available": bool(totals.geography_ready.all()), "eligible_counts_available": bool(totals.turnout_ready.all()),
             "demographic_basis": "No historical demographic composition is supplied or inferred by this counts intake"}
    return totals, long, wide, review, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--elections", nargs="+", default=list(DATES))
    parser.add_argument("--crosswalk", type=Path, default=CROSSWALK)
    parser.add_argument("--download", action="store_true", help="Download missing pinned official archives; verify existing caches without overwriting")
    args = parser.parse_args()
    outputs = {name: [] for name in ["locality_totals", "family_observations", "targets", "mapping_review"]}
    audits = {}
    for election in args.elections:
        source, payload, provenance = acquire_archive(args.project, election, args.download)
        records = payload["result"]["records"]
        raw = pd.DataFrame(records)
        symbols = [column for column in raw if column not in METADATA]
        crosswalk = load_crosswalk(args.crosswalk, election, symbols)
        totals, observations, wide, review, audit = build_targets(raw, crosswalk, election, DATES[election])
        for name, data in zip(outputs, [totals, observations, wide, review]):
            outputs[name].append(data)
        audit.update({"source_path": str(source.relative_to(args.project)), "source_sha256": provenance["sha256"],
                      "source_hash_verified_against_retained_metadata": True,
                      "source_resource_identity_verified": True, "source_complete_result_verified": True,
                      "source_url": provenance["source_url"], "resource_id": provenance["resource_id"]})
        audits[election] = audit
    destination = args.project/"data/processed/historical_model"
    destination.mkdir(parents=True, exist_ok=True)
    for name, parts in outputs.items():
        pd.concat(parts, ignore_index=True).to_csv(destination/(name+".csv"), index=False, encoding="utf-8-sig")
    report = {"schema_version": 1, "crosswalk_sha256": hashlib.sha256(args.crosswalk.read_bytes()).hexdigest(), "elections": audits,
              "limitations": ["Knesset counts only; direct prime-minister votes must never enter these targets.", "Historical codes/names are not asserted equivalent to current Census geography.", "Unknown party-family mappings block training; joint lists use disjoint merged observations, never fabricated splits."]}
    (destination/"readiness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(audits, ensure_ascii=False))


if __name__ == "__main__":
    main()
