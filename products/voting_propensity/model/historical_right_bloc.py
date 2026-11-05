"""Reproducible party-level right/rest targets for a private sensitivity audit.

The versioned crosswalk defines an analytical bloc, independently of government
membership. Ambiguous lists stay outside the primary bloc and enter an explicit
upper classification sensitivity. No existing party-family mapping is changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


METADATA = {"_id", "סמל ישוב", "סמל קלפי", "מספר קלפי", "שם ישוב", "כתובת", "בוחרים", "מצביעים", "פסולים", "כשרים", "סמל ועדה", "ברזל", "קלפי", "ריכוז", "שופט", "בזב"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(value):
    value = float(value or 0)
    if value < 0 or not value.is_integer():
        raise ValueError("Counts and locality codes must be nonnegative integers")
    return int(value)


def aggregate(rows, mapping, election, include_uncertain=False):
    """Reject unmapped positive votes and retain every valid vote in right/rest."""
    grouped = defaultdict(lambda: [0, 0, ""])
    symbols = {k for row in rows for k in row if k not in METADATA}
    for row in rows:
        if "מעטפות" in str(row.get("שם ישוב", "")):
            continue
        code = count(row["סמל ישוב"])
        if code <= 0:
            continue
        counts = {s: count(row.get(s, 0)) for s in symbols}
        if sum(counts.values()) != count(row["כשרים"]):
            raise ValueError("Party counts do not exhaust valid votes")
        missing = [s for s, value in counts.items() if value and s not in mapping]
        if missing:
            raise ValueError("Unclassified positive ballot symbols: " + repr(missing))
        right = sum(value for symbol, value in counts.items() if mapping.get(symbol) == "right" or (include_uncertain and mapping.get(symbol) == "uncertain"))
        grouped[code][0] += right
        grouped[code][1] += count(row["כשרים"])
        grouped[code][2] = str(row.get("שם ישוב", ""))
    return [dict(election=election, locality=code, right_votes=right, rest_votes=valid-right, valid=valid)
            for code, (right, valid, _) in sorted(grouped.items())], grouped


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def run(project, output, crosswalk):
    output.mkdir(parents=True, exist_ok=True)
    mapping_rows = list(csv.DictReader(crosswalk.open(encoding="utf-8-sig")))
    keys = [(x["election"], x["ballot_symbol"]) for x in mapping_rows]
    if len(keys) != len(set(keys)) or any(x["assignment"] not in {"right", "rest", "uncertain"} for x in mapping_rows):
        raise ValueError("Invalid or duplicated classification")
    all_targets, upper_targets, audits = [], [], {}
    for election in ["K16", "K25"]:
        chosen = [x for x in mapping_rows if x["election"] == election]
        mapping = {x["ballot_symbol"]: x["assignment"] for x in chosen}
        if election == "K16":
            path = project / "data/raw/ballots/historical_model/K16_official_datastore.json"
            meta_path = path.with_suffix(".metadata.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if sha256(path) != meta["sha256"]:
                raise ValueError("K16 retained official source hash mismatch")
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["result"]["records"]
            if len(rows) != int(payload["result"]["total"]):
                raise ValueError("Incomplete official K16 archive")
            source = meta["source_url"]
        else:
            path = project / "data/raw/ballots/K25_kalpi_clean.csv"
            rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
            source = "https://data.gov.il/dataset/26f9fa06-fcd7-4173-8df5-65797b63e857/resource/cc223336-07bc-485d-b160-62df92967c0a/download/-25-.csv"
        footers = [row for row in rows if str(row.get("סמל ישוב", "")).strip() == "\x1a"]
        if any(any(value not in {None, "", "\x1a"} for key, value in row.items() if key != "_id") for row in footers):
            raise ValueError("Nonempty source footer cannot be excluded")
        rows = [row for row in rows if row not in footers]
        primary, grouped = aggregate(rows, mapping, election)
        upper, _ = aggregate(rows, mapping, election, True)
        all_targets.extend(primary)
        upper_targets.extend(upper)
        symbol_totals = {s: sum(count(row.get(s, 0)) for row in rows if "מעטפות" not in str(row.get("שם ישוב", "")) and count(row["סמל ישוב"]) > 0) for s in mapping}
        narrow_symbols = {"מחל", "ב", "ל", "שס", "ג"} if election == "K16" else {"מחל", "ט", "ל", "שס", "ג", "ב"}
        examples = []
        for target in primary:
            code = target["locality"]
            if code not in {3797, 1309, 3780, 6100, 3557, 3574, 3640, 3650}:
                continue
            narrow = sum(sum(count(row.get(s, 0)) for s in narrow_symbols) for row in rows if count(row["סמל ישוב"]) == code and "מעטפות" not in str(row.get("שם ישוב", "")))
            examples.append(dict(locality=code, locality_name=grouped[code][2], valid=target["valid"], narrow_right_votes=narrow, broad_right_votes=target["right_votes"], narrow_right_share=narrow/target["valid"], broad_right_share=target["right_votes"]/target["valid"]))
        audits[election] = dict(source_path=str(path.relative_to(project)), source_url=source, source_sha256=sha256(path),
            empty_footer_rows_excluded=len(footers),
            retained_localities=len(primary), retained_valid=sum(x["valid"] for x in primary), right_votes=sum(x["right_votes"] for x in primary),
            upper_classification_right_votes=sum(x["right_votes"] for x in upper),
            all_source_valid=sum(count(row["כשרים"]) for row in rows),
            party_assignments=[dict(x, retained_regular_votes=symbol_totals[x["ballot_symbol"]]) for x in chosen], locality_examples=examples)
    fields = ["election", "locality", "right_votes", "rest_votes", "valid"]
    write_csv(output / "broad_right_targets.csv", all_targets, fields)
    write_csv(output / "broad_right_uncertain_included_targets.csv", upper_targets, fields)
    audit = dict(schema_version="historical-right-v1", research_only=True, crosswalk_sha256=sha256(crosswalk),
                 interpretation="Analytical right/religious bloc including Liberman; not government coalition, observed individual behavior, or pure religiosity.",
                 uncertainty="Primary excludes marked uncertain lists; separate upper-classification target includes all of them. Rest means all votes outside the selected bloc, not ideological left.",
                 geography="All observed regular localities; fitting must intersect the same reviewed population/sample controls. External envelopes excluded.", elections=audits)
    (output / "broad_right_target_provenance.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({e: {k: v for k, v in a.items() if k in {"retained_localities", "retained_valid", "right_votes", "upper_classification_right_votes", "all_source_valid"}} for e, a in audits.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=Path(__file__).with_name("historical_right_bloc_v1.csv"))
    args = parser.parse_args()
    run(args.project, args.output, args.crosswalk)


if __name__ == "__main__":
    main()
