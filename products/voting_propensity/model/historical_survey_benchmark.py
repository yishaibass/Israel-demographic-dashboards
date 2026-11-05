"""Independent INES2003 vote-intention benchmark; never a model calibration.

Raw public survey files stay in the private research output directory. Report
unweighted respondents who name a party, individual religious self-definition,
and descriptive Wilson intervals. These differ from household-based modeled
actual votes and cannot directly replace the fitted historical estimates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import urllib.request

import pandas as pd
from pypdf import PdfReader


BASE = "https://socsci4.tau.ac.il/mu2/ines/wp-content/uploads/sites/4/2023/06/"
PINS = {
    "2003.dta": "8336d9280d6479f84507b740f7eaaadbce069d9cb06c9ecbcac191db70dfac8c",
    "2003.txt": "abf2299b2014893247d039dc67cd17598a69c81a5fb1b2c5a375130db1e39bfd",
    "2003_Erratum.pdf": "e16f5018a2110fb3d68e025fb769e6d25eac75a2145aa0f247d59cb5ac7e877a",
    "2003_Questionnaire_Eng.pdf": "74975bdb9107d9cffa9d1210f6e534ed45c844c6563de23c718a72fd334ba85d",
}
IDENTITY_LABELS = {1: "secular", 2: "traditional", 3: "religious", 4: "haredi"}
NARROW_RIGHT = {1, 5, 6, 7, 8}
BROAD_ADDITIONS = {10, 13, 14, 21}
PARTY_LABELS = {1: "likud", 5: "national union", 6: "national religious party",
                7: "yahadut hatora", 8: "shas", 10: "israel baliya",
                13: "herut", 14: "tzomet", 21: "ahavat israel"}
GROUPS = {1: "hiloni", 2: "masorti", 3: "dati", 4: "haredi"}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def source_files(folder, download=False):
    contents, manifest = {}, []
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in PINS.items():
        path = folder/name
        if not path.exists():
            if not download:
                raise FileNotFoundError("Missing public benchmark source: " + name)
            with urllib.request.urlopen(BASE+name, timeout=60) as response:
                content = response.read()
            if sha256(content) != expected:
                raise ValueError("Public survey source changed; review before updating pin: " + name)
            path.write_bytes(content)
            metadata = dict(source_url=BASE+name, sha256=expected, bytes=len(content),
                            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                            redistribution="Retain source citation; no raw-data redistribution license asserted")
            path.with_name(name+".metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        content = path.read_bytes()
        if sha256(content) != expected:
            raise ValueError("Public survey source hash mismatch: " + name)
        sidecar = path.with_name(name+".metadata.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata["source_url"] != BASE+name or metadata["sha256"] != expected:
                raise ValueError("Public survey source metadata mismatch: " + name)
        contents[name] = content
        manifest.append(dict(name=name, source_url=BASE+name, sha256=expected, bytes=len(content)))
    return contents, manifest


def wilson(successes, total, z=1.959963984540054):
    """Descriptive binomial interval; excludes survey-design/nonresponse error."""
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Invalid respondent counts")
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]


def verify_labels(labels):
    identity = {int(k): str(v).strip().lower() for k, v in labels["b91"].items()}
    parties = {int(k): str(v).strip().lower() for k, v in labels["b63"].items()}
    if identity != IDENTITY_LABELS:
        raise ValueError("B91 self-definition labels differ from the verified codebook")
    if any(parties.get(code) != label for code, label in PARTY_LABELS.items()):
        raise ValueError("B63 party labels differ from the verified codebook")
    sample = {int(k): str(v).strip().lower() for k, v in labels["jew"].items()}
    if sample != {0: "arab", 1: "jew"}:
        raise ValueError("JEW sample identity codes differ")
    return parties


def documentation_checks(contents):
    codebook = contents["2003.txt"].decode("cp1255")
    block = re.search(r"B91\s+Religious self-definition\s+How would you define yourself\?(.*?)B92", codebook, flags=re.S)
    if block is None or any(label.title() not in block.group(1) for label in IDENTITY_LABELS.values()):
        raise ValueError("Codebook does not confirm explicit B91 religious identity")
    erratum = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(contents["2003_Erratum.pdf"])).pages)
    if "B76" not in erratum or "B91" in erratum:
        raise ValueError("Erratum does not match reviewed B76 correction; recheck B91")
    if not re.search(r"B63\s+Respondent party vote if elections were held today\s+In this election", codebook):
        raise ValueError("Codebook does not confirm current-election intention in B63")
    return dict(identity_variable="B91", identity_definition="Individual religious self-definition, Jewish sample only",
                identity_codes=IDENTITY_LABELS, corrected_observance_variable="B76",
                observance_codes={1: "none", 2: "somewhat", 3: "great extent", 4: "all details"},
                identity_is_not_the_reversed_observance_variable=True,
                vote_variable="B63", vote_timing="Intended current-election party if elections were held today; not B66 hypothetical two-ballot vote or B70 recalled1999 vote")


def benchmark(frame, labels):
    parties = verify_labels(labels)
    if set(["jew", "b91", "b63", "wave"])-set(frame):
        raise ValueError("Survey schema incomplete")
    if not set(frame.jew.dropna()) <= {0, 1}:
        raise ValueError("Unknown survey sample identity")
    if not set(frame.b91.dropna()) <= set(IDENTITY_LABELS):
        raise ValueError("Unknown religious self-definition")
    if not set(frame.b63.dropna()) <= set(range(1, 30)) | {96, 97, 98}:
        raise ValueError("Unknown party intention code")
    selected = frame.loc[frame.jew.eq(1)]
    results = []
    for code, group in GROUPS.items():
        sample = selected.loc[selected.b91.eq(code)]
        voters = sample.loc[sample.b63.between(1, 29)]
        if voters.empty:
            raise ValueError("No named-party respondents in a required religious group")
        counts = sample.b63.value_counts().sort_index()
        record = dict(group=group, jewish_sample_members=len(sample), named_party_respondents=len(voters),
                      undecided=int(sample.b63.eq(97).sum()), will_not_vote=int(sample.b63.eq(98).sum()),
                      blank=int(sample.b63.eq(96).sum()), missing_party_response=int(sample.b63.isna().sum()),
                      party_counts=[dict(code=int(k), label=parties[int(k)], respondents=int(v)) for k, v in counts.items()])
        for name, right in [("model_comparable_narrow_right", NARROW_RIGHT),
                            ("broader_right", NARROW_RIGHT | BROAD_ADDITIONS)]:
            n = int(voters.b63.isin(right).sum())
            waves = []
            for wave, part in voters.groupby("wave"):
                successes = int(part.b63.isin(right).sum())
                waves.append(dict(wave=int(wave), right_respondents=successes, named_party_respondents=len(part), share=successes/len(part)))
            record[name] = dict(right_codes=sorted(right), right_respondents=n, share=n/len(voters),
                                descriptive_wilson95=wilson(n, len(voters)), waves=waves)
        results.append(record)
    return dict(jewish_sample_members=len(selected), arab_sample_members=int(frame.jew.eq(0).sum()),
                arab_records_with_populated_b91=int((frame.jew.eq(0) & frame.b91.notna()).sum()),
                sample_filter="JEW==1; B91=1/2/3/4; named-party denominator B63 in 1..29",
                no_weight_variable_detected=not any(re.search(r"weight|wgt|^wt$", str(col), re.I) for col in frame),
                results=results)


def compare_models(report, model_paths):
    comparisons, inputs = [], []
    survey = {row["group"]: row for row in report["results"]}
    for path in model_paths:
        content = path.read_bytes()
        model = pd.read_csv(io.BytesIO(content))
        selected = model.loc[model.election.eq("K16") & model.category.eq("right_religious_families") & model.group.isin(["dati", "haredi"])]
        if selected.empty or selected.duplicated(["variant", "election", "group"]).any():
            raise ValueError("Model benchmark input has missing or duplicate 2003 group estimates")
        if any(set(part.group) != {"dati", "haredi"} for _, part in selected.groupby("variant")):
            raise ValueError("Model benchmark requires both Dati and Haredi for every variant")
        inputs.append(dict(run=path.parent.name, file=path.name, sha256=sha256(content)))
        for row in selected.itertuples():
            observed = survey[row.group]["model_comparable_narrow_right"]
            comparisons.append(dict(run=path.parent.name, variant=row.variant, group=row.group,
                                    modeled_actual_vote_support=float(row.value), survey_pre_election_intention=observed["share"],
                                    model_minus_benchmark_pp=100*(float(row.value)-observed["share"])))
    return comparisons, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-preferences", type=Path, action="append", default=[])
    args = parser.parse_args()
    contents, sources = source_files(args.output/"sources", args.download)
    reader = pd.io.stata.StataReader(io.BytesIO(contents["2003.dta"]), convert_categoricals=False)
    labels = reader.value_labels()
    frame = reader.read()
    report = benchmark(frame, labels)
    report.update(documentation_checks=documentation_checks(contents), sources=sources,
                  study="Arian and Shamir, Israel National Election Study 2003",
                  study_page="https://socsci4.tau.ac.il/mu2/ines/data/our-data/",
                  citation="Arian, Asher, and Michal Shamir. Israeli Election Study, 2003. doi:10.3886/ICPSR20220.v1",
                  fieldwork="2003-01-12 to 2003-01-24; before 2003-01-28 election",
                  weighting="Unweighted published respondent file; no weighting column detected",
                  uncertainty="Wilson intervals describe binomial sample counts only; not survey-design or national confidence intervals",
                  comparability="Intended party among named-party respondents; individual identity differs from modeled household religiosity and actual votes in 195 localities",
                  matched_model_geography_verified=False,
                  geography_limit="National survey benchmark; no verified residence-code crosswalk to the model's 195-locality support",
                  interpretation="Independent benchmark, not a fitted constraint or replacement for historical model estimates",
                  license_note="Public publisher download retained for research; no raw-data redistribution license asserted; raw files remain outside Git")
    report["model_comparisons"], report["model_input_sha256"] = compare_models(report, args.model_preferences)
    report["analysis_source_sha256"] = sha256(Path(__file__).read_bytes())
    (args.output/"ines2003_support_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for row in report["results"]:
        print(json.dumps(dict(group=row["group"], n=row["named_party_respondents"],
                              narrow_right=row["model_comparable_narrow_right"]["share"], broad_right=row["broader_right"]["share"])))


if __name__ == "__main__":
    main()
