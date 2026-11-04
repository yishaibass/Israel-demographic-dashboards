"""Fit all six elections with one common context-model design.

All elections use locality supervision and the same fixed 2022 Census features,
logistic/softmax formulation, and shared coefficients with election deviations.
The existing published historical default is untouched; a comparable five-election
historical average is exported alongside K26 and each historical election.
Only disclosure-safe demographic aggregates are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import context_features as context
import data_pipeline as base
import run_multi_election_average as elections
import run_party_family_final as model

FAMILIES = list(model.FAMILIES)
PARTY_FAMILIES = {
    "Likud": "likud", "Yashar": "liberal_center", "Yachad (Bennett)": "liberal_center",
    "Democrats": "zionist_left", "Joint List": "arab", "Raam": "arab",
    "Yisrael Beiteinu": "secular_national_right", "Otzma Yehudit": "national_religious_right",
    "Religious Zionism-Zehut": "national_religious_right", "Noam": "national_religious_right",
    "Shas": "shas", "United Torah Judaism": "utj",
}
MASS_COLUMNS = ["abstain_mass", "likud_mass", "national_religious_mass", "shas_mass", "utj_mass",
                "arab_mass", "center_mass", "zionist_left_mass", "secular_national_right_mass", "other_mass"]
DATES = {"K21": "2019-04-09", "K22": "2019-09-17", "K23": "2020-03-02", "K24": "2021-03-23", "K25": "2022-11-01", "K26": "2026-10-27"}


def fit_identity(fitted):
    """Bind every aggregate export to the exact fitted parameters and design."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"features": context.MODEL_COLUMNS, "elections": model.ELECTIONS},
                             sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name, tensor in sorted(fitted.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps({"name": name, "shape": list(array.shape), "dtype": str(array.dtype)},
                                 sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def build_targets(current):
    readiness = json.loads((current / "readiness.json").read_text(encoding="utf-8"))
    ballots = pd.read_csv(current / "ballots.csv")
    dimension = pd.read_csv(current / "party_dimension.csv")
    if set(ballots.snapshot_id) != {readiness["snapshot_id"]} or set(dimension.snapshot_id) != {readiness["snapshot_id"]}:
        raise ValueError("Snapshot mismatch")
    columns = [name for name in ballots if name.startswith("votes_")]
    if dimension.ballot_symbol.duplicated().any() or set(name[6:] for name in columns) != set(dimension.ballot_symbol):
        raise ValueError("Party dimension must cover every ballot symbol exactly once")
    if not ballots[columns].sum(axis=1).eq(ballots.valid_votes).all():
        raise ValueError("Party votes must exhaust valid votes")
    mapping = {row.ballot_symbol: PARTY_FAMILIES.get(row.party_id, "other") for row in dimension.itertuples()}
    regular = ballots.loc[~ballots.external_envelope.astype(str).str.lower().eq("true")].copy()
    for family in FAMILIES:
        selected = ["votes_" + symbol for symbol, assigned in mapping.items() if assigned == family]
        regular[family] = regular[selected].sum(axis=1) if selected else 0
    targets = regular.groupby("locality_code", as_index=False)[["eligible_voters", "total_voted", "valid_votes"] + FAMILIES].sum()
    targets = targets.rename(columns={"locality_code": "locality", "eligible_voters": "eligible", "total_voted": "voted", "valid_votes": "valid"})
    targets = targets.loc[(targets.eligible >= 100) & (targets.valid > 0)].copy()
    if not targets[FAMILIES].sum(axis=1).eq(targets.valid).all():
        raise ValueError("Family votes do not exhaust valid votes")
    targets["geo_key"] = targets.locality.astype(str)
    targets["election"] = "K26"
    targets["date"] = "2026-10-27"
    audit = {"snapshot_id": readiness["snapshot_id"], "count_status": readiness["count_status"],
             "count_provenance": "Public normalized CEC mirror; source status is retained, direct CEC retrieval not independently verified",
             "as_of_utc": readiness.get("as_of_utc"), "family_mapping": mapping,
             "party_id_to_family": {row.party_id: mapping[row.ballot_symbol] for row in dimension.itertuples()},
             "mapping_decisions": "User: Yashar and Beyachad to liberal_center; Democrats to zionist_left. All remaining unassigned lists retained in other.",
             "national_valid_votes": int(ballots.valid_votes.sum()),
             "regular_valid_votes": int(regular.valid_votes.sum()), "envelope_rows_excluded": int(len(ballots)-len(regular)),
             "national_family_votes": {family: int(ballots[["votes_" + symbol for symbol, assigned in mapping.items() if assigned == family]].sum().sum()) for family in FAMILIES}}
    return targets, audit


def build_historical_targets(project):
    """Use the published nine-family crosswalk and full regular-locality counts."""
    mapping = pd.read_excel(project / "data/raw/party_mapping.xlsx", sheet_name="Mapping")
    metadata = {"שם ישוב", "סמל ישוב", "מספר קלפי", "קלפי", "סמל ועדה", "ברזל", "ריכוז", "שופט", "בזב", "מצביעים", "פסולים", "כשרים"}
    outputs, audits = [], {}
    for election, date in list(DATES.items())[:5]:
        path = project / "data/raw/ballots" / (election + "_kalpi_clean.csv")
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        party_columns = [name for name in raw if name not in metadata]
        if not raw[party_columns].sum(axis=1).eq(raw["כשרים"]).all():
            raise ValueError(election + " party votes must exhaust valid votes")
        selected = mapping.loc[mapping.Election.eq(int(election[1:]))]
        by_symbol = {str(row.Letter).strip(): elections.family(row.Name_English) for row in selected.itertuples()}
        families = {symbol: by_symbol.get(symbol, "other") for symbol in party_columns}
        envelope = raw["שם ישוב"].astype(str).str.contains("מעטפות חיצוניות", regex=False)
        regular = raw.loc[~envelope].copy()
        for family in FAMILIES:
            columns = [symbol for symbol, value in families.items() if value == family]
            regular[family] = regular[columns].sum(axis=1) if columns else 0
        target = regular.rename(columns={"סמל ישוב": "locality", "בזב": "eligible", "מצביעים": "voted", "כשרים": "valid"})
        target = target.groupby("locality", as_index=False)[["eligible", "voted", "valid"] + FAMILIES].sum()
        target = target.loc[(target.eligible >= 100) & (target.valid > 0)].copy()
        target["geo_key"] = target.locality.astype(str)
        target["election"], target["date"] = election, date
        if not target[FAMILIES].sum(axis=1).eq(target.valid).all():
            raise ValueError(election + " family votes do not exhaust valid votes")
        outputs.append(target)
        audits[election] = {"source": path.name, "national_valid_votes": int(raw["כשרים"].sum()),
                            "regular_valid_votes": int(regular.valid.sum()) if "valid" in regular else int(regular["כשרים"].sum()),
                            "envelope_rows_excluded": int(envelope.sum()), "family_mapping": families,
                            "national_family_votes": {family: int(raw[[symbol for symbol, value in families.items() if value == family]].sum().sum()) for family in FAMILIES}}
    return pd.concat(outputs, ignore_index=True), audits


def prepare(puf, targets):
    micro = elections.load_all_adults(puf).reset_index(drop=True)
    locality = pd.to_numeric(micro.SmlYishuvPUF, errors="coerce").astype("Int64")
    weights = pd.to_numeric(micro.MishkalPratPUF, errors="coerce").fillna(0)
    logpop = np.log1p(pd.DataFrame({"locality": locality, "weight": weights}).groupby("locality").weight.sum())
    locality_z = (logpop - logpop.mean()) / logpop.std()
    features = context.add_micro_context(micro, locality_z)
    selected = features[context.MODEL_COLUMNS].copy()
    selected["fsu_proxy"] = features.fsu_proxy
    selected["locality"] = locality
    selected["MishkalPratPUF"] = weights
    cells = selected.groupby(["locality"] + context.MODEL_COLUMNS + ["fsu_proxy"], as_index=False, dropna=False).MishkalPratPUF.sum()
    present = set(cells.locality.dropna().astype(int))
    targets = targets.loc[targets.locality.isin(present)].copy().reset_index(drop=True)
    geographies = targets[["geo_key", "locality"]].drop_duplicates().reset_index(drop=True)
    gi = {int(code): i for i, code in enumerate(geographies.locality)}
    cells = cells.loc[cells.locality.isin(gi)].copy()
    cells["g"] = cells.locality.map(gi).astype(int)
    eligible = targets.groupby("locality").eligible.mean()
    population = cells.groupby("locality").MishkalPratPUF.sum()
    cells["weight"] = cells.MishkalPratPUF * cells.locality.map(eligible / population)
    if not np.isfinite(cells[context.MODEL_COLUMNS + ["weight"]].to_numpy(float)).all():
        raise ValueError("Nonfinite features or weights")
    return micro, features, cells, geographies, targets


def cube(micro, joint):
    data = micro.copy()
    income = pd.to_numeric(data.HchnsAvgChdshPratPUF, errors="coerce").fillna(0)
    dimensions = {"pop_group": "KvutzaUchlusiyaPUF", "religiosity": "DatiyutPUF", "age_band": "GilPUF",
                  "academic": "LimudToarAcademiPUF", "district": "MachozPUF"}
    for name, source in dimensions.items():
        data[name] = pd.to_numeric(data[source], errors="coerce").fillna(-1)
    data["income_band"] = np.select([income.eq(0), income.le(7), income.le(14)], [0, 1, 2], default=3)
    data["weight"] = pd.to_numeric(data.MishkalPratPUF, errors="coerce").fillna(0)
    for index, name in enumerate(MASS_COLUMNS):
        data[name] = data.weight * joint[:, index]
    keys = ["pop_group", "religiosity", "income_band", "age_band", "academic", "district"]
    aggregate = data.groupby(keys, as_index=False, dropna=False)[["weight"] + MASS_COLUMNS].sum()
    if np.max(np.abs(aggregate[MASS_COLUMNS].sum(axis=1) - aggregate.weight)) > .1:
        raise ValueError("Cube probability mass does not reconcile")
    return aggregate


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    k26_targets, audit = build_targets(args.current)
    project = args.project or args.current.resolve().parents[3]
    historical_targets, historical_audits = build_historical_targets(project)
    targets = pd.concat([historical_targets, k26_targets], ignore_index=True)
    original_targets = k26_targets.copy()
    micro, features, cells, geos, targets = prepare(args.puf, targets)
    elections.DATES = dict(DATES)
    elections.ELECTIONS = list(DATES)
    model.ELECTIONS = list(DATES)
    localities = np.array(sorted(targets.locality.unique()))
    print(json.dumps({"phase": "prepared", "puf_adults": len(micro), "cells": len(cells), "localities": len(localities)}), flush=True)
    cv_rows = []
    for fold, (train, test) in enumerate(GroupKFold(args.folds).split(localities, groups=localities), 1):
        fitted, info = model.fit(cells, geos, targets, context.MODEL_COLUMNS, False,
                                 set(localities[train]), .05, args.epochs, 100 + fold)
        for row in model.evaluate(fitted, cells, geos, targets, context.MODEL_COLUMNS, set(localities[test])):
            row.update({"fold": fold, **info})
            cv_rows.append(row)
        print(json.dumps({"phase": "fold_complete", "fold": fold, "metrics": cv_rows[-1]}), flush=True)
    fitted, info = model.fit(cells, geos, targets, context.MODEL_COLUMNS, False, set(localities), .05, args.epochs, 42)
    model_fit_id = fit_identity(fitted)
    matrix = torch.tensor(features[context.MODEL_COLUMNS].to_numpy(np.float32))
    fsu = torch.tensor(features.fsu_proxy.to_numpy(np.float32))
    args.output.mkdir(parents=True, exist_ok=True)
    weight = pd.to_numeric(micro.MishkalPratPUF, errors="coerce").fillna(0).to_numpy(float)
    cubes, profile_rows, national_rows, all_coverage = {}, [], [], {}
    historical_mean = None
    maximum_sum_error = 0.
    for election_index, election in enumerate(DATES):
        with torch.no_grad():
            turnout, parties = fitted.probs(matrix, fsu, election_index)
            pt, pc = turnout.numpy(), parties.numpy()
        joint = np.c_[1 - pt, pt[:, None] * pc]
        maximum_sum_error = max(maximum_sum_error, float(np.max(np.abs(joint.sum(axis=1)-1))))
        if not np.isfinite(joint).all() or joint.min() < 0 or maximum_sum_error > 1e-6:
            raise ValueError("Invalid individual probability vectors")
        aggregate = cube(micro, joint)
        payload = {"columns": list(aggregate.columns), "rows": aggregate.round(6).values.tolist(), "snapshot_id": audit["snapshot_id"],
                   "model_fit_id": model_fit_id,
                   "election": election, "source_label": "Common six-election context model; locality supervision for every election"}
        cubes[election] = payload
        if election != "K26":
            if historical_mean is None:
                historical_mean = aggregate.copy()
                historical_mean[MASS_COLUMNS] /= 5
            else:
                if not historical_mean.iloc[:, :7].equals(aggregate.iloc[:, :7]):
                    raise ValueError("Election cubes must have identical population dimensions and weights")
                historical_mean[MASS_COLUMNS] += aggregate[MASS_COLUMNS] / 5
        else:
            aggregate.to_csv(args.output / "k26_probability_cube.csv", index=False)
            (args.output / "k26_probability_cube.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
        profile_rows.append(model.profiles([(election, pt, pc)], micro))
        observed = targets.loc[targets.election.eq(election)]
        source = audit if election == "K26" else historical_audits[election]
        predicted = np.average(joint, axis=0, weights=weight)
        predicted_valid = predicted[1:] / predicted[1:].sum()
        actual = observed[FAMILIES].sum().to_numpy(float) / observed.valid.sum()
        national_actual = np.array([source["national_family_votes"][family] for family in FAMILIES], float) / source["national_valid_votes"]
        national_rows.extend({"election": election, "family": family, "model_share_voters": float(predicted_valid[j]),
                              "matched_regular_actual_share_valid": float(actual[j]), "national_actual_share_valid": float(national_actual[j])}
                             for j, family in enumerate(FAMILIES))
        all_coverage[election] = {"matched_localities": int(observed.locality.nunique()),
                                  "matched_regular_valid_votes": int(observed.valid.sum()),
                                  "national_valid_vote_coverage": float(observed.valid.sum()/source["national_valid_votes"]),
                                  "regular_valid_vote_coverage": float(observed.valid.sum()/source["regular_valid_votes"])}
    historical_payload = {"columns": list(historical_mean.columns), "rows": historical_mean.round(6).values.tolist(),
                          "model_fit_id": model_fit_id,
                          "elections": list(DATES)[:5], "snapshot_id": audit["snapshot_id"],
                          "source_label": "Equal mean of K21-K25 from the common six-election context fit"}
    (args.output / "historical_comparable_cube.json").write_text(json.dumps(historical_payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (args.output / "historical_election_cubes.json").write_text(json.dumps({key: value for key, value in cubes.items() if key != "K26"},
            ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    pd.DataFrame(cv_rows).to_csv(args.output / "cross_validation.csv", index=False)
    pd.concat(profile_rows, ignore_index=True).to_csv(args.output / "demographic_profiles.csv", index=False)
    pd.DataFrame(national_rows).to_csv(args.output / "national_comparison.csv", index=False)
    k26_matched = targets.loc[targets.election.eq("K26")]
    audit.update({"model": "Published regularized logistic turnout and multinomial softmax, jointly fitted across six elections with shared coefficients and election-specific deviations",
                  "model_fit_id": model_fit_id,
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
                  "supervision_geography": "locality", "context_geography": "2022 Census locality x rova",
                  "features": context.MODEL_COLUMNS, "fit": info, "folds": args.folds, "epochs_requested": args.epochs,
                  "historical_average_frozen": True, "historical_comparison": "Separate same-design K21-K25 equal average; published historical default stays frozen",
                  "elections": DATES, "historical_sources": historical_audits,
                  "historical_sd": {"ddof": 0, "definition": "Compute each selected subgroup's ratio separately for each K21-K25 cube, then take arithmetic mean and population SD of those five ratios; standardize K26 against that ratio mean", "mean_cube_definition": "Equal average joint probability masses; its conditional share may differ from the arithmetic mean of election conditional shares"},
                  "all_election_coverage": all_coverage,
                  "puf_rows": len(micro), "weighted_adults": float(weight.sum()),
                  "matched_localities": int(k26_matched.locality.nunique()), "target_localities_before_puf_match": len(original_targets),
                  "unmatched_locality_codes": sorted(int(x) for x in set(original_targets.locality)-set(k26_matched.locality)),
                  "matched_regular_valid_votes": int(k26_matched.valid.sum()),
                  "national_valid_vote_coverage": float(k26_matched.valid.sum()/audit["national_valid_votes"]),
                  "regular_valid_vote_coverage": float(k26_matched.valid.sum()/audit["regular_valid_votes"]),
                  "cube_cells": len(aggregate), "max_probability_sum_error": maximum_sum_error,
                  "cross_validation": cv_rows, "runtime_seconds": time.perf_counter()-started,
                  "limitations": ["Individual propensities are ecological model estimates, not recovered individual votes.",
                                   "All six elections use one locality-supervised design; the frozen published default used finer supervision and is separately labeled.",
                                   "Fixed 2022 Census population and composition do not represent an observed 2026 electoral roll.",
                                   "External-envelope votes are excluded from training and included in separately reported national actuals.",
                                   "Family probabilities condition on valid party choices while turnout includes invalid ballots, as in the original model.",
                                   "Historical SD is descriptive variation across five same-design election estimates, not estimation uncertainty or statistical significance.",
                                   "The comparable historical average is retrospectively fitted jointly with K26; it is not a pre-election forecast.",
                                   "Remaining minor lists use disclosed other family; no coalition assignment is inferred from family mapping."]})
    (args.output / "model_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", "output": str(args.output), "runtime_seconds": audit["runtime_seconds"], "snapshot": audit["snapshot_id"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="Prepared K26 current directory")
    parser.add_argument("--project", type=Path, help="Project root containing historical raw ballots and party mapping; inferred from current directory by default")
    parser.add_argument("--puf", type=Path, default=base.PUF_DEFAULT)
    parser.add_argument("--output", type=Path, required=True, help="Local isolated output directory; never commit microdata")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2:
        parser.error("epochs must be positive and folds at least two")
    run(args)


if __name__ == "__main__":
    main()
