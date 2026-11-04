"""Independent per-election sensitivity using the common published design.

No fitted coefficient is shared across elections. Census features, localities,
folds, objective and penalties are held fixed. Outputs contain aggregates only.
Fold variation is a sensitivity diagnostic, not a confidence interval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import GroupKFold

import fit_k26 as common


@contextmanager
def election_scope(election):
    previous = common.model.ELECTIONS, common.elections.ELECTIONS, common.elections.DATES
    common.model.ELECTIONS = [election]
    common.elections.ELECTIONS = [election]
    common.elections.DATES = {election: common.DATES[election]}
    try:
        yield
    finally:
        common.model.ELECTIONS, common.elections.ELECTIONS, common.elections.DATES = previous


def dataset_identity(targets, election):
    selected = targets.loc[targets.election.eq(election)].sort_values("locality")
    columns = ["election", "locality", "eligible", "voted", "valid"] + common.FAMILIES
    return "sha256:" + hashlib.sha256(selected[columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def collection_identity(fit_ids, dataset_ids, snapshot):
    content = {"fit_mode": "independent", "election_fit_ids": fit_ids,
               "input_dataset_ids": dataset_ids, "snapshot_id": snapshot}
    return "sha256:" + hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_one(cells, geographies, targets, election, localities, epochs, seed):
    """Pass only this election's outcomes to the unmodified estimator."""
    selected = targets.loc[targets.election.eq(election)].copy()
    if selected.empty:
        raise ValueError("No targets for " + election)
    with election_scope(election):
        fitted, info = common.model.fit(cells, geographies, selected, common.context.MODEL_COLUMNS,
                                         False, set(localities), .05, epochs, seed)
        fit_id = common.fit_identity(fitted)
    return fitted, info, fit_id


def probability_cube(fitted, matrix, fsu, micro):
    with torch.no_grad():
        turnout, party = fitted.probs(matrix, fsu, 0)
        pt, pc = turnout.numpy(), party.numpy()
    joint = np.c_[1 - pt, pt[:, None] * pc]
    if not np.isfinite(joint).all() or joint.min() < 0 or np.max(np.abs(joint.sum(axis=1)-1)) > 1e-6:
        raise ValueError("Invalid fitted probability vectors")
    return common.cube(micro, joint)


def subgroup_profiles(frame):
    """Aggregate joint masses before calculating voter-conditional shares."""
    data = frame.copy()
    data["group"] = np.select([
        data.pop_group.eq(2), data.pop_group.eq(1) & data.religiosity.eq(1),
        data.pop_group.eq(1) & data.religiosity.isin([2, 6]),
        data.pop_group.eq(1) & data.religiosity.eq(3),
        data.pop_group.eq(1) & data.religiosity.eq(5)],
        ["arab", "hiloni", "masorti", "dati", "haredi"], default="other")
    records = []
    for label, keys in [("group", ["group"]), ("income", ["group", "income_band"]),
                        ("education", ["group", "academic"]),
                        ("income_education", ["group", "income_band", "academic"])]:
        aggregate = data.groupby(keys, as_index=False, dropna=False)[["weight"] + common.MASS_COLUMNS].sum()
        for row in aggregate.to_dict("records"):
            identifier = "|".join(str(row[key]) for key in keys)
            voters = sum(row[column] for column in common.MASS_COLUMNS[1:])
            base = {"slice_type": label, "slice": identifier, "weighted_adults": row["weight"], "modeled_voter_weight": voters}
            records.append({**base, "outcome": "turnout", "value": 1-row["abstain_mass"]/row["weight"]})
            for family, column in zip(common.FAMILIES, common.MASS_COLUMNS[1:]):
                records.append({**base, "outcome": family, "value": row[column]/voters if voters > 0 else np.nan})
    return pd.DataFrame(records)


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    k26, source_audit = common.build_targets(args.current)
    historical, historical_audits = common.build_historical_targets(args.project or args.current.resolve().parents[3])
    targets = pd.concat([historical, k26], ignore_index=True)
    micro, features, cells, geographies, targets = common.prepare(args.puf, targets)
    matrix = torch.tensor(features[common.context.MODEL_COLUMNS].to_numpy(np.float32))
    fsu = torch.tensor(features.fsu_proxy.to_numpy(np.float32))
    localities = np.array(sorted(targets.locality.unique()))
    folds = list(GroupKFold(args.folds).split(localities, groups=localities))
    fold_memberships = []
    for fold, (train, test) in enumerate(folds, 1):
        membership = {"fold": fold, "train_locality_codes": [int(x) for x in localities[train]],
                      "test_locality_codes": [int(x) for x in localities[test]]}
        membership["membership_id"] = "sha256:" + hashlib.sha256(
            json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        fold_memberships.append(membership)
    coverage = {}
    for election in common.DATES:
        source = source_audit if election == "K26" else historical_audits[election]
        observed = targets.loc[targets.election.eq(election)]
        coverage[election] = {
            "matched_localities": int(observed.locality.nunique()),
            "matched_regular_valid_votes": int(observed.valid.sum()),
            "national_valid_votes": source["national_valid_votes"],
            "regular_valid_votes": source["regular_valid_votes"],
            "national_valid_vote_coverage": float(observed.valid.sum()/source["national_valid_votes"]),
            "regular_valid_vote_coverage": float(observed.valid.sum()/source["regular_valid_votes"]),
            "envelope_rows_excluded": source["envelope_rows_excluded"],
        }
    cv_rows, profile_frames, cubes, fit_ids, dataset_ids, full_infos = [], [], {}, {}, {}, {}
    print(json.dumps({"phase": "prepared", "cells": len(cells), "localities": len(localities)}), flush=True)
    for election in common.DATES:
        dataset_ids[election] = dataset_identity(targets, election)
        for fold, (train, test) in enumerate(folds, 1):
            fitted, info, _ = fit_one(cells, geographies, targets, election, localities[train], args.epochs, 100+fold)
            with election_scope(election):
                rows = common.model.evaluate(fitted, cells, geographies, targets.loc[targets.election.eq(election)],
                                               common.context.MODEL_COLUMNS, set(localities[test]))
            cv_rows.extend({**row, "fold": fold, **info} for row in rows)
            aggregate = probability_cube(fitted, matrix, fsu, micro)
            profiles = subgroup_profiles(aggregate)
            profiles["election"], profiles["fit"] = election, "fold_" + str(fold)
            profile_frames.append(profiles)
        fitted, info, fit_ids[election] = fit_one(cells, geographies, targets, election, localities, args.epochs, 42)
        full_infos[election] = info
        aggregate = probability_cube(fitted, matrix, fsu, micro)
        profiles = subgroup_profiles(aggregate)
        profiles["election"], profiles["fit"] = election, "full"
        profile_frames.append(profiles)
        cubes[election] = {"columns": list(aggregate.columns), "rows": aggregate.round(6).values.tolist(),
                            "election": election, "snapshot_id": source_audit["snapshot_id"], "model_fit_id": fit_ids[election],
                            "input_dataset_id": dataset_ids[election], "fit_mode": "independent",
                            "source_label": "Independent election fit; common locality-supervised design"}
        print(json.dumps({"phase": "election_complete", "election": election}), flush=True)
    collection_id = collection_identity(fit_ids, dataset_ids, source_audit["snapshot_id"])
    for payload in cubes.values():
        payload["collection_id"] = collection_id
    output = args.output or args.current / "individual_model/independent"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "election_cubes.json", cubes)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(output / "cross_validation.csv", index=False)
    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles.to_csv(output / "subgroup_profiles.csv", index=False)
    keys = ["election", "slice_type", "slice", "outcome"]
    sensitivity = profiles.groupby(keys, as_index=False).agg(minimum=("value", "min"), maximum=("value", "max"),
                                                            weighted_adults=("weighted_adults", "first"))
    sensitivity["range_pp"] = (sensitivity.maximum-sensitivity.minimum)*100
    sensitivity.to_csv(output / "subgroup_sensitivity.csv", index=False)
    summary = sensitivity.groupby(["slice_type", "outcome"], as_index=False).agg(
        median_range_pp=("range_pp", "median"), p95_range_pp=("range_pp", lambda x: float(x.quantile(.95))),
        maximum_range_pp=("range_pp", "max"), slices=("range_pp", "size"))
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    joint_cv_path = args.current / "individual_model/cross_validation.csv"
    comparison = cv.groupby("election", as_index=False)[["family_mae_pp", "turnout_mae_pp"]].mean()
    if joint_cv_path.exists():
        joint_cv = pd.read_csv(joint_cv_path).groupby("election", as_index=False)[["family_mae_pp", "turnout_mae_pp"]].mean()
        comparison = comparison.merge(joint_cv, on="election", suffixes=("_independent", "_joint"), validate="one_to_one")
        for metric in ["family_mae_pp", "turnout_mae_pp"]:
            comparison[metric + "_independent_minus_joint"] = comparison[metric+"_independent"]-comparison[metric+"_joint"]
    comparison.to_csv(output / "joint_cv_comparison.csv", index=False)
    joint_cubes = json.loads((args.current / "individual_model/historical_election_cubes.json").read_text(encoding="utf-8"))
    joint_cubes["K26"] = json.loads((args.current / "individual_model/k26_probability_cube.json").read_text(encoding="utf-8"))
    joint_profiles = []
    for election, payload in joint_cubes.items():
        frame = subgroup_profiles(pd.DataFrame(payload["rows"], columns=payload["columns"]))
        frame["election"] = election
        joint_profiles.append(frame)
    delta = profiles.loc[profiles.fit.eq("full")].merge(pd.concat(joint_profiles), on=keys, suffixes=("_independent", "_joint"), validate="one_to_one")
    delta["independent_minus_joint_pp"] = (delta.value_independent-delta.value_joint)*100
    delta.to_csv(output / "joint_subgroup_comparison.csv", index=False)
    audit = {"fit_mode": "independent", "publication_approved": False,
             "snapshot_id": source_audit["snapshot_id"], "collection_id": collection_id,
             "election_fit_ids": fit_ids, "input_dataset_ids": dataset_ids,
             "elections": common.DATES, "features": common.context.MODEL_COLUMNS,
             "puf_rows": len(micro), "weighted_adults": float(pd.to_numeric(micro.MishkalPratPUF).sum()),
             "supervised_localities": len(localities), "epochs": args.epochs, "folds": args.folds,
             "all_election_coverage": coverage, "fold_memberships": fold_memberships,
             "fold_pairing": "Same sorted locality union and GroupKFold as the common six-election pipeline; each held-out locality is absent from training for that election",
             "regularization": {
                 "shared_slope_penalty": .08, "election_deviation_penalty": .05, "intercepts_penalized": False,
                 "interpretation": "The unchanged published estimator is fitted with one election at a time. Each prediction uses shared plus one election-deviation coefficient vector; no fitted parameter or other election's outcome is borrowed. Identical penalty constants do not imply the same effective prior as the multi-election model: the single-election shared/deviation decomposition distributes shrinkage differently.",
             },
             "fit": full_infos, "full_fit_seed": 42, "cv_seeds": [100+x for x in range(1,args.folds+1)],
             "runtime": {"python": platform.python_version(), "torch": torch.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
             "validation": {"joint_cv_comparison": comparison.to_dict("records"),
                            "subgroup_sensitivity": summary.to_dict("records"),
                            "maximum_fold_sensitivity": sensitivity.loc[sensitivity.range_pp.idxmax()].to_dict(),
                            "maximum_absolute_joint_subgroup_difference_pp": float(delta.independent_minus_joint_pp.abs().max())},
             "input_preparation": "Fixed 2022 Census population/context/features prepared once. The common per-locality eligible scaling cancels from turnout and conditional-family aggregation ratios; other elections' votes never enter fit_one.",
             "limitations": ["Ecological modeled propensities do not identify individual votes or validate subgroup rates.",
                             "Sensitivity ranges compare three two-thirds-locality fits with the full fit; they are not confidence intervals.",
                             "The estimator initializes all parameters to zero; repeated seeds would not test random-start sensitivity and were not run.",
                             "Regularization and the 2022 Census context remain common assumptions; fixed composition is not the actual electorate of each year.",
                             "Envelope votes are excluded from supervision. Locality coverage matches the common-design input audit.",
                             "Nationally weighted aggregate CV errors cannot establish fine subgroup accuracy.",
                             "No bootstrap uncertainty intervals were estimated in this bounded sensitivity run."],
             "runtime_seconds": time.perf_counter()-started}
    write_json(output / "model_audit.json", audit)
    print(json.dumps({"phase": "complete", "output": str(output), "runtime_seconds": audit["runtime_seconds"],
                      "comparison": comparison.to_dict("records"), "maximum_fold_sensitivity": audit["validation"]["maximum_fold_sensitivity"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--puf", type=Path, default=common.base.PUF_DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2:
        parser.error("Positive epochs and at least two folds are required")
    run(args)


if __name__ == "__main__":
    main()
