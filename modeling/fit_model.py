"""Fit, validate and export the published five-election voting model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import context_features as context
import run_multi_election_average as elections
import run_party_family_final as model


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=elections.DATA_DIR,
                        help="Directory containing the untracked raw inputs described in data/raw/README.md")
    parser.add_argument("--puf", type=Path, default=None,
                        help="2022 Census PUF CSV; defaults to DATA_DIR/census/census_2022_puf.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    args.puf = args.puf or args.data_dir / "census" / "census_2022_puf.csv"
    args.output.mkdir(parents=True, exist_ok=True)

    targets, source_audit = elections.build_targets(args.data_dir)
    cells, geographies, targets, _, matched_micro, _ = elections.prepare(targets, args)
    cells, locality_z = context.add_context(cells, matched_micro)
    localities = np.array(sorted(targets.locality.unique()))

    cv_rows = []
    splitter = GroupKFold(args.folds)
    for fold, (train, test) in enumerate(splitter.split(localities, groups=localities), 1):
        fitted, info = model.fit(cells, geographies, targets, context.MODEL_COLUMNS, False,
                                 set(localities[train]), .05, args.epochs, 100 + fold)
        for row in model.evaluate(fitted, cells, geographies, targets, context.MODEL_COLUMNS,
                                  set(localities[test])):
            row.update({"fold": fold, **info})
            cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(args.output / "cross_validation.csv", index=False)

    fitted, fit_info = model.fit(cells, geographies, targets, context.MODEL_COLUMNS, False,
                                 set(localities), .05, args.epochs, 42)
    full = elections.load_all_adults(args.puf)
    features = context.add_micro_context(full, locality_z)
    x = torch.tensor(features[context.MODEL_COLUMNS].to_numpy(np.float32))
    fsu = torch.tensor(features.fsu_proxy.to_numpy(np.float32))
    predictions = []
    with torch.no_grad():
        for election_index, election in enumerate(model.ELECTIONS):
            turnout, party = fitted.probs(x, fsu, election_index)
            predictions.append((election, turnout.numpy(), party.numpy()))
    model.export(fitted, predictions, full, args.output / "individual_probabilities.csv.gz")
    model.profiles(predictions, full).to_csv(args.output / "demographic_profiles.csv", index=False)

    audit = {
        **source_audit,
        "model": "regularized logistic turnout and multinomial party-family model",
        "features": context.MODEL_COLUMNS,
        "elections": elections.DATES,
        "party_families": model.FAMILIES,
        "puf_rows": int(len(full)),
        "weighted_adult_population": float(pd.to_numeric(full.MishkalPratPUF, errors="coerce").fillna(0).sum()),
        "training_geographies": int(len(geographies)),
        "election_geography_targets": int(len(targets)),
        "localities": int(targets.locality.nunique()),
        "cross_validation_folds": args.folds,
        "pooling_penalty": .05,
        "fit": fit_info,
    }
    (args.output / "model_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
