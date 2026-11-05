"""Research audit of historical right support; no published files are changed.

Independent-election fits and genuinely perturbed initializations challenge
cross-election pooling and ecological attribution. Their spread is a sensitivity
diagnostic, not a statistical confidence interval. Binary broad-right counts
must come from explicit ballot identities, never from fitted residual support.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

import fit_foundation_history as history
import fit_population_reconstruction as population
from population_donors import expand_missing_cohorts


ELECTIONS = ("K16", "K25")
RIGHT = population.STABLE_CATEGORIES["right_religious_families"]
DIRECT_FEATURES = {"intercept", "arab", "other_pop", "haredi", "dati", "masorti",
                   "income_log_z", "haredi_x_income", "dati_x_income", "arab_x_income"}


def binary_data(data, counts=None):
    """Two available internal slots represent right/rest, not party identities."""
    targets = data.targets.copy()
    if counts is None:
        pieces = [p["id"] for p in data.partition if set(p["families"]) <= RIGHT]
        if any(set(p["families"]) & RIGHT and not set(p["families"]) <= RIGHT for p in data.partition):
            raise ValueError("Historical observation crosses the narrow right boundary")
        targets["right_votes"] = targets[pieces].sum(axis=1)
    else:
        selected = counts.loc[counts.election.eq(data.election)]
        if selected.duplicated("locality").any():
            raise ValueError("Duplicate broad-right locality")
        joined = targets[["locality", "valid"]].merge(selected[["locality", "valid", "right_votes"]],
                       on="locality", how="left", suffixes=("", "_audit"), validate="one_to_one")
        if joined.right_votes.isna().any() or not np.array_equal(joined.valid, joined.valid_audit):
            raise ValueError("Broad-right counts must match every original valid denominator")
        targets["right_votes"] = joined.right_votes.to_numpy()
    if (targets.right_votes < 0).any() or (targets.right_votes > targets.valid).any():
        raise ValueError("Invalid binary vote counts")
    targets["rest_votes"] = targets.valid - targets.right_votes
    partition = [{"id": "right_votes", "families": ["likud"]},
                 {"id": "rest_votes", "families": ["liberal_center"]}]
    return history.ElectionData(data.election, data.cells.copy(), data.geos.copy(), targets,
                                partition, ["likud", "liberal_center"])


def direct_only(data):
    cells = data.cells.copy()
    cells[[name for name in history.FEATURES if name not in DIRECT_FEATURES]] = 0.
    return history.ElectionData(data.election, cells, data.geos.copy(), data.targets.copy(),
                                data.partition, data.available)


@contextmanager
def perturbed_initialization(sd):
    """The normal constructor is all-zero; seeds alone are not multiple starts."""
    original = history.common.model.FamilyModel
    def create(*args, **kwargs):
        fitted = original(*args, **kwargs)
        if sd:
            with torch.no_grad():
                for parameter in fitted.parameters():
                    parameter.add_(torch.randn_like(parameter) * sd)
        return fitted
    history.common.model.FamilyModel = create
    try:
        yield
    finally:
        history.common.model.FamilyModel = original


def support(fitted, data):
    x = torch.tensor(data.cells[history.FEATURES].to_numpy(np.float32))
    fsu = torch.tensor(data.cells.fsu_proxy.to_numpy(np.float32))
    with torch.no_grad():
        turnout, choice = fitted.probs(x, fsu, 0)
        choice = history.normalized_choice(choice, data.available).numpy()
    votes = data.cells.weight.to_numpy(float) * turnout.numpy()
    right = choice[:, [i for i, name in enumerate(history.FAMILIES) if name in RIGHT]].sum(1)
    result = []
    for group, indices in data.cells.groupby("profile_group").indices.items():
        result.append(dict(election=data.election, group=group,
                           right_support=float(np.average(right[indices], weights=votes[indices])),
                           voter_share=float(votes[indices].sum()/votes.sum())))
    return result


def prepare(args):
    controls = pd.read_csv(args.population_controls, low_memory=False)
    factors = pd.read_csv(args.religiosity_factors)
    allocation = pd.read_csv(args.electorate_controls)
    splits = pd.read_csv(args.splits).sort_values("locality")
    supported = set(splits.locality)
    if splits.locality.duplicated().any() or len(supported) != 195:
        raise ValueError("Audit requires the frozen 195-locality historical sample")
    historical, _ = history.common.build_historical_targets(args.project)
    older, _ = history.read_older(args.historical_targets)
    partition = [{"id": family, "families": [family]} for family in history.FAMILIES]
    target_sets = {"K16": older["K16"], "K25": (historical.loc[historical.election.eq("K25")], partition, history.FAMILIES)}
    micro = history.common.elections.load_all_adults(args.puf).reset_index(drop=True)
    logpop = np.log1p(pd.to_numeric(micro.MishkalPratPUF).groupby(pd.to_numeric(micro.SmlYishuvPUF)).sum())
    scale = (float(logpop.mean()), float(logpop.std()))
    micro = micro.loc[micro.SmlYishuvPUF.astype(int).isin(supported)].copy().reset_index(drop=True)
    # Match the original run's donor pool expansion over all ten election dates.
    periods = ["K16", "K18", "K19", "K20", "K21", "K22", "K23", "K24", "K25", "K26"]
    selected = controls.loc[controls.period_id.isin(periods) & controls.locality_code.isin(supported)]
    source, donors = expand_missing_cohorts(micro, selected)
    result, audits = [], []
    for election in ELECTIONS:
        targets, partition, available = target_sets[election]
        targets = targets.loc[targets.locality.isin(supported)].copy().sort_values("locality").reset_index(drop=True)
        if set(targets.locality) != supported:
            raise ValueError("Audit lost a frozen locality")
        year = int(population.electorate.ELECTION_DATES[election][:4])
        dated, audit = population.reconstructed_micro(source, controls, factors, year, election)
        cells, geos, targets = history.prepare_cells(dated, targets, scale)
        priors = population.micro_keys(dated).assign(adult_population=dated.MishkalPratPUF).groupby(
            ["locality_code", "broad_group"], as_index=False).adult_population.sum().assign(year=year)
        cells = population.apply_eligible_mixture(cells, targets, priors, year, election, allocated_controls=allocation)
        result.append(history.ElectionData(election, cells, geos, targets, partition, available))
        audits.append(audit)
        print(json.dumps(dict(phase="prepared", election=election, localities=len(targets), cells=len(cells))), flush=True)
    return result, splits, dict(population_audits=audits, donor_cohorts_filled=donors["missing_cohorts_filled"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "puf", "population-controls", "religiosity-factors", "electorate-controls", "historical-targets", "splits", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--broad-right-targets", type=Path)
    parser.add_argument("--outcomes", nargs="+", choices=["nine_family", "narrow_binary", "broad_binary"], default=["nine_family"])
    parser.add_argument("--designs", nargs="+", choices=["full_context", "direct_demographics_income"], default=["full_context", "direct_demographics_income"])
    parser.add_argument("--starts", type=int, choices=[1, 3], default=1)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Only fit full-sample sensitivity starts; export no holdout metrics")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Positive epoch limit required")
    if "broad_binary" in args.outcomes and args.broad_right_targets is None:
        parser.error("Broad binary fitting requires explicit --broad-right-targets")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {k: v for k, v in vars(args).items() if isinstance(v, Path) and v.is_file()}
    paths.update(historical_targets=args.historical_targets/"targets.csv",
                 audit_source=Path(__file__), engine_source=Path(history.__file__), population_source=Path(population.__file__))
    identities = {name: history.file_identity(path) for name, path in paths.items()}
    datasets, splits, preparation = prepare(args)
    broad = pd.read_csv(args.broad_right_targets) if args.broad_right_targets else None
    rows, metrics, predictions, fits, records = [], [], [], [], {}
    for original in datasets:
        variants = {"nine_family": original, "narrow_binary": binary_data(original)}
        if broad is not None:
            variants["broad_binary"] = binary_data(original, broad)
        variants = {name: data for name, data in variants.items() if name in args.outcomes}
        for outcome, data in variants.items():
            for design in args.designs:
                current = data if design == "full_context" else direct_only(data)
                # Three genuinely different starts; never interpreted as sampling draws.
                for seed, jitter in [(42, 0.), (43, .05), (44, .05)][:args.starts]:
                    with perturbed_initialization(jitter):
                        fitted, info = history.fit_history([current], set(splits.locality), args.epochs, seed)
                    label = dict(outcome=outcome, design=design, seed=seed, initialization_sd=jitter)
                    rows.extend(dict(row, **label) for row in support(fitted, current))
                    fits.append(dict(election=current.election, fold="all", **label, **info))
                    records[f"{current.election}/{outcome}/{design}/{seed}"] = population.model_record(fitted, [current], info)
                # Geographical evaluation uses one fixed zero start and the original folds.
                for fold in ([] if args.skip_validation else sorted(splits.fold.unique())):
                    train = set(splits.loc[splits.fold.ne(fold), "locality"])
                    test = set(splits.loc[splits.fold.eq(fold), "locality"])
                    fitted, info = history.fit_history([current], train, args.epochs, 100+int(fold))
                    scores, preds = history.evaluate(fitted, [current], test)
                    metrics.extend(dict(row, outcome=outcome, design=design, fold=int(fold), **info) for row in scores)
                    predictions.extend(dict(row, outcome=outcome, design=design, fold=int(fold)) for row in preds)
                print(json.dumps(dict(phase="complete_specification", election=current.election, outcome=outcome, design=design)), flush=True)
                pd.DataFrame(rows).to_csv(args.output/"group_support.csv", index=False)
                if metrics:
                    pd.DataFrame(metrics).to_csv(args.output/"cross_validation.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output/"group_support.csv", index=False)
    if metrics:
        pd.DataFrame(metrics).to_csv(args.output/"cross_validation.csv", index=False)
        pd.DataFrame(predictions).to_csv(args.output/"heldout_predictions.csv", index=False)
    history.write_json(args.output/"fitted_models.json", records)
    for name, path in paths.items():
        if history.file_identity(path) != identities[name]:
            raise ValueError("Audit input/source changed during execution: "+name)
    report = dict(research_only=True, published_files_written=False, python=platform.python_version(),
                  runtime_seconds=time.perf_counter()-started, localities=len(splits), elections=list(ELECTIONS),
                  epochs=args.epochs, starts=args.starts, holdout_validation_performed=not args.skip_validation,
                  fits=fits, input_and_source_hashes=identities, preparation=preparation,
                  initialization="Zero start; optional two Gaussian coefficient perturbations SD0.05; seeds alone would be identical under original zero initialization",
                  binary_internal_slots="likud means right_votes; liberal_center means rest_votes; not literal party predictions",
                  design_limit="Direct design masks geographical-share, locality-size, periphery and interaction columns; same 24-slot penalties retained",
                  interpretation="Independent-election and specification sensitivity; not confidence intervals, individual flows or proof of group identification")
    history.write_json(args.output/"readiness.json", report)
    print(json.dumps(dict(phase="complete", runtime_seconds=report["runtime_seconds"])), flush=True)


if __name__ == "__main__":
    main()
