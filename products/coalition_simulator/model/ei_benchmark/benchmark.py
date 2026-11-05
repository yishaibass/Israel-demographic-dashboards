"""Compare ecological turnout models on unseen geographic blocks.

Fit adapters receive training counts only. Their ``predict`` method receives
held-out demographic shares and eligible counts, never held-out turnout. All
elections share one locality/subdistrict split. Separate adapters fit each
election; joint adapters receive only the training localities' histories.
This evaluates within-election aggregate prediction, not future-election tails
or the accuracy of unobserved individual demographic turnout.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time
import types

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

GROUPS = ["haredi", "dati", "masorti", "hiloni", "arab", "druze"]
SHARES = ["share_" + group for group in GROUPS]
LEVELS = (.50, .80, .95)
PACKAGE = "products.coalition_simulator.model.ei_benchmark"


def json_safe(value):
    """Unidentified chain diagnostics remain explicit JSON null, never NaN."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def balanced_panel(frame):
    required = ["election", "locality_code", "eligible_voters", "total_voted"] + SHARES
    if set(required) - set(frame):
        raise ValueError("Missing benchmark input columns")
    if frame[required].isna().any().any() or frame.duplicated(["election", "locality_code"]).any():
        raise ValueError("Missing values or duplicate election/locality identities")
    for column in ["eligible_voters", "total_voted"]:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Eligible and voted counts must be nonnegative integers")
    if not frame.eligible_voters.gt(0).all() or frame.total_voted.gt(frame.eligible_voters).any():
        raise ValueError("Invalid eligible-voter turnout counts")
    x = frame[SHARES].to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any() or not np.allclose(x.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Six demographic shares must sum to one")
    if frame.groupby("locality_code")[SHARES].agg(lambda s: s.max()-s.min()).to_numpy().max() > 1e-10:
        raise ValueError("Fixed demographic shares must agree across locality histories")
    elections = sorted(frame.election.unique(), key=lambda e: int(e[1:]))
    common = set.intersection(*(set(frame.loc[frame.election.eq(e), "locality_code"]) for e in elections))
    if not common:
        raise ValueError("No common locality histories")
    selected = frame.loc[frame.locality_code.isin(common)].copy()
    return selected.sort_values(["election", "locality_code"]).reset_index(drop=True), {
        "source_rows": len(frame), "balanced_rows": len(selected), "balanced_localities": len(common),
        "elections": elections, "excluded_rows": len(frame)-len(selected),
    }


def geographic_folds(frame, geography, folds=3, seed=20261104):
    """Use only identities and eligibility to balance whole subdistrict blocks."""
    if folds < 2 or geography.locality_code.duplicated().any():
        raise ValueError("Require unique geography identities and at least two folds")
    mass = frame.groupby("locality_code", as_index=False).eligible_voters.sum()
    lookup = geography[["locality_code", "subdistrict_code"]]
    split = mass.merge(lookup, on="locality_code", how="left", validate="one_to_one")
    if split.subdistrict_code.isna().any():
        raise ValueError("Every benchmark locality requires a source subdistrict")
    blocks = split.groupby("subdistrict_code").eligible_voters.sum().sort_index()
    if len(blocks) < folds:
        raise ValueError("Too few geographic blocks")
    rng = np.random.default_rng(seed)
    order = pd.DataFrame({"block": blocks.index, "mass": blocks.values, "tie": rng.random(len(blocks))})
    order = order.sort_values(["mass", "tie"], ascending=[False, True])
    totals = np.zeros(folds)
    assigned = {}
    for row in order.itertuples():
        lightest = np.flatnonzero(totals == totals.min())
        fold = int(rng.choice(lightest))
        assigned[row.block] = fold
        totals[fold] += row.mass
    split["fold"] = split.subdistrict_code.map(assigned).astype(int)
    return split.sort_values("locality_code").reset_index(drop=True)


@dataclass
class WLSFit:
    group_mean_draws: np.ndarray
    predictive_kind: str = "expected_counts"

    def predict(self, x, eligible, *, seed):
        return (np.asarray(x) @ self.group_mean_draws.mean(axis=0) * eligible)[None, :]

    def expected_rates(self, x):
        return np.asarray(x) @ self.group_mean_draws.mean(axis=0)


@dataclass
class WLSResidualFit:
    """Training-only empirical rate errors, rather than coefficient intervals."""
    group_mean_draws: np.ndarray
    residuals: np.ndarray
    residual_weights: np.ndarray
    samples: int
    predictive_kind: str = "new_locality_counts"

    @property
    def diagnostics(self):
        return {"method": "Eligible-weighted empirical training residuals added to new-locality WLS rates, clipped to [0,1], then rounded to voted counts",
                "posterior": False, "extra_bernoulli_noise_added": False,
                "limitation": "Residuals already include observed count variation; this benchmark does not separately model denominator-dependent residual variance or coefficient uncertainty"}

    def predict(self, x, eligible, *, seed):
        rng = np.random.default_rng(seed)
        residual = rng.choice(self.residuals, size=(self.samples, len(x)), p=self.residual_weights)
        expected = np.asarray(x) @ self.group_mean_draws.mean(axis=0)
        return np.rint(np.clip(expected[None, :]+residual, 0, 1)*eligible[None, :]).astype(np.int64)

    def expected_rates(self, x):
        expected = np.asarray(x) @ self.group_mean_draws.mean(axis=0)
        return np.clip(expected[:, None]+self.residuals[None, :], 0, 1) @ self.residual_weights


def fit_bounded_wls(x, eligible, voted, *, seed, budget, group_names, locality_ids):
    if len(x) < len(GROUPS) or np.linalg.matrix_rank(x) != len(GROUPS):
        raise ValueError("Training demographic design lacks six-group rank")
    weights = np.asarray(eligible, float)
    sw = np.sqrt(weights / weights.sum())
    result = lsq_linear(x*sw[:, None], voted/weights*sw, bounds=(.2, .95))
    if not result.success:
        raise ValueError("Bounded WLS failed")
    return WLSFit(result.x[None, :])


def fit_wls_residual(x, eligible, voted, *, seed, budget, group_names, locality_ids):
    fitted = fit_bounded_wls(x, eligible, voted, seed=seed, budget=budget,
                             group_names=group_names, locality_ids=locality_ids)
    weights = np.asarray(eligible, float)
    errors = voted/weights-np.asarray(x)@fitted.group_mean_draws[0]
    return WLSResidualFit(fitted.group_mean_draws, errors, weights/weights.sum(),
                          max(1000, int(budget.get("draws", 1000))*int(budget.get("chains", 1))))


def weighted_interval_score(observed, median, intervals):
    """Bracher et al. (2021), Eq. 2: median + central 50/80/95 intervals.

    https://doi.org/10.1371/journal.pcbi.1008618
    Returns count units; the scorer converts each locality to percentage points
    before averaging with its eligible population weight.
    """
    y = np.asarray(observed, float)
    score = .5*np.abs(y-np.asarray(median, float))
    for level in LEVELS:
        lower, upper = (np.asarray(v, float) for v in intervals[level])
        alpha = 1-level
        interval = upper-lower+(2/alpha)*np.maximum(lower-y, 0)+(2/alpha)*np.maximum(y-upper, 0)
        score += (alpha/2)*interval
    return score/(len(LEVELS)+.5)


def prediction_rows(name, election, fold, identifiers, eligible, observed, draws, kind, *, identifier="locality_code", expected_counts=None):
    genuine = kind == "new_locality_counts"
    means = draws.mean(axis=0) if expected_counts is None else expected_counts
    medians = np.median(draws, axis=0) if genuine else np.full(len(eligible), np.nan)
    intervals = {level: np.quantile(draws, [(1-level)/2, (1+level)/2], axis=0) for level in LEVELS} if genuine else {}
    rows = []
    for j, code in enumerate(identifiers):
        row = {"model": name, "election": election, "fold": fold, identifier: code,
               "eligible_voters": int(eligible[j]), "observed_count": int(observed[j]),
               "observed_rate": observed[j]/eligible[j], "predicted_count": float(means[j]),
               "predicted_rate": float(means[j]/eligible[j]), "median_count": float(medians[j]) if genuine else None,
               "predictive_kind": kind}
        for level in LEVELS:
            tag = str(int(level*100))
            row["lower_"+tag] = float(intervals[level][0, j]) if genuine else None
            row["upper_"+tag] = float(intervals[level][1, j]) if genuine else None
        rows.append(row)
    return rows


def score_predictions(predictions, *, by_fold=False, unit="localities"):
    rows = []
    keys = ["model", "election"]+(["fold"] if by_fold else [])
    for identity, group in predictions.groupby(keys, sort=True):
        model, election = identity[:2]
        weights = group.eligible_voters.to_numpy(float)
        errors = group.predicted_rate.to_numpy()-group.observed_rate.to_numpy()
        row = {"model": model, "election": election, unit: len(group),
               "folds": int(group.fold.nunique()), "eligible_voters": int(weights.sum()),
               "weighted_mae_pp": float(np.average(np.abs(errors), weights=weights)*100),
               "weighted_rmse_pp": float(np.sqrt(np.average(errors**2, weights=weights))*100)}
        if by_fold:
            row["fold"] = int(identity[2])
        for level in LEVELS:
            tag = str(int(level*100))
            lower, upper = group["lower_"+tag], group["upper_"+tag]
            if lower.notna().all() and upper.notna().all():
                covered = group.observed_count.between(lower, upper)
                row["eligible_weighted_coverage_"+tag] = float(np.average(covered, weights=weights))
                row[("region_coverage_" if unit == "regions" else "locality_coverage_")+tag] = float(covered.mean())
                row["eligible_weighted_width_"+tag+"_pp"] = float(np.average((upper-lower)/weights, weights=weights)*100)
            else:
                row["eligible_weighted_coverage_"+tag] = None
                row[("region_coverage_" if unit == "regions" else "locality_coverage_")+tag] = None
                row["eligible_weighted_width_"+tag+"_pp"] = None
        if group.median_count.notna().all() and all(group["lower_"+str(int(level*100))].notna().all() for level in LEVELS):
            intervals = {level: (group["lower_"+str(int(level*100))], group["upper_"+str(int(level*100))]) for level in LEVELS}
            wis = weighted_interval_score(group.observed_count, group.median_count, intervals)
            row["weighted_wis_pp"] = float(np.average(wis/weights, weights=weights)*100)
        else:
            row["weighted_wis_pp"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_elections(scores):
    """Equal election weights; each election first uses eligible-locality weights."""
    metrics = [c for c in scores if c.startswith(("weighted_", "eligible_weighted_", "locality_coverage_", "region_coverage_"))]
    rows = []
    for model, frame in scores.groupby("model", sort=True):
        row = {"model": model, "elections": len(frame), "scored_elections": ",".join(frame.election),
               "election_weighting": "equal", "minimum_folds_per_election": int(frame.folds.min())}
        for metric in metrics:
            row["mean_"+metric] = float(frame[metric].mean()) if frame[metric].notna().all() else None
        checks = frame.mcmc_diagnostics_passed
        row["mcmc_diagnostics_passed"] = None if checks.isna().all() else bool(checks.fillna(False).all())
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_gate(diagnostics):
    """One predeclared benchmark screen, preserving adapter-native diagnostics."""
    if "rhat_max" not in diagnostics:
        return None
    names = ["rhat", "ess_bulk", "ess_tail"]
    counts = diagnostics.get("diagnostic_value_counts", {})
    nonfinite = diagnostics.get("nonfinite_counts", {})
    if diagnostics.get("all_selected_finite") is not True or any(counts.get(name, 0) <= 0 or nonfinite.get(name, -1) != 0 for name in names):
        return False
    values = [diagnostics.get(key, np.nan) for key in ["rhat_max", "ess_bulk_min", "ess_tail_min"]]
    chains = diagnostics.get("chains", diagnostics.get("budget", {}).get("chains", 0))
    return bool(np.isfinite(values).all() and values[0] <= 1.01 and values[1] >= 400 and values[2] >= 200
                and chains >= 2 and diagnostics.get("divergences", -1) == 0)


def validate_sampler_request(models, nuts_sampler, dense_mass):
    if dense_mass and (nuts_sampler != "numpyro" or "marginal_joint" not in models or set(models)-{"wls", "marginal_joint"}):
        raise ValueError("Dense mass is supported only for marginal_joint with NumPyro, optionally with the wls baseline")


@dataclass
class JointElectionView:
    fitted: object
    election: str
    regional_ids: np.ndarray
    locality_ids: np.ndarray

    @property
    def group_mean_draws(self):
        return self.fitted.group_mean_draws_for_election(self.election)

    @property
    def predictive_kind(self):
        return self.fitted.predictive_kind

    @property
    def diagnostics(self):
        return self.fitted.diagnostics

    def predict(self, x, eligible, *, seed):
        return self.fitted.predict_election(self.election, x, eligible, seed=seed,
                                            regional_ids=self.regional_ids.copy(), locality_ids=self.locality_ids.copy())

    def expected_rates(self, x):
        if hasattr(self.fitted, "expected_rates_for_election"):
            return self.fitted.expected_rates_for_election(self.election, x)
        return np.asarray(x) @ self.group_mean_draws.mean(axis=0)


def cache_context(training, fitter, seed, budget):
    columns = ["election", "locality_code", "subdistrict_code", "eligible_voters", "total_voted"]+SHARES
    payload = training[columns].sort_values(["election", "locality_code"]).to_csv(index=False).encode("utf-8")
    upstream = getattr(sys.modules.get(fitter.__module__), "ADAPTER_METADATA", {})
    module = sys.modules.get(fitter.__module__)
    modules = [module]
    if fitter.__module__.startswith(PACKAGE):
        modules.append(sys.modules.get(PACKAGE+".rebane_adapter"))
    sources, loaded_code = {}, {}
    def constant(value):
        if isinstance(value, types.CodeType):
            return {"bytecode": value.co_code.hex(), "constants": [constant(v) for v in value.co_consts],
                    "names": value.co_names, "variables": value.co_varnames, "freevars": value.co_freevars,
                    "argcount": value.co_argcount, "kwonly": value.co_kwonlyargcount, "flags": value.co_flags}
        if isinstance(value, (tuple, list)):
            return [constant(v) for v in value]
        if isinstance(value, frozenset):
            return sorted([constant(v) for v in value], key=lambda v: json.dumps(v, sort_keys=True))
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)
    def function_bytes(function):
        return json.dumps({"code": constant(function.__code__), "defaults": constant(function.__defaults__),
                           "kwdefaults": {k: constant(v) for k, v in (function.__kwdefaults__ or {}).items()}}, sort_keys=True).encode()
    for item in modules:
        if item is None or not getattr(item, "__file__", None):
            continue
        path = Path(item.__file__)
        sources[item.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()
        digest = hashlib.sha256()
        for name, value in sorted(vars(item).items()):
            if inspect.isfunction(value) and value.__module__ == item.__name__:
                digest.update(name.encode()+function_bytes(value))
            elif inspect.isclass(value) and value.__module__ == item.__name__:
                for method, function in sorted(vars(value).items()):
                    function = function.fget if isinstance(function, property) else function
                    if inspect.isfunction(function):
                        digest.update((name+"."+method).encode()+function_bytes(function))
        loaded_code[item.__name__] = digest.hexdigest()
    return {"training_sha256": hashlib.sha256(payload).hexdigest(), "seed": seed, "budget": dict(budget or {}),
            "adapter_module_source_sha256": sources, "loaded_python_code_sha256": loaded_code,
            "upstream": upstream}


def evaluate(frame, geography, adapters=None, *, folds=3, seed=20261104, budget=None,
             elections=None, selected_folds=None, joint_adapters=None, cache_directory=None,
             reuse_cache=False, cache_loaders=None, prediction_seed_offset=0):
    panel, audit = balanced_panel(frame)
    splits = geographic_folds(panel, geography, folds, seed)
    panel = panel.merge(splits[["locality_code", "subdistrict_code", "fold"]], on="locality_code", validate="many_to_one")
    selected = audit["elections"] if elections is None else list(elections)
    if set(selected)-set(audit["elections"]):
        raise ValueError("Requested election is absent from the balanced panel")
    fold_ids = list(range(folds)) if selected_folds is None else list(selected_folds)
    if not fold_ids or not set(fold_ids) <= set(range(folds)):
        raise ValueError("Requested folds are invalid")
    fitters = {"bounded_wls": fit_bounded_wls, "wls_empirical_residual": fit_wls_residual}
    fitters.update(adapters or {})
    if set(fitters) & set(joint_adapters or {}):
        raise ValueError("Joint and separate adapters require distinct model names")
    fitters.update(joint_adapters or {})
    joint_cache = {}
    predictions, regional_predictions, groups, fits = [], [], [], []
    for election in selected:
        election_index = audit["elections"].index(election)
        current = panel.loc[panel.election.eq(election)]
        for fold in fold_ids:
            training = current.loc[current.fold.ne(fold)]
            heldout = current.loc[current.fold.eq(fold)]
            if not len(training) or not len(heldout):
                raise ValueError("Empty training or held-out fold")
            train_x = training[SHARES].to_numpy(float)
            train_n = training.eligible_voters.to_numpy(np.int64)
            train_y = training.total_voted.to_numpy(np.int64)
            test_x = heldout[SHARES].to_numpy(float)
            test_n = heldout.eligible_voters.to_numpy(np.int64)
            for name, fitter in fitters.items():
                model_index = {"bounded_wls": 0, "wls_empirical_residual": 1, "pyei": 2, "rebane": 3, "rebane_joint": 4, "marginal_joint": 5}.get(
                    name, 6+int(hashlib.sha256(name.encode()).hexdigest()[:4], 16) % 88)
                is_joint = name in (joint_adapters or {})
                fit_seed = seed+(0 if is_joint else election_index*10000)+fold*100+model_index
                started = time.perf_counter()
                reused = False
                cached = False
                cache_file = None
                fit_training = panel.loc[panel.fold.ne(fold)].drop(columns="fold").copy() if is_joint else training
                if cache_directory is not None and name not in ["bounded_wls", "wls_empirical_residual"]:
                    folder = Path(cache_directory)/name
                    folder.mkdir(parents=True, exist_ok=True)
                    cache_file = folder/("fold_"+str(fold)+("_joint" if is_joint else "_"+election)+".npz")
                context = cache_context(fit_training, fitter, fit_seed, budget) if cache_file is not None else None
                if cache_file is not None and reuse_cache and cache_file.exists() and (not is_joint or (name, fold) not in joint_cache):
                    if not cache_file.with_suffix(".json").exists() or json.loads(cache_file.with_suffix(".json").read_text(encoding="utf-8")) != context:
                        raise ValueError("Posterior cache context differs from training data, adapter, seed, or budget: "+cache_file.name)
                    loader = (cache_loaders or {}).get(name)
                    if loader is None:
                        raise ValueError("Adapter cannot reload its saved posterior parameters: "+name)
                    loaded = loader(cache_file)
                    cached = True
                    if is_joint:
                        joint_cache[(name, fold)] = loaded
                if is_joint:
                    key = (name, fold)
                    reused = key in joint_cache
                    if not reused:
                        # Every history of every held-out locality is absent.
                        joint_cache[key] = fitter(fit_training, seed=fit_seed, budget=dict(budget or {}), group_names=list(GROUPS))
                    prediction_fit = copy.copy(joint_cache[key])
                    if hasattr(prediction_fit, "predictive_seed"):
                        prediction_fit.predictive_seed += int(prediction_seed_offset)
                    fitted = JointElectionView(prediction_fit, election, heldout.subdistrict_code.to_numpy(), heldout.locality_code.to_numpy())
                elif cached:
                    fitted = loaded
                else:
                    fitted = fitter(train_x.copy(), train_n.copy(), train_y.copy(), seed=fit_seed,
                                    budget=dict(budget or {}), group_names=list(GROUPS),
                                    locality_ids=training.locality_code.to_numpy().copy())
                parameter_fit = joint_cache[(name, fold)] if is_joint else fitted
                if cache_file is not None and not reused and not cached and hasattr(parameter_fit, "save_parameters"):
                    parameter_fit.save_parameters(cache_file)
                    cache_file.with_suffix(".json").write_text(json.dumps(json_safe(context), sort_keys=True, indent=2)+"\n", encoding="utf-8")
                group_draws = np.asarray(fitted.group_mean_draws, float)
                if group_draws.ndim != 2 or group_draws.shape[1] != 6 or not len(group_draws) or not np.isfinite(group_draws).all() or ((group_draws < 0) | (group_draws > 1)).any():
                    raise ValueError(name+": invalid six-group probability draws")
                # This call cannot pass held-out turnout to an adapter.
                draws = np.asarray(fitted.predict(test_x.copy(), test_n.copy(), seed=fit_seed+1+int(prediction_seed_offset)), float)
                if draws.ndim != 2 or draws.shape[1] != len(heldout) or not len(draws) or not np.isfinite(draws).all() or (draws < 0).any() or (draws > test_n).any():
                    raise ValueError(name+": invalid held-out count predictions")
                genuine = getattr(fitted, "predictive_kind", "") == "new_locality_counts"
                if genuine and (len(draws) < 20 or not np.allclose(draws, np.round(draws))):
                    raise ValueError(name+": predictive coverage needs integer count draws")
                kind = getattr(fitted, "predictive_kind", "unspecified")
                rates = np.asarray(fitted.expected_rates(test_x) if hasattr(fitted, "expected_rates") else test_x @ group_draws.mean(axis=0))
                if rates.shape != (len(heldout),) or not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
                    raise ValueError(name+": invalid analytic expected rates")
                expected_counts = rates*test_n
                predictions.extend(prediction_rows(name, election, fold, heldout.locality_code.to_numpy(),
                                                    test_n, heldout.total_voted.to_numpy(), draws, kind, expected_counts=expected_counts))
                region_ids = heldout.subdistrict_code.to_numpy()
                regions = np.unique(region_ids)
                # Sum the same posterior draw across members: never sum row quantiles.
                regional_draws = np.stack([draws[:, region_ids == code].sum(axis=1) for code in regions], axis=1)
                regional_n = np.array([test_n[region_ids == code].sum() for code in regions])
                regional_y = np.array([heldout.total_voted.to_numpy()[region_ids == code].sum() for code in regions])
                regional_expected = np.array([expected_counts[region_ids == code].sum() for code in regions])
                regional_predictions.extend(prediction_rows(name, election, fold, regions, regional_n, regional_y,
                                                            regional_draws, kind, identifier="subdistrict_code", expected_counts=regional_expected))
                for j, group in enumerate(GROUPS):
                    groups.append({"model": name, "election": election, "fold": fold, "group": group,
                                   "mean": float(group_draws[:, j].mean()), "p025": float(np.quantile(group_draws[:, j], .025)),
                                   "p975": float(np.quantile(group_draws[:, j], .975)), "draws": len(group_draws)})
                diagnostics = getattr(fitted, "diagnostics", {})
                fits.append({"model": name, "election": election, "fold": fold, "seed": fit_seed,
                             "training_localities": len(training), "heldout_localities": len(heldout),
                             "runtime_seconds": time.perf_counter()-started,
                             "fit_scope": "all observed training elections jointly" if is_joint else "one election",
                             "fit_reused_from_previous_scored_election": reused,
                             "posterior_cache_reused_without_refit": cached,
                             "posterior_parameter_cache": str(cache_file.relative_to(Path(cache_directory).parent)) if cache_file is not None and cache_file.exists() else None,
                             "diagnostics": diagnostics, "benchmark_convergence_gate_passed": convergence_gate(diagnostics)})
    predictions = pd.DataFrame(predictions)
    regional_predictions = pd.DataFrame(regional_predictions)
    audit.update({"split_seed": seed, "folds": folds, "evaluated_folds": fold_ids, "evaluated_elections": selected,
                  "subdistrict_blocks": int(splits.subdistrict_code.nunique()), "fits": fits,
                  "budget": dict(budget or {}), "prediction_interval_levels": list(LEVELS),
                  "prediction_seed_offset": int(prediction_seed_offset),
                  "point_prediction_definition": "Analytic adapter expected rates where available; otherwise Bayesian group means marginalize unseen-unit Normal/Dirichlet effects. Empirical WLS methods use the exact eligible-weighted clipped training-residual mixture mean. Monte Carlo counts supply only medians/intervals/WIS, not point MAE/RMSE.",
                  "benchmark_convergence_policy": "For MCMC eligibility: positive diagnostic counts and explicit all-selected-finite certification with zero nonfinite Rhat/bulk-ESS/tail-ESS values; at least2 chains (4preferred), global/hyperparameter and group-functional Rhat <=1.01, bulkESS >=400, tailESS >=200, zero divergences. All-latent diagnostics separately retained. Missing finite certification in legacy caches is UNVERIFIED/not eligible. Failed fits are pilot/exploratory only. Non-MCMC baselines have no MCMC gate.",
                  "wis_definition": "Bracher et al.2021: [0.5*|y-median| + sum(alpha/2*interval_score_alpha)]/(3+0.5), alpha=.5,.2,.05. Per-locality count scores divided by eligible and multiplied by100; eligible-weighted within election, equal election weighting across years.",
                  "wis_source": "https://doi.org/10.1371/journal.pcbi.1008618",
                  "practical_value_policy": "Prespecified before full candidate scores: >=5% lower equal-election mean WIS versus WLS empirical residual at the regional-total or locality level (state which level), locality RMSE no more than5% worse, same-level WIS improves in at least2of3 geographic folds, and strict MCMC gate passes when MCMC applies. Regional-total WIS is primary for cross-locality correlation; locality WIS/RMSE secondary. Secondary point-only value: >=5% lower locality RMSE with strict MCMC gate when MCMC applies. No p-value, future-election tail calibration, or individual-group truth claim.",
                  "regional_score_definition": "For each predictive draw, sum voted counts of all held-out localities in a subdistrict, then compute its predictive median/intervals and WIS against summed observed count. Pairing across localities preserves model-implied correlation.",
                  "score_target": "Eligible-weighted held-out locality turnout rate; observed voted count for interval coverage",
                  "validation_scope": "Within-election geographic holdout; whole locality histories share the split. No future-election calibration or individual group truth is observed.",
                  "joint_model_scope": "Joint adapters use all observed elections of training localities only, including years later than a scored historical election. Source subdistrict IDs are supplied as outcome-free geographic predictors; held-out outcomes are never supplied. This is retrospective geographic prediction, not time-forward forecasting; all years of held-out localities and all held-out subdistricts are excluded from fitting.",
                  "group_composition_uncertainty_included": False,
                  "population_basis": "Eligible voters; fixed K25/2022 Census group shares are proxies, not observed group eligibility counts",
                  "limitations": ["Only three geographic folds; aggregate prediction does not verify individual demographic turnout.",
                                  "Historical coefficients and future national election shock probabilities are separate targets.",
                                  "Model intervals condition on the chosen likelihood, hierarchy, and Census proxies.",
                                  "Boundary .2 to .95 is the existing bounded-WLS benchmark specification; Bayesian adapters retain their native probability support.",
                                  "WLS empirical predictive intervals use training residual rates, weighted by eligibility, clipped to [0,1] and converted to counts; they are a research benchmark, not existing production intervals."]})
    scores = score_predictions(predictions)
    def model_gate(row):
        checks = [fit["benchmark_convergence_gate_passed"] for fit in fits if fit["model"] == row.model and fit["election"] == row.election and (not hasattr(row, "fold") or fit["fold"] == row.fold)]
        return None if all(check is None for check in checks) else all(check is True for check in checks)
    scores["mcmc_diagnostics_passed"] = [model_gate(row) for row in scores.itertuples()]
    fold_scores = score_predictions(predictions, by_fold=True)
    fold_scores["mcmc_diagnostics_passed"] = [model_gate(row) for row in fold_scores.itertuples()]
    regional_scores = score_predictions(regional_predictions, unit="regions")
    regional_scores["mcmc_diagnostics_passed"] = [model_gate(row) for row in regional_scores.itertuples()]
    regional_fold_scores = score_predictions(regional_predictions, by_fold=True, unit="regions")
    regional_fold_scores["mcmc_diagnostics_passed"] = [model_gate(row) for row in regional_fold_scores.itertuples()]
    return {"splits": splits, "predictions": predictions, "scores": scores, "fold_scores": fold_scores,
            "aggregate_scores": aggregate_elections(scores),
            "regional_predictions": regional_predictions, "regional_scores": regional_scores,
            "regional_fold_scores": regional_fold_scores, "regional_aggregate_scores": aggregate_elections(regional_scores),
            "group_summary": pd.DataFrame(groups), "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--geography", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["wls", "pyei", "rebane", "rebane_joint", "marginal_joint", "regional_empirical"], default=["wls"])
    parser.add_argument("--elections", nargs="+")
    parser.add_argument("--selected-folds", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=.9)
    parser.add_argument("--nuts-sampler", choices=["pymc", "numpyro"], default="pymc")
    parser.add_argument("--chain-method", choices=["parallel", "sequential", "vectorized"], default="sequential")
    parser.add_argument("--dense-mass", action="store_true", help="Use a dense NumPyro NUTS metric in the marginal adapter; model likelihood/priors unchanged")
    parser.add_argument("--reuse-cache", action="store_true", help="Reload matching saved posterior parameters; reject stale training/seed/budget/adapter context")
    parser.add_argument("--prediction-seed-offset", type=int, default=0, help="Resample predictions without changing fitted-posterior cache context")
    args = parser.parse_args()
    try:
        validate_sampler_request(args.models, args.nuts_sampler, args.dense_mass)
    except ValueError as error:
        parser.error(str(error))
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    adapters, joint_adapters, upstream, cache_loaders = {}, {}, {}, {}
    for name in args.models:
        if name != "wls":
            module = importlib.import_module(PACKAGE+"."+({"rebane_joint": "rebane_joint", "marginal_joint": "marginal_adapter", "regional_empirical": "regional_empirical"}.get(name, name+"_adapter")))
            if name in ["rebane_joint", "marginal_joint", "regional_empirical"]:
                joint_adapters[name] = module.fit_joint
            else:
                adapters[name] = module.fit
            upstream[name] = getattr(module, "ADAPTER_METADATA", {})
            if hasattr(module, "load_parameters"):
                cache_loaders[name] = module.load_parameters
    budget = {key: getattr(args, key) for key in ["draws", "tune", "chains", "cores", "target_accept", "nuts_sampler", "chain_method", "dense_mass"]}
    results = evaluate(pd.read_csv(args.panel), pd.read_csv(args.geography), adapters,
                       seed=args.seed, budget=budget, elections=args.elections, selected_folds=args.selected_folds,
                       joint_adapters=joint_adapters, cache_directory=args.output/"fit_cache",
                       reuse_cache=args.reuse_cache, cache_loaders=cache_loaders, prediction_seed_offset=args.prediction_seed_offset)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ["splits", "predictions", "scores", "fold_scores", "aggregate_scores", "group_summary",
                "regional_predictions", "regional_scores", "regional_fold_scores", "regional_aggregate_scores"]:
        results[key].to_csv(args.output/(key+".csv"), index=False, encoding="utf-8-sig")
    results["audit"].update({"panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                             "geography_sha256": hashlib.sha256(args.geography.read_bytes()).hexdigest(),
                             "runtime_python": platform.python_version(), "upstream_adapters": upstream})
    (args.output/"report.json").write_text(json.dumps(json_safe(results["audit"]), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(results["scores"][["model", "election", "localities", "weighted_mae_pp", "weighted_rmse_pp", "eligible_weighted_coverage_95"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
