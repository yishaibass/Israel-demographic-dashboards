"""Adapter for Reimo Rebane's pinned MIT-licensed aggregate turnout EI model.

Upstream code remains unmodified in a temporary dependency cache. The author EI
model uses group and locality probit random intercepts. Held-out predictions draw
new locality effects from their posterior population scale, without held-out votes.
This is within-election inference, not a future-election forecasting model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr

UPSTREAM_COMMIT = "a1dd3da5e65ad05c95265ddedf3b6c519c0f1642"
SOURCE_SHA256 = "a6b9a352fd808afe0894e441425f54c0e1d4a3168cf2368d5e4a8a9b670eed76"
LICENSE_SHA256 = "7262d6d97a4c63e2c957964abb6391953f87fa3f4e823685c746ea316e80b590"
ADAPTER_METADATA = {
    "candidate": "rebane_author_EI",
    "upstream_repository": "https://github.com/rrebane/turnout-model-evaluation",
    "upstream_commit": UPSTREAM_COMMIT,
    "source_path": "turnout_models/models.py",
    "source_sha256": SOURCE_SHA256,
    "license": "MIT", "copyright": "Copyright (c) 2025 Reimo Rebane",
    "license_sha256": LICENSE_SHA256,
    "reused": "Unmodified upstream model_EI, build_linear_predictor, poststratify, and PyMC model graph",
    "adapted": "Six-group fractional Census composition allocated to integer group counts preserving each locality eligible total; unseen-locality posterior prediction; diagnostics and benchmark interface",
    "likelihood": "Upstream locality Binomial turnout with weighted six-group probit cell probabilities",
    "effects": "Hierarchical group and locality random intercepts; no interactions, temporal process or common election-shock model",
    "group_mean_definition": "Population-marginal probit turnout averaged over a new Normal locality effect, conditional on posterior group/intercept/locality-scale parameters",
    "predictive_kind": "new_locality_counts",
}


def upstream_cache():
    override = os.environ.get("REBANE_UPSTREAM_DIR")
    return Path(override) if override else Path(tempfile.gettempdir()) / "israel-turnout-ei-deps" / ("rebane-"+UPSTREAM_COMMIT)


def verify_upstream(cache):
    cache = Path(cache)
    for name, expected in [("models.py", SOURCE_SHA256), ("LICENSE", LICENSE_SHA256)]:
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError("Pinned Rebane dependency missing: " + str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned Rebane dependency hash mismatch: " + name)
    return cache / "models.py"


def load_upstream(cache=None):
    path = verify_upstream(upstream_cache() if cache is None else cache)
    spec = importlib.util.spec_from_file_location("_rebane_turnout_"+UPSTREAM_COMMIT, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_arrays(x, eligible, voted=None):
    x = np.asarray(x, dtype=float)
    n = np.asarray(eligible)
    if x.ndim != 2 or x.shape[1] != 6 or n.shape != (len(x),):
        raise ValueError("Expected locality x six-group shares and one eligible count per locality")
    if not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid group shares")
    if not np.isfinite(n).all() or (n <= 0).any() or not np.equal(n, np.floor(n)).all():
        raise ValueError("Eligible counts must be positive integers")
    n = n.astype(np.int64)
    x = x/x.sum(axis=1, keepdims=True)
    if voted is not None:
        y = np.asarray(voted)
        if y.shape != n.shape or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all() or (y < 0).any() or (y > n).any():
            raise ValueError("Voted counts must be integers between zero and eligible")
        return x, n, y.astype(np.int64)
    return x, n


def integer_group_counts(x, eligible):
    x, n = validate_arrays(x, eligible)
    raw = x*n[:, None]
    counts = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw-counts), axis=1, kind="stable")
    for locality, remaining in enumerate(n-counts.sum(axis=1)):
        counts[locality, order[locality, :remaining]] += 1
    if not np.array_equal(counts.sum(axis=1), n):
        raise ValueError("Integer group counts do not preserve eligible totals")
    return counts


def build_inputs(x, eligible, voted, group_names, locality_ids):
    x, n, y = validate_arrays(x, eligible, voted)
    names = [str(name) for name in group_names]
    ids = [str(value) for value in locality_ids]
    if len(names) != 6 or len(set(names)) != 6 or len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError("Unique six group names and unique locality identifiers are required")
    counts = integer_group_counts(x, n)
    census = pd.DataFrame({"group": np.tile(names, len(x)), "locality": np.repeat(ids, 6), "N": counts.reshape(-1)})
    margins = pd.DataFrame({"locality": ids, "voted": "Yes", "N": y})
    return census, margins, counts


def population_group_means(eta, locality_scale):
    """Analytic E[Phi(eta + U)], U ~ Normal(0, locality_scale²)."""
    return ndtr(np.asarray(eta)/np.sqrt(1+np.asarray(locality_scale)[:, None]**2))


def diagnostic_values(result):
    """Flatten ArviZ Dataset or current DataTree diagnostic outputs."""
    nodes = result.subtree if hasattr(result, "subtree") else [result]
    arrays = []
    for node in nodes:
        dataset = node.to_dataset() if hasattr(node, "subtree") else node
        arrays.extend(variable.to_numpy().reshape(-1) for variable in dataset.data_vars.values())
    return np.concatenate(arrays) if arrays else np.array([np.nan])


def diagnostic_metrics(rhat, bulk, tail):
    """Retain completeness checks before reducing selected diagnostics."""
    values = {"rhat": np.asarray(rhat), "ess_bulk": np.asarray(bulk), "ess_tail": np.asarray(tail)}
    counts = {name: int(value.size) for name, value in values.items()}
    nonfinite = {name: int((~np.isfinite(value)).sum()) for name, value in values.items()}
    return {"rhat_max": float(np.max(values["rhat"])) if counts["rhat"] else float("nan"),
            "ess_bulk_min": float(np.min(values["ess_bulk"])) if counts["ess_bulk"] else float("nan"),
            "ess_tail_min": float(np.min(values["ess_tail"])) if counts["ess_tail"] else float("nan"),
            "all_selected_finite": bool(all(counts.values()) and not any(nonfinite.values())),
            "diagnostic_value_counts": counts, "nonfinite_counts": nonfinite}


def sampling_diagnostics(idata, variables, group_means, budget, runtime_seconds, pymc_version, latent_variables=None):
    import arviz as az
    import xarray as xr

    def metrics(data, var_names=None):
        rhat = diagnostic_values(az.rhat(data, var_names=var_names))
        bulk = diagnostic_values(az.ess(data, var_names=var_names, method="bulk"))
        tail = diagnostic_values(az.ess(data, var_names=var_names, method="tail"))
        return diagnostic_metrics(rhat, bulk, tail)

    global_metrics = metrics(idata, variables)
    chains, draws = idata.posterior.sizes["chain"], idata.posterior.sizes["draw"]
    shape = np.asarray(group_means).shape
    dims = ("chain", "draw")+tuple("dim_"+str(i) for i in range(len(shape)-1))
    derived = xr.Dataset({"group_mean": (dims, np.asarray(group_means).reshape((chains, draws)+shape[1:]))})
    group_metrics = metrics(derived)
    posterior_names = set(idata.posterior.data_vars)
    locality_metrics = metrics(idata, ["outcome_locality_effect"]) if "outcome_locality_effect" in posterior_names else None
    latent_names = latent_variables or (["outcome_locality_effect"] if locality_metrics is not None else variables)
    all_latent_metrics = metrics(idata, latent_names)
    divergences = int(idata.sample_stats.diverging.sum().item())
    selected_scopes = (global_metrics, group_metrics)
    result = {"rhat_max": float(np.max([scope["rhat_max"] for scope in selected_scopes])),
              "ess_bulk_min": float(np.min([scope["ess_bulk_min"] for scope in selected_scopes])),
              "ess_tail_min": float(np.min([scope["ess_tail_min"] for scope in selected_scopes])),
              "all_selected_finite": all(scope["all_selected_finite"] for scope in selected_scopes),
              "diagnostic_value_counts": {key: sum(scope["diagnostic_value_counts"][key] for scope in selected_scopes)
                                          for key in ("rhat", "ess_bulk", "ess_tail")},
              "nonfinite_counts": {key: sum(scope["nonfinite_counts"][key] for scope in selected_scopes)
                                   for key in ("rhat", "ess_bulk", "ess_tail")},
              "diagnostic_policy": "all-selected-finite-v1",
              "global_hyper": global_metrics, "population_group_means": group_metrics,
              "locality_latents": locality_metrics, "all_latents": all_latent_metrics, "divergences": divergences,
              "posterior_draws": len(group_means), "budget": budget,
              "runtime_seconds": runtime_seconds, "pymc_version": pymc_version}
    result["convergence_screen_pass"] = bool(result["all_selected_finite"] and np.isfinite(result["rhat_max"]) and result["rhat_max"] <= 1.01
        and result["ess_bulk_min"] >= 400 and result["ess_tail_min"] >= 200 and divergences == 0 and chains >= 2)
    return result


@dataclass
class FittedRebane:
    eta_draws: np.ndarray
    locality_scale_draws: np.ndarray
    group_mean_draws: np.ndarray
    diagnostics: dict
    metadata: dict
    predictive_kind: str = "new_locality_counts"

    def save_parameters(self, path):
        np.savez_compressed(path, eta_draws=self.eta_draws, locality_scale_draws=self.locality_scale_draws,
                            group_mean_draws=self.group_mean_draws,
                            diagnostics=np.array(json.dumps(self.diagnostics)), metadata=np.array(json.dumps(self.metadata)))

    def predict(self, x_test, eligible_test, *, seed):
        x, n = validate_arrays(x_test, eligible_test)
        rng = np.random.default_rng(seed)
        output = np.empty((len(self.eta_draws), len(x)), dtype=np.int64)
        for start in range(0, len(x), 128):
            stop = min(start+128, len(x))
            locality_effect = rng.normal(size=(len(self.eta_draws), stop-start))*self.locality_scale_draws[:, None]
            rates = ndtr(self.eta_draws[:, None, :]+locality_effect[:, :, None])
            probability = np.einsum("slg,lg->sl", rates, x[start:stop])
            output[:, start:stop] = rng.binomial(n[start:stop][None, :], np.clip(probability, 0, 1))
        return output


def load_parameters(path):
    with np.load(path, allow_pickle=False) as saved:
        return FittedRebane(saved["eta_draws"], saved["locality_scale_draws"], saved["group_mean_draws"],
                            json.loads(saved["diagnostics"].item()), json.loads(saved["metadata"].item()))


def fit(x, eligible, voted, *, seed, budget, group_names, locality_ids):
    started = time.perf_counter()
    import pymc as pm

    author = load_upstream()
    x, n, y = validate_arrays(x, eligible, voted)
    census, margins, counts = build_inputs(x, n, y, group_names, locality_ids)
    options = {"draws": 500, "tune": 500, "chains": 2, "cores": 1, "target_accept": .9}
    options.update(dict(budget))
    allowed = {key: options[key] for key in ["draws", "tune", "chains", "cores", "target_accept"]}
    if "nuts_sampler" in options:
        allowed["nuts_sampler"] = options["nuts_sampler"]
    if "nuts_sampler_kwargs" in options:
        allowed["nuts_sampler_kwargs"] = options["nuts_sampler_kwargs"]
    elif options.get("nuts_sampler") == "numpyro":
        allowed["nuts_sampler_kwargs"] = {"chain_method": options.get("chain_method", "vectorized")}
    if allowed["chains"] < 2:
        raise ValueError("At least two chains are required for convergence diagnostics")
    model, _ = author.model_EI(census, margins, input_vars=["group", "locality"], outcome_var="voted", interactions=False, multilevel=True)
    with model:
        idata = pm.sample(**allowed, random_seed=seed, progressbar=False, return_inferencedata=True,
                          compute_convergence_checks=True, discard_tuned_samples=True)
    posterior = idata.posterior
    intercept = posterior["outcome_intercept"].stack(sample=("chain", "draw")).transpose("sample").to_numpy()
    effects = posterior["outcome_group_effect"].sel(group=list(group_names)).stack(sample=("chain", "draw")).transpose("sample", "group").to_numpy()
    scale = posterior["outcome_locality_tau"].stack(sample=("chain", "draw")).transpose("sample").to_numpy()
    eta = intercept[:, None]+effects
    means = population_group_means(eta, scale)
    variables = ["outcome_intercept", "outcome_group_effect", "outcome_group_tau", "outcome_locality_tau"]
    diagnostics = sampling_diagnostics(idata, variables, means, allowed, time.perf_counter()-started, pm.__version__)
    metadata = {**ADAPTER_METADATA, "integer_allocation_max_group_count_error": float(np.abs(counts-x*n[:, None]).max()),
                "training_localities": len(x), "group_names": list(group_names),
                "population_group_means_are_eligible_group_probabilities": True,
                "unseen_locality_effects": "New independent Normal effects with posterior locality scale, shared by the six groups within each test locality; posterior parameters shared across all predictions",
                "future_election_tails_claimed": False}
    return FittedRebane(eta, scale, means, diagnostics, metadata)
