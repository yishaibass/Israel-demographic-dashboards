"""Joint turnout EI using the pinned author's hierarchical probit builder.

Selective group-by-election deviations extend the author design without adding
all locality-by-group or locality-by-election interactions. Common election and
persistent locality effects induce dependence. Future-election draws are a
separate conditional, exchangeable-election experiment, not calibrated rarity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .rebane_adapter import (
    ADAPTER_METADATA as AUTHOR_METADATA, integer_group_counts,
    load_upstream, population_group_means, sampling_diagnostics, validate_arrays,
)

ADAPTER_METADATA = {
    **AUTHOR_METADATA, "candidate": "rebane_joint_selective",
    "reused": "Pinned unmodified author build_linear_predictor and its Normal, HalfNormal and ZeroSumNormal priors",
    "adapted": "Observed locality-election cell construction; selective group-by-election interaction; original weighted-margin Binomial likelihood; held-out and new-election prediction",
    "effects": "Shared group, persistent locality and subdistrict intercepts; common election, group-by-election and subdistrict-by-election deviations; no locality-by-election or group-by-locality interactions",
    "likelihood": "Locality-election Binomial turnout with weighted six-group probit probabilities and the documented joint regional hierarchy",
    "group_mean_definition": "Superpopulation mean-function group probabilities integrating new locality, subdistrict and subdistrict-election effects; not finite-Israel group turnout totals",
    "future_assumption": "New election and group deviations are exchangeable draws from learned hierarchical scales; no temporal autoregression or political regime process is fitted",
}


def build_model(training_panel, group_names, priors=None):
    import pymc as pm

    author = load_upstream()
    frame = training_panel.copy().sort_values(["election", "locality_code"]).reset_index(drop=True)
    if frame.duplicated(["election", "locality_code"]).any():
        raise ValueError("Duplicate training election-locality observations")
    if "subdistrict_code" not in frame or frame.subdistrict_code.isna().any():
        raise ValueError("Validated CBS subdistrict context is required")
    if (frame.groupby("locality_code").subdistrict_code.nunique() > 1).any():
        raise ValueError("Each locality must retain one fixed geographic identity")
    groups = [str(group) for group in group_names]
    if len(groups) != 6 or len(set(groups)) != 6:
        raise ValueError("Six unique group names are required")
    x, n, y = validate_arrays(frame[["share_"+group for group in groups]].to_numpy(), frame.eligible_voters, frame.total_voted)
    counts = integer_group_counts(x, n)
    elections = sorted(frame.election.unique(), key=lambda e: int(e[1:]))
    localities = sorted(frame.locality_code.astype(str).unique())
    regions = sorted(frame.subdistrict_code.astype(str).unique())
    if len(elections) < 2:
        raise ValueError("Joint model requires at least two training elections")
    indices = {"group": np.tile(np.arange(6), len(frame)),
               "locality": np.repeat(pd.Categorical(frame.locality_code.astype(str), categories=localities).codes, 6),
               "election": np.repeat(pd.Categorical(frame.election, categories=elections).codes, 6),
               "subdistrict": np.repeat(pd.Categorical(frame.subdistrict_code.astype(str), categories=regions).codes, 6)}
    coords = {"group": groups, "locality": localities, "subdistrict": regions, "election": elections, "observation": np.arange(len(frame))}
    with pm.Model(coords=coords) as model:
        main, coefs = author.build_linear_predictor(indices, ["group", "locality", "subdistrict", "election"], False, True,
                                                     prefix="outcome", priors=priors)
        mu = main
        for variables in [["group", "election"], ["subdistrict", "election"]]:
            pair_base, coefs = author.build_linear_predictor(indices, variables, False, True,
                                                              prefix="outcome", priors=priors, reuse_coefs=coefs)
            pair, coefs = author.build_linear_predictor(indices, variables, True, True,
                                                         prefix="outcome", priors=priors, reuse_coefs=coefs)
            mu = mu+pair-pair_base
        probabilities = pm.math.invprobit(mu.reshape((len(frame), 6)))
        margin_prob = (probabilities*counts).sum(axis=1)/n
        pm.Binomial("margin_obs", n=n, p=margin_prob, observed=y, dims="observation")
    return model, {"elections": elections, "locality_ids": localities, "regional_ids": regions, "group_names": groups,
                   "training_rows": len(frame), "priors": dict(priors or {}),
                   "integer_allocation_max_group_count_error": float(np.abs(counts-x*n[:, None]).max())}


@dataclass
class FittedJointRebane:
    intercept_draws: np.ndarray
    group_effect_draws: np.ndarray
    election_effect_draws: np.ndarray
    group_election_draws: np.ndarray
    locality_effect_draws: np.ndarray
    locality_scale_draws: np.ndarray
    election_scale_draws: np.ndarray
    group_election_scale_draws: np.ndarray
    regional_effect_draws: np.ndarray
    regional_election_draws: np.ndarray
    regional_scale_draws: np.ndarray
    regional_election_scale_draws: np.ndarray
    elections: list
    locality_ids: list
    regional_ids: list
    predictive_seed: int
    diagnostics: dict
    metadata: dict
    predictive_kind: str = "new_locality_counts"

    def eta(self, election):
        if election not in self.elections:
            raise ValueError("Election not fitted; future election prediction requires the explicit new-election method")
        index = self.elections.index(election)
        return (self.intercept_draws[:, None]+self.group_effect_draws+
                self.election_effect_draws[:, index, None]+self.group_election_draws[:, index, :])

    def group_mean_draws_for_election(self, election):
        total_scale = np.sqrt(self.locality_scale_draws**2+self.regional_scale_draws**2+self.regional_election_scale_draws**2)
        return population_group_means(self.eta(election), total_scale)

    def _new_unit_draws(self, labels, kind, scale, election=""):
        labels = [str(value) for value in labels]
        unique = sorted(set(labels))
        draws = {}
        for value in unique:
            token = (str(self.predictive_seed)+"|"+kind+"|"+election+"|"+value).encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
            draws[value] = np.random.default_rng(seed).normal(size=len(scale))*scale
        return np.column_stack([draws[value] for value in labels])

    def predict_election(self, election, x_test, eligible_test, *, seed, regional_ids, locality_ids=None):
        x, n = validate_arrays(x_test, eligible_test)
        if len(regional_ids) != len(x):
            raise ValueError("Geographic context must match held-out localities")
        eta = self.eta(election)
        regional = self._new_unit_draws(regional_ids, "region", self.regional_scale_draws)
        regional_year = self._new_unit_draws(regional_ids, "region_year", self.regional_election_scale_draws, election)
        region_lookup = {value: i for i,value in enumerate(self.regional_ids)}
        for j, value in enumerate(regional_ids):
            if str(value) in region_lookup:
                index = region_lookup[str(value)]
                regional[:, j] = self.regional_effect_draws[:, index]
                regional_year[:, j] = self.regional_election_draws[:, self.elections.index(election), index]
        if locality_ids is None:
            locality_ids = [str(seed)+"_"+str(i) for i in range(len(x))]
        if len(locality_ids) != len(x):
            raise ValueError("Locality context must match prediction rows")
        local = self._new_unit_draws(locality_ids, "locality", self.locality_scale_draws)
        locality_lookup = {value: i for i,value in enumerate(self.locality_ids)}
        for j, value in enumerate(locality_ids):
            if str(value) in locality_lookup:
                local[:, j] = self.locality_effect_draws[:, locality_lookup[str(value)]]
        rng = np.random.default_rng(seed)
        output = np.empty((len(eta), len(x)), dtype=np.int64)
        for start in range(0, len(x), 128):
            stop = min(start+128, len(x))
            effects = regional[:, start:stop]+regional_year[:, start:stop]+local[:, start:stop]
            rates = ndtr(eta[:, None, :]+effects[:, :, None])
            probability = np.einsum("slg,lg->sl", rates, x[start:stop])
            output[:, start:stop] = rng.binomial(n[start:stop][None, :], np.clip(probability, 0, 1))
        return output

    def predict_new_election_group_rates(self, target_election, x, eligible, locality_ids, regional_ids, *, seed):
        if target_election in self.elections or int(target_election[1:]) <= max(int(e[1:]) for e in self.elections):
            raise ValueError("New-election prediction must exclude the target and all later election outcomes from training")
        x, n = validate_arrays(x, eligible)
        ids = [str(value) for value in locality_ids]
        if len(ids) != len(x) or len(set(ids)) != len(ids):
            raise ValueError("Unique locality identifiers must match prediction rows")
        regions = [str(value) for value in regional_ids]
        if len(regions) != len(x):
            raise ValueError("Regional identifiers must match prediction rows")
        rng = np.random.default_rng(seed)
        samples = len(self.intercept_draws)
        common_shock = rng.normal(size=samples)*self.election_scale_draws
        deviations = rng.normal(size=(samples, 6))*self.group_election_scale_draws[:, None]
        deviations -= deviations.mean(axis=1, keepdims=True)
        eta = self.intercept_draws[:, None]+self.group_effect_draws+common_shock[:, None]+deviations
        lookup = {value: index for index,value in enumerate(self.locality_ids)}
        regional_lookup = {value: index for index,value in enumerate(self.regional_ids)}
        region_effects = self._new_unit_draws(regions, "region", self.regional_scale_draws)
        region_year = self._new_unit_draws(regions, "future_region_year", self.regional_election_scale_draws, target_election+"_"+str(seed))
        for j, region in enumerate(regions):
            if region in regional_lookup:
                region_effects[:, j] = self.regional_effect_draws[:, regional_lookup[region]]
        totals = np.zeros((samples, 6))
        denominator = (x*n[:, None]).sum(axis=0)
        for start in range(0, len(x), 128):
            stop = min(start+128, len(x))
            effects = rng.normal(size=(samples, stop-start))*self.locality_scale_draws[:, None]
            for j, value in enumerate(ids[start:stop]):
                if value in lookup:
                    effects[:, j] = self.locality_effect_draws[:, lookup[value]]
            effects += region_effects[:, start:stop]+region_year[:, start:stop]
            probabilities = ndtr(eta[:, None, :]+effects[:, :, None])
            totals += np.einsum("slg,lg->sg", probabilities, x[start:stop]*n[start:stop, None])
        if (denominator <= 0).any():
            raise ValueError("Every predicted group must have positive eligible mass")
        return totals/denominator

    def save_parameters(self, path):
        arrays = {name: value for name,value in vars(self).items() if isinstance(value, np.ndarray)}
        info = {name: value for name,value in vars(self).items() if not isinstance(value, np.ndarray)}
        np.savez_compressed(path, **arrays, info=np.array(json.dumps(info)))


def load_parameters(path):
    with np.load(path, allow_pickle=False) as saved:
        info = json.loads(saved["info"].item())
        return FittedJointRebane(**{name: saved[name] for name in saved.files if name != "info"}, **info)


def fit_joint(training_panel, *, seed, budget, group_names):
    import pymc as pm

    started = time.perf_counter()
    options = {"draws": 500, "tune": 500, "chains": 2, "cores": 1, "target_accept": .9}
    options.update(dict(budget))
    priors = options.pop("priors", None)
    allowed = {key: options[key] for key in ["draws", "tune", "chains", "cores", "target_accept"]}
    for key in ["nuts_sampler", "nuts_sampler_kwargs"]:
        if key in options:
            allowed[key] = options[key]
    if allowed.get("nuts_sampler") == "numpyro" and "nuts_sampler_kwargs" not in allowed:
        allowed["nuts_sampler_kwargs"] = {"chain_method": options.get("chain_method", "vectorized")}
    if allowed["chains"] < 2:
        raise ValueError("At least two chains are required")
    model, inputs = build_model(training_panel, group_names, priors)
    with model:
        idata = pm.sample(**allowed, random_seed=seed, progressbar=False, return_inferencedata=True,
                          compute_convergence_checks=True, discard_tuned_samples=True)
    posterior = idata.posterior

    def values(name, *dims):
        return posterior[name].stack(sample=("chain", "draw")).transpose("sample", *dims).to_numpy()

    fitted = FittedJointRebane(
        values("outcome_intercept"), values("outcome_group_effect", "group"),
        values("outcome_election_effect", "election"), values("outcome_election:group_effect", "election", "group"),
        values("outcome_locality_effect", "locality"), values("outcome_locality_tau"),
        values("outcome_election_tau"), values("outcome_election:group_tau"),
        values("outcome_subdistrict_effect", "subdistrict"), values("outcome_election:subdistrict_effect", "election", "subdistrict"),
        values("outcome_subdistrict_tau"), values("outcome_election:subdistrict_tau"),
        inputs["elections"], inputs["locality_ids"], inputs["regional_ids"], int(seed), {},
        {**ADAPTER_METADATA, **inputs, "future_election_tails_calibrated": False})
    group_means = np.stack([fitted.group_mean_draws_for_election(election) for election in fitted.elections], axis=1)
    variables = ["outcome_intercept", "outcome_group_effect", "outcome_group_tau", "outcome_locality_tau",
                 "outcome_election_effect", "outcome_election_tau", "outcome_election:group_effect", "outcome_election:group_tau",
                 "outcome_subdistrict_tau", "outcome_election:subdistrict_tau"]
    fitted.diagnostics = sampling_diagnostics(idata, variables, group_means, allowed, time.perf_counter()-started, pm.__version__,
        latent_variables=["outcome_locality_effect", "outcome_subdistrict_effect", "outcome_election:subdistrict_effect"])
    return fitted
