"""Adapt the pinned author EI design with integrated locality overdispersion.

The MIT-licensed upstream linear-predictor builder and its hierarchical priors
are reused unchanged. This adapter replaces the author's locality random
intercept/Binomial observation model with a BetaBinomial locality count model.
It is an explicitly different observation likelihood, not a faster implementation
of the exact native author model. Demographic row margins remain Census proxies.
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
    ADAPTER_METADATA as AUTHOR_METADATA, load_upstream, population_group_means,
    sampling_diagnostics, validate_arrays,
)

ADAPTER_METADATA = {
    **AUTHOR_METADATA,
    "candidate": "marginal_joint_beta_binomial",
    "reused": "Unmodified pinned MIT author build_linear_predictor and its Normal/HalfNormal/ZeroSumNormal design priors",
    "adapted": "New BetaBinomial locality observation likelihood integrates locality overdispersion; removes all locality latent parameters; retains group/subdistrict/election effects and selective group-by-election/subdistrict-by-election interactions",
    "effects": "Group, persistent subdistrict and common election effects plus group-by-election/subdistrict-by-election deviations; no locality latent effects",
    "likelihood": "Voted_i ~ BetaBinomial(eligible_i, kappa*p_i, kappa*(1-p_i)), p_i=sum_g Census_share_ig*Phi(eta_igre)",
    "overdispersion_prior": "kappa ~ LogNormal(log(50),1.5); median50; central95% approximately2.64..946. Within-locality exchangeable intraclass correlation1/(kappa+1), not arbitrary cross-locality covariance",
    "group_mean_definition": "Superpopulation marginal ecological mean-function group probabilities integrated over new subdistrict and subdistrict-election effects; not observed individual turnout or finite-Israel group totals",
    "dependence": "Global posterior parameters and common election effects shared; one persistent new subdistrict effect plus one subdistrict-election shock shared by all member localities; conditional Beta innovations independent across locality-election rows, with no additional locality-specific serial effect",
    "future_election_tails_calibrated": False,
    "graph_optimization": "Evaluate the same probit predictor once per unique subdistrict/election/group and gather to locality rows; priors and BetaBinomial likelihood unchanged. Equivalence evidence: graph_equivalence.json in the local benchmark evidence directory",
}


def context_indices(regional_indices, election_indices):
    """Compress repeated predictor contexts without changing observation rows."""
    contexts, inverse = np.unique(np.column_stack([regional_indices, election_indices]), axis=0, return_inverse=True)
    indices = {"group": np.tile(np.arange(6), len(contexts)),
               "subdistrict": np.repeat(contexts[:, 0], 6),
               "election": np.repeat(contexts[:, 1], 6)}
    return indices, inverse


def build_model(training_panel, group_names, priors=None):
    import pymc as pm

    author = load_upstream()
    frame = training_panel.sort_values(["election", "locality_code"]).reset_index(drop=True)
    if frame.duplicated(["election", "locality_code"]).any():
        raise ValueError("Duplicate election/locality training rows")
    if "subdistrict_code" not in frame or frame.subdistrict_code.isna().any():
        raise ValueError("Source subdistrict context is required")
    if frame.groupby("locality_code").subdistrict_code.nunique().gt(1).any():
        raise ValueError("Locality geography changes across election histories")
    groups = [str(group) for group in group_names]
    if len(groups) != 6 or len(set(groups)) != 6:
        raise ValueError("Six unique demographic groups are required")
    x, n, y = validate_arrays(frame[["share_"+group for group in groups]], frame.eligible_voters, frame.total_voted)
    elections = sorted(frame.election.unique(), key=lambda e: int(e[1:]))
    regions = sorted(frame.subdistrict_code.astype(str).unique())
    if len(elections) < 2:
        raise ValueError("Joint model requires at least two training elections")
    indices, row_context = context_indices(
        pd.Categorical(frame.subdistrict_code.astype(str), categories=regions).codes,
        pd.Categorical(frame.election, categories=elections).codes)
    context_count = len(indices["group"])//6
    with pm.Model(coords={"group": groups, "subdistrict": regions, "election": elections,
                          "observation": np.arange(len(frame))}) as model:
        main, coefficients = author.build_linear_predictor(indices, ["group", "subdistrict", "election"], False, True,
                                                            prefix="outcome", priors=priors)
        predictor = main
        for variables in [["group", "election"], ["subdistrict", "election"]]:
            base, coefficients = author.build_linear_predictor(indices, variables, False, True,
                                                               prefix="outcome", priors=priors, reuse_coefs=coefficients)
            interaction, coefficients = author.build_linear_predictor(indices, variables, True, True,
                                                                      prefix="outcome", priors=priors, reuse_coefs=coefficients)
            predictor = predictor+interaction-base
        group_rates = pm.math.invprobit(predictor.reshape((context_count, 6)))[row_context]
        probability = pm.math.clip((group_rates*x).sum(axis=1), 1e-9, 1-1e-9)
        kappa = pm.LogNormal("overdispersion_kappa", mu=np.log(50.), sigma=1.5)
        pm.BetaBinomial("margin_obs", n=n, alpha=kappa*probability,
                        beta=kappa*(1-probability), observed=y, dims="observation")
    return model, {"elections": elections, "regional_ids": regions, "group_names": groups,
                   "training_rows": len(frame), "training_localities": int(frame.locality_code.nunique()),
                   "priors": dict(priors or {}), "locality_latent_parameters": 0,
                   "predictor_contexts": context_count, "probit_evaluations": context_count*6,
                   "census_share_renormalization_max_abs": float(np.abs(x-frame[["share_"+g for g in groups]].to_numpy()).max())}


@dataclass
class FittedMarginal:
    intercept_draws: np.ndarray
    group_effect_draws: np.ndarray
    election_effect_draws: np.ndarray
    group_election_draws: np.ndarray
    regional_effect_draws: np.ndarray
    regional_election_draws: np.ndarray
    regional_scale_draws: np.ndarray
    regional_election_scale_draws: np.ndarray
    kappa_draws: np.ndarray
    elections: list
    regional_ids: list
    predictive_seed: int
    diagnostics: dict
    metadata: dict
    predictive_kind: str = "new_locality_counts"
    election_scale_draws: np.ndarray | None = None
    group_election_scale_draws: np.ndarray | None = None

    def eta(self, election):
        if election not in self.elections:
            raise ValueError("Election not fitted; this benchmark does not forecast new elections")
        index = self.elections.index(election)
        return self.intercept_draws[:, None]+self.group_effect_draws+self.election_effect_draws[:, index, None]+self.group_election_draws[:, index, :]

    def group_mean_draws_for_election(self, election):
        scale = np.sqrt(self.regional_scale_draws**2+self.regional_election_scale_draws**2)
        return population_group_means(self.eta(election), scale)

    def regional_effects(self, election, regional_ids):
        labels = [str(value) for value in regional_ids]
        known = {value: i for i, value in enumerate(self.regional_ids)}
        effects = {}
        for label in set(labels):
            if label in known:
                index = known[label]
                effects[label] = self.regional_effect_draws[:, index]+self.regional_election_draws[:, self.elections.index(election), index]
            else:
                def normal(kind, year, scale):
                    token = (str(self.predictive_seed)+"|"+kind+"|"+year+"|"+label).encode()
                    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
                    return np.random.default_rng(seed).normal(size=len(scale))*scale
                effects[label] = normal("region", "", self.regional_scale_draws)+normal("region_year", election, self.regional_election_scale_draws)
        return np.column_stack([effects[label] for label in labels])

    def predict_election(self, election, x_test, eligible_test, *, seed, regional_ids, locality_ids=None):
        x, n = validate_arrays(x_test, eligible_test)
        if len(regional_ids) != len(x):
            raise ValueError("Regional IDs must match held-out localities")
        eta = self.eta(election)
        regional = self.regional_effects(election, regional_ids)
        rng = np.random.default_rng(seed)
        output = np.empty((len(eta), len(x)), dtype=np.int64)
        for start in range(0, len(x), 128):
            stop = min(start+128, len(x))
            group_rates = ndtr(eta[:, None, :]+regional[:, start:stop, None])
            p = np.clip(np.einsum("slg,lg->sl", group_rates, x[start:stop]), 1e-9, 1-1e-9)
            kappa = self.kappa_draws[:, None]
            locality_rate = rng.beta(kappa*p, kappa*(1-p))
            output[:, start:stop] = rng.binomial(n[start:stop][None, :], locality_rate)
        return output

    def save_parameters(self, path):
        arrays = {name: value for name, value in vars(self).items() if isinstance(value, np.ndarray)}
        info = {name: value for name, value in vars(self).items() if not isinstance(value, np.ndarray)}
        np.savez_compressed(path, **arrays, info=np.array(json.dumps(info)))


def load_parameters(path):
    with np.load(path, allow_pickle=False) as saved:
        return FittedMarginal(**{name: saved[name] for name in saved.files if name != "info"}, **json.loads(saved["info"].item()))


def sampling_options(budget):
    """Dense mass changes only the NumPyro sampler metric, not model priors."""
    options = {"draws": 500, "tune": 500, "chains": 2, "cores": 1, "target_accept": .95, **dict(budget)}
    allowed = {key: options[key] for key in ["draws", "tune", "chains", "cores", "target_accept"]}
    if "nuts_sampler" in options:
        allowed["nuts_sampler"] = options["nuts_sampler"]
    if options.get("dense_mass") and allowed.get("nuts_sampler") != "numpyro":
        raise ValueError("Dense-mass option requires the NumPyro backend")
    if allowed.get("nuts_sampler") == "numpyro":
        allowed["nuts_sampler_kwargs"] = {"chain_method": options.get("chain_method", "vectorized")}
        if options.get("dense_mass"):
            allowed["nuts_sampler_kwargs"]["dense_mass"] = True
    if allowed["chains"] < 2:
        raise ValueError("At least two chains are required")
    return allowed


def fit_joint(training_panel, *, seed, budget, group_names):
    import pymc as pm

    started = time.perf_counter()
    options = {"draws": 500, "tune": 500, "chains": 2, "cores": 1, "target_accept": .95, **dict(budget)}
    allowed = sampling_options(options)
    model, inputs = build_model(training_panel, group_names, options.get("priors"))
    with model:
        trace = pm.sample(**allowed, random_seed=seed, progressbar=False,
                          return_inferencedata=True, compute_convergence_checks=True)
    posterior = trace.posterior

    def values(name, *dims):
        return posterior[name].stack(sample=("chain", "draw")).transpose("sample", *dims).to_numpy()

    fitted = FittedMarginal(values("outcome_intercept"), values("outcome_group_effect", "group"),
        values("outcome_election_effect", "election"), values("outcome_election:group_effect", "election", "group"),
        values("outcome_subdistrict_effect", "subdistrict"), values("outcome_election:subdistrict_effect", "election", "subdistrict"),
        values("outcome_subdistrict_tau"), values("outcome_election:subdistrict_tau"), values("overdispersion_kappa"),
        inputs["elections"], inputs["regional_ids"], int(seed), {}, {**ADAPTER_METADATA, **inputs},
        election_scale_draws=values("outcome_election_tau"), group_election_scale_draws=values("outcome_election:group_tau"))
    means = np.stack([fitted.group_mean_draws_for_election(election) for election in fitted.elections], axis=1)
    global_variables = ["outcome_intercept", "outcome_group_effect", "outcome_group_tau", "outcome_election_effect",
                        "outcome_election_tau", "outcome_election:group_effect", "outcome_election:group_tau",
                        "outcome_subdistrict_tau", "outcome_election:subdistrict_tau", "overdispersion_kappa"]
    fitted.diagnostics = sampling_diagnostics(trace, global_variables, means, allowed, time.perf_counter()-started,
        pm.__version__, latent_variables=["outcome_subdistrict_effect", "outcome_election:subdistrict_effect"])
    return fitted
