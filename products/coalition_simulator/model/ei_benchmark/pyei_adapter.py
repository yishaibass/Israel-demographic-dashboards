"""Adapter for the pinned MIT-licensed PyEI multinomial--Dirichlet model.

The original model builder is loaded unmodified from a verified temporary source
cache. Native PyMC NUTS replaces RowByColumnEI.fit's hard-coded NumPyro sampler;
the likelihood and priors are unchanged. New-locality predictions are sampled
from the fitted Dirichlet hierarchy, never from fitted training-locality rates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

UPSTREAM_COMMIT = "214a24a7ca79c6f205b40dc6b720d98aaacdcca4"
MODEL_SHA256 = "02ad8516898ad8b399486a79cc459ef1b44a248eb26a05cdb3e118f3cadd67d5"
LICENSE_SHA256 = "f4aa6554cce32fadcd03c210cb8a4d9953fd524d83459e431d3a5471438cb88d"
ADAPTER_METADATA = {
    "name":"PyEI multinomial-dirichlet", "upstream_commit":UPSTREAM_COMMIT,
    "upstream_url":"https://github.com/mggg/ecological-inference",
    "upstream_model_path":"pyei/r_by_c_models.py", "license":"MIT",
    "model_file_sha256":MODEL_SHA256,"license_file_sha256":LICENSE_SHA256,
    "source_changes":[],
    "wrapper_changes":["Import the original model builder directly, avoiding unrelated plotting/Numba imports.",
                       "Default to native PyMC NUTS; budget nuts_sampler='numpyro' restores the upstream sampling backend, with explicit chain_method forwarding.",
                       "Record a deterministic group mean transform for convergence diagnostics; no prior/likelihood change.",
                       "Generate unseen-locality rates from posterior conc_params, rather than reusing fitted locality b."],
    "likelihood":"Observed voted/abstained counts ~ Multinomial(eligible, sum_group share * locality_group_rate).",
    "priors":"Unmodified upstream: conc_params[g,c] ~ Gamma(alpha=4,beta=1/2); locality rates ~ Dirichlet(conc_params).",
    "group_mean_definition":"Posterior superpopulation mean conc_params[g,voted]/sum_c conc_params[g,c]; not training-locality weighted summaries.",
    "prediction_definition":"New group rates independently drawn per unseen locality from the fitted Dirichlet hierarchy, then Binomial(eligible,demographic mixture).",
    "voted_category_index":0,"population_denominator":"eligible voters including nonvoters",
}


def source_directory():
    configured = os.environ.get("ISRAEL_PYEI_SOURCE_DIR")
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "israel-turnout-ei-3152" / ("pyei-source-"+UPSTREAM_COMMIT)


def load_upstream_builder():
    directory = source_directory()
    for name,expected in [("r_by_c_models.py",MODEL_SHA256),("LICENSE",LICENSE_SHA256)]:
        path = directory/name
        if not path.exists():
            raise FileNotFoundError(f"Pinned PyEI source cache missing {path}; supply original files from commit {UPSTREAM_COMMIT}")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("Pinned PyEI source/license hash mismatch: "+name)
    spec = importlib.util.spec_from_file_location("israel_pinned_pyei_r_by_c_models",directory/"r_by_c_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ei_multinom_dirichlet


def validate_predictors(x,eligible,groups=None):
    x = np.asarray(x,dtype=float)
    eligible = np.asarray(eligible)
    if x.ndim!=2 or eligible.ndim!=1 or len(x)!=len(eligible) or len(x)==0:
        raise ValueError("Expected nonempty x[n,groups] and eligible[n]")
    if groups is not None and x.shape[1]!=groups:
        raise ValueError("Prediction demographic group count differs from training")
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x>1) or not np.allclose(x.sum(axis=1),1,rtol=1e-5,atol=1e-8):
        raise ValueError("Demographic shares must be finite, nonnegative and sum to one; no silent renormalization")
    if not np.isfinite(eligible.astype(float)).all() or np.any(eligible<=0) or np.any(eligible!=np.floor(eligible.astype(float))):
        raise ValueError("Eligible population must contain positive integer counts")
    return x,eligible.astype(np.int64)


@dataclass
class FittedPyEI:
    concentration_draws: np.ndarray
    diagnostics: dict
    group_names: tuple
    locality_ids: tuple
    elapsed_seconds: float
    inference_data: object = None
    predictive_kind: str = "new_locality_counts"

    def __post_init__(self):
        self.concentration_draws = np.asarray(self.concentration_draws,dtype=float)
        a = self.concentration_draws
        if a.ndim!=3 or a.shape[2]!=2 or not np.isfinite(a).all() or np.any(a<=0):
            raise ValueError("Expected finite positive posterior concentrations[draw,group,2]")

    @property
    def group_mean_draws(self):
        return self.concentration_draws[:,:,0]/self.concentration_draws.sum(axis=2)

    def predict(self,x_test,eligible_test,*,seed):
        x,n = validate_predictors(x_test,eligible_test,self.concentration_draws.shape[1])
        rng = np.random.default_rng(seed)
        a = self.concentration_draws
        # Binary Dirichlet is exactly Beta; each unseen locality gets a new rate.
        rates = rng.beta(a[:,None,:,0],a[:,None,:,1],size=(len(a),len(x),a.shape[1]))
        probabilities = np.einsum("slg,lg->sl",rates,x)
        return rng.binomial(n[None,:],probabilities)

    def save_parameters(self,path):
        """Save only sufficient predictive hierarchy draws, without latent b or pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        metadata = {"adapter":ADAPTER_METADATA,"diagnostics":self.diagnostics,
                    "group_names":list(self.group_names),"locality_ids":np.asarray(self.locality_ids).tolist(),
                    "elapsed_seconds":self.elapsed_seconds}
        np.savez_compressed(path,concentration_draws=self.concentration_draws,
                            metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load_parameters(cls,path):
        with np.load(path,allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["adapter"]["upstream_commit"]!=UPSTREAM_COMMIT or metadata["adapter"]["model_file_sha256"]!=MODEL_SHA256:
                raise ValueError("Saved PyEI hierarchy uses a different upstream source")
            diagnostics = dict(metadata["diagnostics"])
            count_keys = ("rhat","ess_bulk","ess_tail")
            verified = (diagnostics.get("all_selected_finite") is True
                and all(diagnostics.get("diagnostic_value_counts",{}).get(key,0)>0 for key in count_keys)
                and all(diagnostics.get("nonfinite_counts",{}).get(key) == 0 for key in count_keys))
            if not verified:
                diagnostics.update(diagnostic_gate_passed=False,
                                   diagnostic_gate_unverified_reason="Selected-diagnostic finiteness was not verified; legacy caches cannot pass the convergence gate.")
            return cls(saved["concentration_draws"].copy(),diagnostics,tuple(metadata["group_names"]),
                       tuple(metadata["locality_ids"]),metadata["elapsed_seconds"])


def trace_diagnostics(trace):
    import arviz as az
    table = az.summary(trace,var_names=["conc_params","group_turnout_mean"],kind="diagnostics",round_to="none")
    latent = az.summary(trace,var_names=["b"],kind="diagnostics",round_to="none")
    stats = trace.sample_stats
    columns = ["r_hat","ess_bulk","ess_tail"]
    selected = table[columns].to_numpy(dtype=float)
    latent_selected = latent[columns].to_numpy(dtype=float)
    diagnostic_names = ["rhat","ess_bulk","ess_tail"]
    nonfinite = {name:int((~np.isfinite(selected[:,i])).sum()) for i,name in enumerate(diagnostic_names)}
    all_finite = bool(selected.size>0 and np.isfinite(selected).all())
    result = {"rhat_max":float(np.max(selected[:,0])),"ess_bulk_min":float(np.min(selected[:,1])),
              "ess_tail_min":float(np.min(selected[:,2])),
              "all_selected_finite":all_finite,"selected_diagnostic_values":int(selected.size),
              "diagnostic_value_counts":{name:len(selected) for name in diagnostic_names},
              "nonfinite_counts":nonfinite,
              "divergences":int(stats.diverging.sum()),
              "diagnostics_scope":"Global conc_params hierarchy and derived group_turnout_mean, governing new-locality predictions; latent b reported separately.",
              "rhat_max_all_latent":float(np.max(latent_selected[:,0])),
              "ess_bulk_min_all_latent":float(np.min(latent_selected[:,1])),
              "ess_tail_min_all_latent":float(np.min(latent_selected[:,2])),
              "all_latent_selected_finite":bool(latent_selected.size>0 and np.isfinite(latent_selected).all()),
              "nonfinite_latent_diagnostic_values":int((~np.isfinite(latent_selected)).sum()),
              "chains":int(trace.posterior.sizes["chain"]),"draws_per_chain":int(trace.posterior.sizes["draw"])}
    if "reached_max_treedepth" in stats:
        result["max_treedepth_hits"] = int(stats.reached_max_treedepth.sum())
    result["diagnostic_gate_passed"] = bool(all_finite and result["chains"]>=2 and np.isfinite(result["rhat_max"])
        and result["rhat_max"]<=1.01 and result["ess_bulk_min"]>=400 and result["ess_tail_min"]>=200 and result["divergences"]==0)
    return result


def sampling_engine_options(settings):
    sampler = settings.get("nuts_sampler","pymc")
    if sampler not in {"pymc","numpyro"}:
        raise ValueError("Supported PyEI NUTS engines are pymc and numpyro")
    if sampler=="pymc":
        return {"nuts_sampler":"pymc"}
    method = settings.get("chain_method","vectorized")
    if method not in {"vectorized","parallel"}:
        raise ValueError("NumPyro chain_method must be vectorized or parallel")
    return {"nuts_sampler":"numpyro","nuts_sampler_kwargs":{"chain_method":method}}


def load_parameters(path):
    """Module-level cache loader used by the shared benchmark harness."""
    return FittedPyEI.load_parameters(path)


def fit(x,eligible,voted,*,seed,budget=None,group_names=None,locality_ids=None):
    x,n = validate_predictors(x,eligible)
    y = np.asarray(voted)
    if y.shape!=n.shape or not np.isfinite(y.astype(float)).all() or np.any(y<0) or np.any(y>n) or np.any(y!=np.floor(y.astype(float))):
        raise ValueError("Voted must be integer counts between zero and eligible")
    y = y.astype(np.int64)
    fractions = np.vstack([y/n,(n-y)/n])
    if not np.array_equal(np.rint(fractions*n),np.vstack([y,n-y])):
        raise ValueError("Upstream count reconstruction would change observed counts")
    names = tuple(group_names) if group_names is not None else tuple(str(i) for i in range(x.shape[1]))
    ids = tuple(locality_ids) if locality_ids is not None else tuple(range(len(x)))
    if len(names)!=x.shape[1] or len(ids)!=len(x) or len(set(ids))!=len(ids):
        raise ValueError("Group/locality identities do not match model input")
    settings = dict(budget or {})
    engine_options = sampling_engine_options(settings)
    import pymc as pm
    builder = load_upstream_builder()
    started = time.perf_counter()
    model = builder(x.T,fractions,n)
    with model:
        concentration = model["conc_params"]
        pm.Deterministic("group_turnout_mean",concentration[:,0]/concentration.sum(axis=1))
        trace = pm.sample(draws=int(settings.get("draws",250)),tune=int(settings.get("tune",250)),
            chains=int(settings.get("chains",2)),cores=int(settings.get("cores",1)),
            target_accept=float(settings.get("target_accept",.95)),random_seed=int(seed),
            **engine_options,progressbar=False,return_inferencedata=True,
            compute_convergence_checks=True)
    posterior = trace.posterior["conc_params"].stack(sample=("chain","draw"))
    draws = np.moveaxis(posterior.values,-1,0)
    diagnostics = trace_diagnostics(trace)
    import pytensor
    diagnostics.update(python=platform.python_version(),pymc=pm.__version__,pytensor=pytensor.__version__,
                       pytensor_cxx=str(pytensor.config.cxx),
                       nuts_sampler=engine_options["nuts_sampler"],
                       chain_method=engine_options.get("nuts_sampler_kwargs",{}).get("chain_method"),
                       upstream_commit=UPSTREAM_COMMIT,training_localities=len(x),
                       training_eligible=int(n.sum()),runtime_seconds=time.perf_counter()-started)
    if engine_options["nuts_sampler"]=="numpyro":
        import jax,numpyro
        diagnostics.update(jax=jax.__version__,numpyro=numpyro.__version__,
                           jax_devices=[str(device) for device in jax.devices()],
                           jax_x64=bool(jax.config.x64_enabled))
    return FittedPyEI(draws,diagnostics,names,ids,diagnostics["runtime_seconds"],trace)
