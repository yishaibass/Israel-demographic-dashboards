"""Activate isolated research dependencies and verify a known posterior."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import urllib.request


def activate(dependencies: Path | None = None) -> Path:
    folder = dependencies or Path(os.environ.get(
        "TURNOUT_EI_DEPENDENCIES",
        str(Path(tempfile.gettempdir()) / "israel-turnout-ei-3152" / "site-packages"),
    ))
    if not (folder / "pymc").is_dir():
        raise FileNotFoundError("Isolated PyMC dependencies are missing: " + str(folder))
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=FAST_RUN")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return folder


def prepare_sources() -> list[dict]:
    """Fetch only pinned, hash-verified upstream code and complete MIT notices."""
    activate()
    root = str(Path(__file__).resolve().parents[4])
    if root not in sys.path:
        sys.path.insert(0, root)
    package = "products.coalition_simulator.model.ei_benchmark."
    rebane = importlib.import_module(package + "rebane_adapter")
    pyei = importlib.import_module(package + "pyei_adapter")
    sources = [
        (rebane.upstream_cache(), "rrebane/turnout-model-evaluation", rebane.UPSTREAM_COMMIT,
         [("turnout_models/models.py", "models.py", rebane.SOURCE_SHA256),
          ("LICENSE", "LICENSE", rebane.LICENSE_SHA256)]),
        (pyei.source_directory(), "mggg/ecological-inference", pyei.UPSTREAM_COMMIT,
         [("pyei/r_by_c_models.py", "r_by_c_models.py", pyei.MODEL_SHA256),
          ("LICENSE", "LICENSE", pyei.LICENSE_SHA256)]),
    ]
    verified = []
    for folder, repository, commit, files in sources:
        for upstream_path, filename, expected in files:
            path = folder / filename
            url = f"https://raw.githubusercontent.com/{repository}/{commit}/{upstream_path}"
            if path.exists():
                content = path.read_bytes()
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "israel-turnout-research"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Pinned upstream source/license hash mismatch: " + url)
            if not path.exists():
                folder.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            verified.append({"url": url, "sha256": expected, "license": "MIT"})
    return verified


def reference_check(dense_mass: bool = False) -> dict:
    """Check sampling against the exact Beta(62,42) conjugate posterior."""
    activate()
    import arviz as az
    import numpy as np
    import pymc as pm
    from scipy.stats import beta

    started = time.perf_counter()
    with pm.Model():
        probability = pm.Beta("probability", alpha=2, beta=2)
        pm.Binomial("voted", n=100, p=probability, observed=60)
        trace = pm.sample(
            draws=1000, tune=500, chains=4, cores=1, random_seed=42,
            progressbar=False, nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": "vectorized", "dense_mass": bool(dense_mass)},
        )
    samples = trace.posterior["probability"].values.reshape(-1)
    diagnostics = az.summary(trace, var_names=["probability"], kind="diagnostics", round_to="none")
    actual_mean, actual_variance = float(samples.mean()), float(samples.var(ddof=1))
    expected_mean, expected_variance = beta.stats(62, 42, moments="mv")
    quantile_error = float(np.max(np.abs(np.quantile(samples, [.025, .5, .975]) - beta.ppf([.025, .5, .975], 62, 42))))
    result = {
        "python": platform.python_version(),
        "dense_mass": bool(dense_mass),
        "packages": {name: metadata.version(name) for name in ["pymc", "pytensor", "arviz", "jax", "numpyro", "numpy", "scipy", "pandas"]},
        "pytensor_declared_python_support": metadata.metadata("pytensor").get("Requires-Python"),
        "experimental_metadata_override_required": sys.version_info[:2] >= (3, 15),
        "reference_posterior": "Beta(62,42): Beta(2,2) prior, 60 of 100 observed votes",
        "posterior_mean": actual_mean, "analytic_mean": float(expected_mean),
        "posterior_variance": actual_variance, "analytic_variance": float(expected_variance),
        "maximum_quantile_absolute_error": quantile_error,
        "rhat": float(diagnostics.r_hat.max()),
        "ess_bulk": float(diagnostics.ess_bulk.min()),
        "ess_tail": float(diagnostics.ess_tail.min()),
        "divergences": int(trace.sample_stats.diverging.sum()),
        "runtime_seconds": time.perf_counter() - started,
    }
    checks = {
        "mean": abs(actual_mean-float(expected_mean)) < .01,
        "variance": abs(actual_variance/float(expected_variance)-1) < .15,
        "quantiles": quantile_error < .01,
        "rhat": result["rhat"] <= 1.01,
        "bulk_ess": result["ess_bulk"] >= 400,
        "tail_ess": result["ess_tail"] >= 200,
        "divergences": result["divergences"] == 0,
    }
    result.update(checks=checks, passed=all(checks.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-upstream", action="store_true",
                        help="Fetch verified pinned model sources and complete MIT licenses into the temporary cache")
    parser.add_argument("--dense-mass", action="store_true", help="Check the actual dense-mass NumPyro runtime")
    args = parser.parse_args()
    sources = prepare_sources() if args.prepare_upstream else None
    result = reference_check(dense_mass=args.dense_mass)
    if sources is not None:
        result["verified_upstream_sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    if not result["passed"]:
        raise SystemExit("Analytic posterior/runtime checks failed")


if __name__ == "__main__":
    main()
