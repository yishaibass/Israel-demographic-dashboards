"""Exploratory training-residual regional dependence with unchanged WLS fits.

This resamples predictive errors, not coefficients or future national shocks.
Region selection is shared within each predicted region. Its marginal residual
mixture exactly equals eligible-weighted locality residual resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .rebane_adapter import validate_arrays


MODEL_DEFINITION = {
    "candidate": "regional_empirical",
    "status": "exploratory; specified before scores",
    "point_estimator": "Independent eligible-weighted bounded WLS per election, bounds .2/.95, unchanged from benchmark",
    "prediction": "For each draw and target subdistrict, select one training subdistrict by that election's eligible total; sample eligible-weighted locality residuals within that source region independently for target localities; clip fitted rate plus residual to [0,1] and round eligible times rate",
    "marginals": "Two-stage residual mixture equals the existing eligible-weighted empirical WLS residual distribution exactly; analytic clipped-mixture expected rates are identical",
    "dependence": "Source-region choice is shared by target-region members; stable seeded source-region uniforms are shared across elections. Residual draws are conditionally independent within the selected source region",
    "extra_binomial_noise": False,
    "posterior_group_uncertainty": False,
    "future_national_election_tail_claim": False,
    "limits": "Uses in-sample training residuals; assumes transportable regional error distributions and does not identify every spatial covariance or demographic individual rate",
    "practical_gate": "Exploratory comparison using prespecified benchmark value thresholds: at least5% lower regional WIS, improvement in at least2of3 folds, locality RMSE no more than5% worse. No MCMC diagnostics apply to empirical resampling",
}

ADAPTER_METADATA = MODEL_DEFINITION


def _uniforms(seed, label, samples):
    token = (str(seed)+"|source-region|"+str(label)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little")).random(samples)


@dataclass
class FittedRegionalEmpirical:
    elections: dict
    samples: int
    predictive_seed: int
    predictive_kind: str = "new_locality_counts"

    @property
    def diagnostics(self):
        return {"posterior": False, "method": MODEL_DEFINITION,
                "coefficient_uncertainty_included": False,
                "fit_scope": "Each election fitted independently; panel API carries geography and paired source-region randomness only"}

    def group_mean_draws_for_election(self, election):
        return self.elections[election]["coefficients"][None, :]

    def expected_rates_for_election(self, election, x):
        state = self.elections[election]
        expected = np.asarray(x) @ state["coefficients"]
        return np.clip(expected[:, None]+state["residuals"][None, :], 0, 1) @ state["weights"]

    def predict_election(self, election, x_test, eligible_test, *, seed, regional_ids, locality_ids=None):
        x, n = validate_arrays(x_test, eligible_test)
        labels = np.asarray(regional_ids).astype(str)
        if labels.shape != (len(x),):
            raise ValueError("Regional IDs must match target localities")
        state = self.elections[election]
        source_ids = state["regional_ids"]
        expected = x @ state["coefficients"]
        output = np.empty((self.samples, len(x)), dtype=np.int64)
        rng = np.random.default_rng(seed)
        for target in np.unique(labels):
            columns = np.flatnonzero(labels == target)
            choice = np.minimum(np.searchsorted(np.cumsum(state["regional_weights"]),
                                _uniforms(self.predictive_seed, target, self.samples), side="right"), len(source_ids)-1)
            for index, source in enumerate(source_ids):
                rows = np.flatnonzero(choice == index)
                if not len(rows):
                    continue
                mask = state["regions"] == source
                weights = state["weights"][mask]
                residuals = rng.choice(state["residuals"][mask], size=(len(rows), len(columns)), p=weights/weights.sum())
                output[np.ix_(rows, columns)] = np.rint(np.clip(expected[columns][None, :]+residuals, 0, 1)*n[columns][None, :]).astype(np.int64)
        return output


def fit_joint(training_panel, *, seed, budget, group_names):
    from .benchmark import fit_bounded_wls

    required = ["election", "locality_code", "subdistrict_code", "eligible_voters", "total_voted"]+["share_"+group for group in group_names]
    if set(required)-set(training_panel):
        raise ValueError("Training panel lacks required outcome, composition or geography fields")
    if training_panel.duplicated(["election", "locality_code"]).any() or training_panel.subdistrict_code.isna().any():
        raise ValueError("Unique election/locality rows and complete source geography required")
    states = {}
    for election, frame in training_panel.sort_values(["election", "locality_code"]).groupby("election"):
        x, n, y = validate_arrays(frame[["share_"+g for g in group_names]], frame.eligible_voters, frame.total_voted)
        fit = fit_bounded_wls(x, n, y, seed=seed, budget=budget, group_names=group_names, locality_ids=frame.locality_code)
        regions = frame.subdistrict_code.astype(str).to_numpy()
        names = np.unique(regions)
        weights = n/n.sum()
        states[election] = {"coefficients": fit.group_mean_draws[0], "residuals": y/n-x@fit.group_mean_draws[0],
                            "weights": weights, "regions": regions, "regional_ids": names,
                            "regional_weights": np.array([weights[regions == region].sum() for region in names])}
    return FittedRegionalEmpirical(states, max(1000, int(budget.get("draws", 4000))*int(budget.get("chains", 1))), int(seed))


def main():
    import pandas as pd
    from . import benchmark

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "geography", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    definition = {**MODEL_DEFINITION, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/"definition.json").write_text(json.dumps(definition, indent=2)+"\n", encoding="utf-8")
    started = time.perf_counter()
    result = benchmark.evaluate(pd.read_csv(args.panel), pd.read_csv(args.geography),
        joint_adapters={"regional_empirical": fit_joint}, seed=args.seed, budget={"draws": args.draws, "chains": 1})
    for key, value in result.items():
        if key != "audit":
            value.to_csv(args.output/(key+".csv"), index=False, encoding="utf-8-sig")
    result["audit"].update({"model_definition": definition, "runtime_python": platform.python_version(),
                           "runtime_seconds": time.perf_counter()-started,
                           "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                           "geography_sha256": hashlib.sha256(args.geography.read_bytes()).hexdigest()})
    (args.output/"report.json").write_text(json.dumps(benchmark.json_safe(result["audit"]), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(result["regional_aggregate_scores"].to_json(orient="records"))


if __name__ == "__main__":
    main()
