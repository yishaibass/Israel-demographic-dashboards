"""Synthetic checks of benchmark identification, geographic splits, and leakage."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from products.coalition_simulator.model.ei_benchmark.benchmark import (
    GROUPS, SHARES, JointElectionView, WLSResidualFit, balanced_panel, convergence_gate, evaluate, fit_bounded_wls, geographic_folds, validate_sampler_request, weighted_interval_score,
)


def synthetic():
    rng = np.random.default_rng(31)
    probabilities = np.linspace(.35, .85, 6)
    rows, geography = [], []
    for block in range(9):
        for j in range(8):
            code = block*100+j
            x = rng.dirichlet(np.ones(6))
            geography.append({"locality_code": code, "subdistrict_code": block})
            for election in ["K21", "K22", "K26"]:
                rows.append({"locality_code": code, "election": election, "eligible_voters": 100000,
                             "total_voted": int(round(100000*x@probabilities)), **dict(zip(SHARES, x))})
    return pd.DataFrame(rows), pd.DataFrame(geography), probabilities


class BenchmarkTests(unittest.TestCase):
    def test_joint_expected_rate_hook_preserves_native_fallback(self):
        class Native:
            def group_mean_draws_for_election(self, election):
                return np.full((20, 6), .5)
        class Empirical(Native):
            def expected_rates_for_election(self, election, x):
                return np.full(len(x), .625)
        x = np.eye(6)[:2]
        args = ("K26", np.array([11, 11]), np.array([1, 2]))
        np.testing.assert_array_equal(JointElectionView(Native(), *args).expected_rates(x), [.5, .5])
        np.testing.assert_array_equal(JointElectionView(Empirical(), *args).expected_rates(x), [.625, .625])

    def test_dense_mass_cannot_be_silently_ignored_by_an_unsupported_model(self):
        validate_sampler_request(["wls", "marginal_joint"], "numpyro", True)
        for models, backend in [(["marginal_joint"], "pymc"), (["pyei"], "numpyro"), (["rebane", "marginal_joint"], "numpyro"), (["wls"], "numpyro")]:
            with self.assertRaisesRegex(ValueError, "supported only"):
                validate_sampler_request(models, backend, True)

    def test_bounded_wls_recovers_synthetic_group_turnout(self):
        frame, _, truth = synthetic()
        one = frame.loc[frame.election.eq("K26")]
        fitted = fit_bounded_wls(one[SHARES].to_numpy(), one.eligible_voters.to_numpy(), one.total_voted.to_numpy(),
                                 seed=1, budget={}, group_names=GROUPS, locality_ids=one.locality_code.to_numpy())
        np.testing.assert_allclose(fitted.group_mean_draws[0], truth, atol=2e-5)

    def test_whole_subdistrict_and_all_election_histories_share_folds(self):
        frame, geo, _ = synthetic()
        panel, _ = balanced_panel(frame)
        split = geographic_folds(panel, geo)
        self.assertTrue(split.groupby("subdistrict_code").fold.nunique().eq(1).all())
        joined = frame.merge(split, on="locality_code")
        self.assertTrue(joined.groupby("locality_code").fold.nunique().eq(1).all())
        pd.testing.assert_frame_equal(split, geographic_folds(panel.sample(frac=1, random_state=2), geo.sample(frac=1, random_state=3)))

    def test_heldout_outcomes_do_not_affect_split_fit_or_prediction(self):
        frame, geo, _ = synthetic()
        first = evaluate(frame, geo, elections=["K26"], selected_folds=[0])
        heldout = set(first["splits"].loc[first["splits"].fold.eq(0), "locality_code"])
        changed = frame.copy()
        changed.loc[changed.locality_code.isin(heldout), "total_voted"] = 12345
        second = evaluate(changed, geo, elections=["K26"], selected_folds=[0])
        pd.testing.assert_frame_equal(first["splits"], second["splits"])
        np.testing.assert_array_equal(first["predictions"].predicted_count, second["predictions"].predicted_count)
        pd.testing.assert_frame_equal(first["group_summary"], second["group_summary"])
        self.assertNotEqual(first["scores"].weighted_mae_pp.iloc[0], second["scores"].weighted_mae_pp.iloc[0])

    def test_predictive_coverage_scores_heldout_counts_not_group_posteriors(self):
        frame, geo, _ = synthetic()

        class Prediction:
            predictive_kind = "new_locality_counts"
            group_mean_draws = np.full((40, 6), .01)  # Deliberately wrong group estimates.

            def predict(self, x, eligible, *, seed):
                return np.tile(np.where(np.arange(40)[:, None] < 20, 0, eligible), (1, 1))

        def fit(x, eligible, voted, *, seed, budget, group_names, locality_ids):
            return Prediction()

        result = evaluate(frame, geo, {"synthetic": fit}, elections=["K26"], selected_folds=[0])
        row = result["scores"].loc[result["scores"].model.eq("synthetic")].iloc[0]
        self.assertEqual(row.eligible_weighted_coverage_95, 1)
        self.assertEqual(row.eligible_weighted_width_95_pp, 100)
        baseline = result["scores"].loc[result["scores"].model.eq("bounded_wls")].iloc[0]
        self.assertTrue(pd.isna(baseline.eligible_weighted_coverage_95))

    def test_missing_geography_and_noninteger_observations_are_rejected(self):
        frame, geo, _ = synthetic()
        with self.assertRaisesRegex(ValueError, "requires a source subdistrict"):
            evaluate(frame, geo.iloc[1:])
        frame["total_voted"] = frame.total_voted.astype(float)
        frame.loc[0, "total_voted"] = 3.5
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            balanced_panel(frame)

    def test_common_mcmc_gate_requires_tail_ess_and_all_chain_diagnostics(self):
        good = {"rhat_max": 1.005, "ess_bulk_min": 500, "ess_tail_min": 300, "chains": 2, "divergences": 0,
                "all_selected_finite": True, "diagnostic_value_counts": {name: 6 for name in ["rhat", "ess_bulk", "ess_tail"]},
                "nonfinite_counts": {name: 0 for name in ["rhat", "ess_bulk", "ess_tail"]}}
        self.assertTrue(convergence_gate(good))
        for bad in [{"ess_tail_min": 199}, {"ess_bulk_min": 399}, {"rhat_max": float("nan")}, {"divergences": 1}, {"chains": 1}]:
            self.assertFalse(convergence_gate({**good, **bad}))
        self.assertIsNone(convergence_gate({}))
        self.assertFalse(convergence_gate({key: value for key, value in good.items() if key != "all_selected_finite"}))
        self.assertFalse(convergence_gate({**good, "all_selected_finite": False,
                                           "nonfinite_counts": {"rhat": 1, "ess_bulk": 0, "ess_tail": 0}}))

    def test_weighted_interval_score_matches_analytic_median_and_tail_penalties(self):
        intervals = {.5: (np.array([8., 8.]), np.array([12., 12.])),
                     .8: (np.array([5., 5.]), np.array([15., 15.])),
                     .95: (np.array([0., 0.]), np.array([20., 20.]))}
        np.testing.assert_allclose(weighted_interval_score([12., 16.], [10., 10.], intervals), [1., 3.])

    def test_vacuous_intervals_are_penalized_even_with_full_coverage(self):
        sharp = {level: (np.array([9.]), np.array([11.])) for level in [.5, .8, .95]}
        vacuous = {level: (np.array([0.]), np.array([100.])) for level in [.5, .8, .95]}
        self.assertGreater(weighted_interval_score([10.], [10.], vacuous).item(),
                           weighted_interval_score([10.], [10.], sharp).item())

    def test_joint_adapter_fits_once_and_never_sees_heldout_history(self):
        frame, geo, _ = synthetic()
        calls = []

        class Joint:
            predictive_kind = "new_locality_counts"
            diagnostics = {}

            def group_mean_draws_for_election(self, election):
                return np.full((40, 6), .5)

            def predict_election(self, election, x, eligible, *, seed, regional_ids, locality_ids):
                self_test.assertEqual(len(regional_ids), len(x))
                self_test.assertEqual(len(locality_ids), len(x))
                self_test.assertFalse(set(regional_ids)&set(calls[0].subdistrict_code))
                return np.random.default_rng(seed).binomial(eligible, .5, size=(40, len(x)))

        def fit_joint(training, *, seed, budget, group_names):
            calls.append(training.copy())
            return Joint()

        self_test = self
        result = evaluate(frame, geo, elections=["K21", "K26"], selected_folds=[0], joint_adapters={"joint": fit_joint})
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0].election), {"K21", "K22", "K26"})
        heldout = set(result["splits"].loc[result["splits"].fold.eq(0), "locality_code"])
        self.assertFalse(set(calls[0].locality_code)&heldout)
        self.assertEqual(len(result["fold_scores"].loc[result["fold_scores"].model.eq("joint")]), 2)

    def test_regional_intervals_sum_paired_draws_and_preserve_shared_shocks(self):
        frame, geo, _ = synthetic()

        class Shared:
            predictive_kind = "new_locality_counts"
            group_mean_draws = np.full((1000, 6), .5)

            def predict(self, x, eligible, *, seed):
                return np.rint(np.where(np.arange(1000)[:, None] % 2, .2, .8)*eligible)

        def fit(*args, **kwargs):
            return Shared()

        result = evaluate(frame, geo, {"shared": fit}, elections=["K26"], selected_folds=[0])
        regional = result["regional_predictions"].loc[result["regional_predictions"].model.eq("shared")]
        np.testing.assert_allclose((regional.upper_95-regional.lower_95)/regional.eligible_voters, .6)
        original = frame.loc[frame.election.eq("K26")].merge(geo, on="locality_code").groupby("subdistrict_code").total_voted.sum()
        np.testing.assert_array_equal(regional.observed_count, original.loc[regional.subdistrict_code])

    def test_posterior_cache_avoids_refit_but_rejects_changed_training_counts(self):
        frame, geo, _ = synthetic()
        calls = []

        class Saved:
            predictive_kind = "expected_counts"
            group_mean_draws = np.full((1, 6), .5)

            def predict(self, x, eligible, *, seed):
                return eligible[None, :]*.5

            def save_parameters(self, path):
                np.savez_compressed(path, group_mean_draws=self.group_mean_draws)

        def fitter(*args, **kwargs):
            calls.append(True)
            return Saved()

        def loader(path):
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["group_mean_draws"], Saved.group_mean_draws)
            return Saved()

        with tempfile.TemporaryDirectory() as folder:
            options = dict(elections=["K26"], selected_folds=[0], cache_directory=Path(folder),
                           cache_loaders={"cached": loader})
            first = evaluate(frame, geo, {"cached": fitter}, **options)
            second = evaluate(frame, geo, {"cached": fitter}, reuse_cache=True, prediction_seed_offset=7, **options)
            self.assertEqual(len(calls), 1)
            pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
            heldout = set(first["splits"].loc[first["splits"].fold.eq(0), "locality_code"])
            changed = frame.copy()
            index = changed.index[changed.election.eq("K26") & ~changed.locality_code.isin(heldout)][0]
            changed.loc[index, "total_voted"] += 1
            with self.assertRaisesRegex(ValueError, "Posterior cache context differs"):
                evaluate(changed, geo, {"cached": fitter}, reuse_cache=True, **options)

    def test_point_predictions_do_not_depend_on_predictive_monte_carlo_seed(self):
        frame, geo, _ = synthetic()
        first = evaluate(frame, geo, elections=["K26"], selected_folds=[0])
        second = evaluate(frame, geo, elections=["K26"], selected_folds=[0], prediction_seed_offset=9)
        np.testing.assert_array_equal(first["predictions"].predicted_count, second["predictions"].predicted_count)

    def test_empirical_residual_point_mean_integrates_clipping_exactly(self):
        fitted = WLSResidualFit(np.full((1, 6), .9), np.array([-.2, .3]), np.array([.25, .75]), 1000)
        np.testing.assert_allclose(fitted.expected_rates(np.eye(6)[:1]), [.25*.7+.75*1.])


if __name__ == "__main__":
    unittest.main()
