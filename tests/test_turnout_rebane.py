from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from products.coalition_simulator.model.ei_benchmark.rebane_adapter import (
    FittedRebane, build_inputs, integer_group_counts, population_group_means,
    diagnostic_metrics, sampling_diagnostics, load_parameters, validate_arrays, verify_upstream,
)
from products.coalition_simulator.model.ei_benchmark.rebane_joint import FittedJointRebane, load_parameters as load_joint_parameters


class RebaneAdapterTests(unittest.TestCase):
    def test_any_nonfinite_selected_diagnostic_is_explicit_and_fails(self):
        for rhat, bulk, tail, key in [([1., np.nan], [500., 600.], [300., 400.], "rhat"),
                                      ([1., 1.], [500., np.inf], [300., 400.], "ess_bulk"),
                                      ([1., 1.], [500., 600.], [300., np.nan], "ess_tail")]:
            with self.subTest(key=key):
                result = diagnostic_metrics(rhat, bulk, tail)
                self.assertFalse(result["all_selected_finite"])
                self.assertEqual(result["nonfinite_counts"][key], 1)
                self.assertEqual(result["diagnostic_value_counts"][key], 2)
        self.assertTrue(np.isnan(diagnostic_metrics([1., np.nan], [500.], [300.])["rhat_max"]))
        self.assertFalse(diagnostic_metrics([], [], [])["all_selected_finite"])

    def test_nonfinite_group_functional_cannot_hide_behind_finite_global_diagnostics(self):
        derived = object()
        def output(values):
            return SimpleNamespace(data_vars={"p": SimpleNamespace(to_numpy=lambda: np.asarray(values))})
        def rhat(data, **kwargs):
            return output([1., np.nan] if data is derived else [1., 1.])
        az = SimpleNamespace(rhat=rhat, ess=lambda data, method, **kwargs: output([500., 600.] if method == "bulk" else [300., 400.]))
        xr = SimpleNamespace(Dataset=lambda data: derived)
        idata = SimpleNamespace(posterior=SimpleNamespace(sizes={"chain": 2, "draw": 10}, data_vars={"p": None}),
                                sample_stats=SimpleNamespace(diverging=np.zeros((2, 10), dtype=bool)))
        with patch.dict("sys.modules", {"arviz": az, "xarray": xr}):
            result = sampling_diagnostics(idata, ["p"], np.full((20, 6), .5), {"chains": 2}, 0., "synthetic")
        self.assertFalse(result["all_selected_finite"])
        self.assertFalse(result["convergence_screen_pass"])
        self.assertEqual(result["nonfinite_counts"]["rhat"], 1)
        self.assertEqual(result["diagnostic_value_counts"]["rhat"], 4)
        self.assertTrue(np.isnan(result["rhat_max"]))

    def test_finite_selected_diagnostics_record_complete_counts(self):
        result = diagnostic_metrics([1., 1.005], [500., 600.], [300., 400.])
        self.assertTrue(result["all_selected_finite"])
        self.assertEqual(result["diagnostic_value_counts"], {"rhat": 2, "ess_bulk": 2, "ess_tail": 2})
        self.assertEqual(result["nonfinite_counts"], {"rhat": 0, "ess_bulk": 0, "ess_tail": 0})
        self.assertEqual(result["rhat_max"], 1.005)
        self.assertEqual(result["ess_bulk_min"], 500.)
        self.assertEqual(result["ess_tail_min"], 300.)

    def test_fractional_counts_preserve_every_registered_voter_and_margin_identity(self):
        x = np.array([[.1, .2, .2, .2, .2, .1], [.31, .09, .11, .19, .17, .13]])
        n = np.array([11, 101])
        counts = integer_group_counts(x, n)
        np.testing.assert_array_equal(counts.sum(axis=1), n)
        self.assertLess(np.abs(counts-x*n[:, None]).max(), 1)
        names = ["haredi", "dati", "masorti", "hiloni", "arab", "druze"]
        census, margins, _ = build_inputs(x, n, [8, 77], names, [10, 20])
        np.testing.assert_array_equal(census.groupby("locality").N.sum().to_numpy(), n)
        self.assertEqual(margins.set_index("locality").loc["20", "N"], 77)
        self.assertEqual(set(census.group), set(names))

    def test_new_locality_marginal_is_probit_normal_integral(self):
        eta = np.array([[0., .4, -.2, .8, -.5, .1]])
        scale = np.array([.7])
        rng = np.random.default_rng(11)
        integration = ndtr(eta+rng.normal(0, scale[0], size=(100000, 1))).mean(axis=0)
        np.testing.assert_allclose(population_group_means(eta, scale)[0], integration, atol=.002)

    def test_heldout_prediction_draws_observable_counts_without_outcomes(self):
        eta = np.tile(np.linspace(-.5, .5, 6), (500, 1))
        scale = np.full(500, .3)
        fitted = FittedRebane(eta, scale, population_group_means(eta, scale), {}, {})
        x = np.array([[1., 0, 0, 0, 0, 0], [.1, .2, .2, .2, .2, .1]])
        first = fitted.predict(x, [100, 200], seed=17)
        second = fitted.predict(x, [100, 200], seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (500, 2))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        self.assertTrue(((first >= 0) & (first <= np.array([100, 200]))).all())
        self.assertEqual(fitted.predictive_kind, "new_locality_counts")

    def test_incoherent_counts_and_shares_fail_before_sampling(self):
        x = np.full((2, 6), 1/6)
        for n, y in [([0, 100], [0, 70]), ([100, 100], [101, 70]), ([100, 100], [70.1, 70])]:
            with self.assertRaises(ValueError):
                validate_arrays(x, n, y)
        with self.assertRaises(ValueError):
            validate_arrays(x*2, [100, 100], [70, 70])

    def test_dependency_tampering_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "models.py").write_text("# changed dependency\n")
            (path / "LICENSE").write_text("# changed license\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_upstream(path)

    def joint(self):
        samples = 2000
        return FittedJointRebane(
            intercept_draws=np.zeros(samples), group_effect_draws=np.zeros((samples,6)),
            election_effect_draws=np.zeros((samples,2)), group_election_draws=np.zeros((samples,2,6)),
            locality_effect_draws=np.zeros((samples,1)), locality_scale_draws=np.zeros(samples),
            election_scale_draws=np.full(samples,.2), group_election_scale_draws=np.full(samples,.1),
            regional_effect_draws=np.zeros((samples,1)), regional_election_draws=np.zeros((samples,2,1)),
            regional_scale_draws=np.full(samples,.5), regional_election_scale_draws=np.zeros(samples),
            elections=["K21","K25"], locality_ids=["1"], regional_ids=["11"], predictive_seed=17,
            diagnostics={}, metadata={})

    def test_joint_predictions_share_new_region_effect(self):
        fitted = self.joint()
        x = np.tile([1.,0,0,0,0,0],(3,1))
        draws = fitted.predict_election("K25",x,[100000]*3,seed=1,regional_ids=["newA","newA","newB"],locality_ids=["a","b","c"])
        correlations = np.corrcoef(draws.T)
        self.assertGreater(correlations[0,1], .99)
        self.assertLess(abs(correlations[0,2]), .1)
        self.assertTrue(((draws>=0)&(draws<=100000)).all())

    def test_joint_future_prediction_requires_target_outcomes_absent(self):
        fitted = self.joint()
        for target in ["K21","K24","K25"]:
            with self.assertRaisesRegex(ValueError,"exclude the target"):
                fitted.predict_new_election_group_rates(target,np.eye(6),[100]*6,list(range(6)),["11"]*6,seed=1)
        rates = fitted.predict_new_election_group_rates("K26",np.eye(6),[100]*6,list(range(6)),["11"]*6,seed=1)
        self.assertEqual(rates.shape,(2000,6))
        self.assertTrue(((rates>=0)&(rates<=1)).all())

    def test_parameter_caches_reproduce_predictions_without_refitting(self):
        eta=np.zeros((20,6));scale=np.full(20,.2)
        native=FittedRebane(eta,scale,population_group_means(eta,scale),{"rhat_max":1.0},{"license":"MIT"})
        x=np.tile([1.,0,0,0,0,0],(2,1))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"native.npz"
            native.save_parameters(path)
            restored=load_parameters(path)
            np.testing.assert_array_equal(native.predict(x,[100,100],seed=3),restored.predict(x,[100,100],seed=3))
            joint=self.joint();path=Path(folder)/"joint.npz";joint.save_parameters(path)
            restored_joint=load_joint_parameters(path)
            np.testing.assert_array_equal(
                joint.predict_election("K25",x,[100,100],seed=3,regional_ids=["newA","newA"]),
                restored_joint.predict_election("K25",x,[100,100],seed=3,regional_ids=["newA","newA"]))


if __name__ == "__main__":
    unittest.main()
