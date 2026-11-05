import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.special import ndtr

from products.coalition_simulator.model.ei_benchmark.marginal_adapter import FittedMarginal, context_indices, load_parameters, sampling_options


def fitted(samples=4000, region_scale=0., kappa=50.):
    return FittedMarginal(np.zeros(samples), np.zeros((samples, 6)), np.zeros((samples, 2)),
        np.zeros((samples, 2, 6)), np.zeros((samples, 1)), np.zeros((samples, 2, 1)),
        np.full(samples, region_scale), np.zeros(samples), np.full(samples, kappa),
        ["K25", "K26"], ["known"], 19, {}, {})


class MarginalAdapterTests(unittest.TestCase):
    def test_dense_mass_is_forwarded_only_to_numpyro_sampler_configuration(self):
        options = sampling_options({"nuts_sampler": "numpyro", "dense_mass": True, "chain_method": "vectorized"})
        self.assertEqual(options["nuts_sampler_kwargs"], {"chain_method": "vectorized", "dense_mass": True})
        self.assertNotIn("dense_mass", sampling_options({"nuts_sampler": "numpyro"})["nuts_sampler_kwargs"])
        with self.assertRaisesRegex(ValueError, "requires the NumPyro"):
            sampling_options({"nuts_sampler": "pymc", "dense_mass": True})

    def test_context_compression_reconstructs_dense_nonzero_predictor_in_original_order(self):
        regions = np.array([2, 0, 2, 1, 0, 2, 1, 0])
        elections = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        rng = np.random.default_rng(42)
        group = rng.normal(size=6)
        region = rng.normal(size=3)
        year = rng.normal(size=2)
        group_year = rng.normal(size=(2, 6))
        region_year = rng.normal(size=(2, 3))
        dense = group[None, :]+region[regions, None]+year[elections, None]+group_year[elections]+region_year[elections, regions, None]
        indices, inverse = context_indices(regions, elections)
        compressed = (group[indices["group"]]+region[indices["subdistrict"]]+year[indices["election"]]
                      +group_year[indices["election"], indices["group"]]+region_year[indices["election"], indices["subdistrict"]]).reshape(-1, 6)
        np.testing.assert_allclose(ndtr(compressed)[inverse], ndtr(dense), atol=1e-15)
        self.assertLess(len(compressed), len(regions))

    def test_integrated_beta_binomial_has_expected_mean_and_overdispersion(self):
        model = fitted(samples=50000)
        draws = model.predict_election("K26", np.eye(6)[:1], np.array([1000]), seed=32, regional_ids=["new"])
        self.assertAlmostEqual(draws.mean(), 500, delta=2)
        expected = 1000*.5*.5*(1000+50)/(1+50)
        self.assertAlmostEqual(draws.var(), expected, delta=expected*.04)
        self.assertGreater(draws.var(), 10*1000*.5*.5)

    def test_same_region_effect_is_shared_and_different_regions_are_independent(self):
        model = fitted(region_scale=.5)
        effects = model.regional_effects("K26", ["A", "A", "B"])
        np.testing.assert_array_equal(effects[:, 0], effects[:, 1])
        self.assertLess(abs(np.corrcoef(effects[:, 0], effects[:, 2])[0, 1]), .06)

    def test_predictive_draws_are_counts_and_unknown_elections_are_rejected(self):
        model = fitted()
        x = np.full((2, 6), 1/6)
        first = model.predict_election("K26", x, [100, 200], seed=3, regional_ids=["A", "B"])
        np.testing.assert_array_equal(first, model.predict_election("K26", x, [100, 200], seed=3, regional_ids=["A", "B"]))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        self.assertTrue(((first >= 0) & (first <= [100, 200])).all())
        with self.assertRaisesRegex(ValueError, "Election not fitted"):
            model.eta("K27")

    def test_posterior_cache_roundtrip_preserves_predictions(self):
        model = fitted()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"posterior.npz"
            model.save_parameters(path)
            restored = load_parameters(path)
            args = ("K26", np.eye(6)[:1], [1000])
            np.testing.assert_array_equal(model.predict_election(*args, seed=4, regional_ids=["A"]),
                                          restored.predict_election(*args, seed=4, regional_ids=["A"]))

    def test_weak_kappa_prior_spans_material_overdispersion_without_fixing_it(self):
        prior = np.random.default_rng(7).lognormal(np.log(50), 1.5, 100000)
        self.assertAlmostEqual(np.median(prior), 50, delta=1)
        self.assertLess(np.quantile(prior, .025), 3)
        self.assertGreater(np.quantile(prior, .975), 900)


if __name__ == "__main__":
    unittest.main()
