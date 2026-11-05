import unittest
import numpy as np
import pandas as pd

from products.coalition_simulator.model.ei_benchmark.regional_empirical import FittedRegionalEmpirical, fit_joint
from products.coalition_simulator.model.ei_benchmark.benchmark import GROUPS, fit_wls_residual


class RegionalEmpiricalTests(unittest.TestCase):
    def state(self, samples=60000):
        state = {"coefficients": np.full(6, .5), "residuals": np.array([-.2, -.2, .2, .2]),
                 "weights": np.array([.1, .2, .3, .4]), "regions": np.array(["a", "a", "b", "b"]),
                 "regional_ids": np.array(["a", "b"]), "regional_weights": np.array([.3, .7])}
        return FittedRegionalEmpirical({"K25": state, "K26": state}, samples, 42)

    def test_residual_marginals_match_baseline_and_region_choice_is_shared(self):
        fit = self.state()
        x = np.tile([1., 0, 0, 0, 0, 0], (3, 1))
        draws = fit.predict_election("K26", x, np.full(3, 1000), seed=1, regional_ids=["same", "same", "other"])
        self.assertAlmostEqual((draws[:, 0] == 300).mean(), .3, delta=.008)
        np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
        self.assertLess(abs(np.corrcoef(draws[:, 0], draws[:, 2])[0, 1]), .02)
        np.testing.assert_allclose(fit.expected_rates_for_election("K26", x), .58)
        across_years = fit.predict_election("K25", x, np.full(3, 1000), seed=999, regional_ids=["same", "same", "other"])
        np.testing.assert_array_equal(draws, across_years)

    def test_clipping_counts_and_exact_analytic_mixture_mean(self):
        fit = self.state(1000)
        fit.elections["K26"]["coefficients"] = np.full(6, .95)
        x = np.eye(6)
        expected = .3*.75+.7*1.
        np.testing.assert_allclose(fit.expected_rates_for_election("K26", x), expected)
        first = fit.predict_election("K26", x, np.arange(1, 7), seed=7, regional_ids=["a"]*6)
        np.testing.assert_array_equal(first, fit.predict_election("K26", x, np.arange(1, 7), seed=7, regional_ids=["a"]*6))
        self.assertTrue(np.issubdtype(first.dtype, np.integer))
        self.assertTrue(((first >= 0) & (first <= np.arange(1, 7))).all())
        with self.assertRaises(ValueError):
            fit.predict_election("K26", x, np.arange(1, 7), seed=7, regional_ids=["a"])

    def test_fit_matches_baseline_and_other_election_outcomes_do_not_leak(self):
        rng = np.random.default_rng(5)
        x = rng.dirichlet(np.ones(6), 60)
        n = np.full(60, 10000)
        y = np.rint(n*(x@np.linspace(.3, .8, 6)+rng.normal(0, .01, 60))).astype(int)
        rows = []
        for election in ["K25", "K26"]:
            for i in range(60):
                rows.append({"election": election, "locality_code": i, "subdistrict_code": i//10,
                             "eligible_voters": n[i], "total_voted": y[i], **dict(zip(["share_"+g for g in GROUPS], x[i]))})
        panel = pd.DataFrame(rows)
        fitted = fit_joint(panel, seed=1, budget={"draws": 4000}, group_names=GROUPS)
        baseline = fit_wls_residual(x, n, y, seed=1, budget={"draws": 4000}, group_names=GROUPS, locality_ids=np.arange(60))
        np.testing.assert_allclose(fitted.group_mean_draws_for_election("K26"), baseline.group_mean_draws)
        np.testing.assert_allclose(fitted.expected_rates_for_election("K26", x), baseline.expected_rates(x), atol=1e-15)
        changed = panel.copy()
        changed.loc[changed.election.eq("K25"), "total_voted"] = 0
        refitted = fit_joint(changed, seed=1, budget={"draws": 4000}, group_names=GROUPS)
        np.testing.assert_array_equal(fitted.elections["K26"]["coefficients"], refitted.elections["K26"]["coefficients"])
        np.testing.assert_array_equal(fitted.elections["K26"]["residuals"], refitted.elections["K26"]["residuals"])


if __name__ == "__main__":
    unittest.main()
