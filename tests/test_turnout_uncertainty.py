import unittest

import numpy as np
import pandas as pd

from products.coalition_simulator.model.turnout_uncertainty import (
    GROUPS, SHARES, bootstrap, comovement, covariance_identity, draw_multiplicities,
    fit_panel, prepare_panel, summary_rows,
)


class TurnoutUncertaintyTests(unittest.TestCase):
    def frame(self):
        rows = []
        for t, election in enumerate(["K21", "K25", "K26"]):
            for group in range(6):
                for member in range(10):
                    rows.append({"election": election, "locality_code": group*10+member,
                                 "eligible_voters": 1000, "total_voted": 1000*(.3+.08*group+.002*member+.02*t),
                                 **dict(zip(SHARES, np.eye(6)[group]))})
        return pd.DataFrame(rows)

    def test_entire_geographic_block_receives_one_shared_draw_count(self):
        labels = np.repeat(np.arange(8), 5)
        counts = draw_multiplicities(labels, np.random.default_rng(42))
        for block in np.unique(labels):
            self.assertEqual(len(np.unique(counts[labels == block])), 1)
        self.assertEqual(sum(counts[labels == block][0] for block in np.unique(labels)), 8)

    def test_locality_history_pairing_preserves_common_temporal_increment(self):
        panel = prepare_panel(self.frame())
        counts = draw_multiplicities(np.arange(60), np.random.default_rng(3))
        estimates = fit_panel(panel, counts)
        np.testing.assert_allclose(estimates[1]-estimates[0], np.full(6, .02), atol=1e-12)
        np.testing.assert_allclose(estimates[2]-estimates[1], np.full(6, .02), atol=1e-12)

    def test_zero_weight_and_missing_group_rank_are_rejected(self):
        panel = prepare_panel(self.frame())
        with self.assertRaisesRegex(ValueError, "zero weight or rank-deficient"):
            fit_panel(panel, np.zeros(60))
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            fit_panel(panel, np.r_[np.ones(10), np.zeros(50)])

    def test_deterministic_repeat_and_six_group_probability_bounds(self):
        panel = prepare_panel(self.frame())
        first, a = bootstrap(panel, np.arange(60), 40, 17)
        second, b = bootstrap(panel, np.arange(60), 40, 17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(a, b)
        self.assertEqual(a["attempted"], a["accepted"]+a["rejected"])
        self.assertTrue(((first >= .2) & (first <= .95)).all())

    def test_analytic_paired_covariance_identity(self):
        rng = np.random.default_rng(1)
        common = rng.normal(0, .03, size=(1000, 1, 6))
        draws = common+rng.normal(0, .01, size=(1000, 3, 6))
        current, mean, covariance, change = covariance_identity(draws, 2, [0, 1])
        np.testing.assert_allclose(change, current+mean-2*covariance, atol=1e-12)
        self.assertTrue((change < current+mean).all())

    def test_shared_coefficient_error_does_not_inflate_historical_scatter(self):
        panel = prepare_panel(self.frame())
        point = fit_panel(panel)
        common = np.linspace(-.01, .01, 100)[:, None, None]
        samples = point[None, :, :]+common
        rows = summary_rows(panel, point, samples, "test")
        for row in rows:
            self.assertAlmostEqual(row["estimation_error_contribution_to_historical_variance_pp2"], 0., places=12)
            self.assertAlmostEqual(row["bootstrap_sd_change_pp"], 0., places=12)

    def test_composition_must_be_shared_across_years(self):
        frame = self.frame()
        frame.loc[0, "share_haredi"] = .9
        frame.loc[0, "share_dati"] = .1
        with self.assertRaisesRegex(ValueError, "shared fixed locality composition"):
            prepare_panel(frame)

    def test_comovement_matches_pearson_correlations_with_time_varying_weights(self):
        frame = self.frame()
        rng = np.random.default_rng(12)
        frame["eligible_voters"] *= 1+rng.uniform(0, 1, len(frame))
        frame["total_voted"] = frame.eligible_voters*(.5+rng.uniform(-.1, .1, len(frame)))
        panel = prepare_panel(frame)
        point = fit_panel(panel)
        labels = np.repeat(np.arange(6), 10)
        actual = comovement(panel, point, labels)
        residual = panel["turnout"]-point @ panel["x"].T
        residual -= residual.mean(axis=0)
        residual -= np.array([np.average(row, weights=w) for row,w in zip(residual, panel["weights"])])[:, None]
        correlations = np.corrcoef(residual.T)
        left, right = np.triu_indices(len(labels), 1)
        weights = panel["weights"].mean(axis=0)
        pair_weights = weights[left]*weights[right]
        within = labels[left] == labels[right]
        for mask, key in [(within, "weighted_mean_within_block_residual_pair_correlation"),
                          (~within, "weighted_mean_across_block_residual_pair_correlation")]:
            expected = np.average(correlations[left[mask], right[mask]], weights=pair_weights[mask])
            self.assertAlmostEqual(actual[key], expected, places=12)


if __name__ == "__main__":
    unittest.main()
