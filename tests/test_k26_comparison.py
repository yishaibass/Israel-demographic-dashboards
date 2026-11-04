import unittest

import numpy as np
import pandas as pd

from products.coalition_simulator.model.k26_comparison import GROUPS, HISTORICAL, SHARES, estimate, standardized_change


class K26ComparisonTests(unittest.TestCase):
    def inputs(self):
        a = np.array([.5, .55, .6, .65, .7, .75])
        k = np.array([.2, .3, .4, .5, .6, .7])
        ballots, historical, baseline = [], [], []
        for i, group in enumerate(GROUPS):
            historical.append({"election": "K25", "locality_code": i + 1, "eligible_voters": 1000,
                               "is_envelope": False, "demog_source": "PUF",
                               **dict(zip(SHARES, np.eye(6)[i]))})
            ballots.append({"snapshot_id": "test", "locality_code": i + 1, "external_envelope": False,
                            "eligible_voters": 1000, "total_voted": round(1000*a[i]), "invalid_votes": 0,
                            "valid_votes": round(1000*a[i]), "votes_coalition": round(1000*a[i]*k[i]),
                            "votes_other": round(1000*a[i]*(1-k[i]))})
            baseline.append({"group": group,
                             **{"theta_vote_" + e: .5 + j*.01 for j, e in enumerate(HISTORICAL)},
                             **{"theta_bibi_" + e: .4 for e in HISTORICAL},
                             "theta_vote_mean": .52, "theta_vote_sd": np.std([.5, .51, .52, .53, .54]),
                             "theta_bibi_mean": .4, "theta_bibi_sd": 0})
        dimension = pd.DataFrame({"ballot_symbol": ["coalition", "other"],
                                  "bloc_core": ["Netanyahu bloc", "Other"],
                                  "amcha_included_in_coalition": [False, False]})
        return pd.DataFrame(ballots), dimension, pd.DataFrame(historical), pd.DataFrame(baseline), {"snapshot_id": "test", "count_status": "partial"}, a, k

    def test_recovers_group_rates_and_preserves_frozen_baseline(self):
        ballots, dimension, historical, baseline, readiness, a, k = self.inputs()
        saved = baseline.copy(deep=True)
        result = estimate(ballots, dimension, historical, baseline, readiness, bootstrap=12)
        for index, group in enumerate(GROUPS):
            turnout = next(row for row in result["rows"] if row["group"] == group and row["parameter"] == "theta_vote")
            support = next(row for row in result["rows"] if row["group"] == group and row["parameter"] == "theta_bibi")
            self.assertAlmostEqual(turnout["k26_estimate"], a[index])
            self.assertAlmostEqual(support["k26_estimate"], k[index])
            self.assertAlmostEqual(turnout["change_vs_mean_sd"], (a[index]-.52)/np.std([.5, .51, .52, .53, .54]))
            self.assertIsNone(support["change_vs_mean_sd"])
            self.assertTrue(support["sd_unavailable_reason"])
            self.assertTrue(0 <= support["bootstrap_p10"] <= support["bootstrap_p90"] <= 1)
        self.assertAlmostEqual(result["fit"]["turnout_r2"], 1)
        self.assertAlmostEqual(result["fit"]["coalition_per_eligible_r2"], 1)
        pd.testing.assert_frame_equal(saved, baseline)

    def test_envelopes_and_unmatched_localities_only_enter_national_totals(self):
        ballots, dimension, historical, baseline, readiness, _, _ = self.inputs()
        extras = pd.DataFrame([
            {"snapshot_id": "test", "locality_code": 9000, "external_envelope": True, "eligible_voters": 0,
             "total_voted": 100, "invalid_votes": 0, "valid_votes": 100, "votes_coalition": 25, "votes_other": 75},
            {"snapshot_id": "test", "locality_code": 999, "external_envelope": False, "eligible_voters": 200,
             "total_voted": 100, "invalid_votes": 0, "valid_votes": 100, "votes_coalition": 25, "votes_other": 75},
        ])
        total_coalition = int(ballots.votes_coalition.sum()) + 50
        result = estimate(pd.concat([ballots, extras], ignore_index=True), dimension, historical, baseline, readiness, bootstrap=2)
        self.assertEqual(result["coverage"]["k26_localities_matched"], 6)
        self.assertEqual(result["coverage"]["k26_envelope_rows_excluded"], 1)
        self.assertEqual(result["coverage"]["k26_unmatched_locality_codes"], [999])
        self.assertEqual(result["national_observed"]["coalition_votes"], total_coalition)

    def test_snapshot_and_party_exhaustiveness_are_enforced(self):
        ballots, dimension, historical, baseline, readiness, _, _ = self.inputs()
        with self.assertRaisesRegex(ValueError, "snapshots differ"):
            estimate(ballots, dimension, historical, baseline, {**readiness, "snapshot_id": "wrong"}, bootstrap=2)
        with self.assertRaisesRegex(ValueError, "missing symbols"):
            estimate(ballots, dimension.iloc[:1], historical, baseline, readiness, bootstrap=2)

    def test_nearly_zero_sd_is_unavailable(self):
        self.assertEqual(standardized_change(.9, 1, 1e-16)[0], None)
        self.assertEqual(standardized_change(.1, 0, 1e-9)[0], None)


if __name__ == "__main__":
    unittest.main()
