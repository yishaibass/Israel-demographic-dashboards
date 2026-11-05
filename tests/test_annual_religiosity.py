import unittest

import numpy as np
import pandas as pd

from products.voting_propensity.model.annual_religiosity import (
    CATEGORIES, apply_relative_factors, bridge_historical_trends,
    fit_relative_trends, hes_adults, survey_holdout_validation, weighted_distribution,
)


def observed_fixture():
    rows = []
    for year in range(2016, 2024):
        for age, sex in [("all", "all"), ("18_24", "male")]:
            p = .1+.004*(year-2016)
            shares = {"haredi": p, "religious": .15, "traditional": .25, "mixed": .03, "other": .02, "unknown": .01, "secular": .54-p}
            for category in CATEGORIES:
                rows.append({"year": year, "age_band": age, "sex": sex, "population_group": "jewish",
                             "religiosity": category, "share": shares[category], "kish_effective_household_n": 1000.})
    return pd.DataFrame(rows)


class AnnualReligiosityTests(unittest.TestCase):
    def test_adult_join_uses_person_weights_and_preserves_other(self):
        households = pd.DataFrame({"s_seker": [2022, 2022], "misparmb": [1, 2], "ramatdatiyut": [5, None], "religion": [1, 5], "nationality": [1, 3]})
        persons = pd.DataFrame({"s_seker": [2022]*3, "misparmb": [1, 1, 2], "prat": [1, 2, 1],
                                "age_group": [4, 5, 15], "min": [1, 2, 1], "nationality": [1, 1, 3], "weight": [99., 2., 3.]})
        adults, audit = hes_adults(households, persons, 2022)
        self.assertEqual(adults.weight.tolist(), [2., 3.])
        self.assertEqual(adults.religiosity.tolist(), ["mixed", "unknown"])
        self.assertEqual(adults.population_group.tolist(), ["jewish", "other"])
        self.assertEqual(audit["weighted_adults"], 5.)
        with self.assertRaisesRegex(ValueError, "keys"):
            hes_adults(pd.concat([households, households.iloc[:1]]), persons, 2022)

    def test_precision_clusters_households_not_repeated_members(self):
        split = pd.DataFrame({"household": ["a", "a", "b"], "religiosity": ["haredi", "haredi", "secular"], "weight": [2., 2., 4.]})
        unsplit = split.iloc[[0, 2]].copy()
        unsplit.loc[0, "weight"] = 4.
        a = pd.DataFrame(weighted_distribution(split)).set_index("religiosity")
        b = pd.DataFrame(weighted_distribution(unsplit)).set_index("religiosity")
        self.assertEqual(a.loc["haredi", "share"], .5)
        self.assertEqual(a.loc["haredi", "kish_effective_household_n"], 2.)
        self.assertEqual(a.loc["haredi", "cluster_standard_error"], .5)
        np.testing.assert_allclose(a.cluster_standard_error, b.cluster_standard_error)
        self.assertAlmostEqual(a.share.sum(), 1.)

    def test_anchor_and_residual_factors_are_exact_one(self):
        factors = fit_relative_trends(observed_fixture())
        np.testing.assert_array_equal(factors.loc[factors.year.eq(2022), "relative_factor"], 1.)
        np.testing.assert_array_equal(factors.loc[factors.religiosity.isin(["other", "unknown"]), "relative_factor"], 1.)
        self.assertTrue(factors.loc[factors.age_band.ne("all"), "cell_weight_in_pooled_slope"].between(0, 1).all())
        self.assertTrue(factors.loc[factors.year.eq(2026)&factors.religiosity.eq("haredi"), "relative_factor"].gt(1).all())

    def test_apply_retains_locality_age_sex_total_and_residuals(self):
        factors = fit_relative_trends(observed_fixture())
        cells = pd.DataFrame({"locality_code": [1]*7, "subquarter_code": [1]*7, "age_band": ["18_24"]*7,
                              "sex": ["male"]*7, "population_group": ["jewish"]*7, "household_religiosity": CATEGORIES,
                              "person_weight": [10., 20., 15., 8., 2., 3., 4.]})
        anchor = apply_relative_factors(cells, factors, 2022)
        np.testing.assert_allclose(anchor.person_weight, cells.person_weight)
        changed = apply_relative_factors(cells, factors, 2026)
        self.assertAlmostEqual(changed.person_weight.sum(), cells.person_weight.sum())
        for category in ["other", "unknown"]:
            self.assertEqual(changed.loc[changed.household_religiosity.eq(category), "person_weight"].iloc[0], cells.loc[cells.household_religiosity.eq(category), "person_weight"].iloc[0])

    def test_historical_bridge_uses_actual_relative_ss_change(self):
        observed = observed_fixture()
        recent = fit_relative_trends(observed)
        rows = []
        for year in [2003, *range(2016, 2022)]:
            for group, p in [("haredi", .05 if year == 2003 else .1), ("religious", .15), ("traditional", .3), ("secular", .5 if year == 2003 else .45)]:
                rows.append({"year": year, "religiosity": group, "share": p})
        factors, overlap = bridge_historical_trends(recent, observed, pd.DataFrame(rows))
        early = factors.loc[factors.year.eq(2003)&factors.age_band.eq("all")&factors.religiosity.eq("haredi"), "relative_factor"].iloc[0]
        edge = recent.loc[recent.year.eq(2016)&recent.age_band.eq("all")&recent.religiosity.eq("haredi"), "relative_factor"].iloc[0]
        self.assertAlmostEqual(early, .5*edge)
        self.assertEqual(len(overlap), 6*4)
        post2019 = observed.loc[observed.year.ge(2019)]
        primary = fit_relative_trends(post2019, years=range(2019, 2027))
        bridged, primary_overlap = bridge_historical_trends(primary, post2019, pd.DataFrame(rows))
        self.assertEqual(set(primary_overlap.measurement_bridge_edge_year), {2019})
        self.assertEqual(len(primary_overlap), 3*4)
        self.assertTrue(bridged.loc[bridged.year.lt(2019), "measurement_bridge_edge_year"].eq(2019).all())

    def test_holdout_excludes2022_identity_baseline(self):
        validation = survey_holdout_validation(observed_fixture())
        self.assertEqual(set(validation.held_year), {2019, 2022, 2023})
        anchor = validation.loc[validation.held_year.eq(2022)]
        self.assertTrue(anchor.frozen2022_prediction.isna().all())
        self.assertTrue(anchor.frozen_absolute_error_pp.isna().all())
        self.assertTrue(anchor.status.eq("baseline_identity_excluded").all())

    def test_heldout_survey_values_cannot_change_heldout_prediction(self):
        observed = observed_fixture()
        a = survey_holdout_validation(observed)
        changed = observed.copy()
        changed.loc[changed.year.eq(2019)&changed.religiosity.eq("haredi"), "share"] += .1
        changed.loc[changed.year.eq(2019)&changed.religiosity.eq("secular"), "share"] -= .1
        b = survey_holdout_validation(changed)
        np.testing.assert_array_equal(a.loc[a.held_year.eq(2019), "pooled_trend_prediction"], b.loc[b.held_year.eq(2019), "pooled_trend_prediction"])


if __name__ == "__main__":
    unittest.main()
