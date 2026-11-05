"""Synthetic contracts for controlled population/history comparisons."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"products/voting_propensity/model"))
import fit_foundation_history as f


def synthetic(election="K25"):
    cells = pd.DataFrame({name: np.zeros(4) for name in f.FEATURES})
    cells["intercept"] = 1.
    cells["arab"] = [0., 1., 0., 1.]
    cells["fsu_proxy"] = 0.
    cells["weight"] = [60., 40., 50., 50.]
    cells["population_weight"] = cells.weight
    cells["g"] = [0, 0, 1, 1]
    cells["locality"] = [1, 1, 2, 2]
    cells["profile_group"] = ["hiloni", "arab", "hiloni", "arab"]
    geos = pd.DataFrame({"locality": [1, 2], "geo_key": ["1", "2"]})
    target = pd.DataFrame({"locality": [1, 2], "geo_key": ["1", "2"], "eligible": [100, 100],
                           "voted": [70, 65], "valid": [68, 63], "election": election})
    for j, family in enumerate(f.FAMILIES):
        target[family] = [68/9, 63/9]
    return f.ElectionData(election, cells, geos, target,
                          [{"id": family, "families": [family]} for family in f.FAMILIES], f.FAMILIES)


class FoundationTests(unittest.TestCase):
    def test_joint_partition_exhaustive_not_fake_split(self):
        parts = [{"id": "joint", "families": ["likud", "secular_national_right"]},
                 {"id": "other", "families": ["other"]}]
        matrix = f.validate_partition(parts, ["likud", "secular_national_right", "other"])
        p = torch.full((1, 9), 1/9)
        p = f.normalized_choice(p, ["likud", "secular_national_right", "other"])
        np.testing.assert_allclose((p@torch.tensor(matrix)).numpy(), [[2/3, 1/3]])
        self.assertEqual(float(p[0, f.FAMILIES.index("shas")]), 0.)
        with self.assertRaises(ValueError):
            f.validate_partition(parts + [{"id": "overlap", "families": ["likud"]}], ["likud", "secular_national_right", "other"])

    def test_same_design_matches_unmodified_estimator(self):
        data = synthetic()
        fitted, _ = f.fit_history([data], {1, 2}, 3, 19)
        with f.election_scope(["K25"]):
            original, _ = f.common.model.fit(data.cells, data.geos, data.targets, f.FEATURES, False, {1, 2}, .05, 3, 19)
        for name, value in fitted.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[name], atol=2e-7, rtol=2e-5)

    def test_heldout_outcomes_cannot_change_fit(self):
        first = synthetic()
        second = copy.deepcopy(first)
        second.targets.loc[1, "voted"] = 80
        second.targets.loc[1, f.FAMILIES] = [55] + [1]*8
        a, _ = f.fit_history([first], {1}, 3, 12)
        b, _ = f.fit_history([second], {1}, 3, 12)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)

    def test_annual_group_weights_preserve_unknowns_by_exclusion(self):
        micro = pd.DataFrame({"SmlYishuvPUF": [1, 1, 2, 2, 3], "KvutzaUchlusiyaPUF": [1, 2, 1, 0, 1],
                              "MishkalPratPUF": [10., 10., 15., 5., 20.]})
        margins = pd.DataFrame([{"year": 2020, "locality_code": loc, "dimension": "population_group", "category": cat, "value": value}
                                for loc in [1, 2, 3] for cat, value in [("arab", 70), ("jewish_and_other", 30)]])
        result, audit = f.calibrate_annual_weights(micro, margins, 2020)
        self.assertEqual(set(result.SmlYishuvPUF), {1})
        np.testing.assert_allclose(result.MishkalPratPUF, [6, 14])
        self.assertEqual(audit["excluded_unsupported_or_unknown_localities"], 2)
        self.assertEqual(f.calibrate_annual_weights(micro, margins, 2026)[1]["status"], "unsupported_year")

    def test_each_election_uses_own_weights(self):
        first = synthetic("K24")
        second = synthetic("K25")
        second.cells.loc[:, "weight"] = [10., 90., 80., 20.]
        a = first.tensors()
        b = second.tensors()
        self.assertFalse(torch.equal(a["w"], b["w"]))
        with f.election_scope(["K24", "K25"]):
            model = f.common.model.FamilyModel(24, 9, False)
        with torch.no_grad():
            model.turn_shared[f.FEATURES.index("arab")] = 1.
        pred_a, _ = f.predicted(model, first, a, 0)
        pred_b, _ = f.predicted(model, second, b, 1)
        self.assertGreater(float(pred_b[0].detach()), float(pred_a[0].detach()))

    def test_age_calibration_keeps_ethnic_margin_and_matches_fractional_age(self):
        micro = pd.DataFrame({"SmlYishuvPUF": [1]*6, "KvutzaUchlusiyaPUF": [1]*3+[2]*3,
                              "MinPUF": [1, 2, 1]*2, "GilPUF": [3, 6, 7]*2, "MishkalPratPUF": [10.]*6})
        margins = pd.DataFrame([{"year": 2022, "locality_code": 1, "dimension": "retirement_age", "category": category, "value": value}
                                for category, value in [("working_age_adult", 60), ("retirement_age", 40)]])
        result, audit = f.calibrate_retirement(micro, margins, 2022)
        self.assertEqual(audit["localities"], 1)
        self.assertAlmostEqual(float(result.MishkalPratPUF.sum()), 100., places=5)
        self.assertAlmostEqual(float((result.MishkalPratPUF*f.retirement_score(result)).sum()), 40., places=5)
        self.assertAlmostEqual(float(result.loc[result.KvutzaUchlusiyaPUF.eq(2), "MishkalPratPUF"].sum()), 50., places=5)
        self.assertEqual(f.retirement_score(micro).tolist(), [0, .5, 1, 0, .5, 1])

    def test_exact65_age_margin_preferred_and_legacy_scenario_separate(self):
        micro = pd.DataFrame({"SmlYishuvPUF": [1]*6, "KvutzaUchlusiyaPUF": [1]*3+[2]*3,
                              "MinPUF": [1, 2, 1]*2, "GilPUF": [3, 6, 7]*2, "MishkalPratPUF": [10.]*6})
        rows = [("adult_age", "age18_64", 80), ("adult_age", "age65_plus", 20),
                ("retirement_age", "working_age_adult", 60), ("retirement_age", "retirement_age", 40)]
        margins = pd.DataFrame([dict(year=2026, locality_code=1, dimension=d, category=c, value=v) for d,c,v in rows])
        exact, audit = f.calibrate_retirement(micro, margins, 2026)
        self.assertIn("exact_65plus", audit["status"])
        self.assertAlmostEqual(float(exact.loc[exact.GilPUF.eq(7), "MishkalPratPUF"].sum()), 20., places=5)
        legacy, audit = f.calibrate_retirement(micro, margins, 2026, prefer_exact_age=False)
        self.assertIn("fractional", audit["status"])
        self.assertAlmostEqual(float((legacy.MishkalPratPUF*f.retirement_score(legacy)).sum()), 40., places=5)

    def test_anchored_growth_preserves2022_adult_base_and_zerozero(self):
        micro = pd.DataFrame({"SmlYishuvPUF": [1,1,2,2,3,3], "KvutzaUchlusiyaPUF": [1,2]*3,
                              "MishkalPratPUF": [12.,8.]*3, "GilPUF": [3]*6, "MinPUF": [1]*6})
        rows = []
        for year in [2022, 2026]:
            for locality in [1,2,3]:
                counts = {"jewish_and_other": 30., "arab": 70.} if locality == 1 else {"jewish_and_other": 0., "arab": 0.}
                if year == 2026 and locality == 1:
                    counts = {"jewish_and_other": 60., "arab": 70.}
                if year == 2026 and locality == 3:
                    counts["arab"] = 5.
                rows.extend(dict(year=year, locality_code=locality, dimension="population_group", category=cat, value=value, unit="persons") for cat,value in counts.items())
        margins = pd.DataFrame(rows)
        anchor, _ = f.calibrate_growth_weights(micro, margins, 2022)
        np.testing.assert_array_equal(anchor.MishkalPratPUF, micro.MishkalPratPUF)
        updated, audit = f.calibrate_growth_weights(micro, margins, 2026)
        np.testing.assert_allclose(updated.MishkalPratPUF, [24,8,12,8])
        self.assertEqual(audit["positive_year_over_zero_anchor_localities"], [3])
        self.assertEqual(audit["zero_zero_growth_factor_one_cells"], 2)

    def test_joint_family_profiles_never_export_invented_components(self):
        data = synthetic("K19")
        data.partition = [{"id": "likud__secular_national_right", "families": ["likud", "secular_national_right"]}]+[
            {"id": family, "families": [family]} for family in f.FAMILIES if family not in ["likud", "secular_national_right"]]
        with f.election_scope(["K19"]):
            model = f.common.model.FamilyModel(24, 9, False)
        rows = f.subgroup_profiles(model, [data], data, "fixed_2022_common_reference")
        self.assertFalse(any(row["outcome"] in ["likud", "secular_national_right"] for row in rows))
        joint = next(row for row in rows if row["outcome"] == "likud__secular_national_right")
        self.assertEqual(joint["party_target_support"], "joint_family_aggregate_target_only")
        profiles = pd.DataFrame([dict(variant="test", estimand="reference", group="arab", outcome="likud", election=e, value=.2)
                                 for e in ["K16", "K20", "K26"]] + [dict(variant="test", estimand="reference", group="arab", outcome="turnout", election=e, value=.6)
                                 for e in ["K16", "K19", "K20", "K26"]])
        sd = f.history_sd(profiles)
        row = sd.loc[sd.outcome.eq("likud")].iloc[0]
        self.assertEqual(row.n_elections, 2)
        self.assertEqual(row.excluded_elections_without_separate_target, '["K19"]')


if __name__ == "__main__":
    unittest.main()
