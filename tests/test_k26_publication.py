import copy
import json
import unittest

from products.voting_propensity.k26_publication import integrate, merge_cube


class K26PublicationTests(unittest.TestCase):
    def inputs(self):
        columns = ["pop_group", "religiosity", "income_band", "age_band", "academic", "district", "weight", "abstain_mass", "likud_mass", "fiscal_net_per_hh"]
        baseline = {"columns": columns, "rows": [[1, 1, 2, 3, 1, 4, 100, 40, 60, 12345]]}
        current = {"columns": columns[:-1], "rows": [[1, 1, 2, 3, 1, 4, 100, 20, 80]]}
        return baseline, current

    def test_merge_preserves_baseline_and_fiscal_values(self):
        baseline, current = self.inputs()
        before = copy.deepcopy(baseline)
        result = merge_cube(baseline, current)
        self.assertEqual(result["rows"][0][-3:], [20, 80, 12345])
        self.assertEqual(baseline, before)

    def test_missing_cell_and_changed_population_are_rejected(self):
        baseline, current = self.inputs()
        current["rows"][0][0] = 2
        with self.assertRaisesRegex(ValueError, "every historical"):
            merge_cube(baseline, current)
        baseline, current = self.inputs()
        current["rows"][0][6] = 101
        with self.assertRaisesRegex(ValueError, "population weights"):
            merge_cube(baseline, current)

    def test_nonconserving_probability_mass_is_rejected(self):
        baseline, current = self.inputs()
        current["rows"][0][7] = 30
        with self.assertRaisesRegex(ValueError, "mass conservation"):
            merge_cube(baseline, current)

    def test_mixed_fit_audit_is_rejected_even_at_same_snapshot(self):
        with self.assertRaisesRegex(ValueError, "same fitted model"):
            integrate("", {"snapshot_id": "same", "model_fit_id": "a"},
                      {"snapshot_id": "same", "model_fit_id": "b"}, {})

    def test_historical_cubes_must_use_the_current_fitted_model(self):
        baseline, current = self.inputs()
        current.update(snapshot_id="same", model_fit_id="a")
        history = {e: {**current, "model_fit_id": "b"} for e in ["K21", "K22", "K23", "K24", "K25"]}
        page = '<script>const CUBE=' + json.dumps(baseline) + '</script>'
        with self.assertRaisesRegex(ValueError, "different fitted models"):
            integrate(page, current, {"snapshot_id": "same", "model_fit_id": "a"}, history)


if __name__ == "__main__":
    unittest.main()
