import copy
import json
import unittest

from products.voting_propensity.k26_publication import ELECTIONS, integrate, merge_cube


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

    def publication_inputs(self):
        baseline, current = self.inputs()
        current.update(snapshot_id="same", model_fit_id="joint", election="K26")
        audit = {"snapshot_id": "same", "model_fit_id": "joint", "matched_localities": 200, "national_valid_vote_coverage": .8}
        history = {e: {**copy.deepcopy(current), "election": e} for e in ELECTIONS[:-1]}
        page = '<html><body><script>const CUBE=' + json.dumps(baseline) + ';</script><section class="analysis-section"></section></body></html>'
        independent = {e: {**copy.deepcopy(current), "election": e, "model_fit_id": "fit-" + e, "collection_id": "collection"} for e in ELECTIONS}
        independent_audit = {"snapshot_id": "same", "fit_mode": "independent", "publication_approved": True, "collection_id": "collection", "election_fit_ids": {e: "fit-" + e for e in ELECTIONS}}
        return baseline, page, current, audit, history, independent, independent_audit

    def test_all_elections_comparison_controls_and_frozen_payload(self):
        baseline, page, cube, audit, history, _, _ = self.publication_inputs()
        result = integrate(page, cube, audit, history)
        self.assertIn('const CUBE=' + json.dumps(baseline), result)
        self.assertNotIn('id="modelFitMethod"', result)
        for election in ELECTIONS:
            self.assertEqual(result.count('value="' + election + '"'), 3)
        self.assertIn('id="comparisonElectionA"', result)
        self.assertIn('id="comparisonElectionB"', result)
        self.assertEqual(result, integrate(result, cube, audit, history))

    def test_independent_bundle_approved_and_idempotent(self):
        _, page, cube, audit, history, independent, independent_audit = self.publication_inputs()
        result = integrate(page, cube, audit, history, independent, independent_audit)
        self.assertIn('id="modelFitMethod"', result)
        self.assertIn('"independent"', result)
        self.assertEqual(result, integrate(result, cube, audit, history, independent, independent_audit))

    def test_independent_mixed_or_stale_fits_rejected(self):
        for mutation, message in [
            (lambda c, a: c["K22"].update(snapshot_id="stale"), "snapshot is stale"),
            (lambda c, a: a.update(snapshot_id="stale"), "snapshot is stale"),
            (lambda c, a: c["K22"].update(model_fit_id="fit-K21"), "fit identities differ"),
            (lambda c, a: c["K22"].update(collection_id="mixed"), "fit identities differ"),
            (lambda c, a: c["K22"].update(election="K21"), "fit identities differ"),
            (lambda c, a: a.update(publication_approved=False), "not approved"),
            (lambda c, a: c.pop("K21"), "all six"),
            (lambda c, a: a["election_fit_ids"].update(K22="fit-K21"), "distinct election"),
        ]:
            with self.subTest(message=message):
                _, page, cube, audit, history, independent, independent_audit = self.publication_inputs()
                mutation(independent, independent_audit)
                with self.assertRaisesRegex(ValueError, message):
                    integrate(page, cube, audit, history, independent, independent_audit)

    def test_independent_pair_required(self):
        _, page, cube, audit, history, independent, _ = self.publication_inputs()
        with self.assertRaisesRegex(ValueError, "supplied together"):
            integrate(page, cube, audit, history, independent)


if __name__ == "__main__":
    unittest.main()
