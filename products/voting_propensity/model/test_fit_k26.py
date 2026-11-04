import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fit_k26 import DATES, FAMILIES, MASS_COLUMNS, build_targets, build_historical_targets, cube, context, fit_identity, model


class K26FamilyTests(unittest.TestCase):
    def test_confirmed_mapping_exhausts_votes_and_excludes_envelopes(self):
        with tempfile.TemporaryDirectory() as folder:
            current = Path(folder)
            (current / "readiness.json").write_text(json.dumps({"snapshot_id": "test", "count_status": "official"}), encoding="utf-8")
            pd.DataFrame({"snapshot_id": ["test"]*4, "ballot_symbol": ["a", "b", "c", "d"],
                          "party_id": ["Yashar", "Yachad (Bennett)", "Democrats", "Unclassified"]}).to_csv(current / "party_dimension.csv", index=False)
            pd.DataFrame([
                {"snapshot_id": "test", "locality_code": 1, "eligible_voters": 200, "total_voted": 100, "valid_votes": 100,
                 "external_envelope": False, "votes_a": 40, "votes_b": 30, "votes_c": 20, "votes_d": 10},
                {"snapshot_id": "test", "locality_code": 2, "eligible_voters": 0, "total_voted": 100, "valid_votes": 100,
                 "external_envelope": True, "votes_a": 40, "votes_b": 30, "votes_c": 20, "votes_d": 10}
            ]).to_csv(current / "ballots.csv", index=False)
            target, audit = build_targets(current)
            self.assertEqual(len(target), 1)
            self.assertEqual(target.iloc[0].liberal_center, 70)
            self.assertEqual(target.iloc[0].zionist_left, 20)
            self.assertEqual(target.iloc[0].other, 10)
            self.assertEqual(target[FAMILIES].sum(axis=1).iloc[0], target.valid.iloc[0])
            self.assertEqual(audit["national_family_votes"]["liberal_center"], 140)
            self.assertEqual(audit["envelope_rows_excluded"], 1)

    def test_cube_preserves_all_population_and_probability_mass(self):
        micro = pd.DataFrame({"KvutzaUchlusiyaPUF": [1, 1, 2], "DatiyutPUF": [1, 1, 0], "GilPUF": [3, 3, 4],
                              "LimudToarAcademiPUF": [1, 1, 0], "MachozPUF": [2, 2, 6],
                              "HchnsAvgChdshPratPUF": [10, 10, 0], "MishkalPratPUF": [2, 3, 4]})
        probabilities = np.full((3, 10), .1)
        aggregate = cube(micro, probabilities)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(aggregate.weight.sum(), 9)
        np.testing.assert_allclose(aggregate[MASS_COLUMNS].sum(axis=1), aggregate.weight)
        self.assertAlmostEqual(aggregate.likud_mass.sum(), .9)

    def test_published_estimator_fits_separate_k26_targets_with_pandas_copy_on_write(self):
        old_elections = model.ELECTIONS
        try:
            model.ELECTIONS = ["K26"]
            cells = pd.DataFrame({name: [1., 1., 1.] if name == "intercept" else [0., 0., 0.]
                                  for name in context.MODEL_COLUMNS})
            cells["g"], cells["weight"], cells["fsu_proxy"] = [0, 1, 2], 100., 0.
            geos = pd.DataFrame({"geo_key": ["1", "2", "3"]})
            targets = pd.DataFrame({"geo_key": ["1", "2", "3"], "locality": [1, 2, 3],
                                    "election": "K26", "eligible": 100., "voted": 70., "valid": 63.,
                                    **{family: 7. for family in FAMILIES}})
            fitted, info = model.fit(cells, geos, targets, context.MODEL_COLUMNS, False, {1, 2, 3}, .05, 2, 42)
            self.assertEqual(info["epochs"], 2)
            self.assertTrue(np.isfinite(info["objective"]))
            self.assertEqual(tuple(fitted.turn_dev.shape), (1, len(context.MODEL_COLUMNS)))
        finally:
            model.ELECTIONS = old_elections

    def test_historical_targets_use_same_locality_unit_and_exhaustive_families(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            ballots = project / "data/raw/ballots"
            ballots.mkdir(parents=True)
            mapping = []
            for election in list(DATES)[:5]:
                mapping.extend([{"Election": int(election[1:]), "Letter": "a", "Name_English": "Likud"},
                                {"Election": int(election[1:]), "Letter": "b", "Name_English": "Yesh Atid"}])
                pd.DataFrame({"שם ישוב": ["Town", "Town", "מעטפות חיצוניות"], "סמל ישוב": [1, 1, 9999],
                              "מספר קלפי": [1, 800, 1], "בזב": [100, 100, 0], "מצביעים": [50, 50, 50],
                              "פסולים": [0, 0, 0], "כשרים": [50, 50, 50], "a": [20, 20, 20], "b": [25, 25, 25],
                              "unmapped": [5, 5, 5]}).to_csv(ballots / (election + "_kalpi_clean.csv"), index=False)
            pd.DataFrame(mapping).to_excel(project / "data/raw/party_mapping.xlsx", sheet_name="Mapping", index=False)
            targets, audits = build_historical_targets(project)
            self.assertEqual(len(targets), 5)
            self.assertTrue(targets.eligible.eq(200).all())
            self.assertTrue(targets.likud.eq(40).all())
            self.assertTrue(targets.liberal_center.eq(50).all())
            self.assertTrue(targets.other.eq(10).all())
            self.assertTrue(targets[FAMILIES].sum(axis=1).eq(targets.valid).all())
            self.assertEqual(audits["K21"]["national_valid_votes"], 150)
            self.assertEqual(audits["K21"]["envelope_rows_excluded"], 1)

    def test_one_shared_model_has_six_election_deviation_vectors(self):
        previous = model.ELECTIONS
        try:
            model.ELECTIONS = list(DATES)
            fitted = model.FamilyModel(len(context.MODEL_COLUMNS), len(FAMILIES), False)
            self.assertEqual(tuple(fitted.turn_dev.shape), (6, len(context.MODEL_COLUMNS)))
            self.assertEqual(tuple(fitted.choice_dev.shape), (6, len(context.MODEL_COLUMNS), 9))
        finally:
            model.ELECTIONS = previous

    def test_fit_identity_changes_with_parameters_and_election_order(self):
        previous = model.ELECTIONS
        try:
            model.ELECTIONS = list(DATES)
            fitted = model.FamilyModel(len(context.MODEL_COLUMNS), len(FAMILIES), False)
            first = fit_identity(fitted)
            self.assertEqual(first, fit_identity(fitted))
            fitted.turn_shared.data[0] += 1
            self.assertNotEqual(first, fit_identity(fitted))
            changed = fit_identity(fitted)
            model.ELECTIONS = list(reversed(model.ELECTIONS))
            self.assertNotEqual(changed, fit_identity(fitted))
        finally:
            model.ELECTIONS = previous


if __name__ == "__main__":
    unittest.main()
