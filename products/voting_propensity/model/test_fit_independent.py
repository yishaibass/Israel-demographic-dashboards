import unittest

import numpy as np
import pandas as pd

import fit_independent as independent


class IndependentElectionTests(unittest.TestCase):
    def test_other_election_outcomes_do_not_affect_fit_or_dataset_id(self):
        common = independent.common
        common.torch.set_num_threads(1)
        cells = pd.DataFrame({name: [1., 1.] if name == "intercept" else [0., 0.]
                              for name in common.context.MODEL_COLUMNS})
        cells["g"], cells["weight"], cells["fsu_proxy"] = [0, 1], 100., 0.
        geos = pd.DataFrame({"geo_key": ["1", "2"]})
        rows = []
        for election in ["K25", "K26"]:
            for locality in [1, 2]:
                rows.append({"election": election, "locality": locality, "geo_key": str(locality),
                             "eligible": 100, "voted": 70, "valid": 63,
                             **{family: 7 for family in common.FAMILIES}})
        targets = pd.DataFrame(rows)
        previous = list(common.model.ELECTIONS)
        first, _, first_id = independent.fit_one(cells, geos, targets, "K26", [1, 2], 2, 42)
        changed = targets.copy()
        changed.loc[changed.election.eq("K25"), ["voted", "likud"]] = 99999
        _, _, second_id = independent.fit_one(cells, geos, changed, "K26", [1, 2], 2, 42)
        self.assertEqual(first_id, second_id)
        self.assertEqual(previous, common.model.ELECTIONS)
        self.assertEqual(independent.dataset_identity(targets, "K26"), independent.dataset_identity(changed, "K26"))
        self.assertNotEqual(independent.dataset_identity(targets, "K25"), independent.dataset_identity(changed, "K25"))
        self.assertEqual(tuple(first.turn_dev.shape), (1, len(common.context.MODEL_COLUMNS)))

    def test_voter_conditional_subgroup_shares_use_joint_masses(self):
        rows = []
        for turnout, family in [(1., "likud_mass"), (.2, "arab_mass")]:
            row = {"pop_group": 1, "religiosity": 1, "income_band": 1, "academic": 1, "weight": 50,
                   **{column: 0. for column in independent.common.MASS_COLUMNS}}
            row["abstain_mass"] = 50*(1-turnout)
            row[family] = 50*turnout
            rows.append(row)
        profiles = independent.subgroup_profiles(pd.DataFrame(rows))
        broad = profiles.loc[profiles.slice_type.eq("group")]
        self.assertAlmostEqual(broad.loc[broad.outcome.eq("turnout"), "value"].iloc[0], .6)
        self.assertAlmostEqual(broad.loc[broad.outcome.eq("likud"), "value"].iloc[0], 5/6)
        self.assertAlmostEqual(broad.loc[~broad.outcome.eq("turnout"), "value"].sum(), 1.)

    def test_collection_id_binds_each_fit_dataset_and_snapshot(self):
        identity = independent.collection_identity({"K26": "fit"}, {"K26": "data"}, "snapshot")
        self.assertEqual(identity, independent.collection_identity({"K26": "fit"}, {"K26": "data"}, "snapshot"))
        self.assertNotEqual(identity, independent.collection_identity({"K26": "different"}, {"K26": "data"}, "snapshot"))
        self.assertNotEqual(identity, independent.collection_identity({"K26": "fit"}, {"K26": "different"}, "snapshot"))
        self.assertNotEqual(identity, independent.collection_identity({"K26": "fit"}, {"K26": "data"}, "different"))


if __name__ == "__main__":
    unittest.main()
