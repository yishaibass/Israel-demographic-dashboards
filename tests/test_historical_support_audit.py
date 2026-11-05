import unittest

import numpy as np
import pandas as pd
import torch

import historical_support_audit as audit


def example():
    cells = pd.DataFrame({name: [0., 0.] for name in audit.history.FEATURES})
    cells["intercept"] = 1.
    cells["geo_dati_share"] = [.2, .8]
    cells["dati"] = [1., 0.]
    cells["fsu_proxy"] = 0.
    cells["locality"] = [1, 2]
    cells["g"] = [0, 1]
    cells["weight"] = cells["population_weight"] = [100., 100.]
    cells["profile_group"] = ["dati", "haredi"]
    targets = pd.DataFrame(dict(locality=[1, 2], eligible=[100, 100], voted=[90, 90], valid=[80, 80],
                                joint=[60, 70], rest=[20, 10]))
    partition = [{"id": "joint", "families": ["likud", "national_religious_right"]},
                 {"id": "rest", "families": ["liberal_center"]}]
    return audit.history.ElectionData("K16", cells, pd.DataFrame(dict(locality=[1, 2])), targets,
                                     partition, ["likud", "national_religious_right", "liberal_center"])


class AuditTests(unittest.TestCase):
    def test_narrow_target_preserves_joint_identification_and_counts(self):
        data = audit.binary_data(example())
        self.assertEqual(data.targets.right_votes.tolist(), [60, 70])
        self.assertTrue(np.array_equal(data.targets.right_votes+data.targets.rest_votes, data.targets.valid))
        data.tensors()

    def test_broad_target_requires_exact_matching_denominators_and_identities(self):
        counts = pd.DataFrame(dict(election=["K16", "K16"], locality=[1, 2], valid=[80, 80], right_votes=[75, 76]))
        data = audit.binary_data(example(), counts)
        self.assertEqual(data.targets.right_votes.tolist(), [75, 76])
        for corrupt in [counts.assign(valid=[79, 80]), counts.iloc[:1],
                        counts.assign(right_votes=[81, 76]), pd.concat([counts, counts.iloc[:1]])]:
            with self.assertRaises(ValueError):
                audit.binary_data(example(), corrupt)

    def test_rejects_unidentified_boundary(self):
        data = example()
        data.partition[0]["families"] = ["likud", "liberal_center"]
        with self.assertRaises(ValueError):
            audit.binary_data(data)

    def test_direct_design_preserves_individual_demographics_and_original(self):
        data = example()
        direct = audit.direct_only(data)
        self.assertTrue(direct.cells.geo_dati_share.eq(0).all())
        self.assertEqual(direct.cells.dati.tolist(), [1., 0.])
        self.assertEqual(data.cells.geo_dati_share.tolist(), [.2, .8])

    def test_perturbation_is_real_and_constructor_restores_on_error(self):
        original = audit.history.common.model.FamilyModel
        with audit.history.election_scope(["K16"]):
            normal = original(len(audit.history.FEATURES), len(audit.history.FAMILIES), False)
            with self.assertRaises(RuntimeError):
                with audit.perturbed_initialization(.05):
                    perturbed = audit.history.common.model.FamilyModel(len(audit.history.FEATURES), len(audit.history.FAMILIES), False)
                    self.assertFalse(torch.equal(normal.choice_shared, perturbed.choice_shared))
                    raise RuntimeError("test restoration")
        self.assertIs(audit.history.common.model.FamilyModel, original)


if __name__ == "__main__":
    unittest.main()
