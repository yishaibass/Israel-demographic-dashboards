"""Synthetic identity/mass contracts; no private population records."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "products/voting_propensity/model"))
import locality_lifecycle as p


class LifecycleTests(unittest.TestCase):
    def test_exact_merger_day_and_unknown_inventory(self):
        self.assertEqual(p.active_status(3720, "2022-06-06"), "active")
        self.assertEqual(p.active_status(3720, "2022-06-07"), "inactive")
        self.assertEqual(p.active_status(3826, "2022-06-07"), "active")
        self.assertEqual(p.active_status(999, "2026-10-27"), "unreviewed")

    def test_merge_preserves_each_cell_and_input(self):
        old = {3720: np.array([20., 7.]), 3778: np.array([3., 5.])}
        out, audit = p.reconcile_values(old, "2026-10-27")
        np.testing.assert_array_equal(out[3826], [23, 12])
        np.testing.assert_array_equal(old[3720], [20, 7])
        self.assertEqual(audit["input_mass"], audit["output_mass"])
        self.assertNotIn(3720, out)

    def test_overlapping_sources_fail_or_explicitly_select(self):
        source = {3720: 20, 3778: 3, 3826: 25}
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            p.reconcile_values(source, "2026-10-27")
        old, a = p.reconcile_values(source, "2026-10-27", overlap="predecessors")
        new, b = p.reconcile_values(source, "2026-10-27", overlap="successor")
        self.assertEqual(float(old[3826]), 23)
        self.assertEqual(float(new[3826]), 25)
        self.assertEqual(a["discarded_overlap"], 25)
        self.assertEqual(b["discarded_overlap"], 23)

    def test_split_is_code_birth_and_explicit_partition(self):
        self.assertEqual(p.active_status(3815, "2022-10-01"), "inactive")
        self.assertEqual(p.active_status(3815, "2023-06-01"), "uncertain")
        self.assertEqual(p.predecessor_group(3815, "2026-10-27"), (3779,))
        self.assertEqual(p.comparison_family(3815, "2026-10-27"), (3779, 3815))
        out, audit = p.reconcile_values({3779: np.array([60., 40.])}, "2026-10-27", split_weights={3779: 3, 3815: 7})
        np.testing.assert_allclose(out[3779] + out[3815], [60, 40])
        self.assertEqual(audit["output_mass"], 100)
        with self.assertRaisesRegex(ValueError, "existing positive child"):
            p.reconcile_values({3779: 100, 3815: 70}, "2026-10-27", split_weights={3779: 3, 3815: 7})

    def test_historical_child_consolidates_and_reverse_merge_rejected(self):
        out, _ = p.reconcile_values({3779: 30, 3815: 70}, "2020-02-01")
        self.assertEqual(float(out[3779]), 100)
        self.assertNotIn(3815, out)
        with self.assertRaisesRegex(ValueError, "Reverse merger"):
            p.reconcile_values({3826: 100}, "2020-02-01")

    def test_nonlocality_population_retained_separately(self):
        out, a = p.reconcile_values({566: 7, p.OUTSIDE_LOCALITIES: 2}, "2026-10-27")
        self.assertEqual(float(out[p.OUTSIDE_LOCALITIES]), 9)
        self.assertEqual(a["input_mass"], a["output_mass"])
        self.assertNotIn(566, out)
        self.assertEqual(p.active_status(566, "2022-06-01"), "uncertain")

    def test_recognition_year_does_not_invent_physical_founding(self):
        ramat = next(e for e in p.EVENTS if e["successors"] == (4029,))
        self.assertFalse(ramat["physical_founding"])
        self.assertNotIn("physical_founding_year", ramat)
        self.assertEqual(p.active_status(1362, "2021-12-31"), "inactive")
        self.assertEqual(p.active_status(1362, "2022-07-01"), "uncertain")
        self.assertEqual(p.active_status(1362, "2023-01-01"), "active")
        self.assertEqual(p.active_status(849, "2026-10-27"), "unreviewed")
        self.assertIn(849, p.SAME_CODE_CHANGES)
        self.assertEqual(p.REVIEWED_IDENTITIES[3779], "Kochav Yaakov")
        self.assertEqual(p.active_status(3774, "2026-10-27"), "unreviewed")

    def test_invalid_weights_and_counts(self):
        for weights in ({3779: -1, 3815: 2}, {3779: 0, 3815: 0}, {3779: 1}):
            with self.assertRaises(ValueError):
                p.allocate_split(100, weights)
        with self.assertRaises(ValueError):
            p.reconcile_values({3720: float("nan")}, "2026-10-27")


if __name__ == "__main__":
    unittest.main()
