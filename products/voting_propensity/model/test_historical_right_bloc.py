"""Focused integrity checks for party-level historical bloc targets."""
import unittest

from historical_right_bloc import aggregate


class HistoricalRightBlocTests(unittest.TestCase):
    def rows(self):
        return [{"סמל ישוב": "10.0", "שם ישוב": "A", "כשרים": 10, "מחל": 6, "ב": 3, "ק": 1},
                {"סמל ישוב": 10, "שם ישוב": "A", "כשרים": 5, "מחל": 2, "ב": 1, "ק": 2},
                {"סמל ישוב": 0, "שם ישוב": "מעטפות חיצוניות", "כשרים": 99, "מחל": 99}]

    def test_primary_and_uncertain_counts_close_without_envelopes(self):
        mapping = {"מחל": "right", "ב": "uncertain", "ק": "rest"}
        primary, _ = aggregate(self.rows(), mapping, "K16")
        upper, _ = aggregate(self.rows(), mapping, "K16", True)
        self.assertEqual(primary, [dict(election="K16", locality=10, right_votes=8, rest_votes=7, valid=15)])
        self.assertEqual(upper[0]["right_votes"], 12)
        self.assertEqual(upper[0]["rest_votes"], 3)

    def test_unknown_positive_party_fails(self):
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            aggregate(self.rows(), {"מחל": "right", "ב": "rest"}, "K16")

    def test_party_sum_mismatch_fails(self):
        rows = self.rows()
        rows[0]["כשרים"] = 11
        with self.assertRaisesRegex(ValueError, "exhaust"):
            aggregate(rows, {"מחל": "right", "ב": "rest", "ק": "rest"}, "K16")

    def test_fractional_vote_fails(self):
        rows = self.rows()
        rows[0]["מחל"] = 6.5
        with self.assertRaisesRegex(ValueError, "integers"):
            aggregate(rows, {"מחל": "right", "ב": "rest", "ק": "rest"}, "K16")


if __name__ == "__main__":
    unittest.main()
