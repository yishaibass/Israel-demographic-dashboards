import unittest
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

from historical_elections import (DATES, RESOURCES, SOURCE_URLS, acquire_archive,
                                  build_targets, observation_partition, validate_archive)


def crosswalk(review=False):
    return pd.DataFrame([
        {"ballot_symbol": "joint", "families": "national_religious_right|secular_national_right", "aggregation_rule": "sum"},
        {"ballot_symbol": "nrp", "families": "national_religious_right", "aggregation_rule": "single"},
        {"ballot_symbol": "shas", "families": "" if review else "shas", "aggregation_rule": "review" if review else "single"},
    ])


def counts():
    return pd.DataFrame([{"סמל ישוב": 100, "שם ישוב": "Synthetic", "בזב": 100,
                          "מצביעים": 80, "כשרים": 75, "פסולים": 5, "joint": 60, "nrp": 10, "shas": 5}])


class HistoricalElectionTests(unittest.TestCase):
    def archive(self, election="K16", total=1, response_resource=None):
        content = json.dumps({"success": True, "result": {"resource_id": response_resource or RESOURCES[election],
                             "total": total, "records": [{"_id": 1, "synthetic": 1}], "total_was_estimated": False}}).encode()
        metadata = {"election": election, "election_date": DATES[election], "resource_id": RESOURCES[election],
                    "source_url": SOURCE_URLS[election], "records": 1, "sha256": hashlib.sha256(content).hexdigest()}
        return content, metadata

    def test_download_is_complete_then_cached_without_network_or_overwriting(self):
        content, _ = self.archive()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("historical_elections.urllib.request.urlopen", return_value=io.BytesIO(content)) as request:
                path, payload, metadata = acquire_archive(root, "K16", download=True)
                request.assert_called_once_with(SOURCE_URLS["K16"], timeout=60)
            original_metadata = path.with_suffix(".metadata.json").read_bytes()
            with patch("historical_elections.urllib.request.urlopen") as request:
                cached, payload, metadata = acquire_archive(root, "K16", download=True)
                request.assert_not_called()
            self.assertEqual(path, cached)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.with_suffix(".metadata.json").read_bytes(), original_metadata)

    def test_tampered_cache_fails_even_with_download_and_is_not_replaced(self):
        content, metadata = self.archive()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"data/raw/ballots/historical_model/K16_official_datastore.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(content+b" ")
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
            for download in [False, True]:
                with patch("historical_elections.urllib.request.urlopen") as request:
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        acquire_archive(root, "K16", download=download)
                    request.assert_not_called()
            self.assertEqual(path.read_bytes(), content+b" ")

    def test_metadata_and_response_resource_identity_are_checked_independently(self):
        content, metadata = self.archive()
        for field, value in [("election", "K17"), ("resource_id", RESOURCES["K17"]),
                             ("source_url", SOURCE_URLS["K17"]), ("ballot_type", "prime_minister")]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_archive(content, {**metadata, field: value}, "K16")
        wrong, metadata = self.archive(response_resource=RESOURCES["K17"])
        with self.assertRaisesRegex(ValueError, "resource identity"):
            validate_archive(wrong, metadata, "K16")

    def test_paginated_or_estimated_download_never_becomes_cache(self):
        incomplete, metadata = self.archive(total=2)
        with self.assertRaisesRegex(ValueError, "paginated"):
            validate_archive(incomplete, metadata, "K16")
        with tempfile.TemporaryDirectory() as folder:
            with patch("historical_elections.urllib.request.urlopen", return_value=io.BytesIO(incomplete)):
                with self.assertRaisesRegex(ValueError, "paginated"):
                    acquire_archive(Path(folder), "K16", download=True)
            self.assertFalse((Path(folder)/"data").exists())
        payload = json.loads(incomplete)
        payload["result"].update(total=1, total_was_estimated=True)
        estimated = json.dumps(payload).encode()
        with self.assertRaisesRegex(ValueError, "estimated"):
            validate_archive(estimated, {**metadata, "sha256": hashlib.sha256(estimated).hexdigest()}, "K16")
        payload["result"].update(total=2, total_was_estimated=False, records=[{"_id": 1}, {"_id": 1}])
        duplicate = json.dumps(payload).encode()
        with self.assertRaisesRegex(ValueError, "unique datastore row"):
            validate_archive(duplicate, {**metadata, "records": 2, "sha256": hashlib.sha256(duplicate).hexdigest()}, "K16")

    def test_joint_list_merges_all_component_ballots_once(self):
        totals, long, wide, review, audit = build_targets(counts(), crosswalk(), "K16", "2003-01-28")
        identifier = "national_religious_right__secular_national_right"
        self.assertEqual(wide.loc[0, identifier], 70)
        self.assertNotIn("national_religious_right", wide)
        self.assertEqual(long.votes.sum(), totals.valid.sum())
        self.assertTrue(audit["training_ready"])
        self.assertEqual(len(observation_partition(crosswalk())), 8)
        self.assertEqual(set(audit["available_families"]), {"likud", "national_religious_right", "shas", "utj", "arab", "liberal_center", "zionist_left", "secular_national_right", "other"})
        self.assertNotIn("national_religious_right", audit["separately_identifiable_families"])
        self.assertNotIn("secular_national_right", audit["separately_identifiable_families"])

    def test_unreviewed_vote_mass_is_explicit_and_blocks_training(self):
        totals, long, wide, review, audit = build_targets(counts(), crosswalk(True), "K16", "2003-01-28")
        self.assertEqual(audit["unresolved_votes"], 5)
        self.assertFalse(audit["training_ready"])
        self.assertFalse(long.available.any())
        self.assertEqual(long.votes.sum(), 75)
        self.assertEqual(wide.review_shas.iloc[0], 5)

    def test_missing_historical_codes_and_eligible_are_not_invented(self):
        raw = counts().drop(columns=["סמל ישוב", "בזב", "פסולים"])
        totals, long, wide, review, audit = build_targets(raw, crosswalk(), "K17", "2006-03-28")
        self.assertEqual(totals.locality_id.iloc[0], "name:Synthetic")
        self.assertTrue(totals.eligible.isna().all())
        self.assertFalse(audit["training_ready"])
        self.assertEqual(totals.invalid.iloc[0], 5)
        self.assertEqual(long.votes.sum(), 75)

    def test_zero_mass_unreviewed_symbol_is_retained_without_blocking_known_targets(self):
        raw = counts().assign(withdrawn=0)
        mapping = pd.concat([crosswalk(), pd.DataFrame([{"ballot_symbol": "withdrawn", "families": "", "aggregation_rule": "review"}])], ignore_index=True)
        totals, long, wide, review, audit = build_targets(raw, mapping, "K16", "2003-01-28")
        self.assertTrue(audit["training_ready"])
        self.assertEqual(audit["unreviewed_zero_vote_symbols"], ["withdrawn"])
        self.assertEqual(review.regular_votes.sum(), 0)
        self.assertEqual(long.votes.sum(), totals.valid.sum())

    def test_bad_count_rows_and_envelopes_are_audited(self):
        good = counts()
        bad = good.copy();bad["סמל ישוב"] = 200;bad["בזב"] = 60
        envelope = good.copy();envelope["סמל ישוב"] = 0;envelope["שם ישוב"] = "מעטפות כפולות"
        raw = pd.concat([good, bad, envelope], ignore_index=True)
        totals, long, wide, review, audit = build_targets(raw, crosswalk(), "K16", "2003-01-28")
        self.assertEqual(len(totals), 1)
        self.assertEqual(audit["inconsistent_regular_rows_excluded"], 1)
        self.assertEqual(audit["inconsistent_regular_valid_votes"], 75)
        self.assertEqual(audit["envelope_rows_excluded"], 1)
        self.assertEqual(audit["inconsistent_locality_codes"], [200])

    def test_unknown_symbols_and_negative_votes_fail(self):
        with self.assertRaises(ValueError):
            build_targets(counts().assign(unknown=0), crosswalk(), "K16", "2003-01-28")
        raw = counts();raw["joint"] = -1
        with self.assertRaises(ValueError):
            build_targets(raw, crosswalk(), "K16", "2003-01-28")


if __name__ == "__main__":
    unittest.main()
