import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from products.voting_propensity.model.demographic_foundation import (
    FEATURES, NII_SOURCES, annual_catalog_targets, archive_url, classify, collapse_puf, download_nii, population_mix, project_latest_pair, validate_nii_headers, nii_coded_age_targets,
)


class DemographicFoundationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            [8800, 1, 2, 1, 1, 1, 10.],
            [8800, 1, 2, 1, 2, 2, 20.],
            [9999, None, 3, None, 1, 7, 30.],
            [26, 1, 1, 6, 2, 3, 40.],
            [26, 1, 1, 7, 1, 9, 5.],
            [26, 1, 1, 5, 1, 2, 0.],
        ], columns=FEATURES)

    def test_adults_and_weights_reconcile_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic.csv"
            self.frame().to_csv(path, index=False)
            cells, audit = collapse_puf(path, chunksize=2)
            self.assertEqual(audit["weighted_population_all_ages"], 105.)
            self.assertEqual(audit["weighted_adults_18_plus"], 90.)
            self.assertEqual(audit["weighted_unknown_age"], 5.)
            self.assertEqual(cells.puf_records.sum(), 5)
            self.assertEqual(audit["adult_puf_records"], 3)
            second, _ = collapse_puf(path, chunksize=100)
            pd.testing.assert_frame_equal(cells, second)

    def test_other_mixed_unknown_and_masked_geography_retained(self):
        cells = classify(self.frame())
        self.assertEqual(cells.loc[1, "population_group"], "arab")
        self.assertNotIn("druze", cells.population_group.tolist())
        self.assertEqual(cells.loc[2, "population_group"], "other")
        self.assertEqual(cells.loc[2, "household_religiosity"], "unknown")
        self.assertEqual(cells.loc[2, "geography_status"], "masked_locality_pool")
        self.assertEqual(cells.loc[3, "household_religiosity"], "mixed")
        mix = population_mix(cells.loc[cells.person_weight.gt(0)])
        np.testing.assert_allclose(mix.groupby("locality_code").share.sum(), 1.)

    def test_negative_weight_rejected(self):
        frame = self.frame()
        frame.loc[0, "MishkalPratPUF"] = -1
        with self.assertRaisesRegex(ValueError, "Negative"):
            classify(frame)

    def test_official_margins_preserve_missing_and_derive_other_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            record = {"סמל יישוב": 8800, "שם יישוב": "synthetic", "יהודים ואחרים": 120, "מזה: יהודים": 100, "ערבים": None}
            body = json.dumps({"success": True, "result": {"resource_id": "synthetic", "total": 1, "records": [record]}}).encode()
            (raw/"source.json").write_bytes(body)
            source = {"year": 2022, "file": "source.json", "source_id": "synthetic", "sha256": hashlib.sha256(body).hexdigest(), "records": 1}
            result = annual_catalog_targets(raw, [source])
            self.assertEqual(dict(zip(result.category, result.value)), {"jewish_and_other": 120., "jewish": 100., "other": 20.})
            self.assertEqual(result.loc[result.category.eq("other"), "status"].iloc[0], "arithmetic_derived")
            (raw/"source.json").write_bytes(body+b" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                annual_catalog_targets(raw, [source])

    def test_incomplete_source_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            body = json.dumps({"success": True, "result": {"resource_id": "synthetic", "total": 2, "records": []}}).encode()
            (raw/"source.json").write_bytes(body)
            source = {"year": 2022, "file": "source.json", "source_id": "synthetic", "sha256": hashlib.sha256(body).hexdigest(), "records": 0}
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                annual_catalog_targets(raw, [source])

    def test_projections_require_positive_consecutive_anchors_and_remain_proxies(self):
        base = {"locality_code": 1, "dimension": "retirement_age", "category": "retirement_age",
                "population_reference_date_status": "unverified", "source_id": "synthetic"}
        frame = pd.DataFrame([{**base, "year": 2022, "value": 100.}, {**base, "year": 2023, "value": 110.}])
        projected = project_latest_pair(frame)
        self.assertAlmostEqual(projected.value.iloc[0], 110.*1.1**3)
        self.assertEqual(projected.status.iloc[0], "projected_source_year_proxy")
        frame.loc[0, "year"] = 2021
        self.assertTrue(project_latest_pair(frame).empty)
        frame.loc[0, "year"] = 2022
        frame.loc[0, "value"] = 0
        self.assertTrue(project_latest_pair(frame).empty)
        frame.loc[1, "value"] = 0
        self.assertEqual(project_latest_pair(frame).value.iloc[0], 0.)
        self.assertEqual(project_latest_pair(frame).evidence_grade.iloc[0], "two_explicit_zero_anchors_carried_forward")

    def test_sex_specific_age_definition_guard(self):
        table = pd.DataFrame(np.full((5, 5), None))
        table.iloc[0, 0] = "synthetic2023"
        table.iloc[2, 2:5] = ["17-0", "נשים\n59-18\nגברים\n64-18", "נשים +60\nגברים +65"]
        validate_nii_headers(table, 2023)
        table.iloc[2, 4] = "נשים +65\nגברים +65"
        with self.assertRaisesRegex(ValueError, "age header"):
            validate_nii_headers(table, 2023)

    def test_new_coded_ages_keep_suppression_and_verify_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/"nii_annual"
            raw.mkdir()
            source = raw/"2025_monthly_population.xlsx"
            source.write_bytes(b"synthetic workbook stub")
            source.with_suffix(".metadata.json").write_text(json.dumps({"year": 2025, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
            table = pd.DataFrame(np.full((7, 7), None))
            table.iloc[0, 0] = "דצמבר2025"
            table.iloc[1, 1] = "סמל יישוב"
            table.iloc[2, 4:7] = ["17-0", "64-18", "65+"]
            table.iloc[5] = ["one", 1, 1, "1,000", "200", "700", "100"]
            table.iloc[6] = ["two", 2, 1, "100", "20", "***", "***"]
            with patch("pandas.read_excel", return_value=table):
                targets, _ = nii_coded_age_targets(root)
            first = targets.loc[targets.locality_code.eq(1)]
            self.assertEqual(first.loc[first.category.eq("age65_plus"), "value"].iloc[0], 100.)
            self.assertEqual(first.loc[first.category.eq("adult18_plus"), "value"].iloc[0], 800.)
            self.assertEqual(set(first.population_reference_date), {"2025-12-01"})
            self.assertEqual(set(targets.loc[targets.locality_code.eq(2), "category"]), {"age_0_17"})
            table.iloc[5, 3] = "999"
            with patch("pandas.read_excel", return_value=table):
                with self.assertRaisesRegex(ValueError, "reconcile"):
                    nii_coded_age_targets(root)

    def response(self, body):
        response = BytesIO(body)
        response.headers = {"Content-Type": "application/vnd.ms-excel"}
        return response

    def test_pinned_nii_download_and_verified_cache(self):
        body = bytes.fromhex("d0cf11e0")+b"synthetic workbook"
        source = {"filename": "2023_locality_population.xls", "year": 2023,
                  "source_url": "https://www.btl.gov.il/synthetic.xls", "sha256": hashlib.sha256(body).hexdigest(),
                  "bytes": len(body), "publisher": "synthetic official publisher"}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("products.voting_propensity.model.demographic_foundation.NII_SOURCES", [source]):
                with patch("urllib.request.urlopen", return_value=self.response(body)) as fetch:
                    result = download_nii(Path(tmp))
                    self.assertEqual(fetch.call_count, 1)
                self.assertEqual(result[0]["year"], 2023)
                self.assertIn("retrieved_utc", result[0])
                with patch("urllib.request.urlopen") as fetch:
                    download_nii(Path(tmp))
                    fetch.assert_not_called()
                path = Path(tmp)/"nii_annual"/source["filename"]
                path.write_bytes(body+b"tampered")
                with self.assertRaisesRegex(ValueError, "hash"):
                    download_nii(Path(tmp))

    def test_download_rejects_html_wrong_pin_and_cached_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic.xls"
            with patch("urllib.request.urlopen", return_value=self.response(b"<html>blocked</html>")):
                with self.assertRaisesRegex(ValueError, "Expected workbook"):
                    archive_url("https://www.btl.gov.il/synthetic.xls", path)
            self.assertFalse(path.exists())
            body = bytes.fromhex("d0cf11e0")+b"synthetic"
            with patch("urllib.request.urlopen", return_value=self.response(body)):
                with self.assertRaisesRegex(ValueError, "pinned"):
                    archive_url("https://www.btl.gov.il/synthetic.xls", path, expected_sha256="0"*64)
            self.assertFalse(path.exists())
            with patch("urllib.request.urlopen", return_value=self.response(body)):
                archive_url("https://www.btl.gov.il/synthetic.xls", path)
            with self.assertRaisesRegex(ValueError, "URL/hash"):
                archive_url("https://www.btl.gov.il/another.xls", path)

    def test_manifest_has_complete_pinned_primary_series(self):
        self.assertEqual(len(NII_SOURCES), 19)
        self.assertEqual(len({source["filename"] for source in NII_SOURCES}), 19)
        self.assertEqual({source["year"] for source in NII_SOURCES}, {2003, 2006, *range(2009, 2026)})
        for source in NII_SOURCES:
            self.assertTrue(source["source_url"].startswith("https://www.btl.gov.il/"))
            self.assertEqual(len(source["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
