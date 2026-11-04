"""Count reconciliation, explicit exclusions and frozen historical reference tests."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from products.coalition_simulator.model.turnout_history import (
    GROUPS,SHARES,archived_records,balanced_panel,fit_turnout,normalize_counts,summarize_windows,
)


class TurnoutHistoryTests(unittest.TestCase):
    def counts(self):
        return pd.DataFrame({"סמל ישוב":[1,1,0],"שם ישוב":["A","A","External"],
            "בזב":[100,200,0],"מצביעים":[60,120,30],"כשרים":[59,118,29],"פסולים":[1,2,1]})

    def test_reconciles_then_aggregates_and_excludes_external_votes(self):
        frame,audit=normalize_counts(self.counts(),"K20")
        self.assertEqual(frame.eligible_voters.tolist(),[300])
        self.assertEqual(frame.total_voted.tolist(),[180])
        self.assertEqual(audit["source_totals"]["total_voted"],210)
        self.assertEqual(audit["excluded_envelope_or_zero_eligible_rows"],1)

    def test_rejects_nonreconciling_counts(self):
        counts=self.counts();counts.loc[0,"כשרים"]=60
        with self.assertRaisesRegex(ValueError,"valid \\+ invalid"):
            normalize_counts(counts,"K20")

    def test_invalid_locality_is_excluded_visibly_without_correcting_counts(self):
        counts=self.counts();counts.loc[0,"בזב"]=1;counts.loc[1,"בזב"]=1
        with self.assertRaisesRegex(ValueError,"locality voters exceed"):
            normalize_counts(counts,"K18")
        frame,audit=normalize_counts(counts,"K18",exclude_invalid_localities=True)
        self.assertEqual(len(frame),0)
        self.assertEqual(audit["excluded_invalid_locality_codes"],[1])
        self.assertEqual(audit["excluded_invalid_locality_totals"]["total_voted"],180)
        self.assertEqual(audit["source_totals"]["total_voted"],210)

    def test_ballot_overflow_does_not_drop_a_coherent_locality(self):
        counts=self.counts();counts.loc[0,"בזב"]=50
        frame,audit=normalize_counts(counts,"K22")
        self.assertEqual(frame.eligible_voters.tolist(),[250])
        self.assertEqual(len(audit["incoherent_source_rows"]),1)
        self.assertEqual(audit["excluded_invalid_locality_codes"],[])

    def test_fit_uses_eligible_weights(self):
        frame=pd.DataFrame(np.repeat(np.eye(6),2,axis=0),columns=SHARES)
        frame["eligible_voters"]=np.tile([900,100],6)
        frame["total_voted"]=np.tile([270,90],6)
        values,_=fit_turnout(frame)
        np.testing.assert_allclose(values,.36,atol=1e-10)

    def test_reference_excludes_2026_and_uses_population_sd(self):
        coefficients={f"K{k}":np.full(6,.5+.01*(k-21)) for k in range(21,26)}
        coefficients["K26"]=np.full(6,.8)
        rows=summarize_windows(coefficients,{"recent5":[f"K{k}" for k in range(21,26)]})
        self.assertEqual(len(rows),6)
        self.assertAlmostEqual(rows[0]["mean"],.52)
        self.assertAlmostEqual(rows[0]["sd_ddof0"],np.std([.50,.51,.52,.53,.54],ddof=0))
        self.assertAlmostEqual(rows[0]["k26_shift_pp"],28)
        self.assertEqual(rows[0]["distinct_calendar_years"],4)
        with self.assertRaisesRegex(ValueError,"2026 must be excluded"):
            summarize_windows(coefficients,{"bad":["K25","K26"]})

    def test_archive_hash_detects_raw_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"K20.json"
            payload=json.dumps({"success":True,"result":{"records":[{"a":1}],"total":1}}).encode()
            path.write_bytes(payload)
            path.with_suffix(".metadata.json").write_text(json.dumps({"sha256":hashlib.sha256(payload).hexdigest()}))
            self.assertEqual(archived_records(path),[{"a":1}])
            path.write_bytes(payload+b" ")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                archived_records(path)

    def test_balanced_sample_is_same_localities_every_year_and_row_order_invariant(self):
        panel=pd.DataFrame({"election":["K18","K18","K25","K25","K26","K26"],
                            "locality_code":[1,2,1,3,1,2]})
        expected=balanced_panel(panel)
        self.assertEqual(expected.locality_code.tolist(),[1,1,1])
        actual=balanced_panel(panel.sample(frac=1,random_state=42))
        self.assertEqual(set(map(tuple,expected.to_numpy())),set(map(tuple,actual.to_numpy())))

    def test_wrong_resource_identity_fails_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"K20.json"
            payload=json.dumps({"success":True,"result":{"resource_id":"wrong","records":[],"total":0}}).encode()
            path.write_bytes(payload)
            path.with_suffix(".metadata.json").write_text(json.dumps({"sha256":hashlib.sha256(payload).hexdigest(),"resource_id":"expected"}))
            with self.assertRaisesRegex(ValueError,"resource identity mismatch"):
                archived_records(path,"expected")


if __name__=="__main__":
    unittest.main()
