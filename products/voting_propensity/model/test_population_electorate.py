import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from population_electorate import (DRUZE_SHARES_2022, JERUSALEM_STATUS, allocate_electorate,
    fetch_source, jerusalem_adult_rate, jerusalem_citizenship_controls, jerusalem_wholecity_scope_rate,
    normalize_registry, registry_adult_controls,calibrate_national_electorate,rake_register_matrix,verify_registry_source_date)


class PopulationElectorateTests(unittest.TestCase):
    def eligible(self):
        return pd.DataFrame([dict(election="K25", year=2022, locality_code=code, registered_eligible=n, source_id="synthetic_CEC")
                             for code, n in [(3000, 1000), (10, 500), (20, 600), (9999, 50)]])

    def priors(self):
        return pd.DataFrame([dict(year=2022, locality_code=code, broad_group=group, adult_population=count)
            for code, group, count in [(3000,"jewish",600),(3000,"arab",400),(10,"jewish",200),(10,"arab",100),(9999,"jewish",900)]])

    def test_official_citizenship_table_totals_and_adult_rate(self):
        table = jerusalem_citizenship_controls()
        self.assertEqual(table.value.sum(), 370332)
        rate, citizen, adults = jerusalem_adult_rate()
        self.assertEqual(citizen, 18569)
        self.assertEqual(adults, 227887)
        self.assertAlmostEqual(rate, 18569/227887)

    def test_exact_electorate_mass_closure_and_no_false_masked_locality_identity(self):
        fit = allocate_electorate(self.eligible(), self.priors())
        got = fit.groupby("locality_code").estimated_eligible.sum()
        np.testing.assert_array_equal(got.to_numpy(), [500,600,1000,50])
        for code in [20,9999]:
            self.assertEqual(fit.loc[fit.locality_code.eq(code), "broad_group"].tolist(), ["unallocated"])
        jerusalem_arab = fit.loc[fit.locality_code.eq(3000)&fit.broad_group.eq("arab")].iloc[0]
        self.assertLess(jerusalem_arab.estimated_eligible, 60)
        self.assertEqual(jerusalem_arab.citizenship_source_year, 2024)

    def test_register_excess_remains_unknown_in_conservative_scenario(self):
        fit = allocate_electorate(self.eligible(), self.priors(), "unallocated_register_excess")
        town = fit.loc[fit.locality_code.eq(10)]
        self.assertEqual(town.loc[town.broad_group.eq("unallocated_register_gap"), "estimated_eligible"].iloc[0], 200)
        self.assertEqual(town.estimated_eligible.sum(), 500)

    def test_wholecity_scope_keeps_observed_East_citizens_and_flags_bad_universe(self):
        rate,audit=jerusalem_wholecity_scope_rate(250000)
        self.assertAlmostEqual(rate*250000,18569+(250000-227887))
        self.assertEqual(audit["East_noncitizen_or_unconfirmed_adults"],209318)
        self.assertGreater(rate,jerusalem_adult_rate()[0])
        with self.assertRaisesRegex(ValueError,"incompatible"):
            jerusalem_wholecity_scope_rate(220000)

    def test_no_election_outcome_path_and_bad_population_fails(self):
        eligible = self.eligible()
        with_outcomes = eligible.assign(total_voted=-123, votes_any_party=999999)
        pd.testing.assert_frame_equal(allocate_electorate(eligible,self.priors()), allocate_electorate(with_outcomes,self.priors()))
        bad = self.priors();bad.loc[0,"adult_population"] = -1
        with self.assertRaises(ValueError):
            allocate_electorate(eligible,bad)

    def test_joint_calibration_closes_local_register_and_national_resident_citizens(self):
        specs={"K25":{"counts":{"jewish":700.,"arab":300.,"other":100.},"source":"synthetic_official_CBS",
                       "precision":"synthetic exact","total_rounding":0}}
        controls,audit=calibrate_national_electorate(self.eligible(),self.priors(),specifications=specs)
        sums=controls.groupby("broad_group").resident_citizen_estimate.sum()
        np.testing.assert_allclose(sums.loc[["jewish","arab","other"]],[700,300,100],rtol=0,atol=1e-7)
        local=controls.groupby("locality_code").estimated_eligible.sum()
        np.testing.assert_allclose(local.to_numpy(),[500,600,1000,50],rtol=0,atol=1e-8)
        gaps=controls[["locality_code","unallocated_gap"]].drop_duplicates().unallocated_gap.sum()
        self.assertAlmostEqual(gaps,2150-1100,places=6)
        self.assertEqual(audit[0]["unsupported_locality_eligible_mass"],650)
        self.assertTrue(controls.loc[controls.locality_code.eq(20),"prior_support"].str.contains("fallback").all())
        with_outcomes=self.eligible().assign(total_voted=-999,votes_test=999999)
        repeated,_=calibrate_national_electorate(with_outcomes,self.priors(),specifications=specs)
        pd.testing.assert_frame_equal(controls,repeated)
        self.assertTrue(controls.estimated_eligible.ge(0).all())

    def test_impossible_calibration_fails_instead_of_borrowing_population(self):
        with self.assertRaisesRegex(ValueError,"capacity"):
            rake_register_matrix([[1.,0.],[0.,1.]],[10.,10.],[15.,5.])

    def test_audit_distinguishes_full_geography_from_cec_locality_frame(self):
        specs={"K25":{"counts":{"jewish":700.,"arab":300.,"other":100.},"source":"synthetic_official_CBS",
                       "precision":"synthetic exact","total_rounding":0}}
        prior=self.priors()
        outside=pd.DataFrame([dict(year=2022,locality_code=30,broad_group="jewish",adult_population=25.)])
        original,_=calibrate_national_electorate(self.eligible(),prior,specifications=specs)
        same,audits=calibrate_national_electorate(self.eligible(),pd.concat([prior,outside]),specifications=specs)
        pd.testing.assert_frame_equal(original,same)
        audit=audits[0]
        self.assertEqual(audit["cec_locality_frame_resident_adult_priors"]["jewish"],800.)
        self.assertEqual(audit["all_geography_resident_adult_priors"]["jewish"],825.)
        self.assertEqual(audit["resident_prior_outside_cec_locality_frame"]["jewish"],25.)
        self.assertEqual(audit["resident_prior_locality_codes_outside_cec_frame"],[30])
        self.assertEqual(audit["resident_citizen_target_minus_cec_frame_adult_prior"]["jewish"],-100.)
        self.assertEqual(audit["resident_citizen_target_minus_all_geography_adult_prior"]["jewish"],-125.)

    def test_election_period_priors_override_calendar_without_double_counting(self):
        calendar=self.priors().assign(period_id="2022")
        exact=self.priors().assign(period_id="K25")
        exact.loc[exact.broad_group.eq("jewish"),"adult_population"]*=2
        priors=pd.concat([calendar,exact],ignore_index=True)
        result=allocate_electorate(self.eligible(),priors)
        self.assertEqual(set(result.population_prior_period),{"K25"})
        row=result.loc[result.locality_code.eq(3000)&result.broad_group.eq("jewish")].iloc[0]
        self.assertEqual(row.resident_adult_prior,1200)
        specs={"K25":{"counts":{"jewish":700.,"arab":300.,"other":100.},"source":"synthetic_official_CBS","precision":"synthetic exact","total_rounding":0}}
        controls,_=calibrate_national_electorate(self.eligible(),priors,specifications=specs)
        self.assertEqual(controls.loc[controls.locality_code.eq(3000)&controls.broad_group.eq("jewish"),"resident_adult_prior"].iloc[0],1200)

    def registry(self):
        row = {"_id":1,"סמל_ישוב":10,"שם_ישוב":"Synthetic","סמל_נפה":11,"קוד_מועצה_אזורית":1,
               "סהכ":100,"גיל_0_5":10,"גיל_6_18":20,"גיל_19_45":30,"גיל_46_55":15,"גיל_56_64":15,"גיל_65_פלוס":10}
        return {"success":True,"result":{"resource_id":"test","total":1,"records":[row]}}

    def test_registry_is19plus_and_age18_is_explicitly_estimated(self):
        registry = normalize_registry(self.registry(),"2026-11-01","test")
        self.assertNotIn("age18_plus",set(registry.category))
        national = pd.DataFrame([dict(year=2026,age=age,population=2 if age==18 else 1,source_id="synthetic_national") for age in range(6,19)])
        adult = registry_adult_controls(registry,national).iloc[0]
        self.assertEqual(adult.registered_age19_plus_observed,70)
        self.assertAlmostEqual(adult.registered_age18_estimated,20*2/14)
        self.assertFalse(adult.citizenship_restricted)
        with self.assertRaises(ValueError):
            payload=self.registry();payload["result"]["total"]=2
            normalize_registry(payload,"2026-11-01","test")

    def test_source_cache_tampering_fails_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.json";content=b'{}';path.write_bytes(content)
            path.with_name(path.name+".metadata.json").write_text(json.dumps({"source_url":"https://example.invalid/source","sha256":hashlib.sha256(content).hexdigest()}))
            self.assertEqual(fetch_source("https://example.invalid/source",path),content)
            path.write_bytes(b'{} ')
            with self.assertRaisesRegex(ValueError,"hash"):
                fetch_source("https://example.invalid/source",path)

    def test_rolling_registry_source_cannot_be_misdated(self):
        source={"id":"rolling","url":"https://data.gov.il/download/registry__2026_11_01_03_30.csv"}
        verify_registry_source_date(source,"2026-11-01","rolling")
        with self.assertRaisesRegex(ValueError,"pinned reference date"):
            verify_registry_source_date(source,"2026-12-01","rolling")

    def test_mixed_towns_are_fractional_druze_controls(self):
        self.assertEqual(DRUZE_SHARES_2022[8800],.14)
        self.assertEqual(DRUZE_SHARES_2022[481],.57)
        self.assertEqual(DRUZE_SHARES_2022[473],.30)


if __name__=="__main__":
    unittest.main()
