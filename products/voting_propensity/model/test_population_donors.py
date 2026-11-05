"""Fixed donor imputation contracts; synthetic Census profiles only."""
import unittest
import numpy as np
import pandas as pd
import population_donors as p


def micro():
    rows=[]
    for code,district,quarter,group,weight,rel,income,fsu in [
        (10,1,1,1,30,1,5,0),(10,1,2,1,10,1,5,0),
        (20,1,3,2,20,2,4,0),(20,1,3,2,10,5,15,1),
        (30,2,1,2,100,3,20,0),(9999,1,0,2,10,2,7,0)]:
        rows.append(dict(SmlYishuvPUF=code,MachozPUF=district,TatRovaKtvtMegurimPUF=quarter,
                         GilPUF=2,MinPUF=1,KvutzaUchlusiyaPUF=group,MishkalPratPUF=float(weight),DatiyutPUF=rel,
                         HchnsAvgChdshPratPUF=income,YabeshetMotzaMchlkPUF=1 if fsu else 0,OleShnot90PUF=0,
                         TzuratYishuvPUF=300 if code==9999 else 100,unused_feature=7))
    return pd.DataFrame(rows)


class DonorTests(unittest.TestCase):
    def test_same_district_profiles_quarters_and_original_unchanged(self):
        original=micro()
        before=original.copy(deep=True)
        controls=pd.DataFrame([dict(locality_code=10,age_band=2,sex="male",broad_group="arab",adult_population=8.,year=2022)])
        out,audit=p.expand_missing_cohorts(original,controls)
        pd.testing.assert_frame_equal(original,before)
        pd.testing.assert_frame_equal(out.iloc[:len(original)].drop(columns=p.MARKER).reset_index(drop=True),original,check_dtype=False)
        added=out[out[p.MARKER]]
        self.assertAlmostEqual(added.MishkalPratPUF.sum(),8.)
        q=added.groupby("TatRovaKtvtMegurimPUF").MishkalPratPUF.sum()
        np.testing.assert_allclose(q.to_numpy(),[6.,2.])
        self.assertEqual(audit["cohorts"][0]["donor_scope"],"same_district")
        self.assertNotIn(20,set(added.HchnsAvgChdshPratPUF))
        self.assertTrue((added.SmlYishuvPUF==10).all())
        self.assertEqual(audit["estimated_new_model_parameters"],0)

    def test_union_periods_uses2022placeholder_not_sum(self):
        controls=pd.DataFrame([dict(locality_code=10,age_band=2,sex="male",broad_group="arab",adult_population=n,year=y)
                               for y,n in [(2019,20.),(2022,8.),(2026,40.)]])
        out,audit=p.expand_missing_cohorts(micro(),controls)
        self.assertAlmostEqual(out.loc[out[p.MARKER],"MishkalPratPUF"].sum(),8.)
        self.assertEqual(audit["missing_cohorts_filled"],1)

    def test_existing_supported_cohort_adds_no_rows(self):
        controls=pd.DataFrame([dict(locality_code=10,age_band=2,sex="male",broad_group="jewish",adult_population=8.)])
        out,audit=p.expand_missing_cohorts(micro(),controls)
        self.assertEqual(len(out),len(micro()))
        self.assertEqual(audit["added_rows"],0)

    def test_national_fallback_and_absent_cohort_fail(self):
        original=micro()
        original.loc[original.SmlYishuvPUF.eq(10),"MachozPUF"]=7
        controls=pd.DataFrame([dict(locality_code=10,age_band=2,sex="male",broad_group="arab",adult_population=8.)])
        out,audit=p.expand_missing_cohorts(original,controls)
        self.assertEqual(audit["cohorts"][0]["donor_scope"],"national_same_cohort")
        controls.loc[:,"sex"]="female"
        with self.assertRaises(ValueError):
            p.expand_missing_cohorts(original,controls)

    def test_rural_locality_uses_only_pooled9999_not_urban(self):
        controls=pd.DataFrame([dict(locality_code=99,age_band=2,sex="male",broad_group="arab",adult_population=8.)])
        geography=pd.DataFrame([dict(locality_code=99,district_code=1,subdistrict_code=11,puf_locality_type=300)])
        out,audit=p.expand_missing_cohorts(micro(),controls,geography)
        added=out[out[p.MARKER]]
        self.assertEqual(set(added.HchnsAvgChdshPratPUF),{7})
        self.assertEqual(set(added.TzuratYishuvPUF),{300})
        self.assertTrue(audit["cohorts"][0]["donor_rural_only"])
        self.assertEqual(added.MishkalPratPUF.sum(),8.)

    def test_masked_small_urban_form_is_not_relabelled_rural(self):
        original=micro()
        urban=original.loc[original.SmlYishuvPUF.eq(9999)].copy()
        urban["TzuratYishuvPUF"]=185
        urban["HchnsAvgChdshPratPUF"]=9
        original=pd.concat([original,urban],ignore_index=True)
        controls=pd.DataFrame([dict(locality_code=43,age_band=2,sex="male",broad_group="arab",adult_population=8.)])
        geography=pd.DataFrame([dict(locality_code=43,district_code=1,puf_locality_type=185)])
        out,audit=p.expand_missing_cohorts(original,controls,geography)
        self.assertEqual(set(out.loc[out[p.MARKER],"TzuratYishuvPUF"]),{185})
        self.assertEqual(set(out.loc[out[p.MARKER],"HchnsAvgChdshPratPUF"]),{9})
        self.assertFalse(audit["cohorts"][0]["donor_rural_only"])
        out,audit=p.expand_missing_cohorts(original,controls,geography.drop(columns="puf_locality_type"))
        self.assertTrue(out.loc[out[p.MARKER],"TzuratYishuvPUF"].isna().all())
        self.assertFalse(audit["cohorts"][0]["donor_rural_only"])


if __name__=="__main__":
    unittest.main()
