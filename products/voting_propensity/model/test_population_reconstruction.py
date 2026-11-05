"""Population-only engine checks; no election outcomes or private data."""
import unittest
import numpy as np
import pandas as pd

import population_reconstruction as p


class PopulationReconstructionTests(unittest.TestCase):
    def test_lifecycle_uses_one_backbone_and_correct_split_parent(self):
        codes=np.array([3720,3778,3826,3779,3815,3774])
        prior=np.array([2.,3.,99.,10.,98.,7.])[:,None,None,None]*np.ones((6,7,2,3))
        before,_=p.dated_lifecycle_prior(prior,codes,"2021-03-23",{3779:3.,3815:7.})
        self.assertEqual(float(before[2].sum()),0.)
        self.assertEqual(float(before[4].sum()),0.)
        after,audit=p.dated_lifecycle_prior(prior,codes,"2026-10-27",{3779:3.,3815:7.})
        np.testing.assert_allclose(after[2],5.)
        np.testing.assert_allclose(after[3],3.)
        np.testing.assert_allclose(after[4],7.)
        np.testing.assert_array_equal(after[5],prior[5]) #3774isnotKochavYaakov
        self.assertAlmostEqual(float(after.sum()),float(prior.sum())-audit["discarded_synthetic_overlap"])

    def test_local_registry_growth_closes_merge_and_split_families(self):
        codes=np.array([3826,3779,3815,100,101])
        resident={3720:100.,3778:200.,3779:1000.,100:50.}
        old={3720:50.,3778:100.,3779:500.,100:25.}
        current={3826:300.,3779:300.,3815:200.,100:50.,101:30.}
        values,sources=p.local_register_growth_prior(codes,resident,old,current,np.ones(5),"2026-10-27")
        np.testing.assert_allclose(values,[600.,600.,400.,100.,30.])
        self.assertEqual(values[1]+values[2],1000.)
        self.assertIn("no_observed2020_base",sources[-1])

    def test_age_residual_fills_only_unobserved_components(self):
        known=np.array([[10.,np.nan,30.],[10.,20.,30.]])
        filled,unallocated=p.impute_age_residuals(known,np.array([60.,61.]),np.array([[.2,.5,.3],[.2,.5,.3]]),np.array([[False,True,False],[False,False,False]]))
        np.testing.assert_allclose(filled,[[10.,20.,30.],[10.,20.,30.]])
        np.testing.assert_allclose(unallocated,[0.,1.])

    def test_published_local_age_prior_precedes_regional_fallback(self):
        rows=[dict(locality_code=1,dimension="age",category=category,share=value)for category,value in
              [("age0_19",.6),("age20_64",.3),("age65_plus",.1)]]
        national=np.ones((86,2,3))
        fallback=np.ones((2,18))
        result,used,complete=p.local_age_backcast_prior(pd.DataFrame(rows),[1,2],national,national,fallback)
        self.assertEqual(used.tolist(),[True,False])
        self.assertEqual(complete.tolist(),[True,False])
        np.testing.assert_allclose([result[0,:4].sum(),result[0,4:13].sum(),result[0,13:].sum()],[.6,.3,.1])
        np.testing.assert_array_equal(result[1],fallback[1])

    def test_absolute_national_joint_and_relative_local_group_mass(self):
        rng=np.random.default_rng(4)
        prior=rng.uniform(.01,10,(5,7,2,3))
        target=rng.uniform(30,100,(7,2,3))
        actual,audit=p.rake_national_absolute(prior,target)
        np.testing.assert_allclose(actual.sum(axis=0),target,rtol=2e-8,atol=1e-6)
        factor=target.sum(axis=(0,1))/prior.sum(axis=(0,1,2))
        np.testing.assert_allclose(actual.sum(axis=(1,2)),prior.sum(axis=(1,2))*factor[None,:],rtol=1e-10)
        np.testing.assert_allclose(audit["group_level_rescale_factor"],factor)

    def test_founded_after_period_remains_zero_after_global_reconciliation(self):
        prior=np.ones((3,7,2,3))
        active=p.active_localities([1900,2007,np.nan],2003)
        self.assertEqual(active.tolist(),[True,False,True])
        prior[~active]=0
        fitted,_=p.rake_national_absolute(prior,np.ones((7,2,3))*100)
        self.assertEqual(float(fitted[1].sum()),0.)
        self.assertGreater(float(fitted[0].sum()),0.)

    def test_zero_target_group_and_new_positive_group(self):
        prior=np.ones((2,7,2,3))
        prior[0,:,:,1]=0
        desired=np.array([[14.,10.,0.],[14.,14.,14.]])
        supported=p._support_new_groups(prior,desired,np.ones((7,2,3)))
        fitted=p.rake_joint(supported,group=desired)
        np.testing.assert_allclose(fitted.sum(axis=(1,2)),desired)
        self.assertEqual(float(fitted[0,:,:,2].sum()),0.)
        self.assertGreater(float(fitted[0,:,:,1].sum()),0.)

    def test_observed_mask_requires_every_constituent_published65_cell(self):
        cols=["age_65_69","age_70_74","age_75_79","age_80_84","age_85_up"]
        raw=pd.DataFrame({"SEMEL_YISHUV":[1,1,2],"Pop_Total":[100,100,100],"Male_Total":[50,50,50],"Female_Total":[50,50,50],
                          **{name:[2.,2.,2.] for name in cols}})
        raw.loc[1,"age_85_up"]=np.nan
        age,sex,fraction=p.published_age65_targets(raw,[1,2])
        self.assertEqual(age.tolist(),[False,True])
        self.assertEqual(sex.tolist(),[True,True])
        self.assertAlmostEqual(float(fraction[1]),.1)
        # A known bin's source count never receives age-imputation residuals.
        raw["age_20_24"]=np.nan
        np.testing.assert_allclose(p.published_age65_targets(raw,[1,2])[2],fraction)

    def test_election_date_population_uses_observed_anchors_only(self):
        rows=[]
        for year in [2003,2009,2022,2023,2025,2026]:
            for age in range(86):
                for sex in ["male","female"]:
                    for group in ["jewish","arab","other"]:
                        rows.append(dict(year=year,age=age,sex=sex,broad_group=group,
                                         population=100*np.exp(.02*(year-2003))*(100 if year==2026 else 1),
                                         status="projected" if year==2026 else "observed"))
        controls=pd.DataFrame(rows)
        july=p.national_at_date(controls,"2026-07-01")
        october=p.national_at_date(controls,"2026-10-27")
        self.assertTrue((october>july).all())
        self.assertLess(float(october.max()),200.)
        november2022=p.national_at_date(controls,"2022-11-01")
        self.assertTrue((november2022>p.national_at_date(controls,"2022-07-01")).all())
        with self.assertRaises(ValueError):
            p.national_at_date(controls,"2028-01-01")


if __name__=="__main__":
    unittest.main()
