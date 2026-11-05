import unittest

import numpy as np
import pandas as pd

from historical_electorate import historical_controls, historical_specifications


class HistoricalElectorateTests(unittest.TestCase):
    def priors(self):
        return pd.DataFrame([dict(period_id=period,year=year,locality_code=10,
            broad_group=group,adult_population=value)
            for period,year,counts in [("K16",2003,[3600000,700000,200000]),
                                       ("K18",2009,[4000000,800000,300000]),
                                       ("2003",2003,[9999999,1,1])]
            for group,value in zip(["jewish","arab","other"],counts)])

    def test_backcast_uses_dated_group_ratios_and_keeps_observation_status(self):
        spec=historical_specifications(self.priors(),include_estimated_2003_split=True)["K16"]
        expected=np.array([3600000,700000,200000])*np.array([3888000/4000000,672000/800000,240000/300000])
        expected=expected/expected.sum()*4300000
        np.testing.assert_allclose(list(spec["counts"].values()),expected)
        self.assertEqual(spec["group_control_status"],"BACKCAST_2003_group_split_not_observed")
        self.assertEqual(spec["total_rounding"],50000)
        self.assertNotIn("K16",historical_specifications())

    def test_source_counts_and_pre_election_reference_dates(self):
        specs=historical_specifications()
        self.assertEqual(specs["K19"]["reference_date"],"2012-12-26")
        np.testing.assert_allclose(list(specs["K19"]["counts"].values()),[4131000,765000,204000])
        np.testing.assert_allclose(list(specs["K20"]["counts"].values()),[4240000,795000,265000])

    def test_sensitivity_changes_only_2003_and_preserves_both_constraints(self):
        eligible=pd.DataFrame([dict(election=e,year=y,locality_code=10,registered_eligible=n)
            for e,y,n in [("K16",2003,4700000),("K18",2009,5300000)]])
        primary,audits=historical_controls(eligible,self.priors(),include_estimated_2003_split=True)
        sensitivity,_=historical_controls(eligible,self.priors(),include_estimated_2003_split=True,
            split_2003="unadjusted_resident_mix")
        pd.testing.assert_frame_equal(primary.loc[primary.election.eq("K18")].reset_index(drop=True),
            sensitivity.loc[sensitivity.election.eq("K18")].reset_index(drop=True))
        for frame in [primary,sensitivity]:
            np.testing.assert_allclose(frame.groupby("election").estimated_eligible.sum(),[4700000,5300000],rtol=0,atol=1e-7)
            np.testing.assert_allclose(frame.groupby("election").resident_citizen_estimate.sum(),[4300000,4800000],rtol=0,atol=.01)
        self.assertTrue(primary.loc[primary.election.eq("K16"),"national_group_control_status"].str.contains("BACKCAST").all())
        self.assertIn("BACKCAST",audits[0]["group_split_assumption"])

    def test_missing_backcast_support_fails_instead_of_borrowing_another_year(self):
        with self.assertRaisesRegex(ValueError,"2009"):
            historical_specifications(self.priors().loc[lambda x:x.year.eq(2003)],include_estimated_2003_split=True)


if __name__=="__main__":
    unittest.main()
