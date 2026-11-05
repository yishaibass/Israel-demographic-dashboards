"""Accounting conservation and input identities for historical share changes."""
import numpy as np
import pandas as pd
import pytest

from products.voting_propensity.model import historical_trends as trends


def test_population_only_change_is_assigned_to_eligible_mix():
    before = (np.array([.5,.5]),np.array([.6,.6]))
    after = (np.array([.7,.3]),np.array([.6,.6]))
    a,b,c = trends.shapley_changes(trends.voter_shares,before,after)
    np.testing.assert_allclose(c[0],[.2,-.2])
    np.testing.assert_allclose(c[1],0)
    np.testing.assert_allclose(c.sum(0),b-a)


def test_turnout_only_change_keeps_population_contribution_zero():
    before = (np.array([.5,.5]),np.array([.5,.5]))
    after = (np.array([.5,.5]),np.array([1.,.5]))
    _,_,c = trends.shapley_changes(trends.voter_shares,before,after)
    np.testing.assert_allclose(c[0],0)
    np.testing.assert_allclose(c[1],[1/6,-1/6])


def test_two_factor_normalization_interaction_is_shared_symmetrically():
    before = (np.array([.5,.5]),np.array([.5,.5]))
    after = (np.array([.75,.25]),np.array([1.,.5]))
    _,_,c = trends.shapley_changes(trends.voter_shares,before,after)
    np.testing.assert_allclose(c[:,0],[37/168,23/168])
    np.testing.assert_allclose(c.sum(axis=1),0,atol=1e-15)


def test_preference_only_change_is_assigned_to_support():
    p,t = np.array([.6,.4]),np.array([.5,.75])
    before = (p,t,np.array([[1.,0.],[0.,1.]]))
    after = (p,t,np.array([[.8,.2],[.1,.9]]))
    a,b,c = trends.shapley_changes(trends.group_bloc_shares,before,after)
    np.testing.assert_allclose(c[:2],0)
    np.testing.assert_allclose(c[2].sum(0),[-.05,.05])
    np.testing.assert_allclose(c.sum(0),b-a)


def test_three_factor_accounting_conserves_every_group_and_category():
    before = (np.array([.2,.5,.3]),np.array([.4,.7,.8]),np.array([[.8,.1,.1],[.3,.6,.1],[.1,.2,.7]]))
    after = (np.array([.3,.4,.3]),np.array([.7,.6,.9]),np.array([[.7,.2,.1],[.2,.6,.2],[.1,.1,.8]]))
    a,b,c = trends.shapley_changes(trends.group_bloc_shares,before,after)
    np.testing.assert_allclose(c.sum(0),b-a,atol=1e-14)
    np.testing.assert_allclose(c.sum(axis=(1,2)),0,atol=1e-14)
    np.testing.assert_allclose(a.sum(),1)
    np.testing.assert_allclose(b.sum(),1)


def inputs():
    rows,choices = [],[]
    for election,p,t in [("K16",[.4,.6],[.5,.5]),("K25",[.5,.5],[.6,.7]),("K26",[.6,.4],[.7,.8])]:
        voter = trends.voter_shares(p,t)
        for i,group in enumerate(["A","B"]):
            rows.append(dict(variant="synthetic",election=election,group=group,estimated_eligible=1000*p[i],
                             estimated_eligible_share=p[i],turnout=t[i],estimated_votes=1000*p[i]*t[i],voter_share=voter[i]))
            for category,value in zip(trends.CATEGORIES,[.6,.3,.1] if i==0 else [.3,.4,.3]):
                choices.append(dict(variant="synthetic",election=election,group=group,category=category,value=value))
    return pd.DataFrame(rows),pd.DataFrame(choices)


def test_outputs_include_adjacent_and_both_long_run_endpoints():
    composition,preferences=inputs()
    voter,blocs,group_blocs,levels=trends.build(composition,preferences)
    assert len(voter)==8 and len(blocs)==12 and len(group_blocs)==24 and len(levels)==9
    assert set(voter.comparison)=={"adjacent_elections","2003_to_2022","2003_to_2026"}
    np.testing.assert_allclose(voter.change_pp,voter.eligible_mix_component_pp+voter.turnout_component_pp)
    np.testing.assert_allclose(blocs.change_pp,blocs.eligible_mix_component_pp+blocs.turnout_component_pp+blocs.within_group_support_component_pp)
    report=trends.summary(composition,voter,blocs)
    assert report["direct_individual_voter_flows_identified"] is False
    assert report["long_run_largest_group_share_changes"]


def test_inputs_cannot_silently_reconcile_inconsistent_shares_or_identities():
    c,p=inputs()
    bad=c.copy();bad.loc[0,"voter_share"]+=.1
    with pytest.raises(ValueError,match="Voter shares"):
        trends.build(bad,p)
    with pytest.raises(ValueError,match="Duplicate"):
        trends.build(pd.concat([c,c.iloc[:1]]),p)
    with pytest.raises(ValueError,match="groups differ"):
        trends.build(c,p.iloc[1:])


def test_historical_merged_ballot_families_stay_identifiable_in_stable_categories():
    assert trends.stable_category(["likud","secular_national_right"])==trends.CATEGORIES[0]
    assert trends.stable_category(["national_religious_right","shas"])==trends.CATEGORIES[0]
    assert trends.stable_category(["liberal_center","zionist_left"])==trends.CATEGORIES[1]
    assert trends.stable_category(["zionist_left","other"])==trends.CATEGORIES[1]
    with pytest.raises(ValueError,match="cross"):
        trends.stable_category(["likud","liberal_center"])


def test_political_voter_base_is_conditioned_on_each_category():
    composition,preferences=inputs()
    base=trends.political_voter_base(composition,preferences)
    np.testing.assert_allclose(base.groupby(["variant","election","category"]).voter_base_share.sum(),1)
    np.testing.assert_allclose(base.groupby(["variant","election"]).share_of_all_votes.sum(),1)
    # K16: A supplies .4*.6 / (.4*.6 + .6*.3) of right-family voters.
    row=base.loc[base.election.eq("K16") & base.group.eq("A") & base.category.eq(trends.CATEGORIES[0])].iloc[0]
    assert row.voter_base_share==pytest.approx(4/7)
    assert row.share_of_all_votes==pytest.approx(.24)


def test_cli_writes_serializable_summary_and_share_tables(tmp_path,monkeypatch):
    import json
    import sys
    composition,preferences=inputs()
    composition.to_csv(tmp_path/"composition.csv",index=False)
    preferences.to_csv(tmp_path/"preferences.csv",index=False)
    output=tmp_path/"results"
    monkeypatch.setattr(sys,"argv",["historical_trends.py","--composition",str(tmp_path/"composition.csv"),
                                  "--preferences",str(tmp_path/"preferences.csv"),"--output",str(output)])
    trends.main()
    report=json.loads((output/"historical_trends_summary.json").read_text(encoding="utf-8"))
    assert len(list(output.glob("*.csv")))==5
    assert report["long_run_largest_group_share_changes"]


def test_export_roundoff_correction_is_bounded_and_audited():
    composition,preferences=inputs()
    preferences.loc[0,"value"]+=1e-7
    normalized,audit=trends.normalized_preferences(preferences)
    np.testing.assert_allclose(normalized.groupby(trends.IDENTITY).value.sum(),1,atol=1e-15)
    assert audit["max_input_probability_sum_deviation"]==pytest.approx(1e-7)
    assert 0<audit["max_probability_renormalization_delta"]<1e-7
    voter,blocs,_,_=trends.build(composition,preferences)
    np.testing.assert_allclose(blocs.groupby(["variant","from_election","to_election","comparison"]).change_pp.sum(),0,atol=1e-12)
    preferences.loc[0,"value"]+=1e-4
    with pytest.raises(ValueError,match="sum to one"):
        trends.build(composition,preferences)
