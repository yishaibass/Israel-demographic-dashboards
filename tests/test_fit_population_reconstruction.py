"""Population weighting invariants and separate resident/electorate contexts."""
import numpy as np
import pandas as pd
import pytest
import json
from types import SimpleNamespace
import torch

from fit_population_reconstruction import apply_eligible_mixture, reconstructed_micro, model_record, restore_model
import fit_foundation_history as history


def fixtures():
    micro = pd.DataFrame(dict(SmlYishuvPUF=[10]*5, GilPUF=[2]*5, MinPUF=[1]*5,
                              KvutzaUchlusiyaPUF=[1,1,1,2,3], DatiyutPUF=[1,5,None,None,None],
                              MishkalPratPUF=[40.,10.,5.,20.,5.], income_fixture=[1,2,3,4,5]))
    controls = pd.DataFrame([dict(year=2026, period_id="2026", locality_code=10, age_band=2,
                                  sex="male", broad_group=group, adult_population=amount)
                             for group, amount in [("jewish",110), ("arab",30), ("other",0)]])
    factors = pd.DataFrame([dict(year=2026, age_band="18_24", sex="male", religiosity=group,
                                 relative_factor=factor) for group, factor in [("secular",1.),("haredi",2.)]])
    return micro, controls, factors


def test_dated_weights_preserve_cell_mass_and_conditional_structure():
    micro, controls, factors = fixtures()
    updated, audit = reconstructed_micro(micro, controls, factors, 2026)
    assert updated.MishkalPratPUF.sum() == pytest.approx(140)
    assert updated.loc[updated.KvutzaUchlusiyaPUF.eq(1), "MishkalPratPUF"].sum() == pytest.approx(110)
    assert updated.loc[updated.DatiyutPUF.isna() & updated.KvutzaUchlusiyaPUF.eq(1), "MishkalPratPUF"].item() == pytest.approx(10)
    assert (updated.MishkalPratPUF > 0).all()
    assert audit["unrepresented_control_mass_within_PUF_localities"] == 0
    assert audit["selected_period_id"] == "2026"
    assert updated.income_fixture.tolist() == [1,2,3,4]
    pd.testing.assert_series_equal(micro.MishkalPratPUF, pd.Series([40.,10.,5.,20.,5.],name="MishkalPratPUF"))


def test_missing_religiosity_or_duplicate_controls_fail():
    micro, controls, factors = fixtures()
    with pytest.raises(ValueError, match="Incomplete religiosity"):
        reconstructed_micro(micro, controls, factors.iloc[:1], 2026)
    with pytest.raises(ValueError, match="duplicated"):
        reconstructed_micro(micro, pd.concat([controls, controls]), factors, 2026)


def test_eligible_weights_change_mixture_without_changing_resident_context():
    cells = pd.DataFrame(dict(locality=[3000]*3, arab=[0,1,0], other_pop=[0,0,1],
                              population_weight=[600.,400.,100.], weight=[0.,0.,0.],
                              geo_arab_share=[.4,.4,.4], locality_logpop_z=[1.,1.,1.]))
    targets = pd.DataFrame(dict(locality=[3000],eligible=[1000]))
    priors = pd.DataFrame([dict(year=2026,locality_code=3000,broad_group=g,adult_population=p)
                           for g,p in [("jewish",600.),("arab",400.),("other",100.)]])
    updated = apply_eligible_mixture(cells, targets, priors, 2026, "K26")
    assert updated.weight.sum() == pytest.approx(1000)
    assert updated.loc[updated.arab.eq(1), "weight"].item() < 60
    np.testing.assert_array_equal(updated.geo_arab_share, cells.geo_arab_share)
    np.testing.assert_array_equal(updated.locality_logpop_z, cells.locality_logpop_z)
    np.testing.assert_array_equal(updated.population_weight, cells.population_weight)
    resident = apply_eligible_mixture(cells, targets, priors, 2026, "K26",citizen_adjustment=False)
    assert resident.loc[resident.arab.eq(1), "weight"].item() == pytest.approx(1000*400/1100)


def test_calibrated_eligible_controls_are_used_once_and_reject_negative_mass():
    cells = pd.DataFrame(dict(locality=[3000]*3, arab=[0,1,0], other_pop=[0,0,1],
                              population_weight=[600.,400.,100.], weight=[0.,0.,0.]))
    targets = pd.DataFrame(dict(locality=[3000],eligible=[1000]))
    allocated = pd.DataFrame(dict(election=["K26"]*3,locality_code=[3000]*3,
                                  broad_group=["jewish","arab","other"],estimated_eligible=[800.,150.,50.]))
    result = apply_eligible_mixture(cells, targets, pd.DataFrame(), 2026, "K26", allocated_controls=allocated)
    np.testing.assert_allclose(result.weight, [800,150,50])
    allocated.estimated_eligible = [1100.,-150.,50.]
    with pytest.raises(ValueError,match="Invalid allocated"):
        apply_eligible_mixture(cells, targets, pd.DataFrame(), 2026, "K26", allocated_controls=allocated)


def test_exported_coefficients_restore_identical_election_probabilities():
    elections = [SimpleNamespace(election="K25"), SimpleNamespace(election="K26")]
    with history.election_scope([data.election for data in elections]):
        fitted = history.common.model.FamilyModel(len(history.FEATURES),len(history.FAMILIES),False)
    with torch.no_grad():
        fitted.turn_shared[:] = torch.arange(len(history.FEATURES)) / 100
        fitted.turn_dev[1,0] = .3
        fitted.choice_shared[:] = torch.arange(len(history.FEATURES)*len(history.FAMILIES)).reshape(len(history.FEATURES),len(history.FAMILIES)) / 100
    record = json.loads(json.dumps(model_record(fitted,elections,{"model_fit_id":"synthetic"})))
    restored = restore_model(record)
    x = torch.tensor([[.1]*len(history.FEATURES),[.5]*len(history.FEATURES)])
    fsu = torch.zeros(2)
    for ei in range(2):
        for before,after in zip(fitted.probs(x,fsu,ei),restored.probs(x,fsu,ei)):
            assert torch.equal(before,after)
    record["features"] = ["different"]
    with pytest.raises(ValueError,match="schema"):
        restore_model(record)
