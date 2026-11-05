"""Contract tests for unseen-locality prediction and eligible-population counts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from products.coalition_simulator.model.ei_benchmark import pyei_adapter as adapter


class PyEIPredictionTests(unittest.TestCase):
    def fitted(self):
        concentrations=np.tile(np.array([[7.,3.],[2.,8.]]),(400,1,1))
        return adapter.FittedPyEI(concentrations,{},("a","b"),("training-only",),0)

    def test_group_mean_uses_hierarchy_and_voted_first(self):
        np.testing.assert_allclose(self.fitted().group_mean_draws,np.tile([.7,.2],(400,1)))

    def test_predict_is_for_unseen_localities_with_count_bounds(self):
        fitted=self.fitted()
        draws=fitted.predict([[1,0],[0,1]],[100,200],seed=2)
        self.assertEqual(draws.shape,(400,2))
        self.assertTrue(np.issubdtype(draws.dtype,np.integer))
        self.assertTrue(np.all((draws>=0)&(draws<=[100,200])))
        self.assertAlmostEqual(draws[:,0].mean()/100,.7,delta=.03)
        self.assertAlmostEqual(draws[:,1].mean()/200,.2,delta=.03)
        np.testing.assert_array_equal(draws,fitted.predict([[1,0],[0,1]],[100,200],seed=2))

    def test_predict_includes_new_locality_heterogeneity(self):
        draws=self.fitted().predict([[1,0]],[1000],seed=42)[:,0]/1000
        # Binomial-only SD would be about .0145; hierarchy SD is about .138.
        self.assertGreater(draws.std(),.10)

    def test_identical_new_localities_get_independent_rate_draws(self):
        draws=self.fitted().predict([[1,0],[1,0]],[100000,100000],seed=42)
        self.assertLess(abs(np.corrcoef(draws.T)[0,1]),.15)

    def test_rejects_voter_denominator_and_fractional_counts(self):
        with self.assertRaisesRegex(ValueError,"between zero and eligible"):
            adapter.fit([[1,0]],[10],[11],seed=1)
        with self.assertRaisesRegex(ValueError,"positive integer"):
            adapter.validate_predictors([[1,0]],[10.5])

    def test_rejects_silent_share_repair(self):
        with self.assertRaisesRegex(ValueError,"no silent renormalization"):
            adapter.validate_predictors([[.3,.6]],[100])

    def test_checks_pinned_source_and_license_before_import(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"r_by_c_models.py").write_text("not the pinned model")
            (Path(folder)/"LICENSE").write_text("not the pinned license")
            with patch.dict("os.environ",{"ISRAEL_PYEI_SOURCE_DIR":folder}):
                with self.assertRaisesRegex(ValueError,"hash mismatch"):
                    adapter.load_upstream_builder()

    def test_explicit_numpyro_backend_forwards_chain_execution(self):
        self.assertEqual(adapter.sampling_engine_options({}),{"nuts_sampler":"pymc"})
        options=adapter.sampling_engine_options({"nuts_sampler":"numpyro","chain_method":"vectorized"})
        self.assertEqual(options,{"nuts_sampler":"numpyro","nuts_sampler_kwargs":{"chain_method":"vectorized"}})
        with self.assertRaisesRegex(ValueError,"chain_method"):
            adapter.sampling_engine_options({"nuts_sampler":"numpyro","chain_method":"unknown"})

    def test_saved_hierarchy_reproduces_predictive_draws_without_training_outcomes(self):
        import tempfile
        fitted=self.fitted()
        fitted.locality_ids=tuple(np.array([123,456],dtype=np.int64))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"hierarchy.npz"
            fitted.save_parameters(path)
            restored=adapter.load_parameters(path)
            self.assertEqual(restored.locality_ids,(123,456))
            self.assertFalse(restored.diagnostics["diagnostic_gate_passed"])
            np.testing.assert_array_equal(restored.group_mean_draws,fitted.group_mean_draws)
            np.testing.assert_array_equal(restored.predict([[.3,.7]],[1000],seed=3),
                                          fitted.predict([[.3,.7]],[1000],seed=3))
            with np.load(path,allow_pickle=False) as saved:
                self.assertEqual(set(saved.files),{"concentration_draws","metadata"})

    def test_each_missing_selected_diagnostic_rejects_gate_instead_of_skipping_nan(self):
        import pandas as pd
        class Stats(dict):
            __getattr__=dict.__getitem__
        trace=SimpleNamespace(sample_stats=Stats(diverging=np.zeros((2,1000))),
                              posterior=SimpleNamespace(sizes={"chain":2,"draw":1000}))
        good=pd.DataFrame({"r_hat":[1.,1.],"ess_bulk":[900.,900.],"ess_tail":[800.,800.]})
        for column in good.columns:
            with self.subTest(column=column):
                bad=good.copy();bad.loc[1,column]=np.nan
                fake=SimpleNamespace(summary=Mock(side_effect=[bad,good]))
                with patch.dict(sys.modules,{"arviz":fake}):
                    result=adapter.trace_diagnostics(trace)
                self.assertFalse(result["all_selected_finite"])
                self.assertFalse(result["diagnostic_gate_passed"])
                self.assertEqual(result["nonfinite_counts"]["rhat" if column=="r_hat" else column],1)
        with patch.dict(sys.modules,{"arviz":SimpleNamespace(summary=Mock(side_effect=[good,good]))}):
            result=adapter.trace_diagnostics(trace)
        self.assertTrue(result["all_selected_finite"])
        self.assertTrue(result["diagnostic_gate_passed"])
        self.assertEqual(result["diagnostic_value_counts"],{"rhat":2,"ess_bulk":2,"ess_tail":2})


if __name__=="__main__":
    unittest.main()
