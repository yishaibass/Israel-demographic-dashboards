# Reusable turnout-model benchmark

This is an isolated backend experiment on `research/turnout-ei-benchmark`.
It does not write published model parameters, HTML, styles or controls.

## Design agreed before candidate scoring

The input is the balanced panel of 1,110 localities across K18–K26. Population
means eligible voters; outcomes are voted and eligible-but-did-not-vote. Six
group shares use the fixed K25/2022 Census bridge. They are demographic proxies,
not observed group eligibility counts.

Whole CBS subdistricts are assigned to three folds once, using locality identities,
eligibility and a fixed seed. Every locality's election history stays in one fold.
Each fit receives training outcomes only. A held-out prediction receives group
shares and eligible counts, never observed turnout.

Compare these models on the identical splits:

- The existing six-share bounded weighted-least-squares specification.
- A WLS training-residual predictive benchmark. Its intervals predict unseen
  locality counts; production coefficient intervals are not treated as predictions.
- The MIT-licensed PyEI multinomial–Dirichlet model, using its actual model builder.
- The MIT-licensed Rebane aggregate-turnout EI model, using its actual probit builder.
- A joint Rebane prototype with common election, group-by-election and geographic
  effects, retaining its locality random intercepts.
- A smaller adaptation of that builder with a Beta–Binomial count likelihood,
  retaining regional dependence while integrating out locality overdispersion.
  This changes the observation model; it is not the original model fitted faster.
- An exploratory regional empirical-residual candidate, added after the original
  models' runtime screens and before its own scores were inspected. It retains
  WLS point estimates and the baseline's eligible-weighted marginal residual
  distribution. Each new region draws one training region, then its localities
  draw residuals within that source region. The controlled change is geographic
  dependence; the same practical improvement threshold applies.

Record pinned upstream commits, source and license hashes, dependency versions,
runtime, sampling diagnostics and all model adaptations. Use our data; bundled
third-party example datasets are not imported.

The reused sources are [Rebane's turnout models](https://github.com/rrebane/turnout-model-evaluation/tree/a1dd3da5e65ad05c95265ddedf3b6c519c0f1642)
and [MGGG's PyEI](https://github.com/mggg/ecological-inference/tree/214a24a7ca79c6f205b40dc6b720d98aaacdcca4).
Both carry MIT licenses. Complete upstream license notices stay beside the
unmodified cached source; adapters verify both source and notice hashes before import.

## Value criteria

The primary predictive comparison uses eligible-weighted absolute error, RMSE,
coverage and width at 50%, 80% and 95%, and weighted interval score (WIS).
WIS penalizes both unnecessarily wide intervals and missed observations; lower
is better. Locality-weighted coverage and per-fold results are also reported.
The score follows [Bracher et al. (2021)](https://doi.org/10.1371/journal.pcbi.1008618).

A practical uncertainty improvement requires at least a 5% reduction in locality
or subdistrict-total WIS against the empirical-residual benchmark, at most a 5%
worsening in locality RMSE, and improvement in at least two of the three geographic
folds. Always identify the aggregation level showing improvement. Regional
predictions sum the same draw across member localities before computing intervals;
summing marginal interval endpoints would discard dependence. A 5% RMSE improvement
is separately useful evidence for point predictions. These are engineering
selection thresholds, not statistical significance tests or automatic release rules.

An eligible Bayesian result must have at least two chains, no divergences,
R-hat at most 1.01, bulk ESS at least 400 and tail ESS at least 200 for the
global/hierarchical parameters and group functionals governing new-locality
predictions. Every selected diagnostic must be finite; empty or partially missing
diagnostic arrays fail. Legacy caches without completeness counts are unverified.
Record latent-variable mixing separately. Short smoke runs and
failed convergence checks remain explicitly exploratory.
Empirical-residual models use no MCMC, so their MCMC status is not applicable;
they still require the held-out improvement and runtime checks. Their predictive
intervals are empirically evaluated count ranges, not Bayesian credible intervals
or calibrated future-election probabilities.

## Scope and limitations

Geographic holdout assesses within-election aggregate predictions. It does not
verify unobserved individual group behavior or future national-election tails.
Any joint temporal prototype is evaluated separately. A forecast for K26 must
exclude K26 outcomes from training and retain uncertainty in election shocks.

Posterior intervals remain conditional on demographic proxies, likelihood,
hierarchy and priors. Adding a Bayesian engine does not reconstruct historical
population composition. Eight historical elections cannot establish extreme-tail
frequencies independently of those assumptions.

The required interpreter is Python 3.15.2. Dependencies and pinned upstream code
are isolated in a temporary runtime outside the repository. PyTensor currently
declares Python below 3.15: the experimental pure-Python installation requires
an explicit metadata override, analytic-distribution checks and actual model
sampling tests. This is not a claim of upstream Python 3.15 support.

Install `requirements.txt` into an isolated target with `--ignore-requires-python`,
then set `TURNOUT_EI_DEPENDENCIES` to that target. `run.py` activates it before
importing numerical libraries. `runtime.py --output PATH` runs a four-chain
analytic Beta–Binomial posterior check, including mean, variance, quantiles,
R-hat, effective sample sizes and divergences.

From the release repository, prepare the pinned dependencies and run the check:

```text
python -B products/coalition_simulator/model/ei_benchmark/runtime.py --prepare-upstream --output PATH/runtime_check.json
```

Supply the locally generated history panel and official CBS geography to the benchmark:

```text
python -B products/coalition_simulator/model/ei_benchmark/run.py --panel PANEL.csv --geography GEOGRAPHY.csv --output OUTPUT --models wls marginal_joint --draws 1000 --tune 1000 --chains 4 --cores 1 --target-accept .95 --nuts-sampler numpyro --chain-method vectorized
```

The fast regional empirical comparison needs no MCMC:

```text
python -B products/coalition_simulator/model/ei_benchmark/run.py --panel PANEL.csv --geography GEOGRAPHY.csv --output OUTPUT --models wls regional_empirical --draws 4000 --chains 1
```

`--selected-folds 0` runs one pilot fold. Without it, all three folds are scored.
`--elections` selects years to score; a joint model still fits every training
year in the supplied panel. A history-only experiment must remove target and
later election outcomes from the input training panel itself.
Posteriors are cached immediately after fitting; `--reuse-cache` rejects changed
training outcomes, source code, priors, sampling budgets or seeds.
`--prediction-seed-offset 1` resamples predictions from the same posterior,
allowing Monte Carlo sensitivity checks without another fit.

Sampler configuration is separate from the statistical model. The experimental
`--dense-mass` option, supported for `marginal_joint` with NumPyro, asks
[NumPyro NUTS](https://num.pyro.ai/en/latest/mcmc.html)
to adapt a full parameter covariance matrix. It changes the sampling geometry,
not the likelihood or priors, and remains subject to the same convergence gate.
`runtime.py --dense-mass --output PATH/runtime_check_dense.json` checks that
sampler configuration against the same known analytic posterior.

Adapter group summaries describe their explicitly defined population or
new-locality estimands. They are not automatically the eligible-weighted turnout
of the actual Israeli demographic groups and must not replace published group
parameters without a separate population-standardization review.

Results belong in the source project's
`data/processed/k26/current/turnout_research/ei_benchmark/`.

## Completed empirical comparison

The exploratory regional empirical candidate completed all nine elections and
three geographic folds in 19.2 seconds on Python 3.15.2. The original definition
was saved before its scores. It passed the existing practical threshold; this is
evidence for conditional regional prediction, not automatic production adoption.

| Metric, equal election weights | Independent residual baseline | Regional empirical candidate |
|---|---:|---:|
| Regional weighted interval score, pp; lower is better | 2.212 | 1.988 |
| Regional 50% coverage, eligible weighted | 20.9% | 45.9% |
| Regional 80% coverage, eligible weighted | 55.2% | 73.2% |
| Regional 95% coverage, eligible weighted | 76.0% | 96.9% |
| Regional 95% coverage, each region equal | 70.2% | 92.9% |
| Regional 95% interval width, eligible weighted, pp | 13.65 | 18.05 |
| Locality RMSE, pp | 7.4513 | 7.4513 |
| Locality weighted interval score, pp | 3.209 | 3.219 |

Regional score improvement was 10.1%; all three folds and all nine elections
improved. Predictive-seed offsets 0, 1 and 2 gave 10.1%, 10.7% and 10.4% gains,
with all folds positive and analytic point predictions unchanged to numerical
precision (exported rate differences below 2e-15).
Marginal residual distributions are analytically identical; the small realized
locality-score differences arise from finite predictive simulation. There is
no demonstrated locality-accuracy improvement. The 50% and 80% regional intervals
still under-cover; repeated seeds reuse the same geographic validation folds.

This experiment does not change historical SD, estimate a future national shock
distribution or quantify demographic-group coefficient uncertainty. Its shared
uniforms across years are rank coupling, not literal paired locality histories.
The fixed 2022 demographic proxies remain a substantive limitation.

The direct Rebane K26 pilot returned exploratory diagnostics outside the declared
targets (maximum R-hat 1.031; minimum bulk ESS 149). PyEI and the initial centered
joint runs hit their computational caps without completed posteriors. Timeouts
are incomplete computations, not statistical rejections. Exact model/source
identities, budgets and run status remain in the local benchmark evidence.
The context-compressed joint model also reached its 20-minute cap with dense
NumPyro adaptation still in warmup; it supplies no completed posterior or
held-out score. Its graph equivalence and sampling-runtime checks passed,
but no Bayesian replacement is established by these runs.
