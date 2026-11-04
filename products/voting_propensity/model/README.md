# Voting-propensity model

This directory contains the code used to estimate the voting model. Raw Census and ballot-level files are excluded; see the repository `sources/` catalog for provenance, filenames, and required columns.

## Published historical reference

The population base is the 2022 Israel Census Public Use File (PUF). The model estimates a binary logistic probability of turnout and a multinomial softmax probability over nine stable party families. Predictors are population group, Jewish religiosity, standardized income, income-by-group interactions, local Arab/religious/Haredi concentration, municipality size, broad periphery, and group-by-context interactions. Individual probabilities are Census-weighted and aggregated to locality × TatRova, where they are fitted to aggregate election results for 9 April 2019, 17 September 2019, 2 March 2020, 23 March 2021 and 1 November 2022. The legacy code name `rova` refers here to the PUF TatRova field.

Coefficients contain a component shared across elections and regularized election-specific deviations. Three-fold cross-validation holds out whole localities. No unrestricted locality identifier enters a person's published propensity. The published vector is the equal-weight mean of the five election-specific probability vectors.

The fitted data contain 1,201,728 adult PUF rows, representing 6,252,727 weighted adults. Supervision comprises 8,017 election-geography observations across roughly 1,600 geographic areas and 1,200 localities per election. Parties are mapped to nine stable families: Likud, national-religious right, Shas, United Torah Judaism, Arab parties, liberal center, Zionist left, secular national right, and other.

## Code map

- `data_pipeline.py` — reads the untracked Census PUF and constructs demographic cells.
- `run_multi_election_average.py` — reads official election files, applies the documented party crosswalk and constructs geographic targets.
- `context_features.py` — constructs Census-derived geographic context variables.
- `run_party_family_final.py` — defines and fits the turnout and party-family regressions.
- `fit_model.py` — runs geographic cross-validation, refits the full model and exports probabilities.
- `update_dashboard_from_model.py` — aggregates fitted probabilities into the public dashboard cube.

## Reproduction

```bash
python -m pip install -r products/voting_propensity/model/requirements.txt
python products/voting_propensity/model/fit_model.py --data-dir PATH_TO_LOCAL_INPUTS
python products/voting_propensity/model/update_dashboard_from_model.py --puf PATH_TO_CENSUS_PUF --probabilities PATH_TO_LOCAL_OUTPUT/individual_probabilities.csv.gz
```

Generated product outputs are ignored because individual-row probabilities are derived from microdata. Only aggregate, disclosure-safe dashboard content is committed.

## Election fits and publication

The published five-election cube remains the default reference. `fit_k26.py`
jointly refits K21–K26 with one design: whole-locality supervision, fixed 2022
Census features and weights, the same logistic/softmax objective, and shared
coefficients with regularized election deviations. Context features remain
Census locality × TatRova. Whole-locality supervision is used for every election
because K26 ballot-to-TatRova identity is unverified.

Compare K26 with the common-design historical estimates to assess election
movement. The difference between the published reference and the new historical
fit describes a model-version change. The common historical estimates are
retrospective and use K26 through shared coefficients; they are not forecasts.

`fit_independent.py` fits the same specification separately to each election. Each fitted parameter vector uses only that election's outcomes. Census features, party-family definitions and held-out locality folds stay consistent across years. The joint fit above is retained as a benchmark; independent results and their validation are stored under `individual_model/independent/`. All fits use the fixed 2022 Census population, including historical elections.

```bash
python products/voting_propensity/model/fit_independent.py --current PATH_TO_PROJECT/data/processed/k26/current --puf PATH_TO_CENSUS_PUF --output PATH_TO_PROJECT/data/processed/k26/current/individual_model/independent
```

The independent audit records each election's fit identity, paired geographic cross-validation and subgroup sensitivity to the held-out locality folds. Fold sensitivity is not a confidence interval. Good locality predictions do not identify individual votes or establish accuracy for every filtered subgroup. Age and education are available for filtering; the fitted specification uses group, religiosity, income and geographic-context predictors, so filtered differences can reflect their association with those predictors.

The reviewed November 2026 run retains the joint fit as the primary method and exposes independent fits as sensitivity analysis. Across six elections, the independent model's mean party-family error differs from the joint fit by less than 0.04 percentage points. K26 errors are 2.81 points for family shares and 4.44 points for turnout. Subgroup fold ranges reach 15.9 points for broad groups and 47.9 points for income-by-education slices across the six elections; these are model sensitivity ranges, not observed swings or uncertainty intervals. Independent estimates therefore do not support claims of precise subgroup voting rates. The separate run took about 49 seconds on the tested runtime.

From the repository root:

```bash
python products/voting_propensity/model/fit_k26.py --current PATH_TO_PROJECT/data/processed/k26/current --puf PATH_TO_CENSUS_PUF --output PATH_TO_PROJECT/data/processed/k26/current/individual_model
python tools/build.py voting_propensity --k26-current PATH_TO_PROJECT/data/processed/k26/current
```

The publication adapter also requires `voting_propensity_project` in the ignored
local path configuration. Pass `--k26-current` when building `all` to retain this
comparison. The UI keeps the historical cube intact, allows each election to be selected,
and preserves historical fiscal data. Exported aggregate cubes include:

- `k26_probability_cube.json`: K26 under the common design.
- `historical_comparable_cube.json`: equal-weight K21–K25 joint-probability mean
  from the same fit.
- `historical_election_cubes.json`: direct `K21`–`K25` entries containing cubes for
  selected-group historical means and population SD (`ddof=0`).

For conditional party shares, compute each historical election's selected-group
share before taking its equal-weight mean and SD. A ratio formed from averaged
joint masses instead weights elections by their modeled turnout. Label these
distinct summaries explicitly. Historical SD measures variation across five
estimates; it is separate from estimation uncertainty and significance.

The tested runtime is Python **3.15.2**, NumPy 2.5.4, pandas 3.1.0, and PyTorch
2.15.0+cpu. Default validation uses 70 epochs and three folds holding each locality
out across every election. The common run took 36.7 seconds and produced 18
cross-validation rows (six elections × three folds). Read `model_audit.json` for per-election coverage and
cross-validation; K26's available Census match covers 200 localities and 81.71%
of national valid votes (88.72% of regular valid votes). No individual probability
export is written by this comparison fitter.

The user assigns Yashar and Beyachad to `liberal_center`, Democrats to
`zionist_left`, and remaining unassigned minor lists to `other`. Preserve the
disclosed crosswalk; party families do not imply coalition membership.
Source-reported counts come from a secondary CEC mirror. Direct primary retrieval
and legal certification have not been independently verified. Envelope votes
are excluded from fitting and included in separate national comparisons.

## Interpretation

The output is an ecological model: individual propensities are inferred from aggregate geographic outcomes and Census composition, not from observed individual ballots. Results should therefore be interpreted as modeled group tendencies, not identified votes or causal effects.
