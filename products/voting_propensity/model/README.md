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

The fit selector offers each independently fitted election and a combined 2019–2026 joint-model option. The combined option is the equal-weight mean of the six joint-model election probability vectors: average joint masses first, then calculate conditional shares for the selected population. It is not an average of independently fitted results. The original published five-election reference remains the default.

Across six elections, the independent model's mean party-family error differs from the joint fit by less than 0.04 percentage points. K26 errors are 2.81 points for family shares and 4.44 points for turnout. Subgroup fold ranges reach 15.9 points for broad groups and 47.9 points for income-by-education slices across the six elections; these are model sensitivity ranges, not observed swings or uncertainty intervals. Independent estimates therefore do not support claims of precise subgroup voting rates. The separate run took about 49 seconds on the tested runtime.

Comparison selectors use the same fit options and a five-election mean of independent annual rates. SD units always use the population SD of the selected subgroup's five independent K21–K25 rates. A comparison with the combined fit includes differences in fitting approach as well as the elections being summarized.

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

## Research: annual demographics and earlier elections

The isolated `research/election-demographic-foundation` branch adds three
reproducible stages, without writing published parameters, dashboard cubes or UI:

- `demographic_foundation.py` rebuilds explicit 18+ population cells from the
  licensed 2022 PUF. Other population groups and mixed/other/unknown household
  religiosity remain visible. It does not infer an individual's religion from
  their locality. The detailed model already filtered adults; this rebuild
  makes the foundation and the older six-share bridge audit reproducible.
- `historical_elections.py` downloads and verifies primary CEC archives and applies
  the versioned `historical_party_crosswalk_v1.csv`. Unknown positive vote mass
  blocks training. Joint lists produce disjoint merged observations, including
  every ballot touching the merged families; their components are not invented.
- `fit_foundation_history.py` compares the same 24-feature logistic/softmax
  design on immutable whole-locality folds, with year-specific population weights.

Annual CBS counts provide broad population-group changes for 2018–2023.
The preferred sensitivity starts from the known 2022 adult mix and multiplies
each broad group's weights by its same-series population growth since 2022.
It assumes adult group growth follows all-age growth; it does not substitute
an all-age share for a known adult share. Household religiosity and other
within-cell structure remain conditional on 2022. No annual religiosity is observed.

For 2026, official coded NII December 2024/2025 counts provide exact 18–64 and
65+ categories. Each category's one-year count projection is `count_2025² / count_2024`.
The target is December 1, 2026, not the exact election date. NII registry
residents differ from CEC registered voters. Legacy NII tables are retained as
separate diagnostics: their population reference dates are unverified and their
retirement-age categories differ by sex, so they are not treated as exact 65+ data.

Full fits now include 2003, 2009, 2013 and 2015 alongside K21–K26: ten elections,
nine historical. Those older fits retain the fixed-2022 common reference;
they are not claimed to have reconstructed historical demographics. The 2006
archive lacks locality codes and eligible counts, so it is retained without
entering full turnout fits. Official pre-2003 downloads were identified but
could not be retrieved through the available endpoints. Prime-minister-only
ballots are excluded. Kulanu-to-Likud preserves the existing analytical family
convention; family names do not identify literal party membership or coalition status.

### Results and limits

The preferred paired comparison uses 196 identical localities and three frozen
folds, with 70 epochs and full refits. Exact projected 65+ margins cover 193 of
those localities; three retain explicitly recorded population fallbacks.

| Held-out metric, percentage points; equal fold means | Fixed 2022 | Anchored annual update |
|---|---:|---:|
| K26 turnout MAE | 4.48 | 4.52 |
| K26 party-partition MAE | 2.79 | 2.83 |
| Mean turnout MAE across six elections | 4.54 | 4.53 |

The update improves the input definition and its auditability; it establishes no
material predictive gain. Both comparisons use identical outcomes and support.
The source rebuild reproduced all nine data CSV hashes, and primary-count
regeneration preserved all four historical target CSVs.

Within the extended detailed-model fit, the Arab group's K26 turnout is 3.63
historical SD above the previous-five mean and 3.36 SD above the nine-election
mean, using the same 2022 reference population. This is a different model and
population definition from the six-group simulator. These are descriptive
fitted-election distances, not future-election probabilities. Party-family SD
excludes years in which a family was observed only inside a merged list.

Common-reference summaries use the same 200-locality 2022 population;
year-specific summaries describe reweighted matched samples. Neither is an
observed national eligible-voter distribution or observed individual voting behavior.

### Reproduction

Use Python 3.15.2 and install `requirements.txt`. Raw PUF records, population cells,
fitted outputs and raw election files stay outside Git. `PROJECT` is the existing
local voting-data project and `PUF` is the retained licensed person-file path.
The portable launcher also supports embedded Python installations:

```bash
python -B products/voting_propensity/model/run_foundation.py population --puf PUF --bridge PROJECT/data/clean/model_input_v2.csv --raw PROJECT/data/raw/demographic_foundation --output PROJECT/data/processed/demographic_foundation --download
python -B products/voting_propensity/model/run_foundation.py history --project PROJECT --download
python -B products/voting_propensity/model/run_foundation.py fit --project PROJECT --current PROJECT/data/processed/k26/current --puf PUF --annual-targets PROJECT/data/processed/demographic_foundation/annual_targets.csv --annual-projections PROJECT/data/processed/demographic_foundation/annual_projections.csv --historical-targets PROJECT/data/processed/historical_model --output PROJECT/data/processed/demographic_foundation/model_comparison/reproduced --variants fixed2022_recent fixed2022_paired annual_anchored_growth extended_fixed2022 --epochs 70 --folds 3
```

Population acquisition verifies the official CBS catalog and 19 pinned NII files.
Historical acquisition verifies resource identity, retained hashes and complete
unpaginated primary archives. Existing caches fail on tampering and are not silently
replaced. Source URLs, versions, population bases, mapping qualifications, actual
coverage, fitted identities and input/source hashes are retained in local audits.

Primary source entry points are the
[CBS locality catalog](https://data.gov.il/api/3/action/package_show?id=localities-in-israel),
[NII December 2025 tables](https://www.btl.gov.il/Publications/yeshuvim/2025/lastmonth/Pages/default.aspx)
and [CEC historical archive catalog](https://data.gov.il/api/3/action/package_show?id=26f9fa06-fcd7-4173-8df5-65797b63e857).
