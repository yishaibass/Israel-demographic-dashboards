# Voting-propensity model

This directory contains the complete code used to estimate and publish the voting model. Raw Census and ballot-level files are deliberately excluded; see [`../data/raw/README.md`](../data/raw/README.md) for provenance, expected filenames and required columns.

## Method

The population base is the 2022 Israel Census Public Use File (PUF). The model estimates a binary logistic probability of turnout and a multinomial softmax probability over nine stable party families. Predictors are population group, Jewish religiosity, standardized income, income-by-group interactions, local Arab/religious/Haredi concentration, municipality size, broad periphery, and group-by-context interactions. Individual probabilities are Census-weighted and aggregated to the locality × statistical-area level, where they are fitted to official aggregate election results for 9 April 2019, 17 September 2019, 2 March 2020, 23 March 2021 and 1 November 2022.

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
python -m pip install -r modeling/requirements.txt
python modeling/fit_model.py --data-dir data/raw
python modeling/update_dashboard_from_model.py --puf data/raw/census/census_2022_puf.csv --probabilities modeling/outputs/individual_probabilities.csv.gz
```

The generated `modeling/outputs/` directory is ignored because its individual-row probability file is derived from restricted-source microdata. Only aggregate, disclosure-safe dashboard content is committed.

## Interpretation

The output is an ecological model: individual propensities are inferred from aggregate geographic outcomes and Census composition, not from observed individual ballots. Results should therefore be interpreted as modeled group tendencies, not identified votes or causal effects.
