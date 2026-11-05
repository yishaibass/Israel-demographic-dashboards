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

## Research: election-specific population

The `research/population-reconstruction` branch rebuilds the population inputs.
Published parameters, defaults, HTML and dashboard cubes remain unchanged.
All research runs use Python 3.15.2. Raw records and fitted outputs stay outside Git.

### Population design

1. Retain Census 2022 adult age × sex × population-group associations and household
   religiosity. Other population, mixed households and unknown answers remain explicit.
2. Use actual CBS 2020 locality age counts, Census 2008 subdistrict age × sex data,
   annual locality totals and dated PIBA locality age profiles to estimate geographical
   changes. For 2026, locality totals use the same-series PIBA 2020–2026 growth
   against CBS 2020 resident anchors, with explicit fallbacks for new codes. Source-window and
   registry/resident differences remain assumptions in the geographical prior.
   Reviewed mergers and splits conserve their geographical family population;
   code creation is distinguished from physical settlement founding.
3. Reconcile each dated population to CBS national age × sex × Jewish/Arab/Other
   counts. Annual averages are aligned to July 1 for interpolation to election dates.
   Missing historical years are interpolated; October 2026 is projected from the
   latest observed 2025 annual average using the 2023–2025 trend. Local source
   counts are priors; the reconciled local joint counts are estimates.
4. Update household religiosity with weighted HES 2019–2023 trends, centered on the
   2022 Census. Sparse age/sex trends shrink toward national trends. The full
   2016–2023 window is a sensitivity because nationality measurement changed in 2019.
   Earlier years use observed Social Survey 2002–2021 relative changes through an
   explicit 2019 measurement bridge. Individual 20+ self-definition and household
   18+ religiosity are different measures; their equivalence is not asserted.
5. Separately calibrate resident citizen voters to CBS election-specific eligible
   population reports, and registered totals to CEC locality counts. Permanent
   noncitizens and registered citizens abroad are distinct. The primary allocation
   assumes overseas voters have the same group composition as resident citizens
   within their registered locality; a conservative alternative leaves that gap
   unallocated. The 2019–2020 national targets use approximate headlines and
   rounded group percentages; 2021–2026 counts are published in thousands.
   Unsupported CEC localities use a flagged national-group prior in the primary
   calibration; the conservative allocation retains the unidentified gap.
   Exact constraint closure does not identify individual voters.

The current model retains its 24 predictors and regularization. Resident weights
construct locality/subquarter context; citizen-adjusted registered weights enter
its voting mixture. No locality-specific turnout coefficients are added. Income
and origin remain conditional on the 2022 PUF within the reconstructed demographic
cells. Small demographic cohorts absent from a locality’s PUF sample receive
explicit pooled Census donor profiles; their imputed population mass is audited.
Model validation retains identical whole-locality folds and election targets.

### Evidence and limits

The reconstruction exports 24 calendar years, 2003–2026, plus ten election dates.
The active geography changes with documented code creation, retirement and
mergers. The reviewed lifecycle ledger covers specified 2019–2026 events; older
geographical histories and unknown founding dates remain explicit limitations.
Census 2008 supplies 15 inhabited subdistricts; three water polygons are excluded.
Actual 2020 local age data and November 2026 registry profiles improve geographical
priors, rather than supplying observed annual local religiosity or individual votes.
National population constraints are checked separately from electoral prediction.

Population validation withholds observed data before reconstruction. A separate
parameter-free diagnostic backcasts published 2022 local age profiles to 2020
using national age-count changes. On 753 matched localities with complete 2020
65+ source cells, population-weighted error falls from 0.781 to 0.682 percentage
points; equal-locality error improves only 1.1%. These localities cover 50.2% of
2020 source residents. Suppressed/imputed values are excluded from the truth set.
The older regional prior alone is substantially weaker (3.409 points), so available
local 2022 profiles take priority when filling missing 2020 age components.
Survey validation leaves out entire HES years. These checks do not validate every
unobserved local demographic joint distribution.

The CBS administrative Haredi voter estimate is an external benchmark. It uses a
classifier, a different date, and citizen eligibility; it is not automatically
substituted for Census household religiosity or Social Survey individual identity.
The location of national residents outside identifiable locality codes is not
observed; allocating national totals across supported active codes is disclosed.
Older geographical estimates and the early Social Survey measurement bridge have
weaker evidence than the recent population inputs.

### Controlled model results

All comparisons use the same 200 named PUF localities, three frozen locality
folds, 70-epoch limit, seeds, 24 predictors and regularization. These are
retrospective geographical holdouts of the same six elections.

| Held-out error, percentage points | Fixed 2022 population | Reconstructed eligible population |
|---|---:|---:|
| 2026 turnout MAE | 4.503 | 3.949 |
| 2026 party-family MAE | 2.800 | 2.388 |
| Mean turnout MAE across six elections | 4.530 | 3.943 |

Errors are eligible-weighted within each fold/election, then averaged across
three folds; party-family errors use valid-vote weights. Updating resident
composition alone gives 4.525-point 2026 turnout error: most of the improvement
comes from eligibility calibration. The alternative HES window gives 3.952;
the Jerusalem citizenship-scope alternative gives 3.976. These sensitivities do
not exhaust uncertainty in the population reconstruction or ecological estimates.

The full run took 220 seconds on Python 3.15.2. All 92 focused/regression tests
and four subtests passed. Five full coefficient exports restored identical
probabilities through their actual JSON representations. Imputed Census donor
cohorts account for at most 4,536 estimated registered voters in the paired
sample, below 0.1%; the actual mass is recorded separately for each election.
Published dashboard defaults, parameters and HTML were verified unchanged.

### Reproduction and ownership

`run_foundation.py STAGE --help` exposes each stage. Install `requirements.txt`;
`pyogrio` reads the retained CBS 2020 geodatabase without geometry or GeoPandas.
Provide the retained licensed PUF and HES/SS files, the pinned public census
workbooks and the existing project source caches. No raw microdata are in Git.

| Stage | Authoritative implementation | Local output |
|---|---|---|
| `population` | `demographic_foundation.py` | Adult Census cells; pinned annual locality source controls |
| `national-controls` | `national_population_controls.py` | Verified annual national single-age × sex × group controls |
| `source-inputs` | `population_source_inputs.py` | CBS 2020 attributes, 2008 regions and 2022 locality/founding crosswalk |
| `reconstruct --mode anchors` | `population_reconstruction.py` | Census 2022 PUF joint and independent published margins |
| `electorate --download` | `population_electorate.py` | Dated PIBA age profiles, official eligible totals, citizen controls |
| `reconstruct --mode reconstruct` | `population_reconstruction.py` | `adult_controls.csv.gz`, reconciliation and population holdouts |
| `religiosity` | `annual_religiosity.py` | Primary household trend factors, alternative window and source/holdout evidence |
| `electorate --adult-controls ...` | `population_electorate.py` | Nationally calibrated resident-citizen and registered group allocations |
| `fit-reconstructed` | `fit_population_reconstruction.py` | Paired model fits, profiles, group shares and frozen-fold validation |
| `history` | `historical_elections.py` | Verified primary older election targets and disjoint joint-list partitions |
| `historical-electorate` | `historical_electorate.py` | Dated citizen controls, explicit 2003 backcast and alternative allocation |
| `historical-trends` | `historical_trends.py` | Vote-cast shares, political voter-base composition and accounting decompositions |
| `historical-right-bloc` | `historical_right_bloc.py` | Explicit party-level right/rest targets and classification sensitivity |
| `historical-survey` | `historical_survey_benchmark.py` | Independent pinned-source INES 2003 benchmark; no calibration to survey results |
| `historical-support-audit` | `historical_support_audit.py` | Independent-year, outcome-definition and context-specification checks |

Use a separate research output directory. Hash checks reject changed cached inputs;
final fits capture source and input hashes before loading and verify them again
before marking completion. Reconstructed population files are published atomically.
The model comparison consumes `adult_controls.csv.gz`, primary and sensitivity
`religiosity_relative_factors*.csv`, `citizen_electorate_factors.csv`, its full scope-sensitivity alternative, the PUF,
prepared election targets and an immutable locality-fold assignment. Full-fit
coefficients are retained in `fitted_models.json`; JSON round-trip verification
requires identical election probabilities. Outputs remain research artifacts
until publication is approved.

The separate historical controlled comparison fits ten elections: 2003, 2009,
2013, 2015, April and September 2019, 2020, 2021, 2022 and 2026. It uses the same
195 common named localities and frozen three-fold geographical split in every
variant. The six-election results above remain a separate run.
`historical_electorate.py` adds contemporaneous CBS resident-citizen totals and
rounded group shares for 2009, 2013 and 2015. For 2003, only the 4.3-million
resident-citizen total was recovered: the primary split backcasts the 2009
citizen/adult group calibration factors onto dated 2003 adult counts, while a
second complete fit uses the unadjusted 2003 resident group mix. Neither split
is an observed 2003 citizenship distribution. The Jerusalem local seed remains
a spatial prior; national totals are independently calibrated for each election.
The retrieved 2006 archive lacks locality identities/eligible denominators;
other official files containing them could not be retrieved. That election
does not enter full turnout fitting.
Joint ballot lists identify merged party families; their constituents are not
invented. Party-specific historical SD omits unidentifiable constituent years.
Historical SD is descriptive fitted-election variation, not a future tail probability.

### Historical voter shares

`electoral_composition.csv` divides each group's estimated votes cast by all
estimated votes cast, using that election's reconstructed eligible population
and fitted turnout. Its denominator differs from the eligible-population share.
`stable_political_preferences.csv` exports three totals that respect every
historical joint-list identification constraint: right/religious families
(Likud, national-religious right, Shas, UTJ and Liberman), center/left/Other,
and Arab parties. These are analytical families, not coalition membership.
The residual Other family contains some minor right-wing lists. Individual
constituents of merged historical lists are not identified.

`historical_trends.py` consumes those two exports and produces demographic
voter shares, the demographic composition of each political voter base, and
exact accounting decompositions. Two-factor changes separate eligible group
mix from turnout; three-factor political changes also separate modeled support
within groups. These average all factor orders and conserve the total change.
They describe aggregate net changes, not individual voter switching or causal
effects. Within-group changes can include changing composition inside a broad
group. Masorti includes mixed-religiosity households; Arab includes Druze;
Other/unknown remains visible. All estimates refer to the common sample,
rather than observed national demographic vote counts.

The producer retains its actual fitted coefficients and checks identical
probabilities after a JSON round trip. The consumer accepts and audits only
floating-point probability-sum deviations below one part per million;
larger inconsistencies fail. Input and fitting-source hashes are verified
before and after training. Historical outputs remain private research artifacts.

The historical paired run uses a 70-epoch budget: mean held-out turnout MAE
falls from 4.733 to 4.149 percentage points and observed party-partition MAE
from 2.734 to 2.343. The primary model was also rerun for 140 epochs; errors
are 4.087 and 2.305 respectively. The runs reached their epoch limits, so
optimization convergence is not established. Across all elections and groups,
the longer run changes voter shares by at most 0.97 points and political
category support by at most 6.14 points. Fine subgroup levels remain provisional.

| Estimated share of votes cast in the common sample | 2003, 70 epochs | 2022, 70 epochs | Change, points | Change, 140 epochs |
|---|---:|---:|---:|---:|
| Haredi | 7.6% | 10.8% | +3.1 | +3.1 |
| Dati | 10.1% | 11.1% | +1.0 | +1.2 |
| Masorti, including mixed households | 24.5% | 22.0% | -2.5 | -2.1 |
| Hiloni | 38.4% | 34.1% | -4.3 | -4.5 |
| Arab, including Druze | 12.2% | 14.0% | +1.8 | +1.7 |
| Other/unknown | 7.2% | 8.0% | +0.8 | +0.5 |

The alternative 2003 citizen allocation preserves the Haredi, Hiloni and
Masorti trend directions, but reduces the Arab voter-share change to +0.2
points. The latter is not a robust population-allocation finding. Haredi
growth and Hiloni decline continue across the intermediate elections with
reversals; most of the Masorti decline occurs by 2009, followed by a plateau.
The fitted Dati right-family increase of about 18 points is withdrawn as a
validated historical finding. Agreement between training budgets did not test
whether the model attributed votes to the correct demographic group. The
demographic composition of a fitted political voter base also depends on that
attribution and must not be treated as independently verified.

### Independent historical support check

The public [TAU Israel National Election Studies 2003 data](https://socsci4.tau.ac.il/mu2/ines/data/our-data/)
provide a direct check. For Jewish respondents (`JEW=1`), religious self-definition
is `B91`: 3 religious and 4 Haredi. `B63` measures pre-election party intention,
not reported votes after the election. Among respondents naming a party, the
same narrow right-family definition gives 80/93 religious respondents (86%)
and 72/77 Haredi respondents (94%). The legacy model estimates approximately
69% and 88% respectively for 2003. In particular, the Dati starting point
does not pass this independent check. The survey is unweighted in this audit;
sampling uncertainty, intentions versus actual votes, individual versus
household identity, and the model's geographical sample remain distinct.

There is a separate classification problem: the historical Other family
contains right-wing lists, including Sharansky and Herut. Their votes cannot
be recovered by relabeling the entire fitted Other probability. A complete
historical right-wing comparison requires a separately versioned ballot-level
bloc definition and a new fit to those original counts. The existing nine-family
crosswalk and outputs are retained for reproducibility.

The versioned right-bloc crosswalk and count builder retain all votes, exclude
external envelopes, fail on unknown symbols and reconcile right plus rest to
valid votes for every locality. Ambiguous lists are excluded from the primary
definition and included in a separate classification sensitivity. In observed
2003 results, completing the bloc raises right support from 86.7% to 98.4%
in Kedumim, 84.1% to 98.7% in Beit El and 80.5% to 95.5% in Efrat. These are
locality results, not observations of individual religious identity.

The support audit preserves the 195-locality sample, dated population controls
and frozen folds. It fits 2003 and 2022 independently and compares the existing
context design against direct demographic and income terms. With the same
narrow party-family targets and 140-epoch budget, inferred 2003 Dati support
changes from 65.0% to 74.8%; Haredi support changes from 87.5% to 90.8%.
Held-out right-share MAE improves from 4.65 to 4.30 points in 2003, while the
simpler design worsens 2022 error from 5.28 to 6.50. Right-share errors compare
the same aggregate target; average errors across differing party partitions
are not comparable. Thus neither epoch stability nor geographical prediction
accuracy establishes the correctness of subgroup historical support.

For the explicit broad-right binary outcome, the direct demographic/income
candidate gives the following independently fitted endpoint estimates. The
survey was not used as a fitting constraint.

| Broad-right support within group | 2003, 500-epoch budget | 2022, 500-epoch budget |
|---|---:|---:|
| Dati | 94.4% | 95.3% |
| Haredi | 96.0% | 98.1% |

Three genuinely different initial coefficient states agree within 0.005
percentage points; all six fits stop before the 500-epoch cap under the
existing stopping rule. This demonstrates numerical stability for this
specification, not certainty about individual behavior. At the separately
validated 140-epoch budget, direct versus full-context broad-right MAE is
4.27 versus 4.56 points in 2003 and 5.90 versus 4.88 in 2022. The direct
candidate matches the independent 2003 benchmark substantially better but
does not uniformly improve locality predictions. It remains a research
candidate; no production parameters, published defaults or UI are replaced.
The original claims of a large Dati or Haredi shift into the right are not
supported by this audit and must not be presented as established findings.

Primary entries: [CBS locality catalog](https://data.gov.il/api/3/action/package_show?id=localities-in-israel),
[PIBA locality age profiles](https://data.gov.il/api/3/action/package_show?id=residents_in_israel_by_communities_and_age_groups),
[CBS 2026 eligible population report](https://www.cbs.gov.il/he/mediarelease/DocLib/2026/332/24_26_332b.pdf),
[CBS historical Social Survey table](https://www.cbs.gov.il/he/publications/doclib/2024/1905/t_a.xlsx),
[CEC historical archive](https://data.gov.il/api/3/action/package_show?id=26f9fa06-fcd7-4173-8df5-65797b63e857).

Example final controlled run (paths are supplied by the local project):

```bash
python -B products/voting_propensity/model/run_foundation.py fit-reconstructed \
  --project PROJECT --current PROJECT/data/processed/k26/current --puf PUF \
  --population-controls OUTPUT/adult_controls.csv.gz \
  --religiosity-factors OUTPUT/religiosity/religiosity_relative_factors.csv \
  --sensitivity-factors OUTPUT/religiosity/religiosity_relative_factors_full_window_sensitivity.csv \
  --electorate-controls OUTPUT/citizen_electorate_factors.csv \
  --sensitivity-electorate-controls OUTPUT/citizen_electorate_factors_scope_sensitivity.csv \
  --splits FROZEN_LOCALITY_FOLDS --output OUTPUT/model_comparison --epochs 70 --folds 3
```
