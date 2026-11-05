# Coalition simulator

This package owns the separate six-group K26 comparison and publishes the
simulator. K21–K25 historical defaults stay fixed. Set
`coalition_simulator_project` in ignored `config/paths.toml`, then run:

```bash
python products/coalition_simulator/build.py
```

Before publication, estimate the comparison against a local voting project:

```bash
python products/coalition_simulator/model/k26_comparison.py --root PATH_TO_VOTING_PROJECT
```

This requires that project's `data/clean/model_input_v2.csv`, historical parameter
CSVs, and prepared `data/processed/k26/current/` results. It writes JSON and CSV
under that current directory's `model_comparison/`; historical parameters are
checked by hash and remain unchanged. Python 3.15.2 was tested.

`model/k26_party_blocs.json` is the authoritative 2026 coalition definition. It includes Likud, Otzma Yehudit, United Torah Judaism, Shas, Religious Zionism–Zehut, Amcha Israel, Noam, Tzeva Shachor, Haredi Public and Tzomet Beit Israel. Intake, fitting and rendering validate the same contract and hash. Coalition membership is separate from the nine ideological families in the demographic model. Historical definitions and frozen forecast/contest rules remain unchanged.

The fitter carries K25 locality Census composition forward explicitly. It fits
turnout, then coalition votes per eligible voter bounded by turnout. Coalition
support uses all voters, including invalid ballots, under the original simulator
convention. Envelopes remain in national totals and outside group fitting.
The final simulator tab shows six group rows, whole percentages, one-decimal historical SD changes, and projected eligible-voter shares. Use `#2026` to open it directly. Eligible-share changes are percentage points versus the K25/2022 Census base, not relative percent growth or a five-election population average. Detailed changes, fit diagnostics, coverage and conditional bootstrap intervals remain in the analysis JSON/CSV. SD uses `ddof=0` across K21–K25;
values below `1e-6` yield an undefined standardized change.

`k26_dashboard.py` owns the English/Hebrew comparison renderer; the source project's
`scripts/fit_k26_comparison.py` and `scripts/k26_dashboard.py` are compatibility
entry points. Regenerate that project's English/Hebrew HTML before publishing.
The publication adapter consumes `outputs/dashboard/index_he.html` and preserves
the demographic tab, navigation, and Analytics contract. Historical analysis and
the HTML template still run in the configured source project.

Count metadata are source-reported by a secondary CEC mirror. Arithmetic checks
do not establish direct primary verification or legal certification. Group rates
are ecological estimates; historical SD is descriptive variation, separate from
estimation uncertainty.

The election-loading row includes K26 alongside K21–K25. K26 loads its fitted turnout/support and its national below-threshold vote fractions; the same scenario engine handles every year. The population toggle is independent. The renderer checks projected eligible shares against the stored base, growth assumptions and projection horizon. Actual 2026 locality registered-voter counts and turnout enter the behavioral fit; they do not currently reweight the national demographic-share projection.

## Turnout research: longer history and correlated uncertainty

These backend analyses do not write published parameters or UI. `turnout_history.py`
archives official K18/K19/K20 turnout counts with source hashes, fits K18–K26 on
the same fixed K25 locality composition and exports five-, seven- and eight-election
historical references. Every historical reference excludes K26. Both the full
matched sample and a balanced locality panel are reported. Archived raw-count
incoherencies are explicitly excluded by locality, with no count corrections.

```bash
python products/coalition_simulator/model/turnout_history.py --root PATH_TO_PROJECT --download
python products/coalition_simulator/model/turnout_uncertainty.py --panel PATH_TO_PROJECT/data/processed/k26/current/turnout_research/history/locality_panel.csv --geography PATH_TO_PROJECT/data/processed/k26/current/turnout_research/geography/locality_geography.csv --block-column subdistrict_code --stress-block-column district_code --balanced --draws 1000 --output PATH_TO_PROJECT/data/processed/k26/current/turnout_research/uncertainty
```

The geography lookup comes from the official CBS 2022 locality catalog, resource
`199b15db-3bcb-470e-ba03-73364737e352`. Raw responses, exact field mapping and
coordinate normalization are retained with that catalog's source metadata in
the project's `data/raw/ballots/historical_turnout/`.

Resampling preserves each locality's entire election history; geographic blocks
also preserve within-block co-movement. The uncertainty of K26 minus its historical
mean retains their covariance. A separate covariance-aware diagnostic estimates
how much fitted-coefficient noise contributes to historical scatter; individual
voter variance is not added to historical SD. Research outputs live under the
project's `data/processed/k26/current/turnout_research/`.

Ranges are conditional resampling sensitivity results, not calibrated confidence
intervals or probabilities for a future national election. They condition on fixed
2022 demographics, whose historical changes are unobserved. There are few regional
clusters, group exposure is uneven, and rank-deficient resamples are counted and
excluded jointly. Additional localities do not create independent national election
shocks. See the saved reports for covariance components, coverage and exclusions.
The tested runtime is Python 3.15.2.

## Isolated model benchmark

[`model/ei_benchmark/README.md`](model/ei_benchmark/README.md) describes the
research-only comparison of pinned MIT-licensed PyEI/Rebane models and a fast
regional empirical-residual predictor. It records geographic holdouts, numerical
checks, convergence and practical-value gates. These experiments do not write
published parameters or UI.

## Demographic foundation: next research stage

The current six shares originate in 2022 Census PUF household religiosity,
person-weighted within locality/sub-quarter and mapped to statistical areas.
They are proxies for eligible-voter composition. A local provenance audit found
the raw PUF, codebooks and processed bridges, but none of five documented
temporary reconstruction scripts. The retained notes do not verify an adult-only
filter. Rebuilding and versioning that pipeline, with an explicit 18+ denominator,
comes before changing demographic granularity.

Use 2008 and 2022 census anchors, annual age/religion updates and reconciled
[CBS statistical-area crosswalks](https://www.cbs.gov.il/he/cbsNewBrand/Pages/מפתחות-מעבר.aspx).
Separate observed counts, reconstructed shares and 2026 projections. Historical
Jewish religiosity needs its own estimation method: a
[CBS working paper, page 24](https://www.cbs.gov.il/he/publications/DocLib/pw/pw118/wp118.pdf#page=24)
explicitly notes that the 2008 Census did not contain religiosity. Annual
population counts alone cannot supply those missing six-way distributions.

Compare estimator changes on the same time-matched population foundation and
geographic holdouts. Keep the six-group simulator and the richer socioeconomic
nine-family model as distinct use cases; their current coverage and estimands
differ. No foundation rebuild or structural change is published by this audit.
