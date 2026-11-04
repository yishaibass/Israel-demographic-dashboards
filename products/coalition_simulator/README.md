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

The fitter carries K25 locality Census composition forward explicitly. It fits
turnout, then coalition votes per eligible voter bounded by turnout. Coalition
support uses all voters, including invalid ballots, under the original simulator
convention. Envelopes remain in national totals and outside group fitting.
The comparison shows percentage-point and historical SD changes, fit diagnostics,
coverage, and conditional bootstrap intervals. SD uses `ddof=0` across K21–K25;
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
