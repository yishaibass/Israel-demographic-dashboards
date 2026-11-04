# Israel demographic dashboards

Four independently buildable analytical products share a small set of data and publication contracts:

- `products/balance_sheet` — household assets, liabilities, income, consumption, and saving.
- `products/fiscal_position` — taxes, transfers, services, and net household fiscal balance.
- `products/coalition_simulator` — coalition and electoral scenario simulator.
- `products/voting_propensity` — demographic voting-propensity model and explorer.

Published GitHub Pages routes remain `balance-sheet/`, `fiscal-position/`, and `voting-simulator/`.

## Local setup

Copy `config/paths.example.toml` to the ignored `config/paths.toml` and fill in local data and source-project paths. The `sources/` catalog lists required files, access status, and public download links. Raw data and household-level outputs never belong in Git.

## Build

```bash
python tools/build.py all
python tools/build.py all --verify
```

For the common-design six-election voting-propensity comparison, retain prepared results outside
Git and include them explicitly when building:

```bash
python tools/build.py voting_propensity --k26-current PATH_TO_PROJECT/data/processed/k26/current
```

The same option is accepted with `all`. Omitting it publishes the configured
historical source. K21–K25 remains the default in both voting products; K26 is a
separate comparison. The propensity view keeps the published reference and adds
individual fit selection and comparison. Each election uses its independent fit;
the combined 2019–2026 option averages the six joint-model election vectors equally.
Comparison SD uses the selected subgroup's five independent historical rates.
See the [coalition simulator](products/coalition_simulator/README.md)
and [propensity model](products/voting_propensity/model/README.md) for fit commands,
data coverage, and interpretation. Both K26 fits were tested with Python 3.15.2.

Balance sheet and voting propensity have repository-owned analytical code. The
coalition K26 fitter and renderer are also owned here; its historical analysis
and HTML template still require the configured source project. Fiscal position
remains a source adapter. Source-reported K26 count status comes from a secondary
mirror; direct CEC retrieval and legal certification remain unverified.
Only reviewed aggregate results belong in Git. Publication builds prepare files;
deployment is a separate step.
