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

The first migration increment provides one publication entry point and retains each validated product methodology. Balance sheet and voting propensity have repository-owned analytical code. Fiscal position and coalition simulator are source adapters: publishing them still requires the configured legacy source projects on the local machine. Product-level decisions and methods are documented beside the code.
