# Reproducibility and validation

## Build order

1. `build_cbs_panel.py` harmonizes the downloaded longitudinal releases into `processed/`.
2. `pipeline.py` creates aggregate balance-sheet tables in `figures/`.
3. `build_household_pnl.py` pools the downloaded 2021–2023 expenditure surveys and creates household-accounting tables.
4. `build_dashboard.py` embeds the aggregate tables in English and Hebrew HTML files under `output/`.

## Required checks

After building, inspect `figures/fig_household_pnl_checks.json`. The P&L, saving-allocation and cash-flow identities should be zero within floating-point or source-rounding tolerance. Confirm that:

- all three survey years contribute to each published cell;
- there are no missing dashboard cells;
- the pooled adjusted weights reconcile to the 2023 household-population total;
- both language versions build successfully;
- no household-level files appear in Git status.

## Public-release audit

Before committing, scan the staged project for absolute local paths, credentials, downloaded microdata, household identifiers, caches and large intermediate files. Only source, documentation and aggregate dashboard artifacts belong in the repository.
