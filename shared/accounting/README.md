# Household P&L accounting

The accounting layer consumes canonical household component facts and produces two monthly-NIS views:

- `reported`: the HES P&L with reported direct taxes and reported cash benefits.
- `allocated_extended`: the same private P&L, but allocated taxes and calibrated cash benefits replace the corresponding reported fields; allocated public services are added.

Rules enforced against the authoritative `shared/contracts/definitions/component_catalog.json`:

- Allocated taxes replace reported HES tax fields. Reported fields remain reconciliation inputs only.
- Calibrated cash benefits replace raw HES benefit fields.
- Public health and education are separate from private spending.
- VAT and excise already embedded in purchaser-price consumption are reclassified from gross consumption when displayed as allocated tax; total uses do not increase.
- An in-kind public service has an explicit resource fact and an equal matched-use fact. Dashboards display the service once; it does not create cash saving.
- Imputed owner rent and owner-housing consumption are separate matched entries.
- Pension receipts and mortgage interest enter the P&L. Pension contributions and mortgage principal are financing flows and do not; asset and liability stocks are also excluded.

Input grain is `household_key + reference_year + component_id + representation + method_version`. `amount_nis` is non-negative; `frequency_basis` supplies the time basis and accounting side plus value basis determine the sign. Evidence status, source dataset and optional national-control IDs pass through the detail.

`market_adapter.py` consumes the additive household export from the balance-sheet P&L builder. `hes_adapter.py` consumes the existing tax/transfer outputs and the pre-calibration embedded-tax sidecar. `run_real_validation.py` joins the two domains, requires identical household keys and weights, and runs reported and extended identities without writing microdata.

Both domains construct the opaque key from the exact source tuple
`year:s_seker:misparmb`. Producers preserve these three fields and reject a
source module whose `S_Seker` differs from its release year. The interoperability
gate compares the source tuple, opaque key, and adjusted survey weight for every
household.

## Build the market exports

Paths below are placeholders resolved from ignored `config/paths.toml`; neither
raw data nor household exports belong in Git.

```powershell
# 2023-price pooled 2021-2023 export (written by the balance-sheet builder)
python products/balance_sheet/model/build_household_pnl.py `
  --project <balance_sheet_work_directory> `
  --hes-root <hes_data_root> `
  --longitudinal-panel <private_input_directory>/cbs_longitudinal_panel_full.csv `
  --market-export <private_output_directory>/hes_market_pnl_households_2023.pkl

# 2018-price pooled 2016-2018 export
python shared/accounting/export_market_pnl_2018.py `
  --hes-root <hes_data_root> `
  --pooled-weights <fiscal_2018_pooled_weights.pkl> `
  --mortgage-shares <balance_sheet_fig_household_pnl_by_age.csv> `
  --output <private_output_directory>/hes_market_pnl_households_2018.pkl
```

The 2023 export path is required through `--market-export` or
`IDD_HES_MARKET_EXPORT` and must resolve outside the Git worktree. The 2018
adapter transports the balance-sheet model's age-band mortgage interest shares
because HES reports one combined payment.

The longitudinal panel is also required through `--longitudinal-panel` or
`IDD_LONGITUDINAL_PANEL`. Build `cbs_longitudinal_panel_full.csv` locally from
the catalogued CBS longitudinal PUF modules; it is an input, never a repository
default. Reading the locked 2018 pooled-weight pickle requires the dependencies
in `shared/accounting/requirements.txt`, including PyArrow for pandas extension
dtypes stored by the upstream fiscal build.

## Run the real-data gate

```powershell
$env:PYTHONPATH='shared'
python -m accounting.run_real_validation --year <2018-or-2023> `
  --market-export <private_market_export.pkl> `
  --tax <private_pooled_withtax.pkl> `
  --transfer <private_pooled_withtransfer.pkl> `
  --embedded-basis <private_embedded_tax_basis.pkl> `
  --control-source <reviewed_methodology_source>
```

The command prints aggregate checks only and does not persist household rows.

Run focused tests from the canonical-spines root:

```powershell
python -m unittest discover -s shared/accounting/tests -p "test_*.py" -v
```
