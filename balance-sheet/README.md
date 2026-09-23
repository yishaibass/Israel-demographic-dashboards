# Israeli Household Balance Sheet

Reproducible source for the public household balance-sheet dashboard. The project combines Israel CBS household microdata with documented modeling and accounting transformations to present household assets, liabilities, income, consumption, saving and cash flow.

## Data access

Microdata is **not distributed in this repository**. Download the relevant public-use files directly from the Israel Central Bureau of Statistics (CBS), subject to its access terms, and place them in the folder structure documented in [docs/data-layout.md](docs/data-layout.md).

The build uses:

- Household Expenditure Survey releases for 2021, 2022 and 2023.
- Longitudinal Survey releases covering 2012–2023.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Build

From this `balance-sheet` directory:

```bash
python scripts/build_cbs_panel.py --data-root PATH_TO_LONGITUDINAL_RELEASES
python scripts/pipeline.py
python scripts/build_household_pnl.py --hes-root PATH_TO_HES_RELEASES
python scripts/build_dashboard.py
```

The dashboard builder writes:

- `output/index.html` — English
- `output/index_vHe.html` — Hebrew/RTL

For this GitHub Pages site, the Hebrew file is deployed as `index.html` and the English file as `index-en.html`.

See [docs/methodology.md](docs/methodology.md) for the model and accounting approach and [docs/reproducibility.md](docs/reproducibility.md) for validation checks.

## Important scope notes

- Household Expenditure Survey flows and Longitudinal Survey wealth stocks are not joined household by household.
- Components marked `◇` are modeled or statistically overlaid.
- Property transactions are rare; subgroup monthly values are cohort averages rather than the behavior of a typical household.
- The repository excludes all raw and processed household-level data.
