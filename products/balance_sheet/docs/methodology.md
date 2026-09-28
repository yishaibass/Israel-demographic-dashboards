# Methodology

## Overview

The dashboard has two related but distinct inputs:

1. Household flows from the 2021–2023 CBS Household Expenditure Surveys.
2. Household wealth stocks from the 2023 cross-section of the CBS Longitudinal Survey.

They are combined statistically by demographic cell, not joined at household level.

## Three-year flow pool

The project follows a nominal-GDP-per-capita normalization used in the accompanying public analysis workflow:

| Survey year | Nominal GDP per capita (NIS) | Factor to 2023 |
|---|---:|---:|
| 2021 | 167,000 | 191/167 |
| 2022 | 183,000 | 191/183 |
| 2023 | 191,000 | 1.000 |

Monthly monetary fields are multiplied by the applicable factor before accounting identities are constructed. Survey weights receive the same factor, are divided across the three waves, and are then normalized to the 2023 household-population weight total.

## Household accounting

- Economic/accrual saving equals adjusted disposable economic income less economic consumption and expenses.
- Mortgage payments are split into modeled interest expense and principal reduction.
- Property-transaction equity nets gross acquisition and improvements against signed apartment-debt financing.
- Real-estate-related saving includes property-transaction equity, mortgage principal, other housing-loan payments, unclassified related debt movements and assumed community saving.
- Pension saving combines employee contributions, modeled employer contributions, modeled pension returns and withdrawals.
- Broad raw saving/debt fields are retained for audit and cash-flow reconciliation.

## Wealth model

The longitudinal pipeline harmonizes ten survey waves, imputes or models selected components, and calibrates specified national aggregates. The dashboard marks modeled or statistically overlaid values with `◇`.

## Interpretation

Annual survey weights are used throughout. Large property purchases are infrequent, so unconditional subgroup means represent cohort-average capital formation. They should not be interpreted as the recurring payment of a typical household.
