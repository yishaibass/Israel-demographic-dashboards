# Canonical microdata contracts

The project uses three complementary microdata spines rather than attempting record-level linkage:

- HES is the economic household spine for income, consumption, fiscal allocation, equity imputation, and the household P&L.
- The longitudinal survey is the evidence spine for observed assets and debts, model training, and longitudinal validation.
- The Census is the population and geography spine for demographic totals, voting probabilities, and component-level fiscal transport.

`definitions/` contains the authoritative field, component, and record contracts. `keys.py` creates non-identifying, spine-local keys. `validate.py` checks internal consistency without adding a runtime dependency.

Component facts store a generic non-negative `amount_nis`. `frequency_basis` distinguishes monthly flows, annual flows, and point-in-time stocks; `price_basis` and `price_year` make nominal or constant-price values explicit. P&L views consume flows only and normalize annual flows deliberately. Balance-sheet views consume point-in-time stocks only.

## Stable table boundary

Every spine publishes `canonical_household` and `canonical_person`. Analytical models publish long-form `household_component_fact` records. National totals and reusable age/sex/education/sector factors remain separate tables referenced by ID; they are never embedded as dashboard logic.

HES adapters publish market and fiscal facts separately. `hes_market_component_fact_v1` carries reported market income, private consumption, imputed housing service, and financing flows. `fiscal_allocation_component_fact_v1` carries administrative allocations and must reference an external `national_control_v1`. A national control records its source dataset, source reference, and SHA-256; it is never reverse-engineered from allocated household facts.

Dashboards aggregate component facts. They do not calculate allocations, infer missing fields, or combine reported and allocated amounts.

Producers emit leaf facts only: a selected household view cannot contain both a parent total and one of its children. Mortgage principal and pension contributions are financing flows, not consumption expenses. Indirect tax embedded in reported consumption is removed through the explicit contra component before allocated consumption tax is added. Negative self-employment residuals are emitted as a non-negative income-side contra fact rather than recast as household consumption.

`income.market.other` is restricted to the signed HES private-market residual after labor, self-employment, capital, rent, occupational pension, private transfers, and government benefits have been separated. Its positive part is emitted as cash income; the absolute negative part is emitted as `income.market.other_loss`, an income-side contra. It is not a catch-all for a named income source.

If the HES private-transfer residual is slightly negative after separately identifying government benefits, its absolute value is retained as `income.market.private_transfer_loss`. This preserves the published income identity without clipping or moving the residual into government benefits.

Private cash consumption is reconciled exactly to the published total. A positive remainder after named private categories is `expense.private.cash.other`; an exceptional negative remainder is retained at absolute value as the explicit contra `expense.private.cash.reconciliation`. It may not absorb an identified category or allocation.

Net negative private-health product spending caused by refunds or rebates is emitted at absolute value as `expense.private.cash.health_rebate`; it is never clipped or moved into the general consumption reconciliation.

For a tax marked `embedded_in_purchaser_price`, an allocated HES fact must carry its pre-calibration `embedded_tax_basis_nis_monthly`. The extended view reclassifies no more than the allocated tax, that embedded basis, and remaining gross private purchases. The administrative allocation stays unchanged; only the amount demonstrably embedded in reported purchases is removed from private consumption.

Two presentation views share the same component IDs. `reported` selects HES-reported cash taxes and benefits. `allocated_extended` replaces each substitutable reported item with its allocated representation, then adds public in-kind resources and their matching consumption uses. A view must never sum reported and allocated representations of the same component. Private health and education spending remain separate from public in-kind services.

Functional P&L dimensions are orthogonal to funding and delivery. Each fact carries `economic_function`, `payer_funder`, and `delivery_value_type`; these inherit from the authoritative component catalog and are validated. On the resource side, all attributable government in-kind services present as one `public_services_received` income line while retaining their detailed function tags. On the use side, the matched amount rolls into its functional expense section: health, education and children, housing, social care, transport, culture/religion/recreation, or other consumption. Government remains the payer and `in_kind` remains the delivery type on both sides.

Existing `in_kind.other` facts are household-attributed and remain included as general consumption. Truly collective services use the separate memo-only `memo.government.collective_service` component and are excluded from household income, expense, and fiscal-balance totals; they are never forced into a paired household use.

## Evidence and linkage rules

- A key is stable only within its spine and reference year. It is not a cross-survey match.
- Direct observations are not overwritten by modeled values.
- Each fact states whether it is observed, modeled, transported, macro-calibrated, or derived.
- Each fact states whether it is the `reported` or `allocated` representation.
- Household-level data, local paths, and source identifiers never enter Git outputs.
- Component facts use non-negative amounts. Their accounting side controls display and aggregation signs.
