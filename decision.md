# Decisions

- Analytical products live under `products/`; shared code is limited to stable contracts and runtime utilities under `shared/`.
- Existing GitHub Pages routes remain `balance-sheet/`, `fiscal-position/`, and `voting-simulator/`.
- Raw and household-level data stay outside Git. Machine-specific paths live only in ignored `config/paths.toml`.
- Migration preserves current methodology and public outputs before refactoring model internals.
- Product builders publish complete static HTML and test each route's Google Analytics contract.
- Household P&L facts use one canonical component catalog. The `reported` view retains HES direct taxes and cash benefits; `allocated_extended` replaces those fiscal roots with calibrated allocations and adds paired public-service resource/use entries.
- Public and private health and education remain separate. Owner-occupied housing uses a matched imputed resource/use value; mortgage interest is an expense, while mortgage principal and retirement contributions are financing flows.
- Purchaser-price tax reclassification is limited to verified pre-calibration tax embedded in reported purchases. Full allocated tax remains charged and reconciled to its national control.
- The 2018 market-flow adapter uses original 2016–2018 HES modules in 2018 NIS. Because HES reports mortgage principal and interest together, it transports the current balance-sheet model's age-band interest shares.
- Household P&L facts use orthogonal economic-function, payer/funder, delivery/value-type, and representation dimensions inherited from the component catalog.
- Attributable government in-kind resources present as one `Public services received` income line; matched uses roll into functional household expense sections.
- Existing household-allocated other services remain included as general consumption. Non-attributable collective services are memo-only and excluded from household P&L totals.
- Market and fiscal HES facts use the same explicit source tuple `year:s_seker:misparmb` before hashing to an opaque household key. Export producers retain all three fields and reject unexpected survey identifiers.
