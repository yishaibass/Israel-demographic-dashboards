# Decisions

- Analytical products live under `products/`; shared code is limited to stable contracts and runtime utilities under `shared/`.
- Existing GitHub Pages routes remain `balance-sheet/`, `fiscal-position/`, and `voting-simulator/`.
- Raw and household-level data stay outside Git. Machine-specific paths live only in ignored `config/paths.toml`.
- Migration preserves current methodology and public outputs before refactoring model internals.
- Product builders publish complete static HTML and test each route's Google Analytics contract.
