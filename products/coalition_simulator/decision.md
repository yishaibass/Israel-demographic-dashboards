# Coalition-simulator decisions

- Keep the simulator separate from the voting-propensity model.
- Publish a self-contained Hebrew dashboard at `voting-simulator/index.html`.
- Preserve current model logic during the first migration increment.
- Keep demographic analysis as an embedded sibling product; publication integration must not alter the coalition model payload.
- Own the demographic tab, iframe resizing, home navigation and Analytics application in this repository's publication build.
- Increment 1 publishes from the configured `coalition_simulator_project`; the analytical pipeline is not yet independently runnable from a fresh clone.

## 4 November 2026 — Separate K26 comparison

Keep the K21–K25 mean and population SD fixed. Fit K26 separately using the
original constrained two-regression formulation and explicit K25 locality
composition carry-forward. Do not infer ballot identity from unchanged numbers.
Show changes in percentage points and historical SD; suppress SD ratios below
`1e-6`. Preserve source coverage, weak-fit warnings, and conditional bootstrap
intervals. The repository owns `model/k26_comparison.py` and `k26_dashboard.py`;
the configured source project retains the historical analysis and HTML template.

Use the secondary mirror's count-status label only with its verification limits.
Keep declared seats missing until independently verified. K26 group estimates
can populate a scenario while electorate and threshold assumptions remain visible.
