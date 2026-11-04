# Voting-propensity decisions

- Publish aggregate demographic propensities, not individual voting records.
- Keep Census microdata and individual-row probabilities outside Git.
- Publish the disclosure-safe explorer at `voting-simulator/demographics.html`.

## 4 November 2026 — One design across six elections

Keep the equal-weight K21–K25 cube and 2022 population weights as the default.
For comparable election estimates, fit K21–K26 jointly under the same locality
supervision, features, objective, and shared/election-deviation formulation.
Keep the original published historical cube as a separate reference and export
a new comparable historical mean plus per-election cubes for SD. This is a
retrospective fit that uses K26, distinct from a pre-election forecast.
Map Yashar/Beyachad to liberal center, Democrats to Zionist left, and unassigned
minor lists to other under the user's decision. Family coding and coalition
coding remain separate.

The K26 Census match covers 200 localities and 81.71% of national valid votes.
Publish per-election coverage and source caveats. Retain fiscal values. Compare
selected-group conditional shares across individual historical cubes before
forming their equal-weight mean and population SD. A mean joint cube is a
different summary when turnout varies. Use explicit `--k26-current` input and
reject stale snapshot mismatches. Secondary source status is not independent
certification. The tested model runtime is Python 3.15.2.

## Independent election fits and selection

Keep the published five-election reference fixed. Fit each election independently
under the same specification and Census population; retain the joint model as a
benchmark. Compare both approaches on identical locality folds and inspect how
subgroup estimates change when localities are held out. Preserve the diagnostics
with the aggregate cubes. Publication requires a reviewed bundle with matching
snapshot, collection and per-election fit identities.

Use one fit selector for each independently fitted election and a combined
2019–2026 joint fit. Compute the combined option from the equal-weight mean of
all six joint-model probability vectors, then aggregate under demographic filters.
Allow any two fits, or the five-election independent historical mean, to be compared.
Report whole percentages, percentage-point differences and one-decimal historical
SD of the five independent K21–K25 subgroup rates. Comparisons involving the joint
summary also reflect model differences. Keep the frozen reference visibly distinct
from the refitted historical mean.
Keep fiscal values and Census weights unchanged. Geographic validation and fold
sensitivity do not establish individual votes or subgroup confidence intervals.
