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
