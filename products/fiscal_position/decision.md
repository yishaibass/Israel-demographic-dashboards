# Fiscal-position decisions

- Pool the 2021–2023 Household Expenditure Surveys.
- Allocate NII cash transfers using benefit-specific controls and split long-term care between cash and in-kind services.
- Allocate welfare using 2023 program budgets and recipient profiles.
- Allocate health using the official age-by-sex capitation schedule.
- Retain validated 2018 education relative keys, scaled to the 2023 budget.
- Support `karlinsky_flow` and `household_equity` capital-incidence modes and record the selected mode in each run.
- Increment 1 publishes from the configured `fiscal_position_project`; the analytical pipeline is not yet independently runnable from a fresh clone.
- The Hebrew dashboard shell was recovered from the prior online release (`f0423cf:fiscal-position/index.html`, Git blob `303ae1c88b4b60000a1a4de619d6a820fb845a16`). The tracked template carries an empty data object; publication injects current results, minimally updates the four superseded NII, welfare, health and capital-key disclosures, and applies the shared Analytics contract.
