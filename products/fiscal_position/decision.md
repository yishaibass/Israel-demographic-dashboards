# Fiscal-position decisions

- Pool the 2021–2023 Household Expenditure Surveys.
- Allocate NII cash transfers using benefit-specific controls and split long-term care between cash and in-kind services.
- Allocate welfare using 2023 program budgets and recipient profiles.
- Allocate health using the official age-by-sex capitation schedule.
- Retain validated 2018 education relative keys, scaled to the 2023 budget.
- Support `karlinsky_flow` and `household_equity` capital-incidence modes through one selector that applies to both 2018 and 2023; mixed-year modes are invalid. Equity mode uses separately dated artifacts and controls for each year.
- The 2018 equity target uses Longitudinal Survey Wave 6. Asset imputations and tail caps are estimated within Wave 6, and the pension lifecycle curve is anchored to the end-2018 control; no Wave-10 household or 2023 administrative scalar enters the target. End-2018 CBS controls cover deposits, tradable investments and household pension/provident/insurance claims; Bank of Israel controls cover housing and nonhousing debt. No compatible market-value housing-plus-land control was found, so the CBS produced-dwellings value is a lower-bound sensitivity rather than a calibration target.
- The 2018 transport model's survey-weighted out-of-fold diagnostics are log R² 0.594, level R² 0.610, Spearman 0.879 and MAE ₪903 thousand. The non-overlapping Wave-4→Wave-6 test gives log R² 0.446 and Spearman 0.790. The Haredi source cell is small (n=145) and its out-of-fold mean bias is +16.65%. Global-only versus sector-post-stratified calibration changes sector tax estimates by at most 2.70% and decile estimates by at most 0.303%.
- Increment 1 publishes from the configured `fiscal_position_project`; the analytical pipeline is not yet independently runnable from a fresh clone.
- The Hebrew dashboard shell was recovered from the prior online release (`f0423cf:fiscal-position/index.html`, Git blob `303ae1c88b4b60000a1a4de619d6a820fb845a16`). The tracked template carries an empty data object; publication injects current results, minimally updates the four superseded NII, welfare, health and capital-key disclosures, and applies the shared Analytics contract.
