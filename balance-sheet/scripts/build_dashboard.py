"""
build_dashboard.py -- reads figures/*.csv (+ fig_meta.json) written by pipeline.py
and embeds their PARSED contents as JSON into self-contained output/index.html
(English, LTR) and output/index_vHe.html (Hebrew, RTL).

No calculation happens here and none happens in the browser: pipeline.py already
computed every weighted mean, weighted quantile, share, Gini, etc. The JS in this
page only reads the embedded arrays and (a) formats numbers for display, (b) sorts
rows that are already fully computed, (c) re-groups a handful of already-summed
line items in fig_national_balance_sheet.csv into a 3-way RE/Financial/Pension
split for the national donut (the same split fig_composition_by_group.csv already
reports per social group -- this just applies it to the one national total using
the individual line items that are already in that same table), and (d) divides
two already-aggregated totals from that same table to get "real estate share of
national assets" (a display ratio of two headline numbers, not a new statistical
aggregation). Nothing here re-derives a weighted mean, a percentile, a Gini, or
loops over households -- there are no household-level rows in this file at all,
only the ~10 pre-aggregated figure tables.

A file:// page cannot fetch CSVs at runtime, so "HTML reads from CSV" is honoured
by parsing the CSVs at BUILD time here (in Python) and inlining the parsed JSON.

This file is a single, language-parametrized builder (merged from what used to be
build_dashboard.py + build_dashboard_he.py). Running it with no arguments builds
BOTH output/index.html (English) and output/index_vHe.html (Hebrew). The only
things that vary by language are: (1) static copy (STRINGS), (2) a handful of
data-derived display labels (TR, looked up at runtime in the page's own JS via
tr()), (3) the manifest per-file descriptions (MANIFEST_DESC), (4) four KPI-tile
label/note strings (UI), and (5) chart geometry + RTL mirroring, all gated on a
single boolean baked into the page as `const RTL = true|false;`.
"""

import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(HERE, "..", "figures")
OUT_DIR = os.path.join(HERE, "..", "output")

# NOTE: pipeline.py still generates every figure CSV; this list is only the
# subset the front-end actually reads/embeds. fig_networth_by_group,
# fig_networth_by_dimension and fig_ratios_by_group were dropped here (not from
# the pipeline) when their charts were retired in favour of the cross-tab
# explorer + two-way matrix, so their (now-unused) rows no longer bloat index.html.
CSV_FIGURES = [
    "fig_national_balance_sheet.csv",
    "fig_composition_by_group.csv",
    "fig_networth_distribution.csv",
    "fig_income_wealth.csv",
    "fig_trajectory.csv",
    "fig_participation.csv",
    "fig_income_composition_by_group.csv",
    "fig_metric_by_breakdown.csv",
    "fig_metric_matrix.csv",
]

# Optional figure under active development. Keeping it separate lets the
# existing dashboard continue to build before the expenditure-survey pipeline
# has produced the new table; once present it is embedded like every other
# pre-aggregated figure (no browser-side modelling or household aggregation).
OPTIONAL_CSV_FIGURES = [
    "fig_household_pnl_by_age.csv",
    "fig_household_expenses_by_age.csv",
]

# ---------------------------------------------------------------------------
# Language pack
# ---------------------------------------------------------------------------
LANG_META = {
    "en": {"code": "en", "dir": "ltr", "dir_attr": "", "out": "index.html", "rtl": False},
    "he": {"code": "he", "dir": "rtl", "dir_attr": ' dir="rtl"', "out": "index_vHe.html", "rtl": True},
}

# Extra CSS rule only the Hebrew build needs (keeps RTL amounts/numbers readable
# inside otherwise-RTL table cells and SVG text). Empty string for English.
RTL_CSS_EXTRA = {
    "en": "",
    "he": ".stmt .amt, .datatable td, svg text{unicode-bidi:plaintext;}",
}

# Static UI copy -- every string pulled verbatim from the two original builders.
STRINGS = {
    "en": {
        "title": "The Israeli Household Balance Sheet -- CBS Longitudinal Survey 2012-2023",
        "h1": "The Israeli Household Balance Sheet",
        "sub": "Central Bureau of Statistics &middot; Longitudinal Survey (&#1505;&#1511;&#1512; &#1488;&#1512;&#1493;&#1498; &#1496;&#1493;&#1493;&#1495;) &middot; 2012&ndash;2023 &middot; all figures pre-aggregated, weighted by household",
        "aiflag": "AI-generated dashboard &mdash; review before external use.",

        "nav_s1": "1&nbsp;&middot;&nbsp;P&amp;L",
        "nav_s2": "2&nbsp;&middot;&nbsp;Cash flow",
        "nav_s3": "3&nbsp;&middot;&nbsp;Balance sheet",
        "nav_s4": "4&nbsp;&middot;&nbsp;Income &amp; capital",
        "nav_s5": "5&nbsp;&middot;&nbsp;Household P&amp;L by age",
        "nav_s6": "4&nbsp;&middot;&nbsp;Methodology",

        "s1_kicker": "Section 1",
        "s1_head": "The national picture, 2023",
        "s1_intro": "The average Israeli household's balance sheet: what it owns, what it owes, and what that\n    nets out to. Every number below is a population-weighted average or share across the full survey sample.",
        "s1_card1_h3": "The average household balance sheet",
        "s1_card1_csrc": "Source: fig_national_balance_sheet.csv &middot; &#9671; = modelled (see &sect;5)",
        "s1_card2_h3": "National asset composition",
        "s1_card2_csrc": "Source: fig_national_balance_sheet.csv (Real Estate / Financial ex-pension / Pension line items grouped)",

        "s2_kicker": "Section 2",
        "s2_head": "Differences between social segments",
        "s2_intro": "The same national average masks very different balance sheets across social groups &mdash;\n    in how much households are worth, in what that wealth is held in, and in which financial products they use.",
        "s2_exp_h3": "Cross-tab explorer",
        "s2_exp_csrc": "Source: fig_metric_by_breakdown.csv &middot; wave 10 (2023), weighted by household",
        "s2_exp_metric_label": "Metric",
        "s2_exp_breakdown_label": "Breakdown",
        "s2_mx_h3": "Two-way matrix",
        "s2_mx_csrc": "Source: fig_metric_matrix.csv &middot; wave 10 (2023), weighted by household",
        "s2_mx_metric_label": "Metric",
        "s2_mx_dima_label": "Dimension A (rows)",
        "s2_mx_dimb_label": "Dimension B (columns)",
        "s2_card1_h3": "Net worth by social group, 2023 (&#8362;)",
        "s2_card1_csrc": "Source: fig_networth_by_group.csv",
        "s2_card2_h3": "Asset composition by social group, 2023",
        "s2_card2_csrc": "Source: fig_composition_by_group.csv",
        "s2_card3_h3": "Net worth by socio-demographic cut, 2023 (&#8362;)",
        "s2_card3_csrc": "Source: fig_networth_by_dimension.csv",
        "s2_dim_label": "Dimension",
        "s2_card4_h3": "Participation rates by social group, 2023",
        "s2_card4_csrc": "Source: fig_participation.csv",

        "s3_kicker": "Section 3",
        "s3_head": "Distribution within segments",
        "s3_intro": "Averages hide spread. Within every group there are households with little or negative net\n    worth alongside households with a great deal &mdash; and the width of that spread differs by group too.",
        "s3_card1_h3": "Net worth percentile spread by social group, 2023 (&#8362;)",
        "s3_card1_csrc": "Source: fig_networth_distribution.csv",
        "s3_card2_h3": "Balance-sheet ratios by social group, 2023",
        "s3_card2_csrc": "Source: fig_ratios_by_group.csv",

        "s4_kicker": "Section 4",
        "s4_head": "Income, demographics and capital",
        "s4_intro": "Wealth and income are related but distinct. This section lines households up by income decile\n    to see how their asset mix shifts across the income ladder, then breaks monthly income down by source across\n    social groups. (Net worth by income decile is in the Section&nbsp;2 explorer.)",
        "s4_card1_h3": "Mean net worth by income decile, 2023 (&#8362;)",
        "s4_card1_csrc": "Source: fig_income_wealth.csv",
        "s4_card2_h3": "Asset composition by income decile, 2023",
        "s4_card2_csrc": "Source: fig_income_wealth.csv",
        "s4_card3_h3": "Capital-income share by income decile, 2023",
        "s4_card3_csrc": "Source: fig_income_wealth.csv",
        "s4_card4_h3": "Monthly income composition by social group, 2023 (&#8362;/month)",
        "s4_card4_csrc": "Source: fig_income_composition_by_group.csv",

        "s5p_kicker": "Household statements",
        "s5p_head": "Household financial statements",
        "s5p_intro": "Select the five-column household breakdown. Click grouped statement rows to show their components.",
        "s5p_cash_h3": "Cash flow (&#8362;/month)",
        "s5p_economic_h3": "Economic P&amp;L (&#8362;/month)",
        "s5p_card_csrc": "Source: fig_household_pnl_by_age.csv &middot; CBS Household Expenditure Survey 2021–2023; flows are normalized to 2023 using nominal GDP per capita. Wealth overlays refer to 2023 and are marked ◇",
        "s5p_wealth_h3": "Household balance sheet (&#8362;)",
        "s5p_wealth_csrc": "Cohort-level statistical overlays from the longitudinal survey; these are stocks, not monthly flows, and are not household-level joins.",
        "s5p_empty": "The household P&amp;L figure has not been generated yet. Re-run the data pipeline, then rebuild the dashboard.",
        "s5p_group_label": "Column breakdown",
        "s5p_expense_h3": "Parent and child expense detail by age (&#8362;/month)",
        "s5p_expense_csrc": "Source: fig_household_expenses_by_age.csv &middot; select an expense category to compare across age bands",
        "s5p_expense_label": "Expense category",
        "s5p_expense_metric_label": "Allocation view",
        "s5p_income_view_label": "Income detail",
        "s5p_housing_view_label": "Housing detail",
        "s5p_wealth_view_label": "Balance-sheet detail",

        "s5_kicker": "Section 4",
        "s5_head": "Methodology &amp; caveats",
        "s5_intro": "Plain-language notes on how figures in this dashboard were built, and where they should be\n    read with caution. Full detail lives in fig_meta.json.",

        "meth_h4_data": "Data",
        "meth_p_data": "Israel CBS Longitudinal Survey (&#1505;&#1511;&#1512; &#1488;&#1512;&#1493;&#1498; &#1496;&#1493;&#1493;&#1495;), 10 waves, 2012&ndash;2023, tracking the same households\n    over time. All figures are weighted by <code>weight_hh</code> to represent the general household population,\n    not the raw sample.",

        "meth_h4_home": "Home value model",
        "meth_p_home": "Roughly a quarter of homeowners report a subjective home value directly; the rest are filled in &mdash; either\n    with a CBS-provided estimate, or, where neither exists, a hedonic model (a regression on home/household\n    characteristics) predicts a value. That model explains a modest share of the variation in home prices, so\n    modelled home values (and therefore net worth for those households) carry real uncertainty.",

        "stat_r2held": "held-out R&sup2;",
        "stat_r2in": "in-sample R&sup2;",
        "stat_nself": "self-reported",
        "stat_ncbs": "CBS-estimate fill",
        "stat_nmodel": "model-imputed",

        "meth_h4_pension": "Pension model",
        "meth_p_pension": "Pension wealth is not directly observed in the survey (only pension income flows are), so it is\n    modelled across the household life-cycle: contributions accumulate on capped work income up to retirement (age 67)\n    at an assumed real return, and the retirement stock is then drawn down over the remaining retirement years. This\n    replaced an earlier two-regime version that jumped ~59% at retirement and had no drawdown; the level is anchored so\n    the population mean matches the conservative Bank of Israel / Capital Market Authority long-term-savings aggregate\n    (~&#8362;830,000/household &mdash; ~&#8362;2.5T at end-2023 divided across ~3.05M households). Pension values are\n    flagged &#9671; wherever shown as a modelled line.",

        "stat_mean2023": "modelled mean, 2023 (post-cap)",
        "stat_meanpre": "modelled mean, 2023 (pre-cap)",
        "stat_macro": "macro benchmark target",

        "meth_h4_winsor": "Winsorization &amp; the distribution figures",
        "meth_p_winsor": "All amounts are winsorized (capped) at the 99th percentile (income at the 99.5th), floored at zero, to limit\n    the influence of a handful of extreme reported values on every mean shown in this dashboard &mdash; except\n    Section 3's distribution figures (percentiles, Gini, top-10% share), which are built from a separate,\n    less-aggressive 99.5th-percentile cap. That's deliberate: capping the top tail too hard would understate the\n    very inequality those statistics are meant to measure.",
        "meth_winsor_caveat": "<b>Caveat.</b> Amounts are winsorized (capped) at roughly the 99th&ndash;99.5th percentile before these statistics\n    are computed, to limit the influence of extreme reported values (see fig_meta.json, \"winsorization\"). Household\n    surveys such as this one are also well known to under-sample and under-report holdings of the very wealthy, who\n    are rare in any general population sample and less likely to disclose full holdings. Both effects push in the\n    same direction: the net-worth percentiles above and the Gini coefficient shown in this dashboard are more likely\n    to <b>understate</b> true net-worth concentration in Israel than to overstate it.",

        "meth_h4_wave": "Odd-wave rotation",
        "meth_p_wave": "Financial assets and liabilities (deposits, investments, mortgage, consumer loans, additional property, land)\n    are not collected in odd-numbered waves (2012, 2014, 2017, 2019, 2021). Net worth is undefined (not zero) in\n    those years; the figures in this dashboard are drawn from wave 10 (2023), an even wave with full financial data.",

        "meth_h4_sample": "Sample",
        "meth_p_sample": "<b id=\"m-nrows\">&mdash;</b> household-wave observations across all 10 waves, latest wave = wave <b id=\"m-latest\">&mdash;</b>\n    (2023).",

        "meth_h4_limits": "Limitations",
        "meth_limits_list": "    <li>Modelled lines (pension, and imputed home values) carry model uncertainty beyond ordinary sampling error.</li>\n    <li>Winsorization caps the influence of the very top of the distribution, which likely understates true\n      concentration of wealth (see caveat above).</li>\n    <li>Group sample sizes vary widely (Haredi and Masorti have only ~170&ndash;530 households in wave 10, versus\n      ~2,000 for Hiloni); smaller-N estimates are noisier than the headline numbers suggest.</li>",

        "meth_h4_manifest": "Figure manifest",

        "footer_arch": "<b>Architecture:</b> pipeline.py reads the raw panel and applies every model, weight and aggregation, writing tidy\n  pre-computed tables to figures/. build_dashboard.py parses those CSVs and fig_meta.json at build time and embeds\n  the result as JSON into this page. The JavaScript below performs no weighted means, no percentile calculations, no\n  Gini computation and no modelling -- it only formats numbers, sorts already-computed rows, and draws charts from\n  the embedded arrays.",
        "footer_source": "<b>Source:</b> Israel CBS Longitudinal Survey (&#1505;&#1511;&#1512; &#1488;&#1512;&#1493;&#1498; &#1496;&#1493;&#1493;&#1495;), 2012&ndash;2023. AI-generated &mdash; review before external use.",
    },
    "he": {
        "title": "מאזן משק הבית הישראלי — הלמ״ס, סקר ארוך טווח 2012–2023",
        "h1": "מאזן משק הבית הישראלי",
        "sub": "הלשכה המרכזית לסטטיסטיקה &middot; סקר ארוך טווח &middot; 2012&ndash;2023 &middot; כל הנתונים מחושבים ומשוקללים ברמת משק הבית",
        "aiflag": "לוח מחוונים שנוצר בעזרת בינה מלאכותית &mdash; יש לבדוק לפני שימוש חיצוני.",

        "nav_s1": "1&nbsp;&middot;&nbsp;דו״ח רווח והפסד",
        "nav_s2": "2&nbsp;&middot;&nbsp;תזרים מזומנים",
        "nav_s3": "3&nbsp;&middot;&nbsp;מאזן",
        "nav_s4": "4&nbsp;&middot;&nbsp;הכנסה והון",
        "nav_s5": "5&nbsp;&middot;&nbsp;דו״ח משק בית לפי גיל",
        "nav_s6": "4&nbsp;&middot;&nbsp;מתודולוגיה",

        "s1_kicker": "פרק 1",
        "s1_head": "התמונה הארצית, 2023",
        "s1_intro": "המאזן של משק בית ישראלי ממוצע &mdash; מה ברשותו, מה הוא חייב, וכמה נשאר בסוף. כל מספר כאן הוא ממוצע או שיעור משוקלל על פני כלל מדגם הסקר.",
        "s1_card1_h3": "המאזן של משק בית ממוצע",
        "s1_card1_csrc": "מקור: fig_national_balance_sheet.csv &middot; ◇ = מודלי (ראו פרק 5)",
        "s1_card2_h3": "הרכב הנכסים הארצי",
        "s1_card2_csrc": "מקור: fig_national_balance_sheet.csv (שורות נדל״ן / פיננסי ללא פנסיה / פנסיה מקובצות)",

        "s2_kicker": "פרק 2",
        "s2_head": "פערים בין קבוצות חברתיות",
        "s2_intro": "הממוצע הארצי מסתיר מאזנים שונים מאוד בין הקבוצות החברתיות &mdash; בכמה שווה משק הבית, במה מוחזק ההון, ובאילו מוצרים פיננסיים משתמשים.",
        "s2_exp_h3": "מגלה חתכים (Cross-tab)",
        "s2_exp_csrc": "מקור: fig_metric_by_breakdown.csv &middot; גל 10 (2023), משוקלל לפי משק בית",
        "s2_exp_metric_label": "מדד",
        "s2_exp_breakdown_label": "חתך",
        "s2_mx_h3": "מטריצה דו-ממדית",
        "s2_mx_csrc": "מקור: fig_metric_matrix.csv &middot; גל 10 (2023), משוקלל לפי משק בית",
        "s2_mx_metric_label": "מדד",
        "s2_mx_dima_label": "ממד א׳ (שורות)",
        "s2_mx_dimb_label": "ממד ב׳ (עמודות)",
        "s2_card1_h3": "שווי נטו לפי קבוצה חברתית, 2023 (₪)",
        "s2_card1_csrc": "מקור: fig_networth_by_group.csv",
        "s2_card2_h3": "הרכב הנכסים לפי קבוצה חברתית, 2023",
        "s2_card2_csrc": "מקור: fig_composition_by_group.csv",
        "s2_card3_h3": "שווי נטו לפי חתך חברתי-דמוגרפי, 2023 (₪)",
        "s2_card3_csrc": "מקור: fig_networth_by_dimension.csv",
        "s2_dim_label": "מימד",
        "s2_card4_h3": "שיעורי החזקה לפי קבוצה חברתית, 2023",
        "s2_card4_csrc": "מקור: fig_participation.csv",

        "s3_kicker": "פרק 3",
        "s3_head": "פיזור בתוך הקבוצות",
        "s3_intro": "ממוצעים מסתירים את הפיזור. בכל קבוצה יש משקי בית עם שווי נטו נמוך או שלילי לצד משקי בית אמידים מאוד &mdash; ורוחב הפיזור הזה משתנה גם הוא בין הקבוצות.",
        "s3_card1_h3": "פיזור אחוזוני של שווי נטו לפי קבוצה חברתית, 2023 (₪)",
        "s3_card1_csrc": "מקור: fig_networth_distribution.csv",
        "s3_card2_h3": "יחסים פיננסיים לפי קבוצה חברתית, 2023",
        "s3_card2_csrc": "מקור: fig_ratios_by_group.csv",

        "s4_kicker": "פרק 4",
        "s4_head": "הכנסה, דמוגרפיה והון",
        "s4_intro": "הון והכנסה קשורים אך שונים זה מזה. הפרק מסדר את משקי הבית לפי עשירוני הכנסה כדי לראות כיצד תמהיל הנכסים משתנה לאורך סולם ההכנסה, ולאחר מכן מפרק את ההכנסה החודשית לפי מקור ולפי קבוצה חברתית. (שווי נטו לפי עשירון הכנסה זמין במגלה החתכים בפרק&nbsp;2.)",
        "s4_card1_h3": "שווי נטו ממוצע לפי עשירון הכנסה, 2023 (₪)",
        "s4_card1_csrc": "מקור: fig_income_wealth.csv",
        "s4_card2_h3": "הרכב הנכסים לפי עשירון הכנסה, 2023",
        "s4_card2_csrc": "מקור: fig_income_wealth.csv",
        "s4_card3_h3": "שיעור הכנסות ההון לפי עשירון הכנסה, 2023",
        "s4_card3_csrc": "מקור: fig_income_wealth.csv",
        "s4_card4_h3": "הרכב ההכנסה החודשית לפי קבוצה חברתית, 2023 (₪ לחודש)",
        "s4_card4_csrc": "מקור: fig_income_composition_by_group.csv",

        "s5p_kicker": "דוחות משק הבית",
        "s5p_head": "הכנסות, הוצאות וחיסכון של משק הבית לפי גיל",
        "s5p_intro": "דו״ח כלכלי חודשי למשק הבית הממוצע בכל קבוצת גיל. לחיצה על שורות ההכנסה, ההוצאות, החיסכון והמאזן תציג את הרכיבים.",
        "s5p_cash_h3": "תזרים מזומנים (₪ לחודש)",
        "s5p_economic_h3": "דו״ח רווח והפסד כלכלי (₪ לחודש)",
        "s5p_card_csrc": "מקור: fig_household_pnl_by_age.csv &middot; סקר הוצאות משק הבית 2021–2023; התזרימים מותאמים ל-2023 לפי התמ״ג הנומינלי לנפש. נתוני העושר מתייחסים ל-2023 ומסומנים ◇",
        "s5p_wealth_h3": "מאזן משק הבית לפי גיל (₪)",
        "s5p_wealth_csrc": "התאמות סטטיסטיות ברמת קבוצת גיל מהסקר האורכי; אלה יתרות ולא תזרימים חודשיים, ואין חיבור ברמת משק הבית.",
        "s5p_empty": "קובץ דו״ח משק הבית טרם נוצר. יש להריץ מחדש את צינור הנתונים ולאחר מכן לבנות את לוח המחוונים.",
        "s5p_group_label": "חתך עמודות",
        "s5p_expense_h3": "פירוט הוצאות הורים וילדים לפי גיל (₪ לחודש)",
        "s5p_expense_csrc": "מקור: fig_household_expenses_by_age.csv &middot; יש לבחור קטגוריית הוצאה להשוואה בין קבוצות גיל",
        "s5p_expense_label": "קטגוריית הוצאה",
        "s5p_expense_metric_label": "הקצאת ההוצאה",
        "s5p_income_view_label": "פירוט הכנסות",
        "s5p_housing_view_label": "פירוט דיור",
        "s5p_wealth_view_label": "פירוט מאזן",

        "s5_kicker": "פרק 4",
        "s5_head": "מתודולוגיה והסתייגויות",
        "s5_intro": "הסברים בשפה פשוטה על אופן בניית הנתונים בלוח, והיכן כדאי לקרוא אותם בזהירות. הפירוט המלא נמצא ב-fig_meta.json.",

        "meth_h4_data": "הנתונים",
        "meth_p_data": "סקר ארוך טווח של הלמ״ס, 10 גלים, 2012&ndash;2023, העוקב אחר אותם משקי בית לאורך זמן. כל הנתונים משוקללים לפי <code>weight_hh</code> לייצוג אוכלוסיית משקי הבית ולא המדגם הגולמי.",

        "meth_h4_home": "מודל שווי הדירה",
        "meth_p_home": "כרבע מבעלי הדירות מדווחים ישירות על שווי סובייקטיבי לדירתם; היתר מושלמים &mdash; אם באמצעות אומדן של הלמ״ס, ואם, כשאין אף אחד מהשניים, באמצעות מודל הֶדוני (רגרסיה על מאפייני הדירה ומשק הבית) שחוזה את השווי. המודל מסביר חלק צנוע מהשונות במחירי הדירות, ולכן שוויים מודליים של דירות (ואיתם השווי נטו של אותם משקי בית) נושאים אי-ודאות ממשית.",

        "stat_r2held": "R&sup2; על מדגם בקרה",
        "stat_r2in": "R&sup2; על מדגם האימון",
        "stat_nself": "דיווח עצמי",
        "stat_ncbs": "השלמה מאומדן הלמ״ס",
        "stat_nmodel": "זקיפה במודל",

        "meth_h4_pension": "מודל הפנסיה",
        "meth_p_pension": "עושר פנסיוני אינו נצפה ישירות בסקר (זמינים רק זרמי ההכנסה מפנסיה), ולכן הוא מחושב לאורך מחזור החיים של משק הבית: ההפקדות נצברות על הכנסה מעבודה (עד תקרה) עד גיל הפרישה (67) בתשואה ריאלית מונחת, ולאחר מכן צבירת הפרישה נמשכת (מהוונת) לאורך שנות הפרישה שנותרו. מודל זה החליף גרסה קודמת בעלת שני משטרים שקפצה בכ-59% בגיל הפרישה וללא מנגנון משיכה; הרמה מעוגנת כך שהממוצע הארצי יתאים לאמצע-הטווח השמרני של סך נכסי החיסכון לטווח ארוך (בנק ישראל / רשות שוק ההון) — כ-₪830,000 למשק בית (כ-₪2.5 טריליון בסוף 2023 מחולק בכ-3.05 מיליון משקי בית). ערכי הפנסיה מסומנים ב-◇ בכל מקום שבו הם מוצגים כשורה מודלית.",

        "stat_mean2023": "ממוצע מודלי, 2023 (אחרי חיתוך)",
        "stat_meanpre": "ממוצע מודלי, 2023 (לפני חיתוך)",
        "stat_macro": "יעד עוגן המאקרו",

        "meth_h4_winsor": "חיתוך ערכים (Winsorization) ותרשימי הפיזור",
        "meth_p_winsor": "כל הסכומים נחתכים באחוזון ה-99 (הכנסה באחוזון ה-99.5), עם רצפה של אפס, כדי לצמצם את השפעתם של מעט ערכים קיצוניים על כל ממוצע בלוח &mdash; פרט לתרשימי הפיזור בפרק 3 (אחוזונים, ג׳יני, חלק העשירון העליון), הנבנים מחיתוך מתון יותר באחוזון ה-99.5. זו בחירה מכוונת: חיתוך אגרסיבי מדי של הזנב העליון היה ממעיט דווקא באי-השוויון שאותן סטטיסטיקות אמורות למדוד.",
        "meth_winsor_caveat": "<b>הסתייגות.</b> לפני חישוב הנתונים הסכומים נחתכים (winsorization) סביב האחוזון ה-99&ndash;99.5, כדי לצמצם את השפעתם של ערכים קיצוניים שדווחו (ראו fig_meta.json, ״winsorization״). בנוסף, ידוע היטב שסקרי משקי בית כמו זה מדגמים ומדווחים בחסר את החזקותיהם של העשירים מאוד &mdash; נדירים בכל מדגם אוכלוסייה ופחות נוטים לחשוף את מלוא נכסיהם. שתי ההשפעות פועלות באותו כיוון: סביר יותר שהאחוזונים ומקדם הג׳יני המוצגים כאן <b>ממעיטים</b> בריכוזיות ההון האמיתית בישראל מאשר מפריזים בה.",

        "meth_h4_wave": "רוטציית הגלים האי-זוגיים",
        "meth_p_wave": "נכסים והתחייבויות פיננסיים (פיקדונות, השקעות, משכנתה, הלוואות צרכניות, נכס נוסף, קרקע) אינם נאספים בגלים האי-זוגיים (2012, 2014, 2017, 2019, 2021). בשנים אלו השווי נטו אינו מוגדר (לא אפס); הנתונים בלוח זה נלקחים מגל 10 (2023), גל זוגי שבו נאסף מלוא המידע הפיננסי.",

        "meth_h4_sample": "מדגם",
        "meth_p_sample": "<b id=\"m-nrows\">&mdash;</b> תצפיות משק-בית-גל על פני כל 10 הגלים; הגל האחרון = גל <b id=\"m-latest\">&mdash;</b>\n    (2023).",

        "meth_h4_limits": "מגבלות",
        "meth_limits_list": "    <li>שורות מודליות (פנסיה ושוויי דירות זקופים) נושאות אי-ודאות מודל מעבר לטעות הדגימה הרגילה.</li>\n    <li>חיתוך הערכים מגביל את השפעת קצה הפיזור העליון, מה שכנראה ממעיט בריכוזיות ההון (ראו ההסתייגות לעיל).</li>\n    <li>גודל המדגם משתנה מאוד בין הקבוצות (לחרדים ולמסורתיים יש רק כ-170&ndash;530 משקי בית בגל 10, לעומת כ-2,000 לחילונים); אומדנים מבוססי-מדגם קטן רועשים יותר מכפי שנראה מהמספרים הראשיים.</li>",

        "meth_h4_manifest": "רשימת קובצי הנתונים",

        "footer_arch": "<b>ארכיטקטורה:</b> pipeline.py קורא את קובץ הפאנל הגולמי ומיישם את כל המודלים, המשקלים והאגרגציות, וכותב טבלאות מחושבות מראש לתיקיית figures/. build_dashboard.py קורא את קובצי ה-CSV ואת fig_meta.json בזמן הבנייה ומטמיע את התוצאה כ-JSON בדף זה. קוד ה-JavaScript אינו מבצע ממוצעים משוקללים, חישובי אחוזונים, חישוב ג׳יני או מידול &mdash; הוא רק מעצב מספרים, ממיין שורות מחושבות מראש, ומצייר תרשימים מהמערכים המוטמעים.",
        "footer_source": "<b>מקור:</b> סקר ארוך טווח של הלמ״ס, 2012&ndash;2023. נוצר בעזרת בינה מלאכותית &mdash; יש לבדוק לפני שימוש חיצוני.",
    },
}

# Data-derived display labels, looked up at runtime in the page's own JS via
# tr(). English tr() is pure identity -- an empty dict, so unknown keys just
# fall through to the key itself (see the JS `function tr(s)` below). This
# Hebrew dict is copied verbatim from the original build_dashboard_he.py.
TR = {
    "en": {},
    "he": {
        # Groups / scopes
        "All": "כלל האוכלוסייה", "Haredi": "חרדים", "Dati": "דתיים", "Dati (Religious)": "דתיים", "Masorti": "מסורתיים",
        "Hiloni": "חילונים", "Hiloni (Secular)": "חילונים", "Arab": "ערבים",
        # Balance-sheet items
        "Primary Residence": "דירת מגורים", "Additional Properties": "נכסי נדל״ן נוספים",
        "Land & Other Real Estate": "קרקע ונדל״ן אחר", "Total Real Estate": "סך הנדל״ן",
        "Bank Deposits & Savings": "פיקדונות וחיסכון בבנק", "Investment Portfolio": "תיק השקעות",
        "Pension & Retirement Savings": "פנסיה וחיסכון לטווח ארוך", "Total Financial Assets": "סך הנכסים הפיננסיים",
        "Total Assets": "סך הנכסים", "Mortgage": "משכנתה", "Consumer Loans": "הלוואות צרכניות",
        "Total Liabilities": "סך ההתחייבויות", "Net Worth": "שווי נטו",
        # segStarts section headers
        "Assets — Real Estate": "נכסים — נדל״ן", "Assets — Financial": "נכסים — פיננסיים", "Liabilities": "התחייבויות",
        # Asset classes (donut + stacked)
        "Real Estate": "נדל״ן", "Financial ex-pension": "פיננסי (ללא פנסיה)", "Pension": "פנסיה", "Pension ◇": "פנסיה ◇",
        # Donut + tooltip keys
        "Total assets": "סך הנכסים", "Share of assets": "חלק מסך הנכסים", "Mean value": "ערך ממוצע", "Total / month": "סה״כ לחודש",
        # Box-plot tooltip keys
        "p50 (median)": "p50 (חציון)", "Mean": "ממוצע", "Median": "חציון",
        # Legend
        "Bar = mean": "עמודה = ממוצע", "Tick = median": "קו = חציון",
        "Box = p25–p75": "קופסה = p25–p75", "Line = median (p50)": "קו = חציון (p50)", "Whiskers = p10–p90": "שפמים = p10–p90",
        # Dimension names (in the <select>)
        "Education": "השכלה", "Age band": "קבוצת גיל", "Tenure": "צורת מגורים", "Household size": "גודל משק הבית",
        "Occupation": "משלח יד", "Locality SES": "אשכול חברתי-כלכלי (יישוב)",
        "Locality SES cluster (1–10)": "אשכול חברתי-כלכלי (יישוב, 1–10)",
        # Dimension categories
        "No diploma": "ללא תעודה", "Secondary / Matriculation": "תיכונית / בגרות",
        "Post-secondary (non-academic)": "על-תיכונית (לא אקדמית)", "Academic degree": "תואר אקדמי",
        "Under 35": "עד 35", "35-44": "35–44", "45-54": "45–54", "55-64": "55–64", "65-74": "65–74", "75+": "75+",
        "Owner": "בעלות", "Renter": "שכירות", "Other": "אחר",
        "Managers & Professionals": "מנהלים ובעלי מקצועות חופשיים", "Technicians & Clerical": "טכנאים ופקידות",
        "Sales & Service": "מכירות ושירותים", "Skilled Manual & Agriculture": "עבודה מקצועית וחקלאות",
        "Operators & Unskilled": "מפעילים ובלתי-מקצועיים",
        "Low cluster (1-3)": "אשכול נמוך (1–3)", "Mid cluster (4-7)": "אשכול בינוני (4–7)", "High cluster (8-10)": "אשכול גבוה (8–10)",
        # Participation metric labels
        "With a mortgage": "עם משכנתה", "With a pension": "עם פנסיה", "Owns additional property": "בעלות על נכס נוסף",
        "With an investment portfolio": "עם תיק השקעות",
        # Income sources (stacked bar legend)
        "Work": "עבודה", "Govt transfers": "קצבאות והעברות", "Rental": "שכר דירה", "Interest": "ריבית", "Capital gains": "רווחי הון",
        # Household P&L by age
        "Household profile": "פרופיל משק הבית", "Households (sample)": "משקי בית במדגם", "Average household size": "גודל משק בית ממוצע",
        "Average children": "מספר ילדים ממוצע", "Households with children": "משקי בית עם ילדים",
        "Homeowners": "בעלי דירה", "Income": "הכנסות", "Labor income": "הכנסה מעבודה",
        "Pension income": "הכנסה מפנסיה", "Other asset income": "הכנסה אחרת מנכסים",
        "Transfers": "קצבאות והעברות", "Other income": "הכנסה אחרת",
        "Housing asset income": "הכנסה זקופה משירותי דיור", "Net imputed housing income": "הכנסת דיור זקופה נטו", "Mortgage interest": "ריבית משכנתה", "Equity housing service": "שירותי דיור מהון עצמי", "Mortgage principal": "קרן משכנתה", "Other real-estate return": "תשואה זקופה מנדל״ן נוסף",
        "Other financial-asset return": "תשואה זקופה מנכסים פיננסיים אחרים", "Pension asset return": "תשואה זקופה מנכסי פנסיה",
        "Adjusted gross economic income": "הכנסה כלכלית ברוטו מתואמת", "Adjusted disposable economic income": "הכנסה כלכלית פנויה מתואמת",
        "Housing consumption": "צריכת שירותי דיור", "Full mortgage payment": "מלוא תשלום המשכנתה",
        "Direct taxes": "מסים ישירים", "Disposable economic income": "הכנסה כלכלית פנויה",
        "Private transfers paid": "העברות פרטיות ששולמו", "Cash disposable income": "הכנסה פנויה במזומן",
        "Living expenses": "הוצאות מחיה", "Parent consumption": "צריכת הורים",
        "Child consumption": "צריכת ילדים", "Rent paid": "שכר דירה ששולם",
        "Imputed housing consumption": "צריכת דיור זקופה", "Mortgage interest": "ריבית משכנתה",
        "Total economic expenses": "סך ההוצאות הכלכליות", "Total saving": "סך החיסכון",
        "Cash consumption": "צריכה במזומן", "Cash residual saving": "חיסכון שיורי במזומן",
        "Saving allocation": "הקצאת החיסכון", "Mortgage principal": "קרן משכנתה",
        "Pension contributions": "הפקדות לפנסיה", "Training-fund contributions": "הפקדות לקרן השתלמות",
        "Provident-fund contributions": "הפקדות לקופת גמל", "Life/executive-insurance contributions": "הפקדות לביטוח חיים/מנהלים",
        "Other observed financial saving": "חיסכון פיננסי נצפה אחר", "Residual saving": "חיסכון שיורי",
        "Savings rate": "שיעור חיסכון", "Wealth overlay": "תמונת עושר",
        "Mortgage balance ◇": "יתרת משכנתה ◇", "Financial assets ◇": "נכסים פיננסיים ◇",
        "Pension wealth ◇": "עושר פנסיוני ◇", "Housing assets ◇": "נכסי דיור ◇",
        "Net worth ◇": "שווי נטו ◇", "Age band": "קבוצת גיל",
        "Assets": "נכסים", "Other financial assets ◇": "נכסים פיננסיים אחרים ◇",
        "Total assets ◇": "סך הנכסים ◇", "Other debt ◇": "חוב אחר ◇",
        "Total liabilities ◇": "סך ההתחייבויות ◇",
        "Food": "מזון", "Housing service": "שירותי דיור", "Restaurants": "מסעדות",
        "Travel & vacations": "נסיעות וחופשות", "Car purchases": "רכישת כלי רכב",
        "Other durables": "מוצרים בני-קיימא אחרים", "Transport excluding cars": "תחבורה ללא רכישת רכב",
        "Health": "בריאות", "Education & childcare": "חינוך וטיפול בילדים",
        "Leisure & luxury": "פנאי ומותרות", "Other consumption": "צריכה אחרת",
        "Childcare": "טיפול בילדים", "Education": "חינוך",
        "Parent expense": "הוצאות הורים", "Child expense": "הוצאות ילדים", "Total expense": "סך ההוצאה",
        "Collapsed": "מצומצם", "Expanded": "מורחב", "Summary": "סיכום",
        "Household fixed / unallocated": "הוצאה קבועה/לא מוקצית למשק הבית",
        "Parent variable expenses": "הוצאות משתנות של הורים", "Child variable expenses": "הוצאות משתנות של ילדים",
        "Household fixed expenses (excluding housing)": "הוצאות קבועות של משק הבית (ללא דיור)",
        "Housing services": "שירותי דיור", "Imputed rent": "שכר דירה זקוף", "Other housing services": "שירותי דיור אחרים",
        "Pension and other funded contributions": "הפקדות לפנסיה ולחיסכון ממומן אחר",
        "Cash income": "הכנסה במזומן", "Cash living outflows": "הוצאות מחיה במזומן",
        "Income bridge": "גשר הכנסות", "Expense bridge": "גשר הוצאות", "Saving bridge": "גשר חיסכון",
        "Economic disposable income": "הכנסה כלכלית פנויה", "Less: non-cash and valuation income": "בניכוי הכנסה לא-מזומנית והתאמות שווי",
        "Economic expenses": "הוצאות כלכליות", "Less: non-cash expenses": "בניכוי הוצאות לא-מזומניות", "Cash expenses": "הוצאות במזומן",
        "Less: net non-cash saving effect": "בניכוי השפעת החיסכון הלא-מזומנית נטו",
        "Net cash generation before saving allocations": "יצירת מזומן נטו לפני הקצאת חיסכון",
        "Actual funded cash savings": "חיסכון ממומן בפועל במזומן", "Net cash surplus / deficit": "עודף / גירעון מזומנים נטו",
        "Real-estate investment": "השקעה בנדל״ן", "Net home purchase": "רכישת דירה נטו",
        "Other property purchase": "רכישת נדל״ן אחר", "Home capital improvements": "השבחת דירה",
        "Other saving and debt flows": "חיסכון ותנועות חוב אחרות", "Other debt net repayment": "פירעון נטו של חוב אחר",
        "Apartment debt movement": "תנועת חוב בגין דירה",
        "Real estate and housing financing": "נדל״ן ומימון דיור",
        "Net household equity invested in property": "השקעת הון עצמי נטו של משק הבית בנדל״ן",
        "Gross property acquisition and improvements": "רכישת נדל״ן והשבחות ברוטו",
        "Housing debt financing / repayment": "מימון / פירעון חוב לדיור",
        "Financial-asset saving": "חיסכון בנכסים פיננסיים",
        "Other asset and lending flows": "תנועות נכסים והלוואות אחרות",
        "Total identified saving allocation": "סך הקצאת החיסכון המזוהה",
        "Pension share of identified saving": "חלק הפנסיה בחיסכון המזוהה",
        "Other housing-loan repayment": "פירעון הלוואות דיור אחרות", "Net household-asset sales": "מכירת נכסי משק בית נטו",
        "Household loans extended": "הלוואות שנתן משק הבית", "Other contractual financial saving": "חיסכון פיננסי חוזי אחר",
        "Cash flow before funded saving": "תזרים לפני חיסכון ממומן", "Funded cash saving": "חיסכון ממומן במזומן",
        "Free cash flow": "תזרים מזומנים פנוי", "Economic income (including imputed)": "הכנסה כלכלית (כולל זקיפות)",
        "Operating activities": "פעילות שוטפת", "Investing activities": "פעילות השקעה", "Financing activities": "פעילות מימון",
        "Operating cash receipts (excluding pension withdrawals)": "תקבולים מפעילות שוטפת (ללא משיכות פנסיה)",
        "Operating cash expenses": "הוצאות מזומן מפעילות שוטפת", "Cash consumption": "צריכה במזומן",
        "Private transfers paid": "העברות פרטיות ששולמו", "Cash flow from operating activities": "תזרים מזומנים מפעילות שוטפת",
        "Property acquisitions and improvements": "רכישת נדל״ן והשבחות", "Other financial investment": "השקעה פיננסית אחרת",
        "Other investing cash flow": "תזרים השקעה אחר",
        "Proceeds from household-asset sales": "תקבולים ממכירת נכסי משק הבית", "Loans extended by household": "הלוואות שנתן משק הבית",
        "Cash flow from investing activities": "תזרים מזומנים מפעילות השקעה", "Pension withdrawals": "משיכות פנסיה",
        "Pension and long-term-saving contributions": "הפקדות לפנסיה ולחיסכון ארוך טווח",
        "Apartment debt proceeds / repayment": "קבלת / פירעון חוב בגין דירה",
        "Other housing-loan payment": "תשלום הלוואת דיור אחרת", "Cash flow from financing activities": "תזרים מזומנים מפעילות מימון",
        "Net increase / decrease in unallocated cash": "גידול / קיטון נטו במזומן לא מוקצה",
        "Economic saving from P&L": "חיסכון כלכלי מדוח רווח והפסד",
        "Cash-income adjustments": "התאמות הכנסה למזומן", "Remove non-cash and valuation income": "נטרול הכנסה לא-מזומנית והתאמות שווי",
        "Reclassify pension withdrawals to investing": "סיווג מחדש של משיכות פנסיה לפעילות השקעה",
        "Cash-expense adjustments": "התאמות הוצאות למזומן", "Remove non-cash expense differences": "נטרול פערי הוצאה לא-מזומניים",
        "Pension and long-term savings": "פנסיה וחיסכון ארוך טווח",
        "Pension investing": "השקעה בפנסיה", "Other financial-asset saving": "חיסכון בנכסים פיננסיים אחרים",
        "Other asset investing": "השקעה בנכסים אחרים", "Real-estate investing": "השקעה בנדל״ן",
        "Employee pension contribution": "הפקדת עובד לפנסיה", "Modeled employer pension contribution": "הפקדת מעסיק לפנסיה (ממודלת)",
        "Economic labor income": "הכנסה כלכלית מעבודה", "Cash labor income": "הכנסה מעבודה במזומן",
        "Pension saving": "חיסכון פנסיוני", "Real-estate saving": "חיסכון בנדל״ן",
        "Real-estate debt financing / repayment": "מימון / פירעון חוב נדל״ן",
        "Other real-estate debt proceeds / repayment": "תקבולים / פירעון של חובות נדל״ן אחרים",
        "Other saving and lending": "חיסכון והלוואות אחרות",
        "Donations and community financing": "תרומות ומימון קהילתי",
        "Saving reconciliation and allocation": "התאמת החיסכון והקצאתו",
        "Economic / accrual saving (income less consumption)": "חיסכון כלכלי / צבירה (הכנסה פחות צריכה)",
        "Cash saving before funded allocations": "חיסכון מזומן לפני הקצאות ממומנות",
        "Non-cash accrual and valuation bridge": "גשר צבירה ושערוך לא-מזומני",
        "Change in modeled pension wealth ◇": "שינוי בעושר הפנסיוני הממודל ◇",
        "Modeled employer pension contribution ◇": "הפקדת מעסיק לפנסיה (ממודלת) ◇",
        "Modeled pension asset return ◇": "תשואה ממודלת על נכסי פנסיה ◇",
        "Property equity allocation": "הקצאת הון עצמי לנדל״ן", "Real-estate-related saving": "חיסכון הקשור לנדל״ן",
        "Scheduled mortgage principal repaid (modeled)": "פירעון קרן משכנתה שוטף (ממודל)",
        "Net cash equity in property transactions": "הון עצמי מזומן נטו בעסקאות נדל״ן",
        "Gross property acquisitions and improvements": "רכישות והשבחות נדל״ן ברוטו",
        "New apartment debt proceeds / repayment": "תקבולי חוב חדש לדירה / פירעון",
        "Total identified wealth-building allocation": "סך ההקצאה המזוהה לצבירת עושר",
        "Unallocated saving surplus / deficit": "עודף / גירעון חיסכון לא מוקצה",
        "Debt and community memoranda — excluded from identified saving": "תזכירי חוב וקהילה — אינם נכללים בחיסכון המזוהה",
        "Other housing-loan payment (principal + interest)": "תשלום הלוואת דיור אחרת (קרן וריבית)",
        "Unclassified debt movement": "תנועת חוב לא מסווגת", "Unclassified real-estate-related debt movement": "תנועת חוב לא מסווגת הקשורה לנדל״ן",
        "Unclassified debt proceeds / repayment": "תקבולי / פירעון חוב לא מסווג",
        "Donations / assumed community saving": "תרומות / חיסכון קהילתי משוער",
        "Economic / accrual saving rate": "שיעור חיסכון כלכלי / צבירה",        "Economic expenses (including imputed)": "הוצאות כלכליות (כולל זקיפות)", "Economic saving": "חיסכון כלכלי",
        "Total economic income": "סך ההכנסה הכלכלית", "Total expenses": "סך ההוצאות",
        "Fixed / semi-fixed household costs": "עלויות משק בית קבועות / קבועות למחצה",
        "Transportation and durable goods": "תחבורה ומוצרים בני-קיימא", "Transportation excluding car purchases": "תחבורה ללא רכישת רכב",
        "Other durable goods": "מוצרים בני-קיימא אחרים", "Kids / semi-kids expenses": "הוצאות ילדים / תלויות ילדים",
        "Food at home": "מזון בבית", "Education and childcare": "חינוך וטיפול בילדים",
        "Household disposable consumption": "צריכה פנויה של משק הבית", "Restaurants / food out": "מסעדות ואוכל מחוץ לבית",
        "Travel and entertainment": "נסיעות, חופשות ופנאי", "Other expenses": "הוצאות אחרות",
        "Total funded saving": "סך החיסכון הממומן", "Pension share of funded saving": "חלק הפנסיה בחיסכון הממומן", "Surplus / deficit saving": "עודף / גירעון חיסכון",
        "Government transfers and support": "קצבאות ממשלתיות והעברות",
        "Pension economic income": "הכנסה כלכלית מפנסיה", "Pension cash receipts": "תקבולי פנסיה במזומן",
        "Pension asset return (4%)": "תשואה על נכסי פנסיה (4%)", "Pension surplus / drawdown adjustment": "התאמת עודף / משיכה מפנסיה",
        "Net imputed owner-housing income": "הכנסה זקופה נטו מדיור בבעלות",
        "Reported rental / property income": "הכנסה מדווחת משכר דירה / נכס",
        "Modeled financial-asset income": "הכנסה ממודלת מנכסים פיננסיים",
        "Deposits and savings return (1%)": "תשואה על פיקדונות וחסכונות (1%)",
        "Funds, equities and securities return (4%)": "תשואה על קרנות, מניות וניירות ערך (4%)",
        "Other ungrouped income": "הכנסה אחרת שלא סווגה", "Saving": "חיסכון",
        "Age group": "קבוצת גיל", "Religious group": "קבוצה דתית", "Socioeconomic cluster": "אשכול חברתי-כלכלי",
        "Breakdown": "חתך", "Other funded deposits": "הפקדות לחסכונות ממומנים אחרים",
        "Fees and professional services": "עמלות ושירותים מקצועיים", "Tobacco and smoking products": "טבק ומוצרי עישון", "Donations": "תרומות", "Community / Gemach saving": "חיסכון קהילתי / גמ״ח", "Donations & community saving": "תרומות וחיסכון קהילתי",
        "General personal care and miscellaneous": "טיפוח אישי כללי ושונות", "Child personal care and baby products": "טיפוח ילדים ומוצרי תינוקות",
        "Pension balance change": "שינוי ביתרת הפנסיה", "Pension cash withdrawals": "משיכות פנסיה במזומן",
        "Total net-worth saving allocation": "סך הקצאת החיסכון לשווי הנקי",
        "Pension share of net-worth saving": "חלק הפנסיה בחיסכון לשווי הנקי",
        "Clothing & footwear": "הלבשה והנעלה", "Personal care & miscellaneous": "טיפוח אישי ושונות",
        "Other / residual consumption": "צריכה אחרת / שיורית",
        "Deposits and savings ◇": "פיקדונות וחסכונות ◇",
        "Funds, equities and securities ◇": "קרנות, מניות וניירות ערך ◇",
        "Primary residence ◇": "דירת מגורים ◇", "Additional properties ◇": "נכסים נוספים ◇", "Land and other real estate ◇": "קרקע ונדל״ן אחר ◇",
        # Chart valueLabels
        "Mean net worth": "שווי נטו ממוצע", "Capital-income share": "שיעור הכנסות ההון",
        # Ratios table headers
        "Group": "קבוצה", "Leverage ratio": "יחס מינוף", "Equity ratio": "יחס הון עצמי", "Debt / income": "חוב להכנסה",
        # Manifest table headers
        "File": "קובץ", "Description": "תיאור",
        # --- Cross-tab explorer: metric labels ---
        "Net worth": "שווי נטו", "Total assets": "סך הנכסים", "Total real estate": "סך הנדל״ן",
        "Primary residence": "דירת מגורים", "Total financial assets": "סך הנכסים הפיננסיים",
        "Deposits & savings": "פיקדונות וחיסכון", "Investment portfolio": "תיק השקעות",
        "Total liabilities": "סך ההתחייבויות", "Consumer loans": "הלוואות צרכניות",
        "Annual income": "הכנסה שנתית",
        # --- Cross-tab explorer: breakdown labels ---
        "Social group": "קבוצה חברתית", "Income decile": "עשירון הכנסה",
        "Marital status": "מצב משפחתי", "Children": "ילדים",
        "Industry (sector of employment)": "ענף כלכלי (תחום העסקה)",
        # --- Cross-tab explorer: category labels not already translated above ---
        "Not employed / other": "לא מועסק / אחר",
        "Married": "נשוי/אה", "Single": "רווק/ה", "Divorced/Separated": "גרוש/ה או פרוד/ה",
        "Widowed": "אלמן/ה", "No children": "ללא ילדים", "With children": "עם ילדים",
        "Agriculture, forestry & fishing": "חקלאות, ייעור ודיג",
        "Manufacturing & utilities": "תעשייה ותשתיות", "Construction": "בינוי",
        "Wholesale & retail trade": "מסחר סיטונאי וקמעונאי", "Transport & storage": "תחבורה ואחסנה",
        "Hospitality & food service": "אירוח ושירותי מזון", "Information & communication": "מידע ותקשורת",
        "Finance & insurance": "פיננסים וביטוח", "Professional & business services": "שירותים מקצועיים ועסקיים",
        "Public administration & defence": "מנהל ציבורי וביטחון", "Education & teaching": "חינוך והוראה",
        "Health & social work": "בריאות ורווחה", "Arts & other services": "אמנות ושירותים אחרים",
        # Manifest per-file descriptions (by filename, applied separately -- see MANIFEST_DESC)
    },
}

# Manifest per-file description overrides. English uses the CSV's own
# "description" column verbatim (empty override dict); Hebrew overrides with
# MANIFEST_DESC_HE from the original build_dashboard_he.py.
MANIFEST_DESC = {
    "en": {},
    "he": {
        "fig_national_balance_sheet.csv": "מאזן ארצי, גל 10 (2023)",
        "fig_networth_by_group.csv": "אגרגט שווי נטו ומאזן לפי קבוצה חברתית",
        "fig_composition_by_group.csv": "חלקי הרכב הנכסים לפי קבוצה",
        "fig_networth_by_dimension.csv": "שווי נטו לפי חתכים חברתיים-דמוגרפיים",
        "fig_networth_distribution.csv": "אחוזונים, ממוצע, חלק העשירון העליון ומקדם ג׳יני",
        "fig_income_wealth.csv": "הקשר בין עשירון הכנסה לעושר",
        "fig_trajectory.csv": "סדרת זמן על פני 10 גלים, כלל האוכלוסייה והקבוצות",
        "fig_participation.csv": "שיעורי החזקה לפי קבוצה",
        "fig_ratios_by_group.csv": "מינוף / הון עצמי / חוב-להכנסה לפי קבוצה",
        "fig_income_composition_by_group.csv": "הרכב ההכנסה לפי מקור ולפי קבוצה",
        "fig_metric_by_breakdown.csv": "מגלה חתכים: ממוצע כל מדד לפי כל חתך (גל 10)",
        "fig_metric_matrix.csv": "מטריצה דו-ממדית: ממוצע כל מדד בכל תא (חתך א׳ × חתך ב׳), גל 10",
        "fig_household_pnl_by_age.csv": "דו״ח חודשי של הכנסות, הוצאות, חיסכון ועושר לפי קבוצת גיל",
        "fig_household_expenses_by_age.csv": "פירוט הוצאות הורים לפי קטגוריה, קבוצת גיל וקבוצה דמוגרפית",
        "fig_meta.json": "מטא-נתונים: מתודולוגיה, תוויות ועוגני כיול",
    },
}

# The four KPI-tile label/note strings (Section 1). These are literal text
# baked directly into the page's JS in both original builders (not routed
# through TR), so they get their own small per-language JSON blob.
UI = {
    "en": {
        "kpi1_label": "Mean net worth, 2023",
        "kpi1_note": "All households, weighted. Source: fig_networth_distribution.csv (scope=All).",
        "kpi2_label": "Median net worth, 2023",
        "kpi2_note": "Typical household — half are worth more, half less.",
        "kpi3_label": "Real estate share of assets",
        "kpi3_note": "Total Real Estate ÷ Total Assets, national balance sheet.",
        "kpi4_label": "Gini coefficient, net worth",
        "kpi4_note": "0 = equal, 1 = maximal concentration. See Methodology caveat.",
        "exp_title_tpl": "Average {metric} by {breakdown}, 2023 (₪)",
        "exp_n_label": "Sample size (n)",
        "mx_title_tpl": "Average {metric} by {dimA} × {dimB}, 2023 (₪)",
        "mx_legend_low": "Lower",
        "mx_legend_high": "Higher",
        "mx_tiny_note": "cells with fewer than 30 households are greyed (value hidden; n on hover)",
    },
    "he": {
        "kpi1_label": "שווי נטו ממוצע, 2023",
        "kpi1_note": "כלל משקי הבית, משוקלל.",
        "kpi2_label": "שווי נטו חציוני, 2023",
        "kpi2_note": "משק הבית האופייני — למחצית יש יותר, למחצית פחות.",
        "kpi3_label": "שיעור הנדל״ן מסך הנכסים",
        "kpi3_note": "סך הנדל״ן ÷ סך הנכסים, מאזן ארצי.",
        "kpi4_label": "מקדם ג׳יני, שווי נטו",
        "kpi4_note": "0 = שוויון, 1 = ריכוזיות מלאה. ראו הסתייגות במתודולוגיה.",
        "exp_title_tpl": "{metric} ממוצע לפי {breakdown}, 2023 (₪)",
        "exp_n_label": "גודל מדגם (n)",
        "mx_title_tpl": "{metric} ממוצע לפי {dimA} × {dimB}, 2023 (₪)",
        "mx_legend_low": "נמוך",
        "mx_legend_high": "גבוה",
        "mx_tiny_note": "תאים עם פחות מ-30 משקי בית מסומנים באפור (הערך מוסתר; n בריחוף)",
    },
}

# The chart-drawing functions (chartHBarGrouped [dead], chartStacked100,
# chartStackedAbs, chartDonut, chartHBarSingle, chartBoxPlot)
# are ONE shared JS block, identical text for both languages, embedded into
# HTML_TEMPLATE like the rest of the shared script. RTL mirroring is threaded
# purely through the runtime `const RTL = true|false` flag (written from the
# language config): each function computes its geometry in plain LTR
# coordinates and, when RTL is true, mirrors x-coordinates with MX(x)=width-x,
# widens the label gutter (leftPad), and adds direction:'rtl' / flipped
# text-anchor attributes. When RTL is false NONE of the MX() results are
# applied -- the literal string `MX(` is present in both outputs' source but is
# only ever *used* on the Hebrew (RTL=true) path, so the English charts render
# with the exact original LTR geometry (leftPad 175/160/etc, legendW 190) and
# the Hebrew charts with the exact original RTL geometry (leftPad 210/170/etc,
# legendW 250). Both geometries are preserved verbatim; the branch just selects
# which set of numbers/anchors a given build uses at draw time.
CHART_FUNCS = r"""// ---------------------------------------------------------------------------
// Chart: horizontal grouped bars (e.g. mean+median net worth by group)
// NOTE: not called by any render function below in either language build --
// dead code carried over unchanged from both original builders.
// ---------------------------------------------------------------------------
function chartHBarGrouped(containerId, rows, labelKey, series, opts){
  opts = opts||{};
  const container = document.getElementById(containerId);
  const rowH = 54, barH = 17, gap = 4, leftPad = opts.leftPad || (RTL?210:175), rightPad = 85, topPad = 8;
  const width = 860;
  const height = rows.length*rowH + topPad + 6;
  const maxVal = Math.max(...rows.flatMap(r=>series.map(s=>num(r[s.key])||0))) * 1.08;
  const scaleW = width - leftPad - rightPad;
  const x = v => (v/maxVal)*scaleW;
  // RTL mirror: the LTR geometry above is computed as usual (leftPad = gutter
  // size, x() = bar length from that gutter); MX() flips every x-coordinate
  // across the viewBox so the label gutter lands on the right and bars grow
  // leftward from it, without touching any of the length/scale math. When
  // RTL is false, MX() is never applied (plain leftPad-based coords are used).
  const MX = xv => width - xv;

  if(opts.legend!==false){
    container.insertAdjacentHTML('beforeend', legendHtml(series.map(s=>({label:tr(s.label),color:s.color}))));
  }
  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  rows.forEach((r,i)=>{
    const cy = topPad + i*rowH;
    if(RTL) svgEl.appendChild(svCatLabelWrap(MX(leftPad-10), cy+barH+2, tr(r[labelKey]), leftPad-14, true));
    else svgEl.appendChild(svCatLabelWrap(leftPad-10, cy+barH+2, tr(r[labelKey]), leftPad-14, false));
    series.forEach((s,si)=>{
      const v = num(r[s.key])||0;
      const by = cy + si*(barH+gap);
      const w = Math.max(x(v),1.5);
      const rect = sv('rect',{class:'mark', x: RTL?MX(leftPad+w):leftPad, y:by, width:w, height:barH, rx:4, fill:s.color});
      svgEl.appendChild(rect);
      const lbl = fmtNISCompact(v);
      if(RTL) svgEl.appendChild(svText(MX(leftPad+w+6), by+barH-3.5, lbl, 'val-label', {'text-anchor':'start'}));
      else svgEl.appendChild(svText(leftPad+w+6, by+barH-3.5, lbl, 'val-label'));
      rect.addEventListener('pointermove', e=>showTooltip(e, tr(r[labelKey]), series.map(s2=>({k:tr(s2.label), v:fmtNIS(r[s2.key]), color:s2.color}))));
      rect.addEventListener('pointerleave', hideTooltip);
    });
  });
  container.appendChild(svgEl);
}

// ---------------------------------------------------------------------------
// Chart: 100%-stacked horizontal bars (asset composition)
// ---------------------------------------------------------------------------
function chartStacked100(containerId, groups, segmentGetter, segments, opts){
  // groups: array of group labels (already in desired display order)
  // segmentGetter(groupLabel, segKey) -> share (0..1)
  opts = opts||{};
  const container = document.getElementById(containerId);
  const rowH = MOBILE?120:40, barH = MOBILE?64:22, leftPad = opts.leftPad || (RTL?(MOBILE?155:210):(MOBILE?150:175)), rightPad = 14, topPad = 6;
  const width = MOBILE?380:860;
  const height = groups.length*rowH + topPad + 6;
  const scaleW = width - leftPad - rightPad;
  const gapPx = 2;
  const MX = xv => width - xv; // RTL mirror

  container.insertAdjacentHTML('beforeend', legendHtml(segments.map(s=>({label:tr(s.label),color:s.color}))));
  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  groups.forEach((g,i)=>{
    const cy = topPad + i*rowH;
    if(RTL) svgEl.appendChild(svCatLabelWrap(MX(leftPad-10), cy+barH-5, tr(g), leftPad-14, true));
    else svgEl.appendChild(svCatLabelWrap(leftPad-10, cy+barH-5, tr(g), leftPad-14, false));
    let cx = leftPad;
    segments.forEach((s,si)=>{
      const share = segmentGetter(g, s.key) || 0;
      let w = share*scaleW;
      if(si>0) { cx += gapPx; w = Math.max(w-gapPx,0); }
      w = Math.max(w, share>0?1:0);
      const rect = sv('rect',{class:'mark', x: RTL?MX(cx+w):cx, y:cy, width:w, height:barH, rx: si===0||si===segments.length-1?4:0, fill:s.color});
      svgEl.appendChild(rect);
      if(share>=0.15){
        const tx = cx + w/2;
        const t = svText(RTL?MX(tx):tx, cy+barH/2+4, fmtPct(share,0), 'seg-label', {'text-anchor':'middle'});
        svgEl.appendChild(t);
      }
      rect.addEventListener('pointermove', e=>showTooltip(e, tr(g), [{k:tr(s.label), v:fmtPct(share), color:s.color}]));
      rect.addEventListener('pointerleave', hideTooltip);
      cx += w;
    });
  });
  container.appendChild(svgEl);
}

// ---------------------------------------------------------------------------
// Chart: absolute stacked horizontal bars -- segments sum to each row's total,
// shared scale so bar LENGTH is comparable across rows (e.g. monthly income by
// source, where the whole bar = total household income).
// ---------------------------------------------------------------------------
function chartStackedAbs(containerId, rows, labelKey, segments, opts){
  opts = opts||{};
  const container = document.getElementById(containerId);
  const rowH = MOBILE?120:46, barH = MOBILE?64:24, leftPad = opts.leftPad || (RTL?(MOBILE?155:210):(MOBILE?150:175)), rightPad = 88, topPad = 6;
  const width = MOBILE?380:860;
  const height = rows.length*rowH + topPad + 6;
  const scaleW = width - leftPad - rightPad;
  const totals = rows.map(r=>segments.reduce((a,s)=>a+(num(r[s.key])||0),0));
  const maxTotal = Math.max(...totals, 1)*1.02;
  const x = v => (v/maxTotal)*scaleW;
  const MX = xv => width - xv; // RTL mirror

  container.insertAdjacentHTML('beforeend', legendHtml(segments.map(s=>({label:tr(s.label),color:s.color}))));
  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  rows.forEach((r,i)=>{
    const cy = topPad + i*rowH + (rowH-barH)/2 - 3;
    if(RTL) svgEl.appendChild(svCatLabelWrap(MX(leftPad-10), cy+barH-7, tr(r[labelKey]), leftPad-14, true));
    else svgEl.appendChild(svCatLabelWrap(leftPad-10, cy+barH-7, tr(r[labelKey]), leftPad-14, false));
    const total = totals[i];
    let cx = leftPad;
    segments.forEach((s,si)=>{
      const v = num(r[s.key])||0;
      const w = x(v);
      const ww = Math.max(w, v>0?1:0);
      const rect = sv('rect',{class:'mark', x: RTL?MX(cx+ww):cx, y:cy, width:ww, height:barH,
        rx: si===0||si===segments.length-1?4:0, fill:s.color});
      svgEl.appendChild(rect);
      if(w>=40){
        svgEl.appendChild(svText(RTL?MX(cx+w/2):cx+w/2, cy+barH/2+3.5, fmtNISCompact(v), 'seg-label', {'text-anchor':'middle'}));
      }
      rect.addEventListener('pointermove', e=>showTooltip(e, tr(r[labelKey]),
        segments.map(s2=>({k:tr(s2.label), v:fmtNIS(num(r[s2.key])||0), color:s2.color}))
                .concat([{k:tr('Total / month'), v:fmtNIS(total)}])));
      rect.addEventListener('pointerleave', hideTooltip);
      cx += w;
    });
    if(RTL) svgEl.appendChild(svText(MX(cx+6), cy+barH-7, fmtNISCompact(total), 'val-label', {'text-anchor':'start'}));
    else svgEl.appendChild(svText(cx+6, cy+barH-7, fmtNISCompact(total), 'val-label'));
  });
  container.appendChild(svgEl);
}

// ---------------------------------------------------------------------------
// Chart: donut
// ---------------------------------------------------------------------------
function chartDonut(containerId, segs){ // segs: [{label,value,color}]
  const container = document.getElementById(containerId);
  const total = segs.reduce((a,s)=>a+s.value,0);
  // English: legend on the right (legendW 190), donut on the left (cx=size/2).
  // Hebrew: legend sits on the LEFT (natural reading position on an RTL page)
  // and is widened (legendW 250) so the longest segment label
  // ("פיננסי (ללא פנסיה)") fits fully; the donut is shifted right by legendW.
  // MOBILE: legend moves BELOW the donut instead of beside it (a side legend
  // at mobile width would squeeze the donut down to almost nothing), and the
  // donut itself is drawn bigger since it's no longer sharing width with a
  // side legend -- this is the "taller" branch for what is otherwise a
  // roughly-square chart.
  const size = MOBILE ? 300 : 240, cy = size/2, rOuter = MOBILE?128:96, rInner = MOBILE?74:58;
  const legendW = MOBILE ? 0 : (RTL ? 250 : 190);
  const width = MOBILE ? 340 : (size + legendW);
  const cx = MOBILE ? width/2 : (RTL ? (legendW + size/2) : size/2);
  const legendRowH = MOBILE ? 70 : 0;
  const height = MOBILE ? (size + 34 + segs.length*legendRowH) : size;
  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  let angle = -Math.PI/2;
  segs.forEach(s=>{
    const frac = s.value/total;
    const a0 = angle, a1 = angle + frac*2*Math.PI;
    angle = a1;
    const gapA = 0.012;
    const ga0 = a0+gapA/2, ga1 = a1-gapA/2;
    const large = (ga1-ga0) > Math.PI ? 1 : 0;
    const p = (r,a)=>[cx+r*Math.cos(a), cy+r*Math.sin(a)];
    const [x0,y0]=p(rOuter,ga0),[x1,y1]=p(rOuter,ga1),[x2,y2]=p(rInner,ga1),[x3,y3]=p(rInner,ga0);
    const d = `M${x0},${y0} A${rOuter},${rOuter} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${rInner},${rInner} 0 ${large} 0 ${x3},${y3} Z`;
    const path = sv('path',{class:'mark', d, fill:s.color});
    svgEl.appendChild(path);
    path.addEventListener('pointermove', e=>showTooltip(e, tr(s.label), [{k:tr('Share of assets'), v:fmtPct(frac)},{k:tr('Mean value'), v:fmtNIS(s.value)}]));
    path.addEventListener('pointerleave', hideTooltip);
  });
  svgEl.appendChild(svText(cx, cy-5, tr('Total assets'), null, {'text-anchor':'middle','font-size':15}));
  const t2 = svText(cx, cy+(RTL?16:15), fmtNISCompact(total), null, {'text-anchor':'middle','font-size':21,'font-weight':600});
  t2.setAttribute('fill','var(--txt)');
  svgEl.appendChild(t2);
  // side legend with values (below the donut on MOBILE, beside it otherwise)
  let ly = MOBILE ? (size + 34) : (RTL ? 38 : 36);
  segs.forEach(s=>{
    if(MOBILE){
      // one row per segment, stacked below the donut; RTL still gets a
      // mirrored (right-anchored) layout so Hebrew text reads naturally.
      const swX = RTL ? (width - 40 - 16) : 40;
      const sw = sv('rect',{x:swX, y:ly-14, width:16, height:16, rx:2, fill:s.color});
      svgEl.appendChild(sw);
      const textX = RTL ? swX - 10 : swX + 24;
      // 'start' is direction-relative in SVG (not physical): under the RTL
      // direction this Hebrew (and unicode-bidi:plaintext-resolved) text
      // resolves to, 'start' anchors at the PHYSICAL RIGHT and grows left --
      // exactly the "swatch on the right, text growing toward center" layout
      // this branch wants. Under plain LTR direction 'start' anchors at the
      // physical left and grows right, which is what the else-branch wants.
      // So, perhaps counter-intuitively, both branches use the same token.
      svgEl.appendChild(svText(textX, ly, tr(s.label), 'cat-label', {'font-size':22,'text-anchor':'start'}));
      const t = svText(textX, ly+24, `${fmtPct(s.value/total)}  ·  ${fmtNIS(s.value)}`, null, {'font-size':18,'text-anchor':'start'});
      t.setAttribute('fill','var(--dim)');
      svgEl.appendChild(t);
      ly += legendRowH;
    } else if(RTL){
      // right-aligned toward the donut so the color swatch sits closest to the
      // chart it labels; full strings never clipped because legendW (250)
      // comfortably exceeds the longest label's rendered width.
      const swX = legendW - 22;
      const sw = sv('rect',{x:swX, y:ly-12, width:13, height:13, rx:2, fill:s.color});
      svgEl.appendChild(sw);
      svgEl.appendChild(svText(swX-9, ly, tr(s.label), 'cat-label', {'font-size':16,'text-anchor':'start'}));
      const t = svText(swX-9, ly+18, `${fmtPct(s.value/total)}  ·  ${fmtNIS(s.value)}`, null, {'font-size':15,'text-anchor':'start'});
      t.setAttribute('fill','var(--dim)');
      svgEl.appendChild(t);
      ly += 56;
    } else {
      const sw = sv('rect',{x:size+8, y:ly-11, width:13, height:13, rx:2, fill:s.color});
      svgEl.appendChild(sw);
      svgEl.appendChild(svText(size+27, ly, tr(s.label), 'cat-label', {'font-size':16}));
      const t = svText(size+27, ly+16, `${fmtPct(s.value/total)}  ·  ${fmtNIS(s.value)}`, null, {'font-size':15});
      t.setAttribute('fill','var(--dim)');
      svgEl.appendChild(t);
      ly += 52;
    }
  });
  container.appendChild(svgEl);
}

// ---------------------------------------------------------------------------
// Chart: single-series horizontal bar (sortable, optional sequential ramp)
// ---------------------------------------------------------------------------
function chartHBarSingle(containerId, rows, labelKey, valueKey, opts){
  opts = opts||{};
  const container = document.getElementById(containerId);
  const rowH = opts.rowH || (MOBILE?100:34), barH = opts.barH || (MOBILE?56:18),
        leftPad = opts.leftPad || (RTL?(MOBILE?155:170):(MOBILE?150:160)), rightPad = 70, topPad = 6;
  const width = opts.width || (MOBILE?380:860);
  const vals = rows.map(r=>num(r[valueKey])||0);
  const minV = Math.min(0,...vals), maxV = Math.max(...vals)*1.12;
  const height = rows.length*rowH + topPad + 6;
  const scaleW = width - leftPad - rightPad;
  const range = (maxV-minV)||1;
  const x0 = leftPad + (0-minV)/range*scaleW;
  const xOf = v => leftPad + (v-minV)/range*scaleW;
  const MX = xv => width - xv; // RTL mirror

  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  if(minV<0){
    const ax = RTL ? MX(x0) : x0;
    svgEl.appendChild(sv('line',{class:'axis-line', x1:ax, x2:ax, y1:topPad-2, y2:height-2}));
  }
  rows.forEach((r,i)=>{
    const cy = topPad + i*rowH;
    const v = num(r[valueKey])||0;
    const color = opts.colorFn ? opts.colorFn(r) : (opts.color||'#2a78d6');
    if(RTL) svgEl.appendChild(svCatLabelWrap(MX(leftPad-10), cy+barH-4, tr(r[labelKey]), leftPad-14, true));
    else svgEl.appendChild(svCatLabelWrap(leftPad-10, cy+barH-4, tr(r[labelKey]), leftPad-14, false));
    const bx = Math.min(x0, xOf(v)), bw = Math.max(Math.abs(xOf(v)-x0),1.5);
    const rect = sv('rect',{class:'mark', x: RTL?MX(bx+bw):bx, y:cy, width:bw, height:barH, rx:4, fill:color});
    svgEl.appendChild(rect);
    const lbl = opts.fmt ? opts.fmt(v) : fmtNISCompact(v);
    if(RTL){
      const labelX = v>=0 ? MX(xOf(v))-6 : MX(xOf(v))+6;
      // anchor 'start' (ambient direction:rtl) pins the right edge at labelX and
      // extends left, away from a positive (leftward-growing) bar; 'end' does
      // the mirror-image for a negative bar that grows rightward from x0.
      const t = svText(labelX, cy+barH-3.5, lbl, 'val-label', {'text-anchor': v>=0?'start':'end'});
      svgEl.appendChild(t);
    } else {
      const labelX = v>=0 ? xOf(v)+6 : xOf(v)-6;
      const t = svText(labelX, cy+barH-3.5, lbl, 'val-label', {'text-anchor': v>=0?'start':'end'});
      svgEl.appendChild(t);
    }
    // opts.tooltipRows(r) -> extra [{k,v}] rows appended after the value row
    // (used by the cross-tab explorer to show n=). Existing callers pass nothing,
    // so `extra` is [] and their tooltips are unchanged.
    const extra = opts.tooltipRows ? opts.tooltipRows(r) : [];
    rect.addEventListener('pointermove', e=>showTooltip(e, tr(r[labelKey]), [{k:tr(opts.valueLabel||'Value'), v: opts.fmt?opts.fmt(v):fmtNIS(v), color}].concat(extra)));
    rect.addEventListener('pointerleave', hideTooltip);
  });
  container.appendChild(svgEl);
}

// ---------------------------------------------------------------------------
// Chart: box-and-whisker (p10-p90 whiskers, p25-p75 box, p50 median line)
// One fill color for every box (the primary accent); "All" drawn in the
// neutral grey reference color, not colored by group -- the group label is
// already on the axis, so per-group color here would just be decoration.
// ---------------------------------------------------------------------------
function chartBoxPlot(containerId, rows, labelKey, opts){
  opts = opts||{};
  const container = document.getElementById(containerId);
  const rowH = MOBILE?100:46, leftPad = opts.leftPad || (RTL?(MOBILE?155:210):(MOBILE?150:175)), rightPad = 20, topPad = 8;
  const width = MOBILE?380:900;
  const boxH = MOBILE?32:16, capH = MOBILE?18:9;
  const vals = rows.flatMap(r=>[num(r.p10),num(r.p90)]).filter(v=>v!==null);
  const minV = Math.min(0,...vals)*1.05, maxV = Math.max(...vals)*1.08;
  const height = rows.length*rowH + topPad + 6;
  const scaleW = width - leftPad - rightPad;
  const range = (maxV-minV)||1;
  const xOf = v => leftPad + (v-minV)/range*scaleW;
  const x0 = xOf(0);
  const MX = xv => width - xv; // RTL mirror

  container.insertAdjacentHTML('beforeend',
    `<div class="legend">
       <span class="lk"><span class="sw" style="background:${PRIMARY_ACCENT}"></span>${tr('Box = p25–p75')}</span>
       <span class="lk"><span class="ln" style="border-color:var(--txt)"></span>${tr('Line = median (p50)')}</span>
       <span class="lk"><span class="ln" style="border-color:var(--dim)"></span>${tr('Whiskers = p10–p90')}</span>
     </div>`);

  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  const ax = RTL ? MX(x0) : x0;
  svgEl.appendChild(sv('line',{class:'axis-line', x1:ax, x2:ax, y1:topPad-2, y2:height-2}));
  rows.forEach((r,i)=>{
    const cy = topPad + i*rowH + 18;
    const isAll = r[labelKey]==='All';
    const boxColor = isAll ? GROUP_ALL_COLOR : PRIMARY_ACCENT;
    if(RTL) svgEl.appendChild(svCatLabelWrap(MX(leftPad-10), cy+4, tr(r[labelKey]), leftPad-14, true, {'font-style': isAll?'italic':'normal'}));
    else svgEl.appendChild(svCatLabelWrap(leftPad-10, cy+4, tr(r[labelKey]), leftPad-14, false, {'font-style': isAll?'italic':'normal'}));
    const p10=num(r.p10), p25=num(r.p25), p50=num(r.p50), p75=num(r.p75), p90=num(r.p90);
    // In RTL every plotted x is mirrored via MX(); in LTR the plain xOf() value
    // is used. P* below are the final (possibly mirrored) screen coordinates.
    const P10 = RTL?MX(xOf(p10)):xOf(p10), P90 = RTL?MX(xOf(p90)):xOf(p90);
    const P25 = RTL?MX(xOf(p25)):xOf(p25), P75 = RTL?MX(xOf(p75)):xOf(p75), P50 = RTL?MX(xOf(p50)):xOf(p50);

    // whisker: full p10-p90 line + end caps
    svgEl.appendChild(sv('line',{x1:P10, x2:P90, y1:cy, y2:cy, stroke:'var(--dim)', 'stroke-width':1.5}));
    [P10,P90].forEach(v=>{
      svgEl.appendChild(sv('line',{x1:v, x2:v, y1:cy-capH/2, y2:cy+capH/2, stroke:'var(--dim)', 'stroke-width':1.5}));
    });

    // box: p25-p75
    const bx = Math.min(P25,P75), bw = Math.max(Math.abs(P75-P25),2);
    const rect = sv('rect',{class:'mark', x:bx, y:cy-boxH/2, width:bw, height:boxH, rx:3,
      fill:boxColor, 'fill-opacity':isAll?0.45:0.6, stroke:boxColor, 'stroke-width':1.3});
    svgEl.appendChild(rect);

    // median line inside the box
    svgEl.appendChild(sv('line',{x1:P50, x2:P50, y1:cy-boxH/2, y2:cy+boxH/2, stroke:'var(--txt)', 'stroke-width':2}));

    // hover hit-area across the whole whisker span
    const hitX = Math.min(P10,P90)-4;
    const hit = sv('rect',{x:hitX, y:cy-capH/2-6, width:Math.abs(P90-P10)+8, height:capH+12, fill:'transparent', class:'mark'});
    svgEl.appendChild(hit);
    hit.addEventListener('pointermove', e=>showTooltip(e, tr(r[labelKey]), [
      {k:'p10', v:fmtNIS(p10)}, {k:'p25', v:fmtNIS(p25)}, {k:tr('p50 (median)'), v:fmtNIS(p50)},
      {k:'p75', v:fmtNIS(p75)}, {k:'p90', v:fmtNIS(p90)}, {k:tr('Mean'), v:fmtNIS(r.mean)},
    ]));
    hit.addEventListener('pointerleave', hideTooltip);
    if(p10<0){
      const t = RTL
        ? svText(P10+8, cy-capH/2-11, fmtNISCompact(p10), 'val-label', {'text-anchor':'start'})
        : svText(P10-8, cy-capH/2-8, fmtNISCompact(p10), 'val-label', {'text-anchor':'end'});
      svgEl.appendChild(t);
    }
  });
  container.appendChild(svgEl);
}"""


def read_csv_rows(name):
    path = os.path.join(FIGURES_DIR, name)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(lang):
    meta = LANG_META[lang]

    figures = {}
    for name in CSV_FIGURES:
        key = name[:-4]  # strip .csv
        figures[key] = read_csv_rows(name)
    for name in OPTIONAL_CSV_FIGURES:
        key = name[:-4]
        path = os.path.join(FIGURES_DIR, name)
        figures[key] = read_csv_rows(name) if os.path.exists(path) else []

    with open(os.path.join(FIGURES_DIR, "fig_meta.json"), encoding="utf-8") as f:
        meta_data = json.load(f)

    with open(os.path.join(FIGURES_DIR, "_manifest.csv"), newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    figures_json = json.dumps(figures, ensure_ascii=False)
    meta_json = json.dumps(meta_data, ensure_ascii=False)
    manifest_json = json.dumps(manifest, ensure_ascii=False)
    tr_json = json.dumps(TR[lang], ensure_ascii=False)
    manifest_desc_json = json.dumps(MANIFEST_DESC[lang], ensure_ascii=False)
    ui_json = json.dumps(UI[lang], ensure_ascii=False)

    replacements = {
        "__LANG__": meta["code"],
        "__DIR_ATTR__": meta["dir_attr"],
        "__RTL_CSS_EXTRA__": RTL_CSS_EXTRA[lang],
        "__RTL_JS__": "true" if meta["rtl"] else "false",
        "__HOME_LABEL__": "דף הבית" if lang == "he" else "Home",
        "__FIGURES_JSON__": figures_json,
        "__META_JSON__": meta_json,
        "__MANIFEST_JSON__": manifest_json,
        "__TR_JSON__": tr_json,
        "__MANIFEST_DESC_JSON__": manifest_desc_json,
        "__UI_JSON__": ui_json,
        "__CHART_FUNCS_JS__": CHART_FUNCS,
    }
    for key, val in STRINGS[lang].items():
        replacements["__" + key.upper() + "__"] = val

    html = HTML_TEMPLATE
    for token, val in replacements.items():
        html = html.replace(token, val)

    out_path = os.path.join(OUT_DIR, meta["out"])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    sz_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({sz_kb:,.0f} KB)")
    print(f"Embedded {len(figures)} figure tables + meta + manifest, "
          f"total rows = {sum(len(v) for v in figures.values())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "he"], default=None,
                         help="Build only this language. Default: build both.")
    args = parser.parse_args()
    if args.lang:
        build(args.lang)
    else:
        build("en")
        build("he")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="__LANG__"__DIR_ATTR__>
<head>
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-VE1ZYYW7TS"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','G-VE1ZYYW7TS');</script>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root{
  color-scheme: light;
  --bg:#f9f9f7; --panel:#fcfcfb; --panel2:#f5f5f2; --line:#e1e0d9;
  --border:rgba(11,11,11,0.10); --baseline:#c3c2b7;
  --txt:#0b0b0b; --muted:#52514e; --dim:#898781;
  --gold:#2a78d6; --gold-dim:#184f95;
  --amber:#a8631f; --amber-soft:#fbeedc;
  --pos:#1baf7a; --neg:#e34948;
  --g-haredi:#2a78d6; --g-dati:#1baf7a; --g-masdati:#eda100; --g-mastrad:#008300;
  --g-hiloni:#4a3aa7; --g-arab:#e34948; --g-other:#e87ba4; --g-jewna:#eb6834;
  --a-re:#2a78d6; --a-fin:#1baf7a; --a-pen:#eda100;
}
*{box-sizing:border-box;}
html,body{background:var(--bg);color:var(--txt);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:0;}
body{padding-bottom:60px;}
::selection{background:var(--gold);color:#ffffff;}
a{color:var(--gold);}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px;}

header.masthead{border-bottom:2px solid var(--gold);padding:22px 0 16px;position:relative;}
header.masthead h1{font-size:30px;font-weight:700;color:var(--txt);margin:0 0 4px;line-height:1.15;letter-spacing:-0.01em;}
header.masthead .sub{color:var(--muted);font-size:12.5px;letter-spacing:.3px;}
header.masthead .aiflag{color:var(--gold-dim);font-size:10.5px;margin-top:6px;letter-spacing:.5px;}

nav.topnav{position:sticky;top:0;z-index:50;background:rgba(249,249,247,.96);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line);}
nav.topnav .wrap{display:flex;gap:2px;overflow-x:auto;padding:0 24px;}
nav.topnav button{background:none;border:none;border-bottom:2px solid transparent;color:var(--dim);
  font-family:inherit;font-size:11.5px;letter-spacing:1.2px;text-transform:uppercase;padding:13px 14px;
  cursor:pointer;white-space:nowrap;transition:color .15s,border-color .15s;}
nav.topnav button:hover{color:var(--txt);}
nav.topnav button.active{color:var(--gold);border-bottom-color:var(--gold);}
.legacy-hidden{display:none!important;}

section.pagesec{padding:38px 0 14px;border-bottom:1px solid var(--line);}
section.pagesec:last-of-type{border-bottom:none;}
.sec-kicker{color:var(--gold-dim);font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;}
.sec-head{font-size:23px;font-weight:700;color:var(--txt);margin:0 0 6px;letter-spacing:-0.01em;}
.sec-intro{color:var(--muted);font-size:14.5px;line-height:1.7;max-width:820px;margin-bottom:22px;}

.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px;}
.kpi-tile{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px 18px;}
.kpi-tile .lab{font-size:11.5px;letter-spacing:1.1px;text-transform:uppercase;color:var(--dim);margin-bottom:8px;}
.kpi-tile .val{font-size:29px;font-weight:700;color:var(--txt);line-height:1;font-variant-numeric:tabular-nums;}
.kpi-tile .val .unit{font-size:15px;color:var(--gold);}
.kpi-tile .note{font-size:11.5px;color:var(--dim);margin-top:7px;line-height:1.5;}

.grid2{display:grid;grid-template-columns:1.15fr .85fr;gap:18px;align-items:start;}
.grid2b{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;}

.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px 20px 16px;margin-bottom:18px;}
.card h3{font-size:16px;font-weight:700;color:var(--txt);margin:0 0 2px;
  padding-left:11px;border-left:3px solid var(--gold);}
.card .csrc{font-size:11.5px;color:var(--dim);margin:6px 0 12px 14px;}
.chart-body{margin-top:6px;}

.legend{display:flex;flex-wrap:wrap;gap:5px 16px;margin:8px 0 10px 14px;font-size:13px;color:var(--muted);}
.legend .lk{display:inline-flex;align-items:center;gap:6px;}
.legend .sw{width:11px;height:11px;border-radius:2px;display:inline-block;flex-shrink:0;}
.legend .ln{width:14px;height:0;border-top:2px solid;display:inline-block;flex-shrink:0;}

svg.viz{width:100%;height:auto;display:block;overflow:visible;font-family:inherit;}
svg.viz text{fill:var(--muted);}
svg.viz .axis-line{stroke:var(--baseline);stroke-width:1;}
svg.viz .grid{stroke:var(--line);stroke-width:1;}
svg.viz .cat-label{fill:var(--muted);font-size:16px;}
svg.viz .val-label{fill:var(--txt);font-size:15px;font-weight:600;font-variant-numeric:tabular-nums;}
svg.viz .val-label.gold{fill:var(--gold);}
svg.viz .seg-label{fill:#ffffff;font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;}
svg.viz .axis-tick{fill:var(--dim);font-size:13px;}
svg.viz rect.mark{cursor:pointer;}
svg.viz rect.mark:hover{filter:brightness(1.08);}
svg.viz circle.mark:hover{filter:brightness(1.1);}

#tooltip{position:fixed;pointer-events:none;background:#ffffff;border:1px solid var(--border);
  border-radius:8px;padding:8px 11px;font-size:11.3px;color:var(--txt);z-index:1000;display:none;
  box-shadow:0 6px 18px rgba(11,11,11,.14);max-width:260px;line-height:1.6;}
#tooltip .tt-title{color:var(--gold);font-weight:700;margin-bottom:3px;}
#tooltip .tt-row{display:flex;justify-content:space-between;gap:14px;}
#tooltip .tt-row .k{color:var(--dim);}
#tooltip .tt-row .v{color:var(--txt);font-weight:600;}
#tooltip .tt-row .v.keyed::before{content:'';display:inline-block;width:9px;height:9px;border-radius:2px;
  margin-right:5px;vertical-align:middle;background:var(--sw,var(--gold));}

.selector-row{margin:0 0 14px 14px;display:flex;align-items:center;gap:10px;}
.selector-row label{font-size:11.5px;letter-spacing:1px;text-transform:uppercase;color:var(--gold-dim);}
.selector-row select{background:var(--panel);color:var(--txt);border:1px solid var(--baseline);border-radius:6px;
  padding:6px 10px;font-family:inherit;font-size:13px;cursor:pointer;}
.selector-row select:hover,.selector-row select:focus{border-color:var(--gold);outline:none;}

/* --- Two-way matrix heatmap ------------------------------------------------ */
.matrix-legend{display:flex;align-items:center;gap:9px;margin:2px 0 12px 14px;
  font-size:12px;color:var(--muted);flex-wrap:wrap;}
.matrix-legend .grad{width:150px;height:12px;border-radius:3px;border:1px solid var(--border);}
.matrix-legend .mx-note{color:var(--dim);font-size:11.5px;}
.matrix-scroll{position:relative;overflow-x:auto;-webkit-overflow-scrolling:touch;
  width:100%;min-width:0;max-width:100%;}
table.matrix{border-collapse:separate;border-spacing:0;font-variant-numeric:tabular-nums;
  unicode-bidi:plaintext;}
table.matrix th,table.matrix td{border:1px solid var(--line);padding:6px 9px;text-align:center;
  white-space:nowrap;font-size:13px;min-width:66px;}
table.matrix th{color:var(--dim);font-weight:700;font-size:11.5px;background:var(--panel);}
table.matrix thead th{position:sticky;top:0;z-index:2;}
table.matrix th.rowhead{position:sticky;left:0;z-index:3;background:var(--panel);
  text-align:right;color:var(--txt);font-weight:600;font-size:12px;}
table.matrix th.corner{position:sticky;left:0;top:0;z-index:4;background:var(--panel);}
table.matrix td.cell{color:#0b0b0b;font-weight:600;cursor:pointer;}
table.matrix td.cell:hover{outline:2px solid var(--gold);outline-offset:-2px;}
table.matrix td.cell.tiny{color:var(--dim);cursor:pointer;}
table.matrix td.cell.empty{cursor:default;}
[dir="rtl"] table.matrix th.rowhead,[dir="rtl"] table.matrix th.corner{left:auto;right:0;}
[dir="rtl"] table.matrix th.rowhead{text-align:left;}
[dir="rtl"] .matrix-legend{margin:2px 14px 12px 0;}

.table-scroll{position:relative;overflow-x:auto;-webkit-overflow-scrolling:touch;}
.table-scroll.is-scrollable::after{content:'';position:absolute;top:0;right:0;bottom:0;width:26px;
  pointer-events:none;background:linear-gradient(to right, transparent, var(--panel));}
[dir="rtl"] .table-scroll.is-scrollable::after{right:auto;left:0;
  background:linear-gradient(to left, transparent, var(--panel));}

table.stmt{width:100%;border-collapse:collapse;font-size:14px;margin-top:6px;}
table.stmt td{padding:5px 4px;border-bottom:1px solid var(--line);}
table.stmt td.amt{text-align:right;font-variant-numeric:tabular-nums;color:var(--txt);}
__RTL_CSS_EXTRA__
table.stmt tr.seg td{padding-top:14px;font-size:10.5px;letter-spacing:1.6px;text-transform:uppercase;
  color:var(--gold);font-weight:600;border-bottom:1px solid var(--gold-dim);padding-bottom:5px;}
table.stmt tr.subtotal td{border-top:1px solid var(--dim);font-weight:600;color:var(--txt);padding-top:7px;}
table.stmt tr.total td{border-top:2px solid var(--gold-dim);border-bottom:2px solid var(--gold-dim);
  font-size:14px;font-weight:600;color:var(--txt);padding:9px 4px;}
table.stmt tr.networth td:first-child{color:var(--txt);font-size:14px;font-weight:700;}
table.stmt tr.networth td.amt{color:var(--gold);font-size:20px;}
table.stmt .modeled-flag{color:var(--gold-dim);margin-left:4px;}

.small-multiples{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;}
.small-multiples .card{margin-bottom:0;}

.datatable{width:100%;border-collapse:collapse;font-size:13px;margin-top:4px;}
.datatable th{text-align:right;color:var(--dim);font-weight:700;font-size:11px;letter-spacing:.6px;
  text-transform:uppercase;border-bottom:1px solid var(--line);padding:6px 7px;}
.datatable th:first-child,.datatable td:first-child{text-align:left;}
.datatable td{text-align:right;padding:5px 7px;border-bottom:1px solid var(--line);color:var(--muted);
  font-variant-numeric:tabular-nums;}

table.pnl{width:max-content;min-width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;}
table.pnl th,table.pnl td{padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right;}
table.pnl thead th{position:sticky;top:0;background:var(--panel);z-index:2;color:var(--txt);font-weight:700;}
table.pnl th:first-child,table.pnl td:first-child{position:sticky;left:0;text-align:left;background:var(--panel);z-index:1;min-width:220px;}
table.pnl thead th:first-child{z-index:3;}
table.pnl tr.seg td{padding-top:15px;color:var(--gold);font-size:10.5px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;border-bottom:1px solid var(--gold-dim);}
table.pnl tr.subtotal td{font-weight:700;color:var(--txt);border-top:1px solid var(--baseline);}
table.pnl tr.keyline td{font-weight:700;color:var(--txt);border-top:2px solid var(--gold-dim);border-bottom:2px solid var(--gold-dim);}
table.pnl tr.keyline td:not(:first-child){color:var(--gold);}
table.pnl tr.percent td:not(:first-child){color:var(--gold-dim);font-weight:700;}
table.pnl tr.expandable td:first-child{cursor:pointer;color:var(--txt);font-weight:700;}
table.pnl tr.expandable td:first-child::before{content:'›';display:inline-block;width:16px;color:var(--gold);font-size:17px;line-height:10px;transition:transform .15s ease;}
table.pnl tr.expandable.open td:first-child::before{transform:rotate(90deg);}
table.pnl tr.detail td:first-child{padding-left:30px;color:var(--muted);}
table.pnl tr.detail-2 td:first-child{padding-left:48px;color:var(--muted);}
[dir="rtl"] table.pnl tr.detail td:first-child{padding-left:10px;padding-right:30px;}
[dir="rtl"] table.pnl tr.detail-2 td:first-child{padding-left:10px;padding-right:48px;}
[dir="rtl"] table.pnl tr.expandable td:first-child::before{transform:rotate(180deg);}
[dir="rtl"] table.pnl tr.expandable.open td:first-child::before{transform:rotate(90deg);}.pnl-empty{padding:30px 16px;text-align:center;color:var(--dim);font-size:14px;line-height:1.6;}
[dir="rtl"] table.pnl th:first-child,[dir="rtl"] table.pnl td:first-child{left:auto;right:0;text-align:right;}

.caveat{background:var(--amber-soft);border:1px solid var(--amber);border-radius:8px;padding:12px 16px;
  font-size:12px;color:#6b4210;line-height:1.6;margin:14px 0;}
.caveat b{color:var(--amber);}

.methodology p, .methodology li{font-size:14.5px;line-height:1.75;color:var(--muted);max-width:860px;}
.methodology h4{color:var(--gold);font-size:16.5px;font-weight:700;margin:22px 0 6px;}
.methodology code{background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:1px 6px;
  color:var(--gold-dim);font-size:13px;}
.methodology .stat-row{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0 4px;}
.methodology .stat-row div{background:var(--panel2);border:1px solid var(--line);border-radius:6px;
  padding:8px 14px;font-size:13px;}
.methodology .stat-row b{display:block;color:var(--txt);font-size:16px;}

footer.pagefoot{max-width:1180px;margin:30px auto 0;padding:18px 24px 0;border-top:1px solid var(--line);
  font-size:10.7px;color:var(--dim);line-height:1.7;}
footer.pagefoot b{color:var(--gold-dim);}

@media (max-width:900px){
  .kpi-row{grid-template-columns:repeat(2,1fr);}
  .grid2,.grid2b,.small-multiples{grid-template-columns:1fr;}
}
/* Mobile legibility pass -- bumps body/caption/table/KPI text to >=~15-16px
   and widens the inline-SVG chart label classes so they stay readable once
   the SVG's viewBox (fixed ~860-900 user units) is rendered into a ~340px-wide
   mobile container (rendered px ~= viewBox px * containerWidth/viewBoxWidth,
   so a ~2.3-2.4x class font-size bump here nets out to a similar *visual*
   size as the desktop render at typical phone widths). Desktop rules above are
   untouched -- everything below is additive and only active <=640px. */
@media (max-width:640px){
  body{font-size:16px;}
  .sec-intro{font-size:15.5px;}
  .card .csrc{font-size:13px;}
  table.stmt{font-size:15px;}
  .datatable{font-size:13.5px;}
  .datatable th{font-size:11.5px;}
  /* Compact, standard-orientation financial statements. At normal phone
     widths all five comparison columns fit; ~320px devices retain the
     existing horizontal-scroll fallback. */
  #sec-pnl,#sec-cash,#sec-wealth{margin-inline:-16px;padding-inline:6px;border-radius:6px;}
  #sec-pnl .table-scroll,#sec-cash .table-scroll,#sec-wealth .table-scroll{width:100%;}
  table.pnl{width:100%;min-width:320px;table-layout:fixed;font-size:10.5px;}
  table.pnl th,table.pnl td{padding:5px 2px;}
  table.pnl th:first-child,table.pnl td:first-child{width:38%;min-width:0;white-space:normal;overflow-wrap:break-word;line-height:1.15;}
  table.pnl th:not(:first-child),table.pnl td:not(:first-child){width:12.4%;white-space:nowrap;font-size:10.5px;}
  #household-wealth-age table.pnl td:not(:first-child){font-size:10px;}
  table.pnl thead th:not(:first-child){white-space:normal;overflow-wrap:break-word;line-height:1.05;font-size:9.5px;}
  table.pnl tr.expandable td:first-child::before{width:8px;font-size:13px;}
  table.pnl tr.detail td:first-child{padding-left:10px;}
  table.pnl tr.detail-2 td:first-child{padding-left:16px;}
  [dir="rtl"] table.pnl tr.detail td:first-child{padding-left:2px;padding-right:10px;}
  [dir="rtl"] table.pnl tr.detail-2 td:first-child{padding-left:2px;padding-right:16px;}
  .kpi-tile .lab{font-size:12px;}
  .kpi-tile .note{font-size:12.5px;}
  nav.topnav .wrap{gap:10px;}
  nav.topnav button{font-size:12.5px;padding:13px 10px;min-height:40px;}
  .selector-row{gap:12px;}
  .selector-row select{min-height:42px;padding:10px 12px;font-size:15px;}
  .methodology p,.methodology li{font-size:15.5px;}
  .legend{font-size:13.5px;}
  .chart-body{overflow-x:auto;min-height:260px;}
  /* Charts that were previously half-width inside .grid2/.grid2b/.small-multiples
     now render full-width at this breakpoint (those grids already collapse to
     1fr at <=900px above), so their viewBox-based SVGs (width:100%;height:auto)
     naturally render taller. The min-height above is just a floor so no chart
     (e.g. a small-multiples tile) ever renders too short to read. */
  /* SVG chart text lives in viewBox user-units, not screen px -- these
     class rules (not inline attrs) win the cascade over the inline
     font-size presentation attributes some charts (e.g. the donut) set,
     since presentation attributes sit below author CSS in specificity. */
  /* Font sizes below are in SVG viewBox *user units*, not screen px --
     rendered px = user-units * (containerWidthPx/viewBoxWidth). The chart JS
     below also switches each chart's viewBox width from ~860-900 (desktop)
     down to ~380 (mobile, MOBILE branch) to get a taller aspect ratio; that
     narrower viewBox means the same container width maps to a BIGGER scale
     factor, so these class font-sizes are scaled down from the original
     38/34/30/30 by roughly that same ratio (380/860 ~= 0.44) to keep the
     rendered (on-screen) text size roughly where it was, not larger. */
  /* Bumped again (17/15/13/13 -> 19/19/19/19): measured rendered (on-screen,
     post-viewBox-scale) sizes at 360-390px showed val-label/seg-label/axis-tick
     landing at ~9-11px -- under the ~11px legibility floor -- because the
     actual chart-container width (and therefore the viewBox-to-screen scale
     factor) is smaller than the ratio this comment block originally assumed,
     especially for the narrower nested small-multiples participation tiles
     (container ~228px, scale ~0.6). Uniform 19px nominal keeps every class
     >=11px actual even in that narrowest (0.6 scale) case. */
  svg.viz .cat-label{font-size:19px;}
  svg.viz .val-label{font-size:19px;}
  svg.viz .seg-label{font-size:19px;}
  svg.viz .axis-tick{font-size:19px;}

  /* --- One-chart-per-screen scroll experience (mobile only) ---------------
     `proximity` (not `mandatory`) is deliberate: several cards on this page
     are taller than one viewport (the multi-chart participation card, the
     income-composition legend+bars, a long Hebrew dimension chart with wide
     category labels), and `mandatory` would fight the user trying to read
     the middle of a tall card by yanking the scroll back to the nearest
     snap point on every gesture. `proximity` still gives the snap-to-next-
     chart feel on the common case (a card that fits ~one screen) without
     that trap. The unit is one CHART per screen: cards holding exactly one
     chart (or one table) become snap targets. Cards nested one level down
     inside the participation small-multiples grid are explicitly opted back
     out below so only the outer card snaps, not each of its 4 mini-charts.
     The two table cards (balance-sheet statement, ratios) DO snap (a table
     can be its own screen); the KPI tile row and the long-form methodology
     section (no .card wrapper there) are left out of the snap flow on
     purpose -- see the build_dashboard.py docstring/report for rationale. */
  html{scroll-snap-type:y proximity;}
  .pagesec > .card, .grid2 > .card, .grid2b > .card{
    scroll-snap-align:start;
    scroll-margin-top:50px; /* clears the sticky top nav so a snapped card's
                                title isn't hidden underneath it */
    min-height:100vh;   /* fallback for browsers without svh/dvh support */
    min-height:100svh;  /* small viewport height -- ignores mobile browser
                            chrome (address bar) show/hide so a snapped
                            chart doesn't get clipped by it */
    display:flex;
    flex-direction:column;
    justify-content:center; /* centers the chart vertically when the card is
                                shorter than one screen; cards taller than a
                                screen (see above) simply overflow normally */
  }
  .small-multiples .card{
    /* opt the 4 inner participation-metric mini-cards back OUT of the
       one-screen treatment above -- only the single outer card that wraps
       the whole small-multiples grid should be a snap target/full screen */
    scroll-snap-align:none;
    min-height:0;
    display:block;
  }
}
@media (max-width:560px){
  .kpi-row{grid-template-columns:1fr;}
  header.masthead h1{font-size:23px;}
}
</style>
</head>
<body>

<header class="masthead"><div class="wrap">
  <h1>__H1__</h1>
  <div class="sub">__SUB__</div>
  <div class="aiflag">__AIFLAG__</div>
</div></header>

<nav class="topnav"><div class="wrap">
  <button data-sec="pnl" class="active">__NAV_S1__</button>
  <button data-sec="cash">__NAV_S2__</button>
  <button data-sec="wealth">__NAV_S3__</button>
  <button data-sec="s6">__NAV_S6__</button>
</div></nav>

<div class="wrap">

<section class="pagesec legacy-hidden" id="sec-s1">
  <div class="sec-kicker">__S1_KICKER__</div>
  <h2 class="sec-head">__S1_HEAD__</h2>
  <p class="sec-intro">__S1_INTRO__</p>

  <div class="kpi-row" id="kpi-row"></div>

  <div class="grid2">
    <div class="card">
      <h3>__S1_CARD1_H3__</h3>
      <div class="csrc">__S1_CARD1_CSRC__</div>
      <div class="table-scroll"><table class="stmt" id="bs-statement"></table></div>
    </div>
    <div class="card">
      <h3>__S1_CARD2_H3__</h3>
      <div class="csrc">__S1_CARD2_CSRC__</div>
      <div class="chart-body" id="donut-national"></div>
    </div>
  </div>
</section>

<section class="pagesec legacy-hidden" id="sec-s2">
  <div class="sec-kicker">__S2_KICKER__</div>
  <h2 class="sec-head">__S2_HEAD__</h2>
  <p class="sec-intro">__S2_INTRO__</p>

  <div class="card">
    <h3 id="explorer-title">__S2_EXP_H3__</h3>
    <div class="csrc">__S2_EXP_CSRC__</div>
    <div class="selector-row">
      <label for="exp-metric">__S2_EXP_METRIC_LABEL__</label>
      <select id="exp-metric"></select>
      <label for="exp-breakdown">__S2_EXP_BREAKDOWN_LABEL__</label>
      <select id="exp-breakdown"></select>
    </div>
    <div class="chart-body" id="chart-explorer"></div>
  </div>

  <div class="card" id="matrix-card">
    <h3 id="matrix-title">__S2_MX_H3__</h3>
    <div class="csrc">__S2_MX_CSRC__</div>
    <div class="selector-row">
      <label for="mx-metric">__S2_MX_METRIC_LABEL__</label>
      <select id="mx-metric"></select>
      <label for="mx-dima">__S2_MX_DIMA_LABEL__</label>
      <select id="mx-dima"></select>
      <label for="mx-dimb">__S2_MX_DIMB_LABEL__</label>
      <select id="mx-dimb"></select>
    </div>
    <div id="matrix-legend" class="matrix-legend"></div>
    <div class="matrix-scroll"><div id="chart-matrix"></div></div>
  </div>

  <div class="card">
    <h3>__S2_CARD2_H3__</h3>
    <div class="csrc">__S2_CARD2_CSRC__</div>
    <div class="chart-body" id="chart-composition-group"></div>
  </div>

  <div class="card">
    <h3>__S2_CARD4_H3__</h3>
    <div class="csrc">__S2_CARD4_CSRC__</div>
    <div class="small-multiples" id="chart-participation"></div>
  </div>
</section>

<section class="pagesec legacy-hidden" id="sec-s3">
  <div class="sec-kicker">__S3_KICKER__</div>
  <h2 class="sec-head">__S3_HEAD__</h2>
  <p class="sec-intro">__S3_INTRO__</p>

  <div class="card">
    <h3>__S3_CARD1_H3__</h3>
    <div class="csrc">__S3_CARD1_CSRC__</div>
    <div class="chart-body" id="chart-percentile-spread"></div>
  </div>
</section>

<section class="pagesec legacy-hidden" id="sec-s4">
  <div class="sec-kicker">__S4_KICKER__</div>
  <h2 class="sec-head">__S4_HEAD__</h2>
  <p class="sec-intro">__S4_INTRO__</p>

  <div class="card">
    <h3>__S4_CARD2_H3__</h3>
    <div class="csrc">__S4_CARD2_CSRC__</div>
    <div class="chart-body" id="chart-income-decile-composition"></div>
  </div>

  <div class="card">
    <h3>__S4_CARD4_H3__</h3>
    <div class="csrc">__S4_CARD4_CSRC__</div>
    <div class="chart-body" id="chart-income-composition"></div>
  </div>
</section>

<section class="pagesec" id="sec-s5">
  <div class="sec-kicker">__S5P_KICKER__</div><h2 class="sec-head">__S5P_HEAD__</h2><p class="sec-intro">__S5P_INTRO__</p>
  <div class="selector-row"><label for="pnl-group">__S5P_GROUP_LABEL__</label><select id="pnl-group"></select></div>
  <div class="card" id="sec-pnl"><h3>__S5P_ECONOMIC_H3__</h3><div class="csrc">__S5P_CARD_CSRC__</div><div class="table-scroll"><div id="household-pnl" data-empty="__S5P_EMPTY__"></div></div></div>
  <div class="card" id="sec-cash"><h3>__S5P_CASH_H3__</h3><div class="csrc">__S5P_CARD_CSRC__</div><div class="table-scroll"><div id="household-cashflow" data-empty="__S5P_EMPTY__"></div></div></div>
  <div class="card" id="sec-wealth"><h3>__S5P_WEALTH_H3__</h3><div class="csrc">__S5P_WEALTH_CSRC__</div><div class="table-scroll"><div id="household-wealth-age" data-empty="__S5P_EMPTY__"></div></div></div>
</section>
<section class="pagesec methodology" id="sec-s6">
  <div class="sec-kicker">__S5_KICKER__</div>
  <h2 class="sec-head">__S5_HEAD__</h2>
  <p class="sec-intro">__S5_INTRO__</p>

  <h4>__METH_H4_DATA__</h4>
  <p>__METH_P_DATA__</p>

  <h4>__METH_H4_HOME__</h4>
  <p>__METH_P_HOME__</p>
  <div class="stat-row">
    <div><b id="m-r2held">&mdash;</b>__STAT_R2HELD__</div>
    <div><b id="m-r2in">&mdash;</b>__STAT_R2IN__</div>
    <div><b id="m-nself">&mdash;</b>__STAT_NSELF__</div>
    <div><b id="m-ncbs">&mdash;</b>__STAT_NCBS__</div>
    <div><b id="m-nmodel">&mdash;</b>__STAT_NMODEL__</div>
  </div>

  <h4>__METH_H4_PENSION__</h4>
  <p>__METH_P_PENSION__</p>
  <div class="stat-row">
    <div><b id="m-mean2023">&mdash;</b>__STAT_MEAN2023__</div>
    <div><b id="m-meanpre">&mdash;</b>__STAT_MEANPRE__</div>
    <div><b id="m-macro">&mdash;</b>__STAT_MACRO__</div>
  </div>

  <h4>__METH_H4_WINSOR__</h4>
  <p>__METH_P_WINSOR__</p>
  <div class="caveat">
    __METH_WINSOR_CAVEAT__
  </div>

  <h4>__METH_H4_WAVE__</h4>
  <p>__METH_P_WAVE__</p>

  <h4>__METH_H4_SAMPLE__</h4>
  <p>__METH_P_SAMPLE__</p>

  <h4>__METH_H4_LIMITS__</h4>
  <ul>
__METH_LIMITS_LIST__
  </ul>

  <h4>__METH_H4_MANIFEST__</h4>
  <div class="table-scroll"><table class="datatable" id="table-manifest"></table></div>
</section>

</div>

<footer class="pagefoot">
  __FOOTER_ARCH__<br>
  __FOOTER_SOURCE__
</footer>

<div id="tooltip"></div>

<script id="figures-data" type="application/json">__FIGURES_JSON__</script>
<script id="meta-data" type="application/json">__META_JSON__</script>
<script id="manifest-data" type="application/json">__MANIFEST_JSON__</script>
<script>
// ---------------------------------------------------------------------------
// All data below is pre-aggregated by pipeline.py and parsed from figures/*.csv
// at build time by build_dashboard.py. Everything in this script is display
// formatting, sorting of already-complete rows, or SVG rendering -- no weighted
// means, no percentiles, no Gini, no modelling, and no per-household loops (the
// embedded arrays only ever contain already-aggregated rows, never raw households).
// ---------------------------------------------------------------------------
const FIGURES = JSON.parse(document.getElementById('figures-data').textContent);
const META = JSON.parse(document.getElementById('meta-data').textContent);
const MANIFEST = JSON.parse(document.getElementById('manifest-data').textContent);

const SVGNS = 'http://www.w3.org/2000/svg';

// RTL flag -- true for the Hebrew build, false for the English build. Drives
// chart mirroring (MX), gutter widths (leftPad) and text direction/anchoring.
const RTL = __RTL_JS__;

// Mobile chart-geometry flag -- mirrors the ≤640px CSS breakpoint. Charts read
// this at DRAW time (not just once) so that a debounced resize/orientation
// listener near the bottom of this script can flip it and redraw every chart
// with the other geometry when the viewport crosses the breakpoint (e.g.
// rotating a phone, or resizing a desktop window down past 640px).
const MOBILE_MQ = window.matchMedia('(max-width:640px)');
let MOBILE = MOBILE_MQ.matches;

// Every chart-drawing call below registers a zero-arg "redraw me" function
// here (after clearing its own container) so the resize watcher can just
// replay this list instead of re-running each section's one-time DOM setup
// (option population, small-multiples card creation, etc.) a second time.
const CHART_RENDERERS = [];

// Wide-table horizontal-scroll affordance. Tables (balance-sheet statement,
// ratios, manifest) are wrapped in a `.table-scroll` div; whenever a wrapper's
// content is actually wider than the wrapper itself, we tag it `.is-scrollable`
// so the CSS edge-fade cue (see `.table-scroll.is-scrollable::after`) appears --
// otherwise a table wider than a mobile viewport would silently clip with no
// visual hint that there's more to see by scrolling. Re-run after every table
// render and on resize (table width can change independent of the MOBILE
// breakpoint, e.g. dragging a desktop window).
function markScrollableTables(){
  document.querySelectorAll('.table-scroll').forEach(el=>{
    if(el.scrollWidth > el.clientWidth + 4) el.classList.add('is-scrollable');
    else el.classList.remove('is-scrollable');
  });
}

// Data-derived display labels for the active language only (see TR in
// build_dashboard.py). English's TR is empty, so tr() is pure identity there.
const TR = __TR_JSON__;
function tr(s){ return (TR[s]!==undefined) ? TR[s] : s; }

// Manifest per-file description overrides for the active language (empty for
// English -- falls back to the CSV's own "description" column).
const MANIFEST_DESC = __MANIFEST_DESC_JSON__;

// KPI tile label/note strings (Section 1) for the active language.
const UI = __UI_JSON__;

// Single primary accent used for EVERY single-series magnitude bar (net worth,
// participation rates, income-decile bars, box-plot fill, etc). Chosen once and
// reused everywhere a "how much" bar needs a color -- never re-picked per chart.
const PRIMARY_ACCENT = '#2a78d6'; // blue -- matches --gold (accent) in the CSS theme

// Fixed categorical color assignment -- one social group = one color everywhere
// this dashboard genuinely needs group *identity* (the over-time trajectories are
// the only remaining case; every other chart that used to color-by-group now uses
// PRIMARY_ACCENT instead, since a per-category color on an axis-labeled bar is
// decorative, not identity). Palette + order = the dataviz skill's validated
// 8-hue dark-mode categorical set (references/palette.md), reordered so no two
// adjacent slots collide (old palette had two greens: Dati #1baf7a vs Masorti
// (Traditional) #3fbf3f -- fixed below by giving them non-adjacent hue families).
const GROUP_ORDER = ['Haredi','Dati (Religious)','Masorti','Hiloni (Secular)','Arab'];
const GROUP_PALETTE = ['#2a78d6','#1baf7a','#eda100','#4a3aa7','#e34948'];
const GROUP_COLORS = {};
GROUP_ORDER.forEach((g,i)=>{ GROUP_COLORS[g] = GROUP_PALETTE[i]; });
const GROUP_ALL_COLOR = '#898781'; // neutral reference for "All"
const ASSET_COLORS = { 'Real Estate':'#2a78d6', 'Financial ex-pension':'#1baf7a', 'Pension':'#eda100' };

function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function num(v){ if(v===null||v===undefined||v==='') return null; const n = Number(v); return isNaN(n) ? null : n; }

function fmtNIS(v){
  const n = num(v);
  if(n===null) return '—';
  const r = Math.round(n);
  return (r<0?'-₪':'₪') + Math.abs(r).toLocaleString('en-US');
}
function fmtNISCompact(v){
  const n = num(v); if(n===null) return '—';
  const abs = Math.abs(n);
  let s;
  if(abs>=1e6) s = (n/1e6).toFixed(2)+'M';
  else if(abs>=1e3) s = (n/1e3).toFixed(0)+'K';
  else s = Math.round(n).toString();
  return '₪'+s;
}
function fmtPct(v, digits){
  const n = num(v);
  if(n===null) return '—';
  return (n*100).toFixed(digits===undefined?1:digits)+'%';
}
function fmtNum(v, digits){
  const n = num(v);
  if(n===null) return '—';
  return n.toFixed(digits===undefined?3:digits);
}

// ---------------------------------------------------------------------------
// Tooltip (shared across all charts)
// ---------------------------------------------------------------------------
const ttEl = document.getElementById('tooltip');
function showTooltip(evt, title, rows){
  let h = `<div class="tt-title">${esc(title)}</div>`;
  rows.forEach(r=>{
    const swatch = r.color ? ` style="--sw:${r.color}"` : '';
    h += `<div class="tt-row"><span class="k">${esc(r.k)}</span><span class="v${r.color?' keyed':''}"${swatch}>${esc(r.v)}</span></div>`;
  });
  ttEl.innerHTML = h;
  ttEl.style.display = 'block';
  moveTooltip(evt);
}
function moveTooltip(evt){
  const pad = 16;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const vw = window.innerWidth, vh = window.innerHeight;
  if(x + 270 > vw) x = evt.clientX - 270 - pad;
  if(y + 140 > vh) y = evt.clientY - 140;
  ttEl.style.left = x+'px'; ttEl.style.top = y+'px';
}
function hideTooltip(){ ttEl.style.display = 'none'; }

// On touch devices there is no true "hover" -- a tap fires pointerdown/move
// (which shows the tooltip via the chart marks' own listeners below) but
// often no 'pointerleave' at all, since there's no pointer left hovering to
// leave. Without this, a tapped tooltip can stay stuck on screen, overlapping
// whatever the user scrolls to next. Hiding on any scroll (capture, so it
// fires before the scroll-snap settles) and on any new touch elsewhere
// (capture phase, so it runs before that touch's own pointermove/show, if
// any -- e.g. tapping a different mark still hides-then-shows correctly in
// the same gesture) keeps the tooltip from ever surviving past its target.
window.addEventListener('scroll', hideTooltip, {passive:true, capture:true});
document.addEventListener('touchstart', hideTooltip, {passive:true, capture:true});

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------
function sv(tag, attrs){
  const e = document.createElementNS(SVGNS, tag);
  for(const k in (attrs||{})) e.setAttribute(k, attrs[k]);
  return e;
}
function svText(x, y, content, cls, attrs){
  const e = sv('text', Object.assign({x, y}, attrs||{}));
  if(cls) e.setAttribute('class', cls);
  e.textContent = content;
  return e;
}

// Off-screen canvas used purely to MEASURE text width in CSS px (matches the
// system-ui stack used throughout the page closely enough for wrap decisions;
// never rendered, never appended to the document).
const _measureCanvas = document.createElement('canvas');
const _measureCtx = _measureCanvas.getContext('2d');
function textWidthPx(str, fontSizePx){
  _measureCtx.font = fontSizePx + 'px system-ui, -apple-system, "Segoe UI", sans-serif';
  return _measureCtx.measureText(str).width;
}
// Greedy word-wrap into at most 2 lines (a 3rd+ line would need more vertical
// room than the mobile row height comfortably gives every chart type here).
// Any words beyond what fits on 2 lines are appended to the 2nd line as-is
// (may still overflow in the rare case of a single very long word -- no
// worse than the un-wrapped original, and every real category label in this
// dataset fits within 2 lines at the mobile font size).
function wrapLabelLines(str, maxWidthPx, fontSizePx){
  if(textWidthPx(str, fontSizePx) <= maxWidthPx) return [str];
  const words = str.split(' ');
  let lines = [];
  let cur = '';
  words.forEach(w=>{
    const candidate = cur ? cur+' '+w : w;
    if(cur && textWidthPx(candidate, fontSizePx) > maxWidthPx){
      lines.push(cur);
      cur = w;
    } else {
      cur = candidate;
    }
  });
  if(cur) lines.push(cur);
  if(lines.length > 2) lines = [lines[0], lines.slice(1).join(' ')];
  return lines;
}

// Category-axis row label that WRAPS onto a 2nd line (instead of clipping
// past the SVG's left edge / a bar it shouldn't overlap) whenever the text is
// wider than the leftPad gutter it has to live in. This only kicks in on
// MOBILE, where the leftPad gutter is narrowest relative to the larger
// (19px, see the <=640px CSS block) mobile category-label font -- desktop's
// wider gutters and smaller 16px font were never observed to overflow.
// `maxWidthPx` should be the caller's available gutter width in the SAME
// viewBox user-units as `x`/`y` (viewBox units render 1:1 with CSS px inside
// an SVG, so textWidthPx's px measurement is directly comparable). The
// multi-line block is anchored so its LAST line lands on the original `y`
// (growing upward), matching every single-line call site's existing
// bottom-of-row baseline.
function svCatLabelWrap(x, y, text, maxWidthPx, rtl, extraAttrs){
  const attrs = Object.assign({'text-anchor':'end'}, rtl ? {direction:'rtl'} : {}, extraAttrs||{});
  const fontSizePx = MOBILE ? 19 : 16;
  if(!MOBILE || !maxWidthPx || textWidthPx(text, fontSizePx) <= maxWidthPx){
    return svText(x, y, text, 'cat-label', attrs);
  }
  const lines = wrapLabelLines(text, maxWidthPx, fontSizePx);
  const lineHeight = fontSizePx + 2;
  const e = sv('text', Object.assign({x, y: y - (lines.length-1)*lineHeight}, attrs));
  e.setAttribute('class', 'cat-label');
  lines.forEach((line,i)=>{
    const tspan = document.createElementNS(SVGNS, 'tspan');
    tspan.setAttribute('x', x);
    if(i>0) tspan.setAttribute('dy', lineHeight);
    tspan.textContent = line;
    e.appendChild(tspan);
  });
  return e;
}
function niceMax(v){
  if(v<=0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v/mag;
  let step;
  if(n<=1) step=1; else if(n<=2) step=2; else if(n<=5) step=5; else step=10;
  return step*mag;
}
function legendHtml(items){ // items: [{label,color,isLine}]
  return '<div class="legend">' + items.map(it=>{
    const sw = it.isLine ? `<span class="ln" style="border-color:${it.color}"></span>` : `<span class="sw" style="background:${it.color}"></span>`;
    return `<span class="lk">${sw}${esc(it.label)}</span>`;
  }).join('') + '</div>';
}

__CHART_FUNCS_JS__

// ---------------------------------------------------------------------------
// Chart: multi-series line, with gaps (nulls skipped, not interpolated).
// Legitimately multi-color (genuinely simultaneous series, not a magnitude
// ramp). Hovering a legend item OR a line highlights that series and dims the
// rest, so an 8-series tangle stays readable.
// NOTE: not called by any render function below in either language build --
// dead code (like chartHBarGrouped above) carried over unchanged.
// ---------------------------------------------------------------------------
function chartLines(containerId, seriesList, opts){
  // seriesList: [{label, color, points:[{x,y}] (y may be null), dashed}]
  opts = opts||{};
  const container = document.getElementById(containerId);
  const width = 900, height = 320, leftPad = 80, rightPad = 20, topPad = 16, botPad = 36;
  const allX = [...new Set(seriesList.flatMap(s=>s.points.map(p=>p.x)))].sort((a,b)=>a-b);
  const allY = seriesList.flatMap(s=>s.points.map(p=>p.y)).filter(v=>v!==null && v!==undefined);
  const minY = Math.min(0,...allY), maxY = niceMax(Math.max(...allY));
  const xMin = allX[0], xMax = allX[allX.length-1];
  const scaleW = width-leftPad-rightPad, scaleH = height-topPad-botPad;
  const xOf = x => leftPad + (x-xMin)/(xMax-xMin)*scaleW;
  const yOf = y => topPad + scaleH - (y-minY)/(maxY-minY)*scaleH;

  const groupEls = []; // one <g> per series -- target of the hover highlight/dim
  function highlight(idx){
    groupEls.forEach((g,i)=>{ g.style.opacity = (i===idx) ? '1' : '0.14'; });
  }
  function resetHighlight(){
    groupEls.forEach(g=>{ g.style.opacity = '1'; });
  }

  if(opts.legend!==false){
    const legendDiv = document.createElement('div');
    legendDiv.className = 'legend';
    legendDiv.innerHTML = seriesList.map((s,i)=>
      `<span class="lk" data-idx="${i}" style="cursor:pointer;"><span class="ln" style="border-color:${s.color}"></span>${esc(s.label)}</span>`
    ).join('');
    container.appendChild(legendDiv);
    legendDiv.querySelectorAll('.lk').forEach(el=>{
      const idx = Number(el.dataset.idx);
      el.addEventListener('pointerenter', ()=>highlight(idx));
      el.addEventListener('pointerleave', resetHighlight);
    });
  }
  const svgEl = sv('svg',{class:'viz','viewBox':`0 0 ${width} ${height}`});
  // gridlines + y ticks
  const ticks = 4;
  for(let i=0;i<=ticks;i++){
    const yv = minY + (maxY-minY)*i/ticks;
    const gy = yOf(yv);
    svgEl.appendChild(sv('line',{class:'grid', x1:leftPad, x2:width-rightPad, y1:gy, y2:gy}));
    svgEl.appendChild(svText(leftPad-8, gy+3, fmtNISCompact(yv), 'axis-tick', {'text-anchor':'end'}));
  }
  allX.forEach(xv=>{
    svgEl.appendChild(svText(xOf(xv), height-8, String(xv), 'axis-tick', {'text-anchor':'middle'}));
  });
  seriesList.forEach((s,idx)=>{
    const g = sv('g', {});
    g.style.transition = 'opacity .15s';
    g.addEventListener('pointerenter', ()=>highlight(idx));
    g.addEventListener('pointerleave', resetHighlight);
    const segs = [];
    let cur = [];
    s.points.forEach(p=>{
      if(p.y===null||p.y===undefined){ if(cur.length){segs.push(cur); cur=[];} }
      else cur.push(p);
    });
    if(cur.length) segs.push(cur);
    segs.forEach(seg=>{
      if(seg.length<2){
        // single point: draw a dot only
        const p = seg[0];
        const c = sv('circle',{cx:xOf(p.x), cy:yOf(p.y), r:5, fill:s.color, stroke:'#f9f9f7','stroke-width':2});
        g.appendChild(c);
        return;
      }
      const d = seg.map((p,i)=>`${i===0?'M':'L'}${xOf(p.x)},${yOf(p.y)}`).join(' ');
      const path = sv('path',{d, fill:'none', stroke:s.color, 'stroke-width':2, 'stroke-linecap':'round', 'stroke-linejoin':'round'});
      if(s.dashed) path.setAttribute('stroke-dasharray','5,4');
      g.appendChild(path);
      seg.forEach(p=>{
        const c = sv('circle',{class:'mark', cx:xOf(p.x), cy:yOf(p.y), r:4.5, fill:s.color, stroke:'#f9f9f7','stroke-width':1.5});
        g.appendChild(c);
        const hit = sv('circle',{cx:xOf(p.x), cy:yOf(p.y), r:12, fill:'transparent', class:'mark'});
        g.appendChild(hit);
        hit.addEventListener('pointermove', e=>showTooltip(e, `${s.label} — ${p.x}`, [{k:opts.yLabel||'Value', v:fmtNIS(p.y), color:s.color}]));
        hit.addEventListener('pointerleave', hideTooltip);
      });
    });
    svgEl.appendChild(g);
    groupEls.push(g);
  });
  container.appendChild(svgEl);
}

// ---------------------------------------------------------------------------
// KPI tiles (Section 1)
// ---------------------------------------------------------------------------
(function renderKPIs(){
  const dist = FIGURES.fig_networth_distribution.find(r=>r.scope==='All');
  const bs = FIGURES.fig_national_balance_sheet;
  const reRow = bs.find(r=>r.item==='Total Real Estate');
  const assetsRow = bs.find(r=>r.item==='Total Assets');
  const reShare = num(reRow.value_nis) / num(assetsRow.value_nis); // ratio of two totals already in one figure table
  const tiles = [
    {label:UI.kpi1_label, value:fmtNIS(dist.mean), note:UI.kpi1_note},
    {label:UI.kpi2_label, value:fmtNIS(dist.p50), note:UI.kpi2_note},
    {label:UI.kpi3_label, value:fmtPct(reShare,0), note:UI.kpi3_note},
    {label:UI.kpi4_label, value:fmtNum(dist.gini,3), note:UI.kpi4_note},
  ];
  const row = document.getElementById('kpi-row');
  row.innerHTML = tiles.map(t=>`<div class="kpi-tile"><div class="lab">${esc(t.label)}</div><div class="val">${t.value}</div><div class="note">${esc(t.note)}</div></div>`).join('');
})();

// ---------------------------------------------------------------------------
// Section 1: balance sheet statement + donut
// ---------------------------------------------------------------------------
(function renderBalanceSheet(){
  const rows = FIGURES.fig_national_balance_sheet;
  const segStarts = {'Primary Residence':'Assets — Real Estate', 'Bank Deposits & Savings':'Assets — Financial', 'Mortgage':'Liabilities'};
  let html = '';
  rows.forEach(r=>{
    if(segStarts[r.item]) html += `<tr class="seg"><td colspan="2">${esc(tr(segStarts[r.item]))}</td></tr>`;
    let cls = '';
    if(r.category==='subtotal') cls='subtotal';
    if(r.category==='total') cls='subtotal';
    if(r.category==='networth') cls='networth total';
    const flag = r.is_modeled==='True' ? '<span class="modeled-flag">◇</span>' : '';
    html += `<tr class="${cls}"><td>${esc(tr(r.item))}${flag}</td><td class="amt">${fmtNIS(r.value_nis)}</td></tr>`;
  });
  document.getElementById('bs-statement').innerHTML = html;
  markScrollableTables();

  function drawDonut(){
    const reV = num(rows.find(r=>r.item==='Total Real Estate').value_nis);
    const depV = num(rows.find(r=>r.item==='Bank Deposits & Savings').value_nis);
    const invV = num(rows.find(r=>r.item==='Investment Portfolio').value_nis);
    const penV = num(rows.find(r=>r.item==='Pension & Retirement Savings').value_nis);
    document.getElementById('donut-national').innerHTML = '';
    chartDonut('donut-national', [
      {label:'Real Estate', value:reV, color:ASSET_COLORS['Real Estate']},
      {label:'Financial ex-pension', value:depV+invV, color:ASSET_COLORS['Financial ex-pension']},
      {label:'Pension ◇', value:penV, color:ASSET_COLORS['Pension']},
    ]);
  }
  drawDonut();
  CHART_RENDERERS.push(drawDonut);
})();

// ---------------------------------------------------------------------------
// Section 2: cross-tab explorer. Two dropdowns (metric x breakdown) pick which
// pre-computed column of fig_metric_by_breakdown to plot, for the chosen
// breakdown's categories. NO calculation here -- it only filters the embedded
// rows, sorts them (natural order for ordinal breakdowns per META.cross_tab,
// else descending by the chosen metric value), and draws a one-colour hbar.
// ---------------------------------------------------------------------------
(function renderExplorer(){
  const rows = FIGURES.fig_metric_by_breakdown;
  const ct = META.cross_tab;
  const TINY_N = 30; // cells below this unweighted sample size are greyed
  const metricSel = document.getElementById('exp-metric');
  const bSel = document.getElementById('exp-breakdown');
  metricSel.innerHTML = ct.metrics.map(m=>`<option value="${esc(m.key)}">${esc(tr(m.label))}</option>`).join('');
  bSel.innerHTML = ct.breakdowns.map(b=>`<option value="${esc(b.key)}">${esc(tr(b.label))}</option>`).join('');
  metricSel.value = ct.default_metric;
  bSel.value = ct.default_breakdown;

  function draw(){
    const mKey = metricSel.value, bKey = bSel.value;
    const mDef = ct.metrics.find(m=>m.key===mKey) || {label:mKey};
    const bDef = ct.breakdowns.find(b=>b.key===bKey) || {label:bKey, ordinal:false, categories:[]};
    // descriptive auto-updating title (not a takeaway)
    document.getElementById('explorer-title').textContent =
      UI.exp_title_tpl.replace('{metric}', tr(mDef.label)).replace('{breakdown}', tr(bDef.label));
    // filter to this breakdown; read the chosen metric column + n
    let sub = rows.filter(r=>r.breakdown===bKey)
                  .map(r=>({category:r.category, val:num(r[mKey]), n:num(r.n_households)}));
    if(bDef.ordinal){
      const order = bDef.categories;
      sub.sort((a,b)=>order.indexOf(a.category)-order.indexOf(b.category)); // natural order
    } else {
      sub.sort((a,b)=>(b.val||0)-(a.val||0)); // descending by value
    }
    document.getElementById('chart-explorer').innerHTML = '';
    chartHBarSingle('chart-explorer', sub, 'category', 'val', {
      leftPad: RTL?(MOBILE?160:300):(MOBILE?150:230),
      colorFn: r => (r.n!==null && r.n < TINY_N) ? 'var(--dim)' : PRIMARY_ACCENT,
      valueLabel: mDef.label,
      tooltipRows: r => [{k:tr(UI.exp_n_label), v: r.n===null ? '—' : String(r.n)}],
    });
  }
  metricSel.addEventListener('change', draw);
  bSel.addEventListener('change', draw);
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 2: two-way matrix. Three dropdowns (metric x dim A x dim B) select a
// pre-computed slice of fig_metric_matrix and render a heatmap grid of the
// metric's weighted mean per (dim A cat x dim B cat) cell. NO calculation here:
// it only reads cell values, computes the min/max of the shown n>=30 cells for
// the sequential colour ramp, formats (compact NIS), and lays out the grid --
// transposing A/B when the user picks a pair in the reverse of its stored order.
// n<30 cells are greyed with the value hidden (n on hover); empty cells blank.
// ---------------------------------------------------------------------------
(function renderMatrix(){
  const rows = FIGURES.fig_metric_matrix;
  const ct = META.cross_tab, mx = META.matrix;
  const TINY_N = mx.tiny_n || 30;

  // Sequential blue ramp, light->dark (dataviz skill light-mode 100..700). Used
  // only to colour-scale already-computed cell values (display formatting).
  const RAMP = ['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5',
                '#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b'];
  function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
  function rampColor(t){
    t = Math.max(0, Math.min(1, t));
    const x = t*(RAMP.length-1), i = Math.floor(x), f = x-i;
    if(i>=RAMP.length-1) return RAMP[RAMP.length-1];
    const a=hex2rgb(RAMP[i]), b=hex2rgb(RAMP[i+1]);
    const c=a.map((v,k)=>Math.round(v+(b[k]-v)*f));
    return '#'+c.map(v=>v.toString(16).padStart(2,'0')).join('');
  }

  const mSel=document.getElementById('mx-metric'),
        aSel=document.getElementById('mx-dima'), bSel=document.getElementById('mx-dimb');
  mSel.innerHTML = ct.metrics.map(m=>`<option value="${esc(m.key)}">${esc(tr(m.label))}</option>`).join('');
  const dimOpts = ct.breakdowns.map(b=>`<option value="${esc(b.key)}">${esc(tr(b.label))}</option>`).join('');
  aSel.innerHTML = dimOpts; bSel.innerHTML = dimOpts;
  mSel.value = mx.default_metric; aSel.value = mx.default_dim_a; bSel.value = mx.default_dim_b;

  function catsOf(key){ const b=ct.breakdowns.find(x=>x.key===key); return b?b.categories:[]; }
  function labelOf(key){ const b=ct.breakdowns.find(x=>x.key===key); return b?b.label:key; }

  function draw(){
    // A == B is not allowed: nudge B to the first breakdown that isn't A.
    if(bSel.value===aSel.value){
      const alt = ct.breakdowns.find(b=>b.key!==aSel.value);
      if(alt) bSel.value = alt.key;
    }
    // disable the A-matching option inside B's dropdown as a visible hint
    Array.from(bSel.options).forEach(o=>{ o.disabled = (o.value===aSel.value); });

    const mKey=mSel.value, aKey=aSel.value, bKey=bSel.value;
    const mDef = ct.metrics.find(m=>m.key===mKey)||{label:mKey};
    document.getElementById('matrix-title').textContent =
      UI.mx_title_tpl.replace('{metric}', tr(mDef.label))
                     .replace('{dimA}', tr(labelOf(aKey)))
                     .replace('{dimB}', tr(labelOf(bKey)));

    // gather cells for this unordered pair, transposing so dim A is across rows,
    // dim B across columns regardless of which order the CSV stored the pair in.
    const cells={}; // `${catA}||${catB}` -> {val, n}
    rows.forEach(r=>{
      let ca, cb;
      if(r.dim_a===aKey && r.dim_b===bKey){ ca=r.cat_a; cb=r.cat_b; }
      else if(r.dim_a===bKey && r.dim_b===aKey){ ca=r.cat_b; cb=r.cat_a; }
      else return;
      cells[ca+'||'+cb] = {val:num(r[mKey]), n:num(r.n_households)};
    });

    // ordered category lists (cross_tab order), filtered to those present here
    const aAll=catsOf(aKey), bAll=catsOf(bKey);
    const rowCats = aAll.filter(ca=>bAll.some(cb=>cells[ca+'||'+cb]!==undefined));
    const colCats = bAll.filter(cb=>aAll.some(ca=>cells[ca+'||'+cb]!==undefined));

    // colour scale over the shown n>=TINY cells that have a numeric value
    let lo=Infinity, hi=-Infinity;
    rowCats.forEach(ca=>colCats.forEach(cb=>{
      const c=cells[ca+'||'+cb];
      if(c && c.n!==null && c.n>=TINY_N && c.val!==null){
        if(c.val<lo) lo=c.val; if(c.val>hi) hi=c.val;
      }
    }));
    const haveScale = isFinite(lo) && isFinite(hi) && hi>lo;

    let h='<table class="matrix"><thead><tr><th class="corner"></th>';
    colCats.forEach(cb=>{ h+=`<th scope="col">${esc(tr(cb))}</th>`; });
    h+='</tr></thead><tbody>';
    rowCats.forEach(ca=>{
      h+=`<tr><th class="rowhead" scope="row">${esc(tr(ca))}</th>`;
      colCats.forEach(cb=>{
        const c=cells[ca+'||'+cb];
        if(c===undefined){ h+='<td class="cell empty"></td>'; return; }
        // n<30 or a non-numeric value -> greyed, value hidden (n on hover)
        if((c.n!==null && c.n<TINY_N) || c.val===null){
          h+=`<td class="cell tiny" style="background:var(--panel2)" data-n="${c.n===null?'':c.n}" `
            +`data-ca="${esc(ca)}" data-cb="${esc(cb)}"></td>`;
          return;
        }
        const t = haveScale ? (c.val-lo)/(hi-lo) : 0.5;
        const bg = rampColor(t);
        const txt = t>0.55 ? '#ffffff' : '#0b0b0b';   // white on the darkest cells
        h+=`<td class="cell" style="background:${bg};color:${txt}" data-v="${c.val}" `
          +`data-n="${c.n===null?'':c.n}" data-ca="${esc(ca)}" data-cb="${esc(cb)}">`
          +`${esc(fmtNISCompact(c.val))}</td>`;
      });
      h+='</tr>';
    });
    h+='</tbody></table>';
    document.getElementById('chart-matrix').innerHTML = h;

    // colour-scale legend
    const grad = `linear-gradient(to right, ${RAMP.join(',')})`;
    document.getElementById('matrix-legend').innerHTML = haveScale
      ? `<span>${esc(tr(UI.mx_legend_low))} ${esc(fmtNISCompact(lo))}</span>`
        + `<span class="grad" style="background:${grad}"></span>`
        + `<span>${esc(fmtNISCompact(hi))} ${esc(tr(UI.mx_legend_high))}</span>`
        + `<span class="mx-note">&middot; ${esc(tr(UI.mx_tiny_note))}</span>`
      : `<span class="mx-note">${esc(tr(UI.mx_tiny_note))}</span>`;

    // per-cell hover tooltip (value + n; greyed cells show n only)
    const aLab=tr(labelOf(aKey)), bLab=tr(labelOf(bKey)), mLab=tr(mDef.label);
    document.querySelectorAll('#chart-matrix td.cell:not(.empty)').forEach(td=>{
      td.addEventListener('pointermove', e=>{
        const ca=tr(td.getAttribute('data-ca')), cb=tr(td.getAttribute('data-cb'));
        const v=td.getAttribute('data-v'), n=td.getAttribute('data-n');
        const trows=[];
        if(v!==null && v!=='') trows.push({k:mLab, v:fmtNIS(num(v)), color:PRIMARY_ACCENT});
        trows.push({k:tr(UI.exp_n_label), v:(n===null||n==='')?'—':String(n)});
        showTooltip(e, `${aLab}: ${ca} · ${bLab}: ${cb}`, trows);
      });
      td.addEventListener('pointerleave', hideTooltip);
    });
  }
  mSel.addEventListener('change', draw);
  aSel.addEventListener('change', draw);
  bSel.addEventListener('change', draw);
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 2: asset composition by group (100% stacked), sorted by RE share desc
// ---------------------------------------------------------------------------
(function renderCompositionByGroup(){
  function draw(){
    const rows = FIGURES.fig_composition_by_group;
    const byGroup = {};
    rows.forEach(r=>{ (byGroup[r.group]=byGroup[r.group]||{})[r.asset_class] = num(r.share_of_assets); });
    const groups = Object.keys(byGroup).sort((a,b)=>(byGroup[b]['Real Estate']||0)-(byGroup[a]['Real Estate']||0));
    document.getElementById('chart-composition-group').innerHTML = '';
    chartStacked100('chart-composition-group', groups, (g,k)=>byGroup[g][k], [
      {key:'Real Estate', label:'Real Estate', color:ASSET_COLORS['Real Estate']},
      {key:'Financial ex-pension', label:'Financial ex-pension', color:ASSET_COLORS['Financial ex-pension']},
      {key:'Pension', label:'Pension ◇', color:ASSET_COLORS['Pension']},
    ]);
  }
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 2: participation rates, small multiples
// ---------------------------------------------------------------------------
(function renderParticipation(){
  const rows = FIGURES.fig_participation;
  const metrics = [
    {key:'pct_with_mortgage', label:'With a mortgage'},
    {key:'pct_with_pension', label:'With a pension'},
    {key:'pct_owns_additional_property', label:'Owns additional property'},
    {key:'pct_with_investment', label:'With an investment portfolio'},
  ];
  const container = document.getElementById('chart-participation');
  metrics.forEach(m=>{
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h3 style="font-size:15px;">${esc(tr(m.label))}</h3><div class="chart-body" id="pp-${m.key}"></div>`;
    container.appendChild(card);
  });
  function draw(){
    metrics.forEach(m=>{
      const sub = [...rows].sort((a,b)=>num(b[m.key])-num(a[m.key]));
      document.getElementById(`pp-${m.key}`).innerHTML = '';
      chartHBarSingle(`pp-${m.key}`, sub, 'group', m.key, {
        width: MOBILE?380:560, rowH: MOBILE?120:30, barH: MOBILE?64:16,
        leftPad: RTL?(MOBILE?155:170):(MOBILE?150:150),
        color:PRIMARY_ACCENT, fmt:v=>fmtPct(v,0), valueLabel:m.label,
      });
    });
  }
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 3: percentile spread (All + groups, sorted by median desc, All pinned first)
// ---------------------------------------------------------------------------
(function renderPercentileSpread(){
  function draw(){
    const rows = FIGURES.fig_networth_distribution;
    const all = rows.find(r=>r.scope==='All');
    const groups = rows.filter(r=>r.scope!=='All').sort((a,b)=>num(b.p50)-num(a.p50));
    document.getElementById('chart-percentile-spread').innerHTML = '';
    chartBoxPlot('chart-percentile-spread', [all, ...groups].map(r=>({...r, [ 'group']: r.scope})), 'group');
  }
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 4: income decile charts
// ---------------------------------------------------------------------------
(function renderIncomeDecile(){
  function draw(){
    const rows = [...FIGURES.fig_income_wealth].sort((a,b)=>Number(a.income_decile)-Number(b.income_decile));
    // Only the asset-composition-by-decile chart is drawn here now. The old
    // mean-net-worth-by-decile bar lives in the Section-2 cross-tab explorer
    // (Net worth x Income decile); the capital-income-share bar was retired.
    const groups2 = rows.map(r=>r.income_decile);
    const byDec = {};
    rows.forEach(r=>{ byDec[r.income_decile] = {'Real Estate':num(r.real_estate_share),'Financial ex-pension':num(r.financial_share),'Pension':num(r.pension_share)}; });
    document.getElementById('chart-income-decile-composition').innerHTML = '';
    chartStacked100('chart-income-decile-composition', groups2, (g,k)=>byDec[g][k], [
      {key:'Real Estate', label:'Real Estate', color:ASSET_COLORS['Real Estate']},
      {key:'Financial ex-pension', label:'Financial ex-pension', color:ASSET_COLORS['Financial ex-pension']},
      {key:'Pension', label:'Pension', color:ASSET_COLORS['Pension']},
    ], {leftPad: MOBILE?48:76});
  }
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 4: monthly income composition by group (stacked, absolute NIS)
// ---------------------------------------------------------------------------
(function renderIncomeComposition(){
  function draw(){
    const rows = FIGURES.fig_income_composition_by_group.filter(r=>r.group!=='All');
    const sorted = [...rows].sort((a,b)=>num(b.mean_total_monthly)-num(a.mean_total_monthly));
    // dark-mode-consistent categorical steps (validated set, see GROUP_PALETTE) --
    // previously mixed light-mode hexes (#eda100, #1baf7a) with dark ones, and used
    // a low-chroma grey for "Govt transfers" which fails the categorical chroma
    // floor (grey reads as "no category," not identity #4) -- swapped for green.
    // Order chosen so the two green-family hues (Govt transfers #008300 and
    // Rental #1baf7a) are never adjacent in the stack.
    const segs = [
      {key:'mean_work', label:'Work', color:'#2a78d6'},
      {key:'mean_govt_transfers', label:'Govt transfers', color:'#008300'},
      {key:'mean_pension', label:'Pension', color:'#eda100'},
      {key:'mean_rental', label:'Rental', color:'#1baf7a'},
      {key:'mean_interest', label:'Interest', color:'#4a3aa7'},
      {key:'mean_capital_gains', label:'Capital gains', color:'#e34948'},
    ];
    document.getElementById('chart-income-composition').innerHTML = '';
    chartStackedAbs('chart-income-composition', sorted, 'group', segs, {legend:true, leftPad: RTL?(MOBILE?155:190):(MOBILE?150:175)});
  }
  draw();
  CHART_RENDERERS.push(draw);
})();

// ---------------------------------------------------------------------------
// Section 5: household P&L and accumulated wealth by age. Every cell is read
// directly from the one-row-per-age-band pipeline output; this renderer only
// orders age bands, applies signs/number formats, and lays out statement rows.
// ---------------------------------------------------------------------------
(function renderHouseholdPnlByAge(){
  const allRows=[...(FIGURES.fig_household_pnl_by_age||[])], expenseRows=[...(FIGURES.fig_household_expenses_by_age||[])];
  const groupSel=document.getElementById('pnl-group');
  const cashEl=document.getElementById('household-cashflow'), flowEl=document.getElementById('household-pnl'), wealthEl=document.getElementById('household-wealth-age');
  if(!allRows.length){ const e=`<div class="pnl-empty">${esc(flowEl.dataset.empty||'')}</div>`; cashEl.innerHTML=e; flowEl.innerHTML=e; wealthEl.innerHTML=e; return; }
  const dimensions=['Age group','Religious group','Socioeconomic cluster'];
  dimensions.filter(d=>allRows.some(r=>r.breakdown_dimension===d)).forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=tr(d);groupSel.appendChild(o);});
  groupSel.value='Age group';
  const open=new Set();
  const mobileHeShort={
    'Other financial-asset saving':'חיסכון בנכ׳ פיננסיים אח׳',
    'Modeled financial-asset income':'הכנסה מנכ׳ פיננסיים ממודלת',
    'Funds, equities and securities return (4%)':'תשואת קרנות וניירות ערך (4%)',
    'Government transfers and support':'העברות ממשלה ותמיכות',
    'Adjusted disposable economic income':'הכנסה כלכלית פנויה מותאמת',
    'Fixed / semi-fixed household costs':'עלויות משק בית קבועות / קב׳ למחצה',
    'Transportation excluding car purchases':'תחבורה ללא רכישת רכב',
    'Household disposable consumption':'צריכה פנויה של משק הבית',
    'Other real-estate debt proceeds / repayment':'תקבולים / פירעון חוב נדל״ן אח׳',
    'Housing debt financing / repayment':'מימון / פירעון חוב דיור',
    'Remove non-cash and valuation income':'נטרול הכנסה לא־מזומנית ושערוכים',
    'Remove non-cash expense differences':'נטרול פערי הוצאה לא־מזומניים',
    'Reclassify pension withdrawals to investing':'סיווג משיכות פנסיה להשקעה',
    'Other financial assets ◇':'נכ׳ פיננסיים אחרים ◇',
    'Funds, equities and securities ◇':'קרנות וניירות ערך ◇'
  };
  const statementLabel=label=>(MOBILE&&document.documentElement.dir==='rtl'&&mobileHeShort[label])?mobileHeShort[label]:tr(label);
  const statementMoney=v=>{
    const n=num(v);if(n===null)return '—';
    const abs=Math.abs(n);let s;
    if(abs>=1e6)s=(abs/1e6).toFixed(1).replace(/\.0$/,'')+'M';
    else if(abs>=1e3)s=(abs/1e3).toFixed(1).replace(/\.0$/,'')+'K';
    else s=Math.round(abs).toString();
    return n<0?'('+s+')':s;
  };
  const pct=v=>{const n=num(v);if(n===null)return '—';const x=Math.abs(n)<=1?n*100:n;const s=Math.abs(x).toFixed(1)+'%';return x<0?'('+s+')':s;};
  const count=v=>{const n=num(v);return n===null?'—':Math.round(n).toLocaleString('en-US');};
  const decimal=v=>{const n=num(v);return n===null?'—':n.toFixed(1);};
  const outflow=v=>{const n=num(v);return n===null?'—':(n===0?statementMoney(0):'('+statementMoney(Math.abs(n))+')');};
  function statement(el,lines,rows){
    let h='<table class="pnl"><thead><tr><th>'+esc(tr('Breakdown'))+'</th>'; rows.forEach(r=>h+=`<th>${esc(tr(r.column_label))}</th>`); h+='</tr></thead><tbody>';
    lines.forEach(line=>{
      if(line.seg){h+=`<tr class="seg"><td colspan="${rows.length+1}">${esc(statementLabel(line.seg))}</td></tr>`;return;}
      if(line.parent&&!(Array.isArray(line.parent)?line.parent:[line.parent]).every(k=>open.has(k)))return;
      const expandable=line.expand?` expandable${open.has(line.expand)?' open':''}`:'';
      const attrs=line.expand?` data-expand="${esc(line.expand)}" role="button" tabindex="0" aria-expanded="${open.has(line.expand)}"`:'';
      h+=`<tr class="${line.cls||''}${expandable}"${attrs}><td>${esc(statementLabel(line.label))}</td>`;
      rows.forEach(r=>h+=`<td>${esc((line.fmt||statementMoney)(r[line.key]))}</td>`); h+='</tr>';
    });
    el.innerHTML=h+'</tbody></table>';
    el.querySelectorAll('tr[data-expand]').forEach(row=>{const toggle=()=>{const k=row.dataset.expand;open.has(k)?open.delete(k):open.add(k);draw();};row.addEventListener('click',toggle);row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle();}});});
  }
  function draw(){
    const selectedDimension=groupSel.value;
    const rows=allRows.filter(r=>r.breakdown_dimension===selectedDimension).sort((a,b)=>(num(a.column_order)||0)-(num(b.column_order)||0));
    if(!rows.length)return;
    const first=(r,keys,f=0)=>{for(const k of keys){const n=num(r[k]);if(n!==null)return n;}return typeof f==='function'?f(r):f;};
    const eFor=label=>expenseRows.filter(e=>e.breakdown_dimension===selectedDimension&&e.column_label===label);
    rows.forEach(r=>{
      r._cash_living=first(r,['cash_living_outflows'],x=>first(x,['cash_consumption'])+first(x,['private_transfers_paid'])+first(x,['mortgage_interest'])); r._cash_before=first(r,['cash_flow_before_funded_saving'],x=>first(x,['cash_disposable_income'])-x._cash_living); r._funded=first(r,['total_funded_cash_saving']); r._free=first(r,['free_cash_flow'],x=>x._cash_before-x._funded);
      const es=eFor(r.column_label), byCat={}; es.forEach(e=>byCat[e.category||e.metric||e.expense_category]=num(e.value_nis)||0); const cat=k=>byCat[k]||0;
      r._housing=cat('Housing service'); r._transport=cat('Transport excluding cars'); r._cars=cat('Car purchases'); r._durables=cat('Other durables'); r._fees_professional=cat('Fees & professional services'); r._transport_durables=r._transport+r._cars+r._durables; r._fixed=r._housing+r._transport_durables+r._fees_professional;
      r._food_home=cat('Food'); r._education=cat('Education')+cat('Childcare'); r._health=cat('Health'); r._clothing=cat('Clothing & footwear'); r._personal_child=cat('Child personal care & baby products'); r._kids=r._food_home+r._education+r._clothing+r._personal_child;
      r._restaurants=cat('Restaurants'); r._travel_entertainment=cat('Travel & vacations')+cat('Leisure & luxury'); r._tobacco=cat('Tobacco & smoking'); r._disposable=r._restaurants+r._travel_entertainment+r._tobacco;
      r._personal_general=cat('Personal care & miscellaneous'); r._other_personal=r._personal_general; r._donations=first(r,['community_gemach_saving'],cat('Donations')); r._other_residual=cat('Other consumption'); r._other_expenses=r._other_personal+r._other_residual+first(r,['private_transfers_paid']); r._expense_total=r._fixed+r._kids+r._health+r._disposable+r._other_expenses;
      r._other_housing=Math.max(0,r._housing-first(r,['rent_paid'])-first(r,['imputed_housing_consumption'])); r._other_funded=first(r,['other_observed_financial_saving']); r._other_fund_deposits=first(r,['training_fund_contributions'])+first(r,['provident_fund_contributions'])+first(r,['life_exec_insurance_contributions'])+r._other_funded;
      r._funded_components=first(r,['identified_transaction_allocation'],x=>first(x,['pension_balance_change'])+first(x,['other_financial_asset_saving_allocation'])+first(x,['real_estate_transaction_allocation']));
      r._networth_alloc=first(r,['total_net_worth_saving_allocation'],r._funded_components); r._pension_share=r._networth_alloc?first(r,['pension_balance_change'])/r._networth_alloc:null; r._surplus_deficit=first(r,['residual_saving'],first(r,['total_saving'])-r._networth_alloc);
    });
 statement(cashEl,[
      {seg:'Operating activities'},
      {key:'total_saving',label:'Economic saving from P&L',cls:'subtotal'},
      {key:'operating_income_cash_adjustment',label:'Cash-income adjustments',expand:'cashincomeadjustment'},
      {key:'remove_noncash_income_cash_flow',label:'Remove non-cash and valuation income',parent:'cashincomeadjustment',cls:'detail'},
      {key:'remove_pension_withdrawals_from_operations',label:'Reclassify pension withdrawals to investing',parent:'cashincomeadjustment',cls:'detail'},
      {key:'operating_expense_cash_adjustment',label:'Cash-expense adjustments',expand:'cashexpenseadjustment'},
      {key:'remove_noncash_expenses_cash_flow',label:'Remove non-cash expense differences',parent:'cashexpenseadjustment',cls:'detail'},
      {key:'cash_flow_operating_activities',label:'Cash flow from operating activities',cls:'keyline'},
      {seg:'Investing activities'},
      {key:'pension_investing_cash_flow',label:'Pension investing',cls:'subtotal',expand:'pensioninvesting'},
      {key:'pension_withdrawals_cash_flow',label:'Pension withdrawals',parent:'pensioninvesting',cls:'detail'},
      {key:'pension_contributions',label:'Employee pension contribution',fmt:outflow,parent:'pensioninvesting',cls:'detail'},
      {key:'other_financial_savings_investing_cash_flow',label:'Other financial-asset saving',cls:'subtotal',expand:'otherfinancialinvesting'},
      {key:'training_fund_contributions',label:'Training-fund contributions',fmt:outflow,parent:'otherfinancialinvesting',cls:'detail'},
      {key:'provident_fund_contributions',label:'Provident-fund contributions',fmt:outflow,parent:'otherfinancialinvesting',cls:'detail'},
      {key:'life_exec_insurance_contributions',label:'Life/executive-insurance contributions',fmt:outflow,parent:'otherfinancialinvesting',cls:'detail'},
      {key:'other_observed_financial_saving',label:'Other observed financial saving',fmt:outflow,parent:'otherfinancialinvesting',cls:'detail'},
      {key:'real_estate_investing_cash_flow',label:'Real-estate investing',cls:'subtotal',expand:'realestateinvesting'},
      {key:'property_investment_cash_flow',label:'Property acquisitions and improvements',parent:'realestateinvesting',cls:'detail',expand:'propertyinvestment'},
      {key:'home_purchase_net',label:'Net home purchase',fmt:outflow,parent:['realestateinvesting','propertyinvestment'],cls:'detail-2'},
      {key:'other_property_purchase',label:'Other property purchase',fmt:outflow,parent:['realestateinvesting','propertyinvestment'],cls:'detail-2'},
      {key:'home_capital_improvements',label:'Home capital improvements',fmt:outflow,parent:['realestateinvesting','propertyinvestment'],cls:'detail-2'},
      {key:'community_real_estate_cash_flow',label:'Donations & community saving',parent:'realestateinvesting',cls:'detail'},
      {key:'other_asset_investing_cash_flow',label:'Other asset investing',cls:'subtotal',expand:'otherinvesting'},
      {key:'household_asset_sales_cash_flow',label:'Proceeds from household-asset sales',parent:'otherinvesting',cls:'detail'},
      {key:'household_loans_extended_cash_flow',label:'Loans extended by household',parent:'otherinvesting',cls:'detail'},
      {key:'cash_flow_investing_activities',label:'Cash flow from investing activities',cls:'keyline'},
      {seg:'Financing activities'},
      {key:'cash_flow_financing_activities',label:'Real-estate debt financing / repayment',cls:'keyline',expand:'realestatefinancing'},
      {key:'mortgage_principal_cash_flow',label:'Mortgage principal',parent:'realestatefinancing',cls:'detail'},
      {key:'housing_debt_financing_cash_flow',label:'Other real-estate debt proceeds / repayment',parent:'realestatefinancing',cls:'detail',expand:'housingfinance'},
      {key:'apartment_debt_movement_cash_flow',label:'Apartment debt proceeds / repayment',parent:['realestatefinancing','housingfinance'],cls:'detail-2'},
      {key:'other_housing_loan_cash_flow',label:'Other housing-loan payment (principal + interest)',parent:['realestatefinancing','housingfinance'],cls:'detail-2'},
      {key:'unclassified_debt_cash_flow',label:'Unclassified debt proceeds / repayment',parent:['realestatefinancing','housingfinance'],cls:'detail-2'},
      {key:'net_change_in_unallocated_cash',label:'Net increase / decrease in unallocated cash',cls:'keyline'}
    ],rows);
    const pnlLines=[
      {seg:'Household profile'},{key:'sample_households',label:'Households (sample)',fmt:count},{key:'mean_household_size',label:'Average household size',fmt:decimal},{key:'mean_children',label:'Average children',fmt:decimal},
      {seg:'Economic income (including imputed)'},{key:'adjusted_gross_economic_income',label:'Total economic income',cls:'subtotal',expand:'income'},
      {key:'labor_economic_income',label:'Economic labor income',parent:'income',cls:'detail',expand:'laborincome'},{key:'labor_income',label:'Cash labor income',parent:['income','laborincome'],cls:'detail-2'},{key:'modeled_employer_pension_contributions',label:'Modeled employer pension contribution',parent:['income','laborincome'],cls:'detail-2'},{key:'transfers_income',label:'Government transfers and support',parent:'income',cls:'detail'},{key:'pension_economic_income',label:'Pension economic income',parent:'income',cls:'detail',expand:'pensionincome'},{key:'pension_income',label:'Pension cash receipts',parent:['income','pensionincome'],cls:'detail-2'},{key:'pension_income_adjustment',label:'Pension surplus / drawdown adjustment',parent:['income','pensionincome'],cls:'detail-2'},{key:'imputed_housing_asset_income',label:'Net imputed owner-housing income',parent:'income',cls:'detail'},{key:'rental_property_income',label:'Reported rental / property income',parent:'income',cls:'detail'},{key:'imputed_financial_asset_income',label:'Modeled financial-asset income',parent:'income',cls:'detail',expand:'financialincome'},{key:'imputed_deposit_savings_income',label:'Deposits and savings return (1%)',parent:['income','financialincome'],cls:'detail-2'},{key:'imputed_nonbank_investment_income',label:'Funds, equities and securities return (4%)',parent:['income','financialincome'],cls:'detail-2'},{key:'other_income',label:'Other income',parent:'income',cls:'detail'},
      {key:'direct_taxes',label:'Direct taxes',fmt:outflow},{key:'adjusted_disposable_economic_income',label:'Adjusted disposable economic income',cls:'keyline'},
      {seg:'Economic expenses (including imputed)'},{key:'_expense_total',label:'Total expenses',fmt:outflow,cls:'keyline'},
      {key:'_fixed',label:'Fixed / semi-fixed household costs',fmt:outflow,expand:'fixed'},{key:'_housing',label:'Housing services',fmt:outflow,parent:'fixed',cls:'detail',expand:'housing'},{key:'rent_paid',label:'Rent paid',fmt:outflow,parent:['fixed','housing'],cls:'detail-2'},{key:'imputed_housing_consumption',label:'Imputed rent',fmt:outflow,parent:['fixed','housing'],cls:'detail-2',expand:'imputedhousing'},{key:'mortgage_interest',label:'Mortgage interest',fmt:outflow,parent:['fixed','housing','imputedhousing'],cls:'detail-2'},{key:'equity_housing_service',label:'Equity housing service',fmt:outflow,parent:['fixed','housing','imputedhousing'],cls:'detail-2'},{key:'_other_housing',label:'Other housing services',fmt:outflow,parent:['fixed','housing'],cls:'detail-2'},{key:'_transport_durables',label:'Transportation and durable goods',fmt:outflow,parent:'fixed',cls:'detail',expand:'transport'},{key:'_transport',label:'Transportation excluding car purchases',fmt:outflow,parent:['fixed','transport'],cls:'detail-2'},{key:'_cars',label:'Car purchases',fmt:outflow,parent:['fixed','transport'],cls:'detail-2'},{key:'_durables',label:'Other durable goods',fmt:outflow,parent:['fixed','transport'],cls:'detail-2'},{key:'_fees_professional',label:'Fees and professional services',fmt:outflow,parent:'fixed',cls:'detail'},
      {key:'_kids',label:'Kids / semi-kids expenses',fmt:outflow,expand:'kids'},{key:'_food_home',label:'Food at home',fmt:outflow,parent:'kids',cls:'detail'},{key:'_education',label:'Education and childcare',fmt:outflow,parent:'kids',cls:'detail'},{key:'_clothing',label:'Clothing & footwear',fmt:outflow,parent:'kids',cls:'detail'},{key:'_personal_child',label:'Child personal care and baby products',fmt:outflow,parent:'kids',cls:'detail'},{key:'_health',label:'Health',fmt:outflow},
      {key:'_disposable',label:'Household disposable consumption',fmt:outflow,expand:'disposable'},{key:'_restaurants',label:'Restaurants / food out',fmt:outflow,parent:'disposable',cls:'detail'},{key:'_travel_entertainment',label:'Travel and entertainment',fmt:outflow,parent:'disposable',cls:'detail'},{key:'_tobacco',label:'Tobacco and smoking products',fmt:outflow,parent:'disposable',cls:'detail'},
      {key:'_other_expenses',label:'Other expenses',fmt:outflow,expand:'otherexp'},{key:'_other_personal',label:'Personal care & miscellaneous',fmt:outflow,parent:'otherexp',cls:'detail'},{key:'_other_residual',label:'Other / residual consumption',fmt:outflow,parent:'otherexp',cls:'detail'},{key:'private_transfers_paid',label:'Private transfers paid',fmt:outflow,parent:'otherexp',cls:'detail'},
      {seg:'Saving reconciliation and allocation'},{key:'transactional_accrual_saving',label:'Economic / accrual saving (income less consumption)',cls:'keyline',expand:'saving'},
      {key:'cash_flow_before_funded_saving',label:'Cash saving before funded allocations',parent:'saving',cls:'detail'},
      {key:'noncash_saving_bridge',label:'Non-cash accrual and valuation bridge',parent:'saving',cls:'detail'},
      {key:'pension_balance_change',label:'Change in modeled pension wealth ◇',parent:'saving',cls:'detail',expand:'pensionsaving'},
      {key:'pension_contributions',label:'Employee pension contribution',parent:['saving','pensionsaving'],cls:'detail-2'},
      {key:'modeled_employer_pension_contributions',label:'Modeled employer pension contribution ◇',parent:['saving','pensionsaving'],cls:'detail-2'},
      {key:'imputed_pension_asset_income',label:'Modeled pension asset return ◇',parent:['saving','pensionsaving'],cls:'detail-2'},
      {key:'pension_income',label:'Pension cash withdrawals',fmt:outflow,parent:['saving','pensionsaving'],cls:'detail-2'},
      {key:'other_financial_asset_saving_allocation',label:'Other financial-asset saving',parent:'saving',cls:'detail',expand:'financialsaving'},
      {key:'training_fund_contributions',label:'Training-fund contributions',parent:['saving','financialsaving'],cls:'detail-2'},
      {key:'provident_fund_contributions',label:'Provident-fund contributions',parent:['saving','financialsaving'],cls:'detail-2'},
      {key:'life_exec_insurance_contributions',label:'Life/executive-insurance contributions',parent:['saving','financialsaving'],cls:'detail-2'},
      {key:'_other_funded',label:'Other observed financial saving',parent:['saving','financialsaving'],cls:'detail-2'},
      {key:'real_estate_transaction_allocation',label:'Real-estate-related saving',parent:'saving',cls:'detail',expand:'realestateallocation'},
      {key:'mortgage_principal',label:'Scheduled mortgage principal repaid (modeled)',parent:['saving','realestateallocation'],cls:'detail-2'},
      {key:'net_property_acquisition_equity',label:'Net cash equity in property transactions',parent:['saving','realestateallocation'],cls:'detail-2',expand:'propertyequity'},
      {key:'real_estate_investment',label:'Gross property acquisitions and improvements',parent:['saving','realestateallocation','propertyequity'],cls:'detail-2',expand:'realestatepurchase'},
      {key:'home_purchase_net',label:'Net home purchase',parent:['saving','realestateallocation','propertyequity','realestatepurchase'],cls:'detail-2'},
      {key:'other_property_purchase',label:'Other property purchase',parent:['saving','realestateallocation','propertyequity','realestatepurchase'],cls:'detail-2'},
      {key:'home_capital_improvements',label:'Home capital improvements',parent:['saving','realestateallocation','propertyequity','realestatepurchase'],cls:'detail-2'},
      {key:'housing_debt_movement',label:'New apartment debt proceeds / repayment',parent:['saving','realestateallocation','propertyequity'],cls:'detail-2'},
      {key:'other_housing_loan_repayment',label:'Other housing-loan payment (principal + interest)',parent:['saving','realestateallocation'],cls:'detail-2'},
      {key:'unclassified_debt_movement',label:'Unclassified real-estate-related debt movement',parent:['saving','realestateallocation'],cls:'detail-2'},
      {key:'community_gemach_saving',label:'Donations / assumed community saving',parent:['saving','realestateallocation'],cls:'detail-2'},
      {key:'identified_transaction_allocation',label:'Total identified wealth-building allocation',parent:'saving',cls:'detail subtotal'},
      {key:'unallocated_transactional_saving',label:'Unallocated saving surplus / deficit',parent:'saving',cls:'detail subtotal'},
      {key:'_pension_share',label:'Pension share of identified saving',fmt:pct,parent:'saving',cls:'detail percent'},
      {key:'savings_rate_pct',label:'Economic / accrual saving rate',fmt:pct,cls:'percent'}
    ];
    statement(flowEl,pnlLines,rows);
    const wealthLines=[{seg:'Assets'},{key:'total_assets_overlay',label:'Total assets ◇',cls:'subtotal',expand:'assets'},{key:'housing_assets_overlay',label:'Housing assets ◇',parent:'assets',cls:'detail',expand:'housingassets'},{key:'primary_home_assets_overlay',label:'Primary residence ◇',parent:['assets','housingassets'],cls:'detail-2'},{key:'additional_property_assets_overlay',label:'Additional properties ◇',parent:['assets','housingassets'],cls:'detail-2'},{key:'land_assets_overlay',label:'Land and other real estate ◇',parent:['assets','housingassets'],cls:'detail-2'},{key:'pension_wealth_overlay',label:'Pension wealth ◇',parent:'assets',cls:'detail'},{key:'other_financial_assets_overlay',label:'Other financial assets ◇',parent:'assets',cls:'detail',expand:'financialassets'},{key:'deposits_savings_overlay',label:'Deposits and savings ◇',parent:['assets','financialassets'],cls:'detail-2'},{key:'nonbank_investments_overlay',label:'Funds, equities and securities ◇',parent:['assets','financialassets'],cls:'detail-2'},{seg:'Liabilities'},{key:'total_debt_overlay',label:'Total liabilities ◇',fmt:outflow,cls:'subtotal',expand:'liabilities'},{key:'mortgage_balance_overlay',label:'Mortgage balance ◇',fmt:outflow,parent:'liabilities',cls:'detail'},{key:'other_debt_overlay',label:'Other debt ◇',fmt:outflow,parent:'liabilities',cls:'detail'},{key:'net_worth_overlay',label:'Net worth ◇',cls:'keyline'}];
    statement(wealthEl,wealthLines,rows); markScrollableTables();
  }
  groupSel.addEventListener('change',draw); draw(); CHART_RENDERERS.push(draw);
})();// ---------------------------------------------------------------------------
// Section 6: methodology stat fill-ins + manifest table
// ---------------------------------------------------------------------------
(function renderMethodology(){
  const hm = META.home_model, pm = META.pension_model;
  document.getElementById('m-r2held').textContent = hm.held_out_r2.toFixed(3);
  document.getElementById('m-r2in').textContent = hm.in_sample_r2.toFixed(3);
  document.getElementById('m-nself').textContent = hm.n_subjective_reported.toLocaleString('en-US');
  document.getElementById('m-ncbs').textContent = hm.n_cbs_estimate_fill.toLocaleString('en-US');
  document.getElementById('m-nmodel').textContent = hm.n_model_imputed.toLocaleString('en-US');
  document.getElementById('m-mean2023').textContent = fmtNIS(pm.mean_2023);
  document.getElementById('m-meanpre').textContent = fmtNIS(pm.mean_2023_before_p99_winsorization);
  document.getElementById('m-macro').textContent = fmtNIS(pm.macro_benchmark_2023);
  document.getElementById('m-nrows').textContent = META.n_rows.toLocaleString('en-US');
  document.getElementById('m-latest').textContent = META.latest_wave;

  let h = `<tr><th>${tr('File')}</th><th>${tr('Description')}</th></tr>`;
  MANIFEST.forEach(r=>{
    const desc = MANIFEST_DESC[r.file] !== undefined ? MANIFEST_DESC[r.file] : r.description;
    h += `<tr><td>${esc(r.file)}</td><td style="text-align:left;">${esc(desc)}</td></tr>`;
  });
  document.getElementById('table-manifest').innerHTML = h;
  markScrollableTables();
})();

// ---------------------------------------------------------------------------
// Mobile <-> desktop chart-geometry breakpoint watcher. Debounced so a drag-
// resize doesn't redraw on every pixel; only actually redraws when MOBILE
// flips (e.g. crossing 640px, or rotating a phone), never on every resize
// event, so it can't double-render. Registered once at script load -- there
// is exactly one 'resize' and one 'orientationchange' listener for the life
// of the page, so there is nothing to leak on repeated calls.
// ---------------------------------------------------------------------------
(function chartResizeWatcher(){
  let resizeTimer = null;
  function handleBreakpointChange(){
    const nowMobile = MOBILE_MQ.matches;
    // Table-scroll affordance can change on ANY resize (window width, table
    // reflow) independent of the chart-geometry breakpoint, so it's checked
    // unconditionally every time, unlike the chart redraw below.
    markScrollableTables();
    if(nowMobile === MOBILE) return; // didn't cross the breakpoint -- nothing to redraw
    MOBILE = nowMobile;
    CHART_RENDERERS.forEach(draw=>{
      try{ draw(); } catch(err){ console.error('Chart redraw failed on breakpoint change:', err); }
    });
  }
  function scheduleCheck(){
    if(resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(handleBreakpointChange, 150);
  }
  window.addEventListener('resize', scheduleCheck);
  window.addEventListener('orientationchange', scheduleCheck);
})();

// ---------------------------------------------------------------------------
// Top nav: scrollspy + click-to-scroll
// ---------------------------------------------------------------------------
(function nav(){
  const buttons = [...document.querySelectorAll('nav.topnav button')];
  const sections = buttons.map(b=>document.getElementById('sec-'+b.dataset.sec));
  buttons.forEach((b,i)=>b.addEventListener('click', ()=>{
    sections[i].scrollIntoView({behavior:'smooth', block:'start'});
  }));
  const io = new IntersectionObserver((entries)=>{
    entries.forEach(en=>{
      if(en.isIntersecting){
        const idx = sections.indexOf(en.target);
        buttons.forEach(b=>b.classList.remove('active'));
        if(idx>=0) buttons[idx].classList.add('active');
      }
    });
  }, {rootMargin:'-30% 0px -60% 0px', threshold:0});
  sections.forEach(s=>io.observe(s));
})();

document.addEventListener('pointermove', moveTooltip);
</script>
<style>.hb-home-back{position:fixed;top:14px;left:14px;z-index:99999;background:#0f172a;color:#fff;padding:9px 16px;border-radius:999px;text-decoration:none;font-family:'Heebo',system-ui,-apple-system,sans-serif;font-size:14px;font-weight:600;box-shadow:0 4px 14px rgba(0,0,0,.28);display:inline-flex;align-items:center;gap:6px}.hb-home-back .hb-lbl{display:inline}@media(max-width:480px){.hb-home-back{padding:0;width:40px;height:40px;gap:0;justify-content:center;border-radius:50%;font-size:18px}.hb-home-back .hb-lbl{display:none}}</style>
<a href="../index.html" class="hb-home-back" aria-label="__HOME_LABEL__">🏠<span class="hb-lbl">__HOME_LABEL__</span></a>
</body>
</html>
"""

if __name__ == "__main__":
    main()








