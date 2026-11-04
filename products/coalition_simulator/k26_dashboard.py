"""Render the separately fitted K26 comparison without changing simulator defaults."""
from __future__ import annotations

import hashlib
import html
import json
import csv
from pathlib import Path


def add_comparison(page: str, root: Path, language: str = "en") -> str:
    source = root / "data/processed/k26/current/model_comparison/six_group_comparison.json"
    if not source.exists():
        return page
    data = json.loads(source.read_text(encoding="utf-8"))
    readiness = json.loads((source.parents[1] / "readiness.json").read_text(encoding="utf-8"))
    baseline_path = root / "data/processed/params_groups.csv"
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    if data["snapshot_id"] != readiness["snapshot_id"] or data["baseline"]["sha256"] != baseline_hash:
        raise ValueError("K26 comparison is stale: rerun scripts/fit_k26_comparison.py")
    he = language == "he"
    labels = ({"haredi": "חרדים", "dati": "דתי־לאומי", "masorti": "מסורתיים", "hiloni": "חילונים", "arab": "ערבים", "druze": "דרוזים"} if he else
              {"haredi": "Haredi", "dati": "Dati", "masorti": "Masorti", "hiloni": "Hiloni", "arab": "Arab", "druze": "Druze"})
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        population = {row["group"]: row for row in csv.DictReader(stream)}
    with (root / "data/processed/config.csv").open(encoding="utf-8-sig", newline="") as stream:
        configuration = {row["key"]: row["value"] for row in csv.DictReader(stream)}
    years = float(configuration["projection_years"])
    headings = (["קבוצה", "חלק מבעלי זכות ההצבעה · תחזית 2026", "הצבעה · ממוצע חמש הבחירות", "הצבעה · אומדן 2026 (ס״ת)", "תמיכה בגוש נתניהו · ממוצע חמש הבחירות", "תמיכה בגוש נתניהו · אומדן 2026 (ס״ת)"] if he else
                ["Group", "Eligible-voter share · 2026 projection", "Turnout · past five elections average", "Turnout · 2026 estimate (SD)", "Netanyahu support · past five elections average", "Netanyahu support · 2026 estimate (SD)"])
    def percent(value):
        return f"{value * 100:.0f}%"
    def estimate(row):
        z = row["change_vs_mean_sd"]
        return percent(row["k26_estimate"]) + (" (—)" if z is None else f" ({z:+.1f} SD)")
    by_group = {}
    for row in data["rows"]:
        by_group.setdefault(row["group"], {})[row["parameter"]] = row
    rows = []
    for group, parameters in by_group.items():
        turnout, support = parameters["theta_vote"], parameters["theta_bibi"]
        base = float(population[group]["eligible_share"])
        projected = float(population[group]["eligible_share_2026"])
        change = round((projected / base - 1) * 100)
        relative = f"{change:+d}%" if change else "0%"
        cells = [labels[group], f"{percent(projected)} ({relative})", percent(turnout["historical_mean"]),
                 estimate(turnout), percent(support["historical_mean"]), estimate(support)]
        rows.append('<tr><td>' + html.escape(cells[0]) + '</td>' + ''.join(f'<td dir="ltr">{html.escape(cell)}</td>' for cell in cells[1:]) + '</tr>')
    title = "2026 מול ממוצע חמש הבחירות הקודמות" if he else "2026 vs past five elections average"
    tab_label = "2026 מול חמש הבחירות הקודמות" if he else "2026 vs past five elections"
    button = "טען אומדני 2026" if he else "Load 2026 estimates"
    note = (f"ממוצע: 2019–2022. חלק מבעלי זכות ההצבעה: בסיס מפקד 2022 ובחירות 2022, תחזית צמיחה ל־{years:g} שנים; בסוגריים שינוי יחסי בחלק הקבוצה לעומת הבסיס. אומדני תמיכה מתוך המצביעים, כולל פסולים. —: ס״ת אינה מוגדרת." if he else
            f"Average: 2019–2022. Eligible shares: 2022 Census/K25 base, {years:g}-year growth projection; parentheses show relative change in share from that base. Support: among voters, including invalid ballots. —: SD undefined.")
    technical = {"snapshot_id": data["snapshot_id"], "count_status": readiness["count_status"], "fit": data["fit"],
                 "coverage": data["coverage"], "limitations": data["limitations"],
                 "population_basis": "K25 eligible rolls and 2022 Census composition", "projection_years": years}
    section = f'''<section class="tab-panel" id="tab-k26">
<div class="card k26-comparison" id="k26Comparison">
<h2>{title}</h2>
<div class="table-scroll-wrap"><div class="table-scroll"><table><thead><tr>{''.join('<th>'+x+'</th>' for x in headings)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<p class="hint">{note}</p>
<button type="button" class="clear-btn" id="loadK26">{button}</button>
</div>
</section>'''
    page = page.replace('</head>', '<style>.tabs-bar{flex-wrap:wrap}.k26-comparison table{min-width:760px}.k26-comparison th{white-space:normal}</style>\n</head>', 1)
    # Append before the existing JS binds tabs; publication adds demographics before this last tab.
    tabs_end = page.index("</div>", page.index('<div class="tabs-bar">'))
    tab = f'  <button class="tab-btn" data-tab="k26">{tab_label}</button>\n'
    page = page[:tabs_end] + tab + page[tabs_end:]
    if page.count('<footer class="foot">') != 1:
        raise ValueError("Simulator comparison footer anchor missing or duplicated")
    page = page.replace('<footer class="foot">', section + '\n<footer class="foot">', 1)
    payload = json.dumps(data["rows"], ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    script = '''<script>
document.getElementById('loadK26').addEventListener('click', function () {
  activeReference = 'K26';
  const rows = __K26_ROWS__;
  rows.forEach(function (row) {
    const kind = row.parameter === 'theta_vote' ? 't' : 'k';
    sdState[kind + '_' + row.group] = null;
    setControlValue(kind, row.group, row.k26_estimate, null);
  });
  document.querySelectorAll('.load-btn').forEach(function (b) { b.classList.remove('active'); });
  document.querySelector('[data-tab="scenario"]').click();
  computeSeats();
});
if (window.location.hash === '#2026') document.querySelector('[data-tab="k26"]').click();
</script>'''.replace("__K26_ROWS__", payload)
    page = page.replace("</body>", '<script type="application/json" id="k26ModelMetadata">' + json.dumps(technical, ensure_ascii=False, allow_nan=False).replace("</", "<\\/") + '</script>\n</body>', 1)
    return page.replace("</body>", script + "\n</body>", 1)
