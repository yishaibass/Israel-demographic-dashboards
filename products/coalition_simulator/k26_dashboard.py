"""Render the separately fitted K26 comparison without changing simulator defaults."""
from __future__ import annotations

import hashlib
import html
import json
import csv
from pathlib import Path
from .model.k26_bloc import canonical_names, coalition_symbols, contract_sha256, load_contract


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
    contract_hash = contract_sha256()
    if any(value.get("party_bloc_contract_sha256") != contract_hash for value in (data, readiness)):
        raise ValueError("K26 coalition definition changed: rebuild prepared results and refit the comparison")
    he = language == "he"
    labels = ({"haredi": "חרדים", "dati": "דתי־לאומי", "masorti": "מסורתיים", "hiloni": "חילונים", "arab": "ערבים", "druze": "דרוזים"} if he else
              {"haredi": "Haredi", "dati": "Dati", "masorti": "Masorti", "hiloni": "Hiloni", "arab": "Arab", "druze": "Druze"})
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        population = {row["group"]: row for row in csv.DictReader(stream)}
    with (root / "data/processed/config.csv").open(encoding="utf-8-sig", newline="") as stream:
        configuration = {row["key"]: row["value"] for row in csv.DictReader(stream)}
    years = float(configuration["projection_years"])
    raw_shares = {group: float(row["eligible_share"]) * (1 + float(configuration["cagr_" + group])) ** years
                  for group, row in population.items()}
    raw_total = sum(raw_shares.values())
    if any(abs(float(population[group]["eligible_share_2026"]) - value / raw_total) > 1e-12
           for group, value in raw_shares.items()):
        raise ValueError("Eligible-voter projections differ from the model growth configuration")
    with (source.parents[1] / "party_actuals.csv").open(encoding="utf-8-sig", newline="") as stream:
        parties = list(csv.DictReader(stream))
    if not parties or {row["snapshot_id"] for row in parties} != {data["snapshot_id"]}:
        raise ValueError("K26 party actuals and model comparison snapshots differ")
    assigned = {row["ballot_symbol"] for row in parties if row["bloc_core"] == "Netanyahu bloc"}
    if assigned != coalition_symbols():
        raise ValueError("K26 party tagging differs from the reviewed coalition definition")
    names = canonical_names()
    if any(row["ballot_symbol"] in names and row["party_id"] != names[row["ballot_symbol"]] for row in parties):
        raise ValueError("K26 coalition party names differ from the reviewed definition")
    valid_totals = {int(row["valid_votes"]) for row in parties}
    if len(valid_totals) != 1 or sum(int(row["votes"]) for row in parties) != next(iter(valid_totals)):
        raise ValueError("K26 party counts do not reconcile to national valid votes")
    valid = next(iter(valid_totals))
    if valid <= 0:
        raise ValueError("K26 valid-vote total must be positive")
    lost = {"w_c": 0, "w_o": 0}
    for party in parties:
        votes = int(party["votes"])
        if votes * 100 < valid * float(configuration["threshold_pct"]):
            coalition = party["bloc_core"] == "Netanyahu bloc" or party["amcha_included_in_coalition"].lower() == "true"
            lost["w_c" if coalition else "w_o"] += votes
    thresholds = {key: votes / valid for key, votes in lost.items()}
    headings = (["קבוצה", "חלק מבעלי זכות ההצבעה · תחזית 2026 (נ״א מול בסיס 2022)", "שיעור הצבעה · ממוצע חמש הבחירות", "שיעור הצבעה · אומדן 2026 (ס״ת מול הממוצע)", "תמיכה בגוש נתניהו · ממוצע חמש הבחירות", "תמיכה בגוש נתניהו · אומדן 2026 (ס״ת מול הממוצע)"] if he else
                ["Group", "Eligible-voter share · 2026 projection (pp vs 2022 base)", "Turnout · past five elections average", "Turnout · 2026 estimate (SD vs average)", "Netanyahu support · past five elections average", "Netanyahu support · 2026 estimate (SD vs average)"])
    def percent(value):
        return f"{value * 100:.0f}%"
    def estimate(row):
        z = row["change_vs_mean_sd"]
        return percent(row["k26_estimate"]) + ("" if z is None else f" ({z:+.1f} SD)")
    by_group = {}
    for row in data["rows"]:
        by_group.setdefault(row["group"], {})[row["parameter"]] = row
    rows = []
    for group, parameters in by_group.items():
        turnout, support = parameters["theta_vote"], parameters["theta_bibi"]
        base = float(population[group]["eligible_share"])
        projected = float(population[group]["eligible_share_2026"])
        change = round((projected - base) * 100, 1)
        points = f"{change:+.1f} pp" if change else "0.0 pp"
        cells = [labels[group], f"{percent(projected)} ({points})", percent(turnout["historical_mean"]),
                 estimate(turnout), percent(support["historical_mean"]), estimate(support)]
        rows.append('<tr><td>' + html.escape(cells[0]) + '</td>' + ''.join(f'<td dir="ltr">{html.escape(cell)}</td>' for cell in cells[1:]) + '</tr>')
    title = "2026 מול ממוצע חמש הבחירות הקודמות" if he else "2026 vs past five elections average"
    tab_label = "2026 מול חמש הבחירות הקודמות" if he else "2026 vs past five elections"
    button = "טען אומדני 2026" if he else "Load 2026 estimates"
    technical = {"snapshot_id": data["snapshot_id"], "count_status": readiness["count_status"], "fit": data["fit"],
                 "party_bloc_definition_id": load_contract()["definition_id"], "party_bloc_contract_sha256": contract_hash,
                 "coverage": data["coverage"], "limitations": data["limitations"],
                 "population_basis": "K25 eligible rolls and 2022 Census composition", "projection_years": years,
                 "population_reweighted_from_2026_locality_counts": False,
                 "behavior_fitted_to_2026_locality_counts": True,
                 "eligible_share_change_unit": "percentage points vs K25/2022 base",
                 "support_denominator": "all voters including invalid ballots",
                 "population": {group: {"base_share": float(row["eligible_share"]),
                                        "projected_share_2026": float(row["eligible_share_2026"]),
                                        "annual_growth_assumption": float(configuration["cagr_" + group])}
                                for group, row in population.items()},
                 "thresholds": thresholds}
    section = f'''<section class="tab-panel" id="tab-k26">
<div class="card k26-comparison" id="k26Comparison">
<h2>{title}</h2>
<div class="table-scroll-wrap"><div class="table-scroll"><table><thead><tr>{''.join('<th>'+x+'</th>' for x in headings)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<button type="button" class="clear-btn" id="loadK26">{button}</button>
</div>
</section>'''
    page = page.replace('</head>', '<style>.tabs-bar{flex-wrap:wrap}.k26-comparison table{min-width:760px}.k26-comparison th{white-space:normal}.foot:empty{display:none}html[lang="en"] #partiesTable[data-election="K26"] th:nth-child(2),html[lang="en"] #partiesTable[data-election="K26"] td:nth-child(2){display:none}</style>\n</head>', 1)
    # Append before the existing JS binds tabs; publication adds demographics before this last tab.
    tabs_end = page.index("</div>", page.index('<div class="tabs-bar">'))
    tab = f'  <button class="tab-btn" data-tab="k26">{tab_label}</button>\n'
    page = page[:tabs_end] + tab + page[tabs_end:]
    if page.count('<footer class="foot">') != 1:
        raise ValueError("Simulator comparison footer anchor missing or duplicated")
    page = page.replace('<footer class="foot">', section + '\n<footer class="foot">', 1)
    previous_election = '<button class="load-btn" data-election="K25">2022</button>'
    if page.count(previous_election) != 1:
        raise ValueError("Simulator election-loading row anchor missing or duplicated")
    page = page.replace(previous_election, '<button class="load-btn" data-election="K26">2026</button>\n    ' + previous_election, 1)
    payload = json.dumps(data["rows"], ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    script = '''<script>
DATA.election_labels.K26 = '2026';
DATA.thresholds.K26 = __K26_THRESHOLDS__;
DATA.parties.K26 = __K26_PARTIES__;
const k26PartyOption = document.createElement('option');
k26PartyOption.value = 'K26';
k26PartyOption.textContent = '2026';
partySelect.insertBefore(k26PartyOption, partySelect.firstChild);
partySelect.value = 'K26';
renderParties('K26');
__K26_ROWS__.forEach(function (row) {
  const key = row.parameter === 'theta_vote' ? 'theta_vote_by_election' : 'theta_bibi_by_election';
  DATA.groups[row.group][key].K26 = row.k26_estimate;
});
document.getElementById('loadK26').addEventListener('click', function () {
  document.querySelector('[data-tab="scenario"]').click();
  document.querySelector('.load-btn[data-election="K26"]').click();
});
if (window.location.hash === '#2026') document.querySelector('[data-tab="k26"]').click();
</script>'''.replace("__K26_ROWS__", payload).replace("__K26_THRESHOLDS__", json.dumps(thresholds, allow_nan=False))
    party_rows = [{"name": party["party_name_he"] if he or party["party_id"].startswith("K26:") else party["party_id"],
                   "bloc": "Coalition" if party["bloc_core"] == "Netanyahu bloc" or party["amcha_included_in_coalition"].lower() == "true" else "Opposition",
                   "share": int(party["votes"]) / valid, "letter": party["ballot_symbol"], "leader": ""}
                  for party in sorted(parties, key=lambda row: int(row["votes"]), reverse=True)]
    script = script.replace("__K26_PARTIES__", json.dumps(party_rows, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    page = page.replace("</body>", '<script type="application/json" id="k26ModelMetadata">' + json.dumps(technical, ensure_ascii=False, allow_nan=False).replace("</", "<\\/") + '</script>\n</body>', 1)
    rendered = page.replace("</body>", script + "\n</body>", 1)
    return "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
