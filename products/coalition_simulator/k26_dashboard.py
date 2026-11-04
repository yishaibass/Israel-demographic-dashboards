"""Render the separately fitted K26 comparison without changing simulator defaults."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path


def add_comparison(page: str, root: Path, language: str = "en") -> str:
    source = root / "data/processed/k26/current/model_comparison/six_group_comparison.json"
    if not source.exists():
        return page
    data = json.loads(source.read_text(encoding="utf-8"))
    readiness = json.loads((source.parents[1] / "readiness.json").read_text(encoding="utf-8"))
    baseline_hash = hashlib.sha256((root / "data/processed/params_groups.csv").read_bytes()).hexdigest()
    if data["snapshot_id"] != readiness["snapshot_id"] or data["baseline"]["sha256"] != baseline_hash:
        raise ValueError("K26 comparison is stale: rerun scripts/fit_k26_comparison.py")
    he = language == "he"
    labels = ({"haredi": "חרדים", "dati": "דתי־לאומי", "masorti": "מסורתיים", "hiloni": "חילונים", "arab": "ערבים", "druze": "דרוזים"} if he else
              {"haredi": "Haredi", "dati": "Dati", "masorti": "Masorti", "hiloni": "Hiloni", "arab": "Arab", "druze": "Druze"})
    headings = (["קבוצה / מדד", "ממוצע 2019–2022", "2022", "אומדן 2026", "שינוי מהממוצע (נ״א)", "שינוי מ־2022 (נ״א)", "שינוי בסטיות תקן", "טווח בוטסטרפ 10–90%"] if he else
                ["Group / parameter", "2019–2022 mean", "2022", "2026 estimate", "Δ mean (pp)", "Δ 2022 (pp)", "Δ historical SD", "Bootstrap 10–90%"])
    rows = []
    for row in data["rows"]:
        parameter = (("הצבעה" if row["parameter"] == "theta_vote" else "תמיכה בגוש נתניהו") if he else
                     ("Turnout" if row["parameter"] == "theta_vote" else "Netanyahu support / all voters"))
        z = row["change_vs_mean_sd"]
        sd = ("לא מוגדר" if he else "Undefined") if z is None else f"{z:+.2f} SD"
        boundary = " *" if row["at_lower_bound"] or row["at_upper_bound"] else ""
        cells = [f"{labels[row['group']]} · {parameter}", f"{100*row['historical_mean']:.1f}%", f"{100*row['k25_estimate']:.1f}%",
                 f"{100*row['k26_estimate']:.1f}%{boundary}", f"{row['change_vs_mean_pp']:+.2f}", f"{row['change_vs_k25_pp']:+.2f}", sd,
                 f"{100*row['bootstrap_p10']:.1f}–{100*row['bootstrap_p90']:.1f}%"]
        rows.append('<tr>' + ''.join(f'<td>{html.escape(cell)}</td>' for cell in cells) + '</tr>')
    title = "2026 לעומת ברירת המחדל" if he else "2026 versus the default"
    button = "החל את אומדני הקבוצות ל־2026" if he else "Apply 2026 group estimates"
    explanation = ("ברירת המחדל נשארת ממוצע חמש הבחירות 2019–2022. השינוי בסטיות תקן מתאר פיזור היסטורי (אוכלוסייה, ddof=0), ואינו מובהקות סטטיסטית. כשסטיית התקן כמעט אפס, השינוי אינו מוגדר. * מקדם בגבול האילוץ." if he else
                   "Defaults remain the five-election 2019–2022 average. SD changes describe historical dispersion (population SD, ddof=0), not statistical significance. Near-zero historical SD is undefined. * Coefficient at a fitted bound.")
    method = ("אומדנים אקולוגיים בהרכב יישובים מ־2022 שנישא קדימה. קולות מעטפות חיצוניות אינם משויכים לקבוצות. טווחי הבוטסטרפ מותנים בהנחת ההרכב. החלת האומדנים משנה רק את התנהגות הקבוצות; הרכב האוכלוסייה וקולות מתחת לחסימה נשארים לפי בחירתכם. המנדטים הם תרחיש מודל." if he else
              "Ecological estimates carry 2022 locality composition forward. External-envelope votes are excluded from group fitting. Bootstrap intervals condition on this composition. Applying estimates changes group behavior; electorate and wasted-vote settings remain your scenario assumptions. Simulated seats are a model scenario.")
    complete = readiness.get("counted_station_fraction", 0) >= 1 and not readiness.get("external_envelopes_pending", True)
    provenance = (("ספירה מלאה מדווחת" if complete else "ספירה חלקית") + " לפי מקור משני; קובץ ועדת הבחירות הישיר והאישור המשפטי לא אומתו." if he else
                  ("Full count reported" if complete else "Partial count reported") + " by a secondary mirror; the direct CEC export and legal certification have not been independently verified.")
    fit = data["fit"]
    coverage = data["coverage"]
    detail = (f"R² turnout = {fit['turnout_r2']:.3f} (weak fit); R² coalition / eligible = {fit['coalition_per_eligible_r2']:.3f}. "
              f"Matched regular eligible voters: {100*coverage['k26_matched_eligible_fraction']:.2f}%; "
              f"national valid votes represented in fit: {100*coverage['k26_matched_valid_vote_fraction_national']:.2f}%.")
    observed = data["national_observed"]
    national = (("נתון ארצי: " if he else "National count: ") + f"{observed['valid_votes']:,}" +
                (" קולות כשרים; גוש נתניהו " if he else " valid votes; Netanyahu bloc ") + f"{100*observed['coalition_share_valid']:.2f}%" +
                (" מהכשרים, כולל עמך ישראל." if he else " of valid votes, including Amcha Israel."))
    section = f'''<section class="card k26-comparison" id="k26Comparison">
<h2>{title}</h2><p>{explanation}</p>
<p><strong>{national}</strong></p><p>{provenance}</p>
<p dir="ltr">{html.escape(data['snapshot_id'])} · {html.escape(data['as_of_utc'])}<br>{detail}</p>
<div style="overflow-x:auto"><table><thead><tr>{''.join('<th>'+x+'</th>' for x in headings)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>{method}</p><button type="button" class="clear-btn" id="loadK26">{button}</button>
<p><a href="https://raw.githubusercontent.com/harelc/elections-vote-transfer/master/ballot26.csv">{'מקור ספירת הקולות' if he else 'Count source'}</a></p>
</section>'''
    # Keep the comparison alongside the interactive scenario, before historical tabs.
    anchor = '  <div id="groupRows"></div>'
    if page.count(anchor) != 1:
        raise ValueError("Simulator comparison anchor is missing or duplicated")
    page = page.replace(anchor, section + '\n' + anchor, 1)
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
  computeSeats();
});
</script>'''.replace("__K26_ROWS__", payload)
    return page.replace("</body>", script + "\n</body>", 1)
