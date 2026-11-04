"""Publish one six-election model comparison alongside the frozen published reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

KEYS = ["pop_group", "religiosity", "income_band", "age_band", "academic", "district"]
BEGIN = "<!-- K26_COMPARISON_BEGIN -->"
END = "<!-- K26_COMPARISON_END -->"


def merge_cube(baseline: dict, current: dict) -> dict:
    """Match demographic cells explicitly; fiscal inputs remain historical."""
    old_index = {name: index for index, name in enumerate(baseline["columns"])}
    new_index = {name: index for index, name in enumerate(current["columns"])}
    required = KEYS + ["weight"] + [name for name in baseline["columns"] if name.endswith("_mass")]
    if set(required) - new_index.keys():
        raise ValueError("K26 cube is missing required population or probability columns")
    def key(row, index):
        return tuple(float(row[index[name]]) for name in KEYS)
    lookup = {key(row, new_index): row for row in current["rows"]}
    if len(lookup) != len(current["rows"]) or len(baseline["rows"]) != len(lookup):
        raise ValueError("Duplicate or unequal K26/historical cube cells")
    rows = []
    for old in baseline["rows"]:
        new = lookup.get(key(old, old_index))
        if new is None:
            raise ValueError("K26 cube does not cover every historical demographic cell")
        if abs(new[new_index["weight"]] - old[old_index["weight"]]) > .01:
            raise ValueError("K26 cube changes the frozen population weights")
        masses = [new[new_index[name]] for name in required if name.endswith("_mass")]
        if any(value < 0 for value in masses) or abs(sum(masses) - new[new_index["weight"]]) > max(.01, new[new_index["weight"]] * 1e-5):
            raise ValueError("K26 cube probabilities fail mass conservation")
        row = list(old)
        for name in required:
            row[old_index[name]] = new[new_index[name]]
        rows.append(row)
    return {"columns": baseline["columns"], "rows": rows}


def integrate(page: str, cube: dict, audit: dict, historical: dict) -> str:
    if cube["snapshot_id"] != audit["snapshot_id"]:
        raise ValueError("K26 cube and fit audit snapshots differ")
    fit_id = cube.get("model_fit_id")
    if not fit_id or audit.get("model_fit_id") != fit_id:
        raise ValueError("K26 cube and audit must come from the same fitted model")
    page = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", page, flags=re.DOTALL)
    start = page.index("const CUBE=") + len("const CUBE=")
    baseline, _ = json.JSONDecoder().raw_decode(page[start:])
    current = merge_cube(baseline, cube)
    elections = ["K21", "K22", "K23", "K24", "K25"]
    if set(historical) != set(elections):
        raise ValueError("Historical comparison requires all five same-design election cubes")
    mass_columns = [name for name in baseline["columns"] if name.endswith("_mass")]
    indices = [baseline["columns"].index(name) for name in mass_columns]
    history_masses = {}
    for election in elections:
        if historical[election]["snapshot_id"] != cube["snapshot_id"]:
            raise ValueError("Historical and K26 comparison fits have different snapshots")
        if historical[election].get("model_fit_id") != fit_id:
            raise ValueError("Historical and K26 comparison cubes come from different fitted models")
        matched = merge_cube(baseline, historical[election])
        history_masses[election] = [[row[index] for index in indices] for row in matched["rows"]]
    baseline_hash = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    panel = '''<div class="card" id="k26ModelPanel">
<label for="modelElection"><strong>תקופת המודל: </strong></label>
<select id="modelElection"><option value="historical">ברירת המחדל שפורסמה — ממוצע 2019–2022</option><option value="K26">2026 — אומדן מהמודל המשותף</option></select>
<p>אותו מודל מותאם לכל שש הבחירות: אותם משתנים, אוכלוסיית מפקד 2022, יעדים ברמת יישוב ופונקציית התאמה. לכל בחירה אומדנים משלה. ברירת המחדל שפורסמה נשמרת כנקודת ייחוס.</p>
<p>סיווג 2026: ישר וביחד — מרכז ליברלי; הדמוקרטים — שמאל ציוני. רשימות קטנות אחרות מוצגות ב״אחרות״. נתוני המסים וההעברות נשארים נתונים היסטוריים.</p>
<p id="k26ModelStatus"></p>
<p id="k26FitCoverage"></p>
<details><summary>השוואת הקבוצה שנבחרה: שינוי לאורך זמן ושינוי באומדן</summary>
<div style="overflow-x:auto"><table style="width:100%"><thead><tr><th>מדד</th><th>ברירת המחדל שפורסמה</th><th>ממוצע 2019–2022 באותו מודל</th><th>אומדן 2026</th><th>שינוי מהממוצע באותו מודל (נ״א)</th><th>שינוי בס״ת</th></tr></thead><tbody id="k26FamilyComparison"></tbody></table></div>
<p>ממוצע וסטיית תקן מחושבים מחמשת שיעורי הבחירות ההיסטוריות בקבוצה שנבחרה, באותה התאמה משותפת. השינוי בס״ת משווה את 2026 לממוצע הזה; הוא אינו מובהקות סטטיסטית. זוהי התאמה בדיעבד המשתמשת גם בנתוני 2026. הפער מול ברירת המחדל עשוי לכלול גם עדכון של האמידה.</p></details>
</div>'''
    marker = '<section class="analysis-section">'
    if marker not in page:
        raise ValueError("Demographic explorer publication anchor missing")
    page = page.replace(marker, BEGIN + panel + END + marker, 1)
    # Both cubes use identical dimensions. Keep the original serialized CUBE intact.
    script = r'''<script>
const K26_FROZEN_ROWS=CUBE.rows;
const K26_FITTED_CUBE=__CUBE__;
const K26_HISTORICAL_MASSES=__HISTORY__;
const K26_MASS_COLUMNS=__MASS_COLUMNS__;
const K26_FIT_METADATA=__META__;
const K26_HISTORICAL_PARTY_NAMES=P.map(p=>p[1]);
const k26Intro=document.querySelector('.top p'),k26GroupFootnote=document.querySelector('.group-footnote p');
const k26HistoricalIntro=k26Intro.textContent,k26HistoricalFootnote=k26GroupFootnote.innerHTML;
const K26_PARTY_NAMES={shas:'ש״ס',utj:'יהדות התורה',likud:'ליכוד',national_religious:'הציונות הדתית–זהות, עוצמה יהודית ונעם',center:'ישר וביחד',zionist_left:'הדמוקרטים',secular_national_right:'ישראל ביתנו',arab:'הרשימה המשותפת ורע״ם',other:'שאר הרשימות, כולל עמך ישראל, הציבור החרדי והמילואימניקים'};
function renderK26Comparison(){
  const selections=Object.entries(F).map(([key,element])=>[ci[key],sel(element)]);
  const selected=K26_FROZEN_ROWS.map((row,index)=>selections.every(([column,allowed])=>allowed.has(String(row[column])))?index:-1).filter(index=>index>=0);
  function values(rows,compact=false){
    let weight=0,abstain=0,masses=Array(P.length).fill(0);
    const column=name=>compact?K26_MASS_COLUMNS.indexOf(name):ci[name];
    const abstainColumn=column('abstain_mass'),partyColumns=P.map(p=>column(p[0]+'_mass'));
    for(const index of selected){
      const row=rows[index];weight+=K26_FROZEN_ROWS[index][ci.weight];abstain+=row[abstainColumn];
      partyColumns.forEach((position,j)=>masses[j]+=row[position]);
    }
    const totalVoted=weight-abstain;
    return [weight?100*totalVoted/weight:null,...masses.map(mass=>totalVoted?100*mass/totalVoted:null)];
  }
  const old=values(K26_FROZEN_ROWS),current=values(K26_FITTED_CUBE.rows),names=['שיעור הצבעה',...P.map(p=>p[2])];
  const history=Object.values(K26_HISTORICAL_MASSES).map(rows=>values(rows,true));
  const fmt=v=>v===null?'—':v.toFixed(2)+'%';
  document.getElementById('k26FamilyComparison').innerHTML=names.map((name,i)=>{
    const available=history.every(e=>e[i]!==null)&&current[i]!==null;
    const mean=available?history.reduce((sum,e)=>sum+e[i],0)/history.length:null;
    const sd=available?Math.sqrt(history.reduce((sum,e)=>sum+(e[i]-mean)**2,0)/history.length):null;
    const delta=available?current[i]-mean:null,z=available&&sd>=.0001?delta/sd:null;
    const signed=value=>value===null?'—':(value>=0?'+':'')+value.toFixed(2);
    return '<tr><td>'+name+'</td><td>'+fmt(old[i])+'</td><td>'+fmt(mean)+'</td><td>'+fmt(current[i])+'</td><td dir="ltr">'+signed(delta)+'</td><td dir="ltr">'+signed(z)+'</td></tr>';
  }).join('');
}
const k26OriginalRenderDemo=renderDemo;
renderDemo=function(){k26OriginalRenderDemo();renderK26Comparison();};
const selector=document.getElementById('modelElection');
function refreshK26PartyLabels(){
  [A,B].forEach(el=>{
    el.querySelectorAll('input').forEach(input=>{
      const party=P.find(p=>p[0]===input.value);
      if(party){input.dataset.label=party[1];input.nextElementSibling.textContent=party[1];}
    });
    const checked=[...el.querySelectorAll('input:checked')],n=el.querySelectorAll('input').length;
    el.querySelector('.trigger').textContent=checked.length===n?'הכול':checked.length===0?'לא נבחר':checked.length===1?checked[0].dataset.label:checked.length+' נבחרו';
  });
}
selector.addEventListener('change',()=>{
  CUBE.rows=selector.value==='K26'?K26_FITTED_CUBE.rows:K26_FROZEN_ROWS;
  P.forEach((p,i)=>p[1]=selector.value==='K26'?K26_PARTY_NAMES[p[0]]:K26_HISTORICAL_PARTY_NAMES[i]);
  k26Intro.textContent=selector.value==='K26'?'אומדני 2026 ממודל אחד לכל שש הבחירות, עם משתנים ואופן התאמה משותפים, על בסיס מפקד 2022. אלו הסתברויות מודליות ולא קולות פרטניים.':k26HistoricalIntro;
  k26GroupFootnote.innerHTML=selector.value==='K26'?'<strong>קבוצות המודל ב־2026:</strong> '+P.map(p=>p[2]+' — '+p[1]).join(' · '):k26HistoricalFootnote;
  refreshK26PartyLabels();
  renderDemo();renderParty();
});
document.getElementById('filters').addEventListener('change',renderK26Comparison);
document.querySelectorAll('#filters [data-x]').forEach(button=>button.addEventListener('click',renderK26Comparison));
document.getElementById('k26ModelStatus').textContent=K26_FIT_METADATA.status;
document.getElementById('k26FitCoverage').textContent=K26_FIT_METADATA.coverage;
renderK26Comparison();
</script>'''
    metadata = {"snapshot_id": cube["snapshot_id"], "model_fit_id": fit_id, "baseline_sha256": baseline_hash,
                "coverage": f"{audit['matched_localities']} יישובים בהתאמה · {100*audit['national_valid_vote_coverage']:.2f}% מהקולות הכשרים הארציים. המודל אינו מכייל את התוצאות לסכום הארצי.",
                "status": f"{cube['snapshot_id']} · מקור ספירה משני; האימות הישיר והאישור המשפטי לא אומתו. אומדנים אקולוגיים, אינם קולות פרטניים."}
    script = script.replace("__CUBE__", json.dumps(current, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    script = script.replace("__HISTORY__", json.dumps(history_masses, separators=(",", ":"), allow_nan=False))
    script = script.replace("__MASS_COLUMNS__", json.dumps(mass_columns))
    script = script.replace("__META__", json.dumps(metadata, ensure_ascii=False).replace("</", "<\\/"))
    return page.replace("</body>", BEGIN + script + END + "</body>", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--historical-cubes", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    page = integrate(args.html.read_text(encoding="utf-8"), json.loads(args.cube.read_text(encoding="utf-8")), json.loads(args.audit.read_text(encoding="utf-8")), json.loads(args.historical_cubes.read_text(encoding="utf-8")))
    (args.output or args.html).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
