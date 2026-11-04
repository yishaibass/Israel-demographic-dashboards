"""Publish one six-election model comparison alongside the frozen published reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

KEYS = ["pop_group", "religiosity", "income_band", "age_band", "academic", "district"]
ELECTIONS = ["K21", "K22", "K23", "K24", "K25", "K26"]
ELECTION_LABELS = ["אפריל 2019", "ספטמבר 2019", "2020", "2021", "2022", "2026"]
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


def integrate(page: str, cube: dict, audit: dict, historical: dict,
              independent: dict | None = None, independent_audit: dict | None = None) -> str:
    if cube["snapshot_id"] != audit["snapshot_id"]:
        raise ValueError("K26 cube and fit audit snapshots differ")
    fit_id = cube.get("model_fit_id")
    if not fit_id or audit.get("model_fit_id") != fit_id:
        raise ValueError("K26 cube and audit must come from the same fitted model")
    page = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", page, flags=re.DOTALL)
    page = re.sub(r'(<header class="top">.*?<h1>.*?</h1>)<p>.*?</p>', r'\1', page, count=1, flags=re.DOTALL)
    page = re.sub(r'(<div class="group-footnote">)<p>.*?</p>', r'\1', page, count=1, flags=re.DOTALL)
    page = page.replace('שיעור הצבעה חזוי', 'שיעור הצבעה')
    page = page.replace('ימין חילוני', 'ליברמן')
    start = page.index("const CUBE=") + len("const CUBE=")
    baseline, _ = json.JSONDecoder().raw_decode(page[start:])
    current = merge_cube(baseline, cube)
    elections = ELECTIONS[:-1]
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
    method_masses = {"joint": {**history_masses, "K26": [[row[index] for index in indices] for row in current["rows"]]}}
    independent_metadata = None
    if independent is not None or independent_audit is not None:
        if independent is None or independent_audit is None:
            raise ValueError("Independent cubes and audit must be supplied together")
        if independent_audit.get("publication_approved") is not True:
            raise ValueError("Independent model is not approved for publication")
        if independent_audit.get("fit_mode") != "independent":
            raise ValueError("Independent audit must identify the independent fit mode")
        if set(independent) != set(ELECTIONS):
            raise ValueError("Independent comparison requires all six election cubes")
        collection_id = independent_audit.get("collection_id")
        fit_ids = independent_audit.get("election_fit_ids", {})
        if not collection_id or set(fit_ids) != set(ELECTIONS) or any(not fit_ids[e] for e in ELECTIONS) or len(set(fit_ids.values())) != len(ELECTIONS):
            raise ValueError("Independent audit requires a collection and distinct election fit identities")
        if independent_audit.get("snapshot_id") != cube["snapshot_id"]:
            raise ValueError("Independent model snapshot is stale")
        independent_masses = {}
        for election in ELECTIONS:
            candidate = independent[election]
            if candidate.get("snapshot_id") != cube["snapshot_id"]:
                raise ValueError("Independent election snapshot is stale")
            if candidate.get("collection_id") != collection_id or candidate.get("model_fit_id") != fit_ids[election] or candidate.get("election") != election:
                raise ValueError("Independent election cube and audit fit identities differ")
            matched = merge_cube(baseline, candidate)
            independent_masses[election] = [[row[index] for index in indices] for row in matched["rows"]]
        method_masses["independent"] = independent_masses
        independent_metadata = {"collection_id": collection_id, "election_fit_ids": fit_ids, "snapshot_id": cube["snapshot_id"]}
    combined_masses = [[sum(method_masses["joint"][election][row_index][column_index] for election in ELECTIONS) / len(ELECTIONS)
                        for column_index in range(len(mass_columns))] for row_index in range(len(baseline["rows"]))]
    baseline_hash = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    election_options = "".join(f'<option value="{election}">{label}</option>' for election, label in zip(ELECTIONS, ELECTION_LABELS))
    election_options += '<option value="joint_combined">2019–2026 · ממוצע התאמה משותפת</option>'
    panel = '''<div class="card" id="k26ModelPanel">
<label for="modelElection"><strong>התאמה: </strong></label>
<select id="modelElection"><option value="historical">ממוצע חמש הבחירות · פורסם</option>__ELECTIONS__</select>
<details><summary>השוואת התאמות</summary>
<label for="comparisonElectionA">א׳: </label><select id="comparisonElectionA"><option value="mean">ממוצע חמש הבחירות</option>__ELECTIONS__</select>
<label for="comparisonElectionB">ב׳: </label><select id="comparisonElectionB"><option value="mean">ממוצע חמש הבחירות</option>__ELECTIONS_B__</select>
<div style="overflow-x:auto"><table style="width:100%"><thead><tr><th>מדד</th><th>ממוצע חמש הבחירות · פורסם</th><th id="comparisonHeaderA">א׳</th><th id="comparisonHeaderB">ב׳</th><th>ב׳ פחות א׳ (נ״א)</th><th>ב׳ פחות א׳ (ס״ת)</th></tr></thead><tbody id="k26FamilyComparison"></tbody></table></div>
</details>
</div>'''.replace("__ELECTIONS__", election_options).replace("__ELECTIONS_B__", election_options.replace('value="K26"', 'value="K26" selected'))
    marker = '<section class="analysis-section">'
    if marker not in page:
        raise ValueError("Demographic explorer publication anchor missing")
    page = page.replace(marker, BEGIN + panel + END + marker, 1)
    # Both cubes use identical dimensions. Keep the original serialized CUBE intact.
    script = r'''<script>
const K26_FROZEN_ROWS=CUBE.rows;
const K26_METHOD_MASSES=__METHOD_MASSES__;
const K26_COMBINED_MASSES=__COMBINED_MASSES__;
const K26_HISTORICAL_MASSES=Object.fromEntries(Object.entries(K26_METHOD_MASSES.joint).filter(([key])=>key!=='K26'));
const K26_MASS_COLUMNS=__MASS_COLUMNS__;
const K26_FIT_METADATA=__META__;
const k26Method=()=>K26_METHOD_MASSES.independent?'independent':'joint';
const K26_ROW_CACHE={};
function k26Rows(election){
  if(election==='historical')return K26_FROZEN_ROWS;
  const method=k26Method(),key=method+':'+election;
  const source=election==='joint_combined'?K26_COMBINED_MASSES:K26_METHOD_MASSES[method][election];
  if(!K26_ROW_CACHE[key])K26_ROW_CACHE[key]=source.map((masses,index)=>{
    const row=K26_FROZEN_ROWS[index].slice();K26_MASS_COLUMNS.forEach((name,j)=>row[ci[name]]=masses[j]);return row;
  });
  return K26_ROW_CACHE[key];
}
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
  const old=values(K26_FROZEN_ROWS),names=['שיעור הצבעה',...P.map(p=>p[2])];
  const fitted=Object.fromEntries(Object.entries(K26_METHOD_MASSES[k26Method()]).map(([key,rows])=>[key,values(rows,true)]));
  fitted.joint_combined=values(K26_COMBINED_MASSES,true);
  const history=['K21','K22','K23','K24','K25'].map(key=>fitted[key]);
  const means=names.map((_,i)=>history.every(e=>e[i]!==null)?history.reduce((sum,e)=>sum+e[i],0)/history.length:null);
  const a=document.getElementById('comparisonElectionA'),b=document.getElementById('comparisonElectionB');
  const av=a.value==='mean'?means:fitted[a.value],bv=b.value==='mean'?means:fitted[b.value];
  document.getElementById('comparisonHeaderA').textContent=a.selectedOptions[0].textContent;
  document.getElementById('comparisonHeaderB').textContent=b.selectedOptions[0].textContent;
  const fmt=v=>v===null?'':v.toFixed(0)+'%';
  document.getElementById('k26FamilyComparison').innerHTML=names.map((name,i)=>{
    const mean=means[i];
    const sd=mean===null?null:Math.sqrt(history.reduce((sum,e)=>sum+(e[i]-mean)**2,0)/history.length);
    const delta=av[i]!==null&&bv[i]!==null?bv[i]-av[i]:null,z=delta!==null&&sd!==null&&sd>=.0001?delta/sd:null;
    const signed=value=>value===null?'':(value>=0?'+':'')+value.toFixed(1);
    return '<tr><td>'+name+'</td><td>'+fmt(old[i])+'</td><td>'+fmt(av[i])+'</td><td>'+fmt(bv[i])+'</td><td dir="ltr">'+signed(delta)+'</td><td dir="ltr">'+signed(z)+'</td></tr>';
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
function updateK26Election(){
  CUBE.rows=k26Rows(selector.value);
  P.forEach(p=>p[1]=p[2]);
  refreshK26PartyLabels();
  renderDemo();renderParty();
}
selector.addEventListener('change',updateK26Election);
['comparisonElectionA','comparisonElectionB'].forEach(id=>document.getElementById(id).addEventListener('change',renderK26Comparison));
document.getElementById('filters').addEventListener('change',renderK26Comparison);
document.querySelectorAll('#filters [data-x]').forEach(button=>button.addEventListener('click',renderK26Comparison));
updateK26Election();
</script>'''
    metadata = {"snapshot_id": cube["snapshot_id"], "model_fit_id": fit_id, "baseline_sha256": baseline_hash,
                "methods": {"joint": {"model_fit_id": fit_id, "snapshot_id": cube["snapshot_id"]}, **({"independent": independent_metadata} if independent_metadata else {})},
                "annual_fit_method": "independent" if independent_metadata else "joint",
                "joint_combined": {"model_fit_id": fit_id, "elections": ELECTIONS, "definition": "Equal arithmetic mean of all six joint-fit election cell probability masses; conditional shares are computed after aggregation."},
                "historical_sd": {"elections": ELECTIONS[:-1], "fit_method": "independent" if independent_metadata else "joint", "definition": "Population standard deviation of the five annual filtered estimates."},
                "coverage": f"{audit['matched_localities']} יישובים בהתאמה · {100*audit['national_valid_vote_coverage']:.2f}% מהקולות הכשרים הארציים. המודל אינו מכייל את התוצאות לסכום הארצי.",
                "status": f"{cube['snapshot_id']} · מקור ספירה משני; האימות הישיר והאישור המשפטי לא אומתו. אומדנים אקולוגיים, אינם קולות פרטניים."}
    script = script.replace("__METHOD_MASSES__", json.dumps(method_masses, separators=(",", ":"), allow_nan=False))
    script = script.replace("__COMBINED_MASSES__", json.dumps(combined_masses, separators=(",", ":"), allow_nan=False))
    script = script.replace("__MASS_COLUMNS__", json.dumps(mass_columns))
    script = script.replace("__META__", json.dumps(metadata, ensure_ascii=False).replace("</", "<\\/"))
    return page.replace("</body>", BEGIN + script + END + "</body>", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--historical-cubes", type=Path, required=True)
    parser.add_argument("--independent-cubes", type=Path)
    parser.add_argument("--independent-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    page = integrate(args.html.read_text(encoding="utf-8"), json.loads(args.cube.read_text(encoding="utf-8")), json.loads(args.audit.read_text(encoding="utf-8")), json.loads(args.historical_cubes.read_text(encoding="utf-8")),
                     json.loads(args.independent_cubes.read_text(encoding="utf-8")) if args.independent_cubes else None,
                     json.loads(args.independent_audit.read_text(encoding="utf-8")) if args.independent_audit else None)
    (args.output or args.html).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
