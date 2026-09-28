"""Build the dashboard cube from fitted probabilities and untracked Census input."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import data_pipeline as base

HERE=Path(__file__).resolve().parent
HTML=HERE.parent/"voting-simulator"/"demographics.html"
DEFAULT_PROBS=HERE/"outputs"/"individual_probabilities.csv.gz"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--puf",type=Path,default=base.PUF_DEFAULT)
    ap.add_argument("--probabilities",type=Path,default=DEFAULT_PROBS)
    ap.add_argument("--source-label",default="all-context five-election model")
    args=ap.parse_args()
    parts=[]
    for d in pd.read_csv(args.puf,usecols=base.RAW_FEATURES,chunksize=250_000,low_memory=False):
        age=pd.to_numeric(d.GilPUF,errors="coerce"); wt=pd.to_numeric(d.MishkalPratPUF,errors="coerce")
        parts.append(d[(age>=2)&(wt>0)].copy())
    micro=pd.concat(parts,ignore_index=True); p=pd.read_csv(args.probabilities)
    if len(micro)!=len(p): raise RuntimeError(f"row mismatch: PUF={len(micro)}, probabilities={len(p)}")
    pop=pd.to_numeric(micro.KvutzaUchlusiyaPUF,errors="coerce").fillna(-1)
    rel=pd.to_numeric(micro.DatiyutPUF,errors="coerce").fillna(-1)
    inc=pd.to_numeric(micro.HchnsAvgChdshPratPUF,errors="coerce").fillna(0)
    micro["pop_group"]=pop; micro["religiosity"]=rel
    micro["income_band"]=np.select([inc.eq(0),inc.le(7),inc.le(14)],[0,1,2],default=3)
    micro["age_band"]=pd.to_numeric(micro.GilPUF,errors="coerce").fillna(-1)
    micro["academic"]=pd.to_numeric(micro.LimudToarAcademiPUF,errors="coerce").fillna(-1)
    micro["district"]=pd.to_numeric(micro.MachozPUF,errors="coerce").fillna(-1)
    micro["raw_weight"]=pd.to_numeric(micro.MishkalPratPUF,errors="coerce").fillna(0)
    base_pcols=["p_abstain","p_likud","p_national_religious_right","p_shas","p_utj","p_arab","p_liberal_center","p_zionist_left","p_secular_national_right","p_other"]
    pcols=[c+"_typical" for c in base_pcols] if "p_abstain_typical" in p.columns else base_pcols
    missing=[c for c in pcols if c not in p.columns]
    if missing: raise RuntimeError(f"probability columns missing: {missing}")
    masscols=["abstain_mass","likud_mass","national_religious_mass","shas_mass","utj_mass","arab_mass","center_mass","zionist_left_mass","secular_national_right_mass","other_mass"]
    for pc,mc in zip(pcols,masscols): micro[mc]=micro.raw_weight*p[pc].to_numpy()
    keys=["pop_group","religiosity","income_band","age_band","academic","district"]
    agg=micro.groupby(keys,as_index=False)[["raw_weight"]+masscols].sum()

    s=HTML.read_text(encoding="utf-8"); start=s.index("const CUBE=")+len("const CUBE=")
    oldobj,end=json.JSONDecoder().raw_decode(s[start:]); old=pd.DataFrame(oldobj["rows"],columns=oldobj["columns"])
    # The full PUF defines the dashboard population; fiscal estimates are joined
    # from the existing household-survey cube where exact cells are available.
    fiscal_count=["fiscal_households","fiscal_weight"]
    fiscal_values=["fiscal_tax_per_hh","fiscal_welfare_per_hh","fiscal_net_per_hh"]
    fiscal=fiscal_count+fiscal_values
    new=agg.rename(columns={"raw_weight":"weight"}).merge(
        old[keys+fiscal],on=keys,how="left",validate="one_to_one")
    exact_fiscal=new.fiscal_net_per_hh.notna()
    # Fiscal estimates came from a different household survey and do not cover
    # every Census cell.  Preserve exact matches, then impute only the per-household
    # fiscal rates from increasingly broad old-cube demographic groups.
    levels=[keys[:-1],keys[:-2],keys[:4],keys[:3],["pop_group","income_band"],["income_band"]]
    for group in levels:
        missing=new.fiscal_net_per_hh.isna()
        if not missing.any(): break
        src=old[old.fiscal_net_per_hh.notna() & old.fiscal_net_per_hh.ne(-1)].copy()
        for col in fiscal_values:
            src["_weighted"]=src[col]*src.weight
            lookup=(src.groupby(group,dropna=False)[["_weighted","weight"]].sum()
                    .assign(_value=lambda z:z._weighted/z.weight)["_value"])
            idx=pd.MultiIndex.from_frame(new.loc[missing,group]) if len(group)>1 else new.loc[missing,group[0]]
            new.loc[missing,col]=lookup.reindex(idx).to_numpy()
    for col in fiscal_values:
        if new[col].isna().any(): new[col]=new[col].fillna(np.average(old.loc[old[col].ne(-1),col],weights=old.loc[old[col].ne(-1),"weight"]))
    new.loc[~exact_fiscal,fiscal_count]=-1
    if new[masscols].isna().any().any(): raise RuntimeError("not every full-PUF cell received fitted probabilities")
    new=new[old.columns]
    err=(new[masscols].sum(axis=1)-new.weight).abs().max()
    if err>0.05: raise RuntimeError(f"mass reconciliation failed: {err}")
    full_population=float(micro.raw_weight.sum())
    if abs(new.weight.sum()-full_population)>0.05: raise RuntimeError("dashboard population does not match full PUF")
    expected=np.array([(micro.raw_weight*p[c].to_numpy()).sum() for c in pcols[1:]])
    expected/=expected.sum(); displayed=new[masscols[1:]].sum().to_numpy(copy=True); displayed/=displayed.sum()
    share_err=float(np.max(np.abs(expected-displayed)))
    if share_err>1e-8: raise RuntimeError(f"national party-share reconciliation failed: {share_err}")
    obj={"columns":list(new.columns),"rows":new.round(6).values.tolist()}
    payload=json.dumps(obj,ensure_ascii=False,separators=(",",":"))
    s=s[:start]+payload+s[start+end:]
    old_sub="אומדני נטייה אישיים המבוססים על נתוני מפקד ותוצאות קלפיות. התוצאות הן הסתברויות מודליות ולא שיוך קולות בפועל."
    new_sub="אומדני נטייה אישיים על בסיס מפקד 2022 והממוצע השווה של הבחירות ב־09.04.2019, 17.09.2019, 02.03.2020, 23.03.2021 ו־01.11.2022. התוצאות הן הסתברויות מודליות ולא שיוך קולות בפועל."
    if old_sub in s: s=s.replace(old_sub,new_sub,1)
    old_context="אומדני נטייה אישיים על בסיס מפקד 2022 והממוצע השווה של הבחירות ב־09.04.2019, 17.09.2019, 02.03.2020, 23.03.2021 ו־01.11.2022. התוצאות הן הסתברויות מודליות ולא שיוך קולות בפועל."
    new_context="אומדני נטייה אישיים ממודל ההקשר הנבחר, על בסיס מפקד 2022 והממוצע השווה של חמש הבחירות בשנים 2019–2022. המודל כולל לאום, דתיות, הכנסה והקשר גאוגרפי רחב, ללא מזהה יישוב. התוצאות הן הסתברויות מודליות ולא שיוך קולות בפועל."
    if old_context in s: s=s.replace(old_context,new_context,1)
    replacements={
        '.segment-detail{min-height:34px;margin:12px auto 0;padding:8px 12px;max-width:520px;border-radius:7px;background:var(--page);color:var(--secondary);font-size:12px;text-align:center}.segment-detail strong{color:var(--text);margin-left:5px}':
        '.segment-detail{min-height:34px;margin:12px auto 0;padding:10px 14px;max-width:620px;border:1px solid var(--grid);border-radius:9px;background:var(--page);color:var(--secondary);font-size:12px;text-align:center}.segment-detail .detail-title{display:block;color:var(--text);font-size:14px;font-weight:800;margin-bottom:3px}.segment-detail .detail-included{display:block;margin-bottom:7px}.segment-detail .detail-values{display:flex;justify-content:center;gap:20px;flex-wrap:wrap}.segment-detail .detail-values strong{color:var(--text);font-size:15px;margin-right:4px}.segment-detail .overall-value{color:var(--muted)}',
        '<div class="segment-detail" id="segmentDetail">לחצו על כל מקטע להצגת שם המפלגה והאחוז המדויק</div><p class="coalition-note">צהוב: חרדים · כחול: קואליציה ללא חרדים · ירוק: אופוזיציה ללא ערבים · אפור: מפלגות ערביות ואחרות</p><p class="muted" id="cellNote"></p><div class="group-footnote"><p><strong>הרכב הקבוצות:</strong> חרדים — ש״ס ויהדות התורה · קואליציה ללא חרדים — הליכוד, הציונות הדתית, עוצמה יהודית, נעם והבית היהודי · אופוזיציה ללא ערבים — יש עתיד, המחנה הממלכתי, העבודה, מרצ וישראל ביתנו · ערבים — רע״ם, חד״ש־תע״ל ובל״ד</p>':
        '<div class="segment-detail" id="segmentDetail">לחצו על מקטע בטור הראשי כדי לראות את המפלגות הכלולות ואת ההשוואה לכלל האוכלוסייה</div><p class="coalition-note">צהוב: מפלגות חרדיות · כחול: ליכוד וימין דתי · ירוק: מרכז, שמאל ציוני וישראל ביתנו · אפור: מפלגות ערביות ואחרות</p><p class="muted" id="cellNote"></p><div class="group-footnote"><p><strong>קבוצות המודל:</strong> חרדים — ש״ס ויהדות התורה · ליכוד — הליכוד, כולנו וזהות · ימין דתי — הציונות הדתית, עוצמה יהודית, נעם, הבית היהודי, האיחוד הלאומי והימין החדש · מרכז — יש עתיד, כחול לבן, המחנה הממלכתי, תקווה חדשה, גשר וימינה · שמאל ציוני — העבודה ומרצ · ימין חילוני — ישראל ביתנו · מפלגות ערביות — רע״ם, חד״ש־תע״ל, הרשימה המשותפת ובל״ד · אחרות — רשימות קטנות שלא סווגו</p>',
        "const P=[['shas','ש״ס','ש״ס'],['utj','יהדות התורה','יהדות התורה'],['likud','ליכוד','ליכוד'],['national_religious','הציונות הדתית, עוצמה, נעם והבית היהודי','ציונות דתית'],['center','מרכז — יש עתיד והמחנה הממלכתי','מרכז'],['zionist_left','שמאל ציוני — העבודה ומרצ','שמאל ציוני'],['secular_national_right','ישראל ביתנו','ישראל ביתנו'],['other','מפלגות אחרות','אחרות'],['arab','מפלגות ערביות','ערביות']];":
        "const P=[['shas','ש״ס','ש״ס'],['utj','יהדות התורה','יהדות התורה'],['likud','הליכוד, כולנו וזהות','ליכוד'],['national_religious','הציונות הדתית, עוצמה יהודית, נעם, הבית היהודי, האיחוד הלאומי והימין החדש','ימין דתי'],['center','יש עתיד, כחול לבן, המחנה הממלכתי, תקווה חדשה, גשר וימינה','מרכז'],['zionist_left','העבודה ומרצ','שמאל ציוני'],['secular_national_right','ישראל ביתנו','ימין חילוני'],['other','רשימות קטנות שלא סווגו','אחרות'],['arab','רע״ם, חד״ש־תע״ל, הרשימה המשותפת ובל״ד','ערביות']];",
        "const ci=Object.fromEntries(CUBE.columns.map((x,i)=>[x,i])),F={};let distributionMode='voters';":
        "const ci=Object.fromEntries(CUBE.columns.map((x,i)=>[x,i])),F={};let distributionMode='voters',demoSelected=[],demoOverall=[];",
        " data-name=\"'+party+'\" data-value=\"'+fmtPct(x.pct)+'\" data-group=\"'+label+'\"":
        " data-name=\"'+party+'\" data-index=\"'+i+'\" data-value=\"'+fmtPct(x.pct)+'\" data-group=\"'+label+'\"",
        "function showSegment(el){document.querySelectorAll('#bars .segment.selected').forEach(x=>x.classList.remove('selected'));el.classList.add('selected');let box=document.getElementById('segmentDetail');box.innerHTML='<strong>'+el.dataset.name+'</strong><span>'+el.dataset.value+' · '+el.dataset.group+'</span>'}":
        "function showSegment(el){document.querySelectorAll('#bars .segment.selected').forEach(x=>x.classList.remove('selected'));el.classList.add('selected');let i=Number(el.dataset.index),a=composition(demoSelected)[i],b=composition(demoOverall)[i],party=P[i],box=document.getElementById('segmentDetail');box.innerHTML='<span class=\"detail-title\">'+party[2]+'</span><span class=\"detail-included\">כולל: '+party[1]+'</span><span class=\"detail-values\"><span>הקבוצה שנבחרה <strong>'+fmtPct(a.pct)+'</strong></span><span class=\"overall-value\">כלל האוכלוסייה <strong>'+fmtPct(b.pct)+'</strong></span></span>'}",
        "labels=P.map(x=>x[2]);document.getElementById('pop')":
        "labels=P.map(x=>x[2]);demoSelected=selected;demoOverall=overall;document.getElementById('pop')",
    }
    for before,after in replacements.items():
        if before in s: s=s.replace(before,after,1)
        elif after not in s: raise RuntimeError('dashboard UI replacement target missing: '+before[:80])
    HTML.write_text(s,encoding="utf-8")
    try: probability_source=str(args.probabilities.resolve().relative_to(HERE.parent.resolve()))
    except ValueError: probability_source=args.probabilities.name
    audit={"dashboard_cells":len(new),"eligible_population_weight":float(new.weight.sum()),
           "max_mass_error":float(err),"probability_source":probability_source,
           "source_label":args.source_label,"full_puf_population_weight":full_population,
           "max_national_party_share_error":share_err,
           "exact_fiscal_cells":int(exact_fiscal.sum()),"imputed_fiscal_cells":int((~exact_fiscal).sum()),
           "national_party_shares":{c.replace("_mass",""):float(v) for c,v in zip(masscols[1:],displayed)},
           "dashboard_updated":True}
    (HERE/"dashboard_model_deployment.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(audit,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
