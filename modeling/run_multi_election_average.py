"""Fit individual voting propensities to five elections on a fixed 2022 Census PUF.

The 2022 Census microdata, weights and geography never change.  Election results
for 2019-04-09 through 2022-11-01 are aggregate supervision.  Demographic slopes
and locality effects are shared; each election has only an intercept shift.  The
published propensity is the equal-weight mean of the five date-specific vectors.
"""
from __future__ import annotations

import argparse, json, math, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import data_pipeline as base

DATA_DIR = Path(os.environ.get("ISRAEL_DASHBOARD_DATA_DIR", base.DATA_DIR))
ELECTION_DIR = DATA_DIR / "elections"

DATES = {"K21":"2019-04-09", "K22":"2019-09-17", "K23":"2020-03-02", "K24":"2021-03-23", "K25":"2022-11-01"}
ELECTIONS = list(DATES)
FAMILIES = ["likud","national_religious_right","shas","utj","arab","liberal_center","zionist_left","secular_national_right","other"]
OUT = Path(__file__).resolve().parent / "multi_election_outputs"

def ballot_key(x):
    try:
        y=float(x); return str(int(y)) if y.is_integer() else ("%.4f"%y).rstrip("0").rstrip(".")
    except Exception: return str(x).strip()

def family(name):
    s=str(name).lower()
    if "likud" in s: return "likud"
    if "kulanu" in s or "zehut" in s: return "likud"
    if s.strip()=="shas": return "shas"
    if "torah judaism" in s or "utj" in s: return "utj"
    if any(x in s for x in ["arab","ra'am","balad","hadash","ta'al"]): return "arab"
    if any(x in s for x in ["yamina","religious zion","otzma","jewish home","national union","new right","noam"]): return "national_religious_right"
    if "yisrael beiteinu" in s: return "secular_national_right"
    if any(x in s for x in ["labor","meretz"]): return "zionist_left"
    if any(x in s for x in ["blue & white","yesh atid","national unity","kulanu","new hope","gesher"]): return "liberal_center"
    return "other"

def build_targets(data_dir=DATA_DIR):
    election_dir=Path(data_dir)/"elections"
    geo=pd.read_csv(election_dir/"model_input_v2.csv",low_memory=False)
    geo=geo[~geo.is_envelope.astype(str).str.lower().eq("true")].copy()
    geo["locality"]=pd.to_numeric(geo.locality_code,errors="coerce").astype("Int64")
    geo["ballot"]=geo.ballot_id.map(ballot_key)
    geo["rova"]=(pd.to_numeric(geo.sa_code,errors="coerce")//10).astype("Int64")
    geo=geo.dropna(subset=["locality","rova"])
    mp=pd.read_excel(election_dir/"party_mapping.xlsx",sheet_name="Mapping")
    rows=[]; audits=[]
    metadata={"party_crosswalk":{}}
    for e in ELECTIONS:
        raw=pd.read_csv(election_dir/f"{e}_kalpi_clean.csv",encoding="utf-8-sig",low_memory=False)
        raw["locality"]=pd.to_numeric(raw["סמל ישוב"],errors="coerce").astype("Int64")
        bid="קלפי" if "קלפי" in raw else "מספר קלפי"
        raw["ballot"]=raw[bid].map(ballot_key)
        fixed={"סמל ועדה","ברזל","שם ישוב","סמל ישוב",bid,"ריכוז","שופט","בזב","מצביעים","פסולים","כשרים","locality","ballot"}
        partycols=[c for c in raw if c not in fixed]
        emap={str(r.Letter).strip():family(r.Name_English) for _,r in mp[mp.Election.eq(int(e[1:]))].iterrows()}
        metadata["party_crosswalk"][DATES[e]]={c:emap.get(c,"other") for c in partycols}
        for c in partycols: raw[c]=pd.to_numeric(raw[c],errors="coerce").fillna(0)
        for f in FAMILIES:
            cc=[c for c in partycols if emap.get(c,"other")==f]
            raw[f]=raw[cc].sum(axis=1) if cc else 0.0
        raw["eligible"]=pd.to_numeric(raw["בזב"],errors="coerce")
        raw["voted"]=pd.to_numeric(raw["מצביעים"],errors="coerce")
        raw["valid"]=pd.to_numeric(raw["כשרים"],errors="coerce")
        z=geo[geo.election.eq(e)][["locality","ballot","rova"]].merge(raw[["locality","ballot","eligible","voted","valid"]+FAMILIES],on=["locality","ballot"],how="inner")
        a=z.groupby(["locality","rova"],as_index=False)[["eligible","voted","valid"]+FAMILIES].sum()
        a=a[(a.eligible>=100)&(a.valid>0)].copy(); a["election"]=e; a["date"]=DATES[e]
        a["geo_key"]=a.locality.astype(str)+"_"+a.rova.astype(str); rows.append(a)
        audits.append({"date":DATES[e],"raw_ballots":len(raw),"matched_ballots":len(z),"geographies":len(a),"localities":a.locality.nunique(),"valid_vote_coverage":a.valid.sum()/raw.valid.sum()})
    metadata["geography_audit"]=audits
    return pd.concat(rows,ignore_index=True),metadata

class MultiElectionModel(torch.nn.Module):
    def __init__(self,p,k,nloc):
        super().__init__(); self.turncols=None
        self.turn=torch.nn.Parameter(torch.zeros(p)); self.choice=torch.nn.Parameter(torch.zeros(p,k))
        self.date_turn=torch.nn.Parameter(torch.zeros(len(ELECTIONS))); self.date_choice=torch.nn.Parameter(torch.zeros(len(ELECTIONS),k))
        self.loc_turn=torch.nn.Parameter(torch.zeros(nloc)); self.loc_choice=torch.nn.Parameter(torch.zeros(nloc,k))
    def cell_probs(self,x,loc,e):
        pt=torch.sigmoid(x@self.turn+self.loc_turn[loc]+self.date_turn[e])
        pc=torch.softmax(x@self.choice+self.loc_choice[loc]+self.date_choice[e],1)
        return pt,pc

def aggregate(pt,pc,w,g,ngeo):
    den=torch.zeros(ngeo).scatter_add_(0,g,w).clamp_min(1e-8)
    tv=torch.zeros(ngeo).scatter_add_(0,g,w*pt)/den
    joint=w[:,None]*pt[:,None]*pc
    ix=(g[:,None]*pc.shape[1]+torch.arange(pc.shape[1])[None,:]).reshape(-1)
    fam=torch.zeros(ngeo*pc.shape[1]).scatter_add_(0,ix,joint.reshape(-1)).reshape(ngeo,-1)
    fam=fam/fam.sum(1,keepdim=True).clamp_min(1e-8)
    return tv,fam

def prepare(targets,args):
    unique=targets.groupby(["geo_key","locality","rova"],as_index=False).eligible.mean()
    cells,unique,fcols,micro=base.load_cells(args.puf,targets)
    present=set(unique.geo_key); targets=targets[targets.geo_key.isin(present)].copy().reset_index(drop=True)
    gi={k:i for i,k in enumerate(unique.geo_key)}; cells["g"]=cells.geo_key.map(gi)
    locs=sorted(unique.locality.unique()); lm={v:i for i,v in enumerate(locs)}
    cells["loc"]=cells.locality.map(lm)
    return cells,unique,targets,fcols,micro,lm

def fit(cells,geos,targets,fcols,lm,train_localities=None,epochs=220,seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x=torch.tensor(cells[fcols].to_numpy(np.float32)); w=torch.tensor(cells.weight.to_numpy(np.float32)); g=torch.tensor(cells.g.to_numpy(np.int64)); loc=torch.tensor(cells["loc"].to_numpy(np.int64))
    model=MultiElectionModel(len(fcols),len(FAMILIES),len(lm)); opt=torch.optim.Adam(model.parameters(),lr=.045)
    geo_ix={k:i for i,k in enumerate(geos.geo_key)}
    obs=[]
    for ei,e in enumerate(ELECTIONS):
        z=targets[targets.election.eq(e)].copy(); z["g"]=z.geo_key.map(geo_ix)
        if train_localities is not None: z=z[z.locality.isin(train_localities)]
        yturn=torch.tensor((z.voted/z.eligible).to_numpy(np.float32))
        yy=z[FAMILIES].to_numpy(np.float32); yy/=yy.sum(1,keepdims=True)
        obs.append((ei,torch.tensor(z.g.to_numpy(np.int64)),yturn,torch.tensor(yy)))
    best=1e9; state=None; stale=0
    for ep in range(epochs):
        opt.zero_grad(); losses=[]
        for ei,ix,yt,yp in obs:
            pt,pc=model.cell_probs(x,loc,ei); at,ap=aggregate(pt,pc,w,g,len(geos))
            tr=at[ix].clamp(1e-6,1-1e-6); losses.append((-(yt*tr.log()+(1-yt)*(1-tr).log())).mean()-(yp*ap[ix].clamp_min(1e-8).log()).sum(1).mean())
        reg=.05*(model.turn[1:].square().mean()+model.choice[1:].square().mean())+.20*(model.loc_turn.square().mean()+model.loc_choice.square().mean())+.35*(model.date_turn.square().mean()+model.date_choice.square().mean())
        loss=torch.stack(losses).mean()+reg; loss.backward(); opt.step(); v=float(loss.detach())
        if v<best-1e-6: best=v; state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=35: break
    model.load_state_dict(state); return model,{"epochs":ep+1,"loss":best,"locality_map":lm,"features":fcols}

def predict_micro(model,micro,fcols,lm):
    x=torch.tensor(base.feature_frame(micro).to_numpy(np.float32)); locn=pd.to_numeric(micro.SmlYishuvPUF,errors="coerce").map(lm).fillna(-1).astype(int).to_numpy()
    known=locn>=0; lt=torch.zeros(len(locn)); lc=torch.zeros((len(locn),len(FAMILIES)))
    if known.any():
        ix=torch.tensor(locn[known]); lt[known]=model.loc_turn[ix].detach(); lc[known]=model.loc_choice[ix].detach()
    allp=[]
    with torch.no_grad():
        for ei in range(len(ELECTIONS)):
            pt=torch.sigmoid(x@model.turn+lt+model.date_turn[ei])
            pc=torch.softmax(x@model.choice+lc+model.date_choice[ei],1)
            joint=torch.cat([(1-pt)[:,None],pt[:,None]*pc],1).numpy(); allp.append(joint)
    return np.stack(allp),locn

def load_all_adults(puf):
    parts=[]
    for d in pd.read_csv(puf,usecols=base.RAW_FEATURES,chunksize=250_000,low_memory=False):
        age=pd.to_numeric(d.GilPUF,errors="coerce"); wt=pd.to_numeric(d.MishkalPratPUF,errors="coerce")
        parts.append(d[(age>=2)&(wt>0)].copy())
    return pd.concat(parts,ignore_index=True)

def weighted_quantile(a,w,q):
    o=np.argsort(a); a=a[o]; w=w[o]; return float(a[np.searchsorted(np.cumsum(w),q*w.sum(),side="left")])

def sanity_report(ens, micro, targets):
    wt=pd.to_numeric(micro.MishkalPratPUF,errors="coerce").fillna(0).to_numpy(float)
    labels=["abstain"]+FAMILIES; rows=[]
    for ei,e in enumerate(ELECTIONS):
        pred=np.average(ens[ei],axis=0,weights=wt)
        z=targets[targets.election.eq(e)]; eligible=z.eligible.sum()
        actual=np.r_[(eligible-z.voted.sum())/eligible,z[FAMILIES].sum().to_numpy()/eligible]
        r={"date":DATES[e],"mean_absolute_calibration_error_pp":float(np.abs(pred-actual).mean()*100)}
        for j,n in enumerate(labels): r["actual_"+n]=float(actual[j]); r["predicted_"+n]=float(pred[j])
        rows.append(r)
    return rows

def dashboard_cube(micro, avg):
    d=micro.copy(); wt=pd.to_numeric(d.MishkalPratPUF,errors="coerce").fillna(0).to_numpy(float)
    pop=pd.to_numeric(d.KvutzaUchlusiyaPUF,errors="coerce").fillna(-1)
    rel=pd.to_numeric(d.DatiyutPUF,errors="coerce").fillna(-1)
    inc=pd.to_numeric(d.HchnsAvgChdshPratPUF,errors="coerce").fillna(0)
    d["pop_group"]=pop; d["religiosity"]=rel
    d["income_band"]=np.select([inc.eq(0),inc.le(7),inc.le(14)],[0,1,2],default=3)
    d["age_band"]=pd.to_numeric(d.GilPUF,errors="coerce").fillna(-1)
    d["academic"]=pd.to_numeric(d.LimudToarAcademiPUF,errors="coerce").fillna(-1)
    d["district"]=pd.to_numeric(d.MachozPUF,errors="coerce").fillna(-1)
    d["weight"]=wt
    names=["abstain_mass","likud_mass","national_religious_mass","shas_mass","utj_mass","arab_mass","center_mass","zionist_left_mass","secular_national_right_mass","other_mass"]
    for j,n in enumerate(names): d[n]=wt*avg[:,j]
    dims=["pop_group","religiosity","income_band","age_band","academic","district"]
    return d.groupby(dims,as_index=False,dropna=False)[["weight"]+names].sum()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--puf",type=Path,default=base.PUF_DEFAULT); ap.add_argument("--data-dir",type=Path,default=DATA_DIR); ap.add_argument("--epochs",type=int,default=220); ap.add_argument("--quick",action="store_true"); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    targets,meta=build_targets(args.data_dir); cells,geos,targets,fcols,micro,lm=prepare(targets,args)
    epochs=60 if args.quick else args.epochs
    models=[]
    for seed in ([42] if args.quick else [42,314,2718]):
        m,info=fit(cells,geos,targets,fcols,lm,epochs=epochs,seed=seed); models.append(m)
    full_micro=load_all_adults(args.puf)
    preds=[predict_micro(m,full_micro,fcols,lm)[0] for m in models]
    ens=np.mean(preds,axis=0); avg=ens.mean(0); wt=pd.to_numeric(micro.MishkalPratPUF,errors="coerce").fillna(0).to_numpy(float)
    micro=full_micro; wt=pd.to_numeric(micro.MishkalPratPUF,errors="coerce").fillna(0).to_numpy(float)
    labels=["abstain"]+FAMILIES
    stability=[]
    for j,n in enumerate(labels):
        rng=ens[:,:,j].max(0)-ens[:,:,j].min(0)
        stability.append({"outcome":n,"weighted_mean_date_range_pp":float(np.average(rng,weights=wt)*100),"weighted_p90_date_range_pp":weighted_quantile(rng,wt,.9)*100,"weighted_share_over_20pp":float(np.average(rng>.20,weights=wt))})
    seedavg=np.array([p.mean(0) for p in preds]); dev=np.abs(seedavg-seedavg.mean(0)).max(0)
    seed_stability={"weighted_mean_abs_max_seed_deviation_pp":float(np.average(dev.mean(1),weights=wt)*100),"weighted_p90_row_max_seed_deviation_pp":weighted_quantile(dev.max(1),wt,.9)*100}
    out=micro[["SmlYishuvPUF","TatRovaKtvtMegurimPUF","GilPUF","MinPUF","KvutzaUchlusiyaPUF","DatiyutPUF","MspShnotLimudPUF","HchnsAvgChdshPratPUF","MishkalPratPUF"]].copy()
    for j,n in enumerate(labels): out["p_"+n]=avg[:,j]
    out.to_csv(OUT/"individual_probabilities_five_election_average.csv.gz",index=False)
    pd.DataFrame(stability).to_csv(OUT/"date_stability.csv",index=False)
    sanity=sanity_report(ens,micro,targets); pd.DataFrame(sanity).to_csv(OUT/"sanity_calibration.csv",index=False)
    cube=dashboard_cube(micro,avg); cube.to_csv(OUT/"dashboard_probability_cube.csv",index=False)
    prob_checks={"minimum_probability":float(avg.min()),"maximum_probability":float(avg.max()),"maximum_row_sum_error":float(np.abs(avg.sum(1)-1).max()),"dashboard_population_weight":float(cube.weight.sum())}
    meta.update({"election_dates":list(DATES.values()),"population_source":"2022 Census PUF fixed for every election","averaging":"equal arithmetic mean of five date-specific joint probability vectors","n_census_rows":len(micro),"n_demographic_cells":len(cells),"n_geographies":len(geos),"sanity_calibration":sanity,"probability_checks":prob_checks,"stability":stability,"seed_stability":seed_stability})
    (OUT/"model_audit.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"rows":len(out),"probability_checks":prob_checks,"sanity_calibration":sanity,"seed_stability":seed_stability,"date_stability":stability},indent=2))

if __name__=="__main__": main()
