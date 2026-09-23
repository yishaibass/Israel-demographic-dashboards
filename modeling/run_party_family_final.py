"""Regularized turnout and party-family probability model."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import run_multi_election_average as multi

OUT=Path(__file__).resolve().parent/"party_family_final_outputs"
FAMILIES=multi.FAMILIES; ELECTIONS=multi.ELECTIONS
BASE=["intercept","arab","other_pop","haredi","dati","masorti"]
INCOME=BASE+["income_log_z","haredi_x_income","dati_x_income","arab_x_income"]
TIERS={"A_nationality_religiosity":BASE,"B_plus_income":INCOME,"C_targeted_fsu":INCOME}
YB=FAMILIES.index("secular_national_right")

class FamilyModel(torch.nn.Module):
    def __init__(self,p,k,targeted_fsu):
        super().__init__(); E=len(ELECTIONS); self.targeted_fsu=targeted_fsu
        self.turn_shared=torch.nn.Parameter(torch.zeros(p)); self.turn_dev=torch.nn.Parameter(torch.zeros(E,p))
        self.choice_shared=torch.nn.Parameter(torch.zeros(p,k)); self.choice_dev=torch.nn.Parameter(torch.zeros(E,p,k))
        self.fsu_shared=torch.nn.Parameter(torch.zeros(1)); self.fsu_dev=torch.nn.Parameter(torch.zeros(E))
    def probs(self,x,fsu,e):
        pt=torch.sigmoid(x@(self.turn_shared+self.turn_dev[e]))
        logits=x@(self.choice_shared+self.choice_dev[e])
        if self.targeted_fsu: logits[:,YB]+=fsu*(self.fsu_shared[0]+self.fsu_dev[e])
        return pt,torch.softmax(logits,1)

def aggregate(pt,pc,w,g,n):
    den=torch.zeros(n).scatter_add_(0,g,w).clamp_min(1e-8)
    turn=torch.zeros(n).scatter_add_(0,g,w*pt)/den
    fam=[]
    for j in range(pc.shape[1]): fam.append(torch.zeros(n).scatter_add_(0,g,w*pt*pc[:,j]))
    fam=torch.stack(fam,1); fam=fam/fam.sum(1,keepdim=True).clamp_min(1e-8)
    return turn,fam

def setup(cells,geos,targets,cols):
    x=torch.tensor(cells[cols].to_numpy(np.float32)); fsu=torch.tensor(cells.fsu_proxy.to_numpy(np.float32)); w=torch.tensor(cells.weight.to_numpy(np.float32)); g=torch.tensor(cells.g.to_numpy(np.int64)); gi={k:i for i,k in enumerate(geos.geo_key)}; obs=[]
    for e in ELECTIONS:
      z=targets[targets.election.eq(e)].copy(); z["g"]=z.geo_key.map(gi); yf=z[FAMILIES].to_numpy(np.float32); yf/=yf.sum(1,keepdims=True)
      obs.append({"z":z,"ix":torch.tensor(z.g.to_numpy(np.int64)),"turn":torch.tensor((z.voted/z.eligible).to_numpy(np.float32)),"fam":torch.tensor(yf)})
    return x,fsu,w,g,obs

def fit(cells,geos,targets,cols,targeted_fsu,train_locs,pool,epochs,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); x,fsu,w,g,obs=setup(cells,geos,targets,cols)
    model=FamilyModel(len(cols),len(FAMILIES),targeted_fsu); opt=torch.optim.Adam(model.parameters(),lr=.04); best=1e9; state=None; stale=0
    for ep in range(epochs):
      opt.zero_grad(); losses=[]
      for ei,o in enumerate(obs):
        m=o["z"].locality.isin(train_locs).to_numpy(); ii=torch.tensor(np.where(m)[0]); pt,pc=model.probs(x,fsu,ei); at,af=aggregate(pt,pc,w,g,len(geos)); ix=o["ix"][ii]
        gw=torch.sqrt(torch.tensor(o["z"].valid.to_numpy(np.float32))[ii]); gw/=gw.mean()
        tl=-(o["turn"][ii]*at[ix].clamp(1e-6,1-1e-6).log()+(1-o["turn"][ii])*(1-at[ix]).clamp(1e-6).log())
        fl=-(o["fam"][ii]*af[ix].clamp_min(1e-8).log()).sum(1); losses.append(((tl+fl)*gw).mean())
      reg=.08*(model.turn_shared[1:].square().mean()+model.choice_shared[1:].square().mean())
      pooling=pool*(model.turn_dev[:,1:].square().mean()+model.choice_dev[:,1:].square().mean()+model.fsu_dev.square().mean())
      loss=torch.stack(losses).mean()+reg+pooling+.08*model.fsu_shared.square().mean(); loss.backward(); opt.step(); val=float(loss.detach())
      if val<best-1e-6: best=val; state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
      else: stale+=1
      if stale>=30: break
    model.load_state_dict(state); return model,{"epochs":ep+1,"objective":best}

def evaluate(model,cells,geos,targets,cols,test_locs):
    x,fsu,w,g,obs=setup(cells,geos,targets,cols); rows=[]
    with torch.no_grad():
      for ei,e in enumerate(ELECTIONS):
        o=obs[ei]; pt,pc=model.probs(x,fsu,ei); at,af=aggregate(pt,pc,w,g,len(geos)); m=o["z"].locality.isin(test_locs).to_numpy(); ii=np.where(m)[0]; ix=o["ix"].numpy()[ii]; vw=o["z"].valid.to_numpy()[ii]; ew=o["z"].eligible.to_numpy()[ii]
        err=np.abs(af.numpy()[ix]-o["fam"].numpy()[ii]); rows.append({"election":e,"date":multi.DATES[e],"family_mae_pp":float(np.average(err.mean(1),weights=vw)*100),"turnout_mae_pp":float(np.average(np.abs(at.numpy()[ix]-o["turn"].numpy()[ii]),weights=ew)*100),"n_geographies":len(ii)})
    return rows

def micro_predict(model,micro,cols):
    ff=multi.base.feature_frame(micro); x=torch.tensor(ff[cols].to_numpy(np.float32)); fsu=torch.tensor(ff.fsu_proxy.to_numpy(np.float32)); out=[]
    with torch.no_grad():
      for ei,e in enumerate(ELECTIONS):
        pt,pc=model.probs(x,fsu,ei); out.append((e,pt.numpy(),pc.numpy()))
    return out

def profiles(pred,micro):
    wt=pd.to_numeric(micro.MishkalPratPUF,errors="coerce").fillna(0).to_numpy(float); pop=pd.to_numeric(micro.KvutzaUchlusiyaPUF,errors="coerce"); rel=pd.to_numeric(micro.DatiyutPUF,errors="coerce"); inc=pd.to_numeric(micro.HchnsAvgChdshPratPUF,errors="coerce").fillna(0); fsu=multi.base.feature_frame(micro).fsu_proxy
    masks={"arab":pop.eq(2),"hiloni":pop.eq(1)&rel.eq(1),"masorti":pop.eq(1)&rel.isin([2,6]),"dati":pop.eq(1)&rel.eq(3),"haredi":pop.eq(1)&rel.eq(5),"dati_low_income":pop.eq(1)&rel.eq(3)&inc.between(1,7),"dati_high_income":pop.eq(1)&rel.eq(3)&inc.ge(15),"fsu_proxy":fsu.eq(1),"not_fsu_proxy":fsu.eq(0)}
    rows=[]
    for e,pt,pc in pred:
      for n,m in masks.items():
        q=m.to_numpy(); r={"election":e,"date":multi.DATES[e],"segment":n,"turnout":float(np.average(pt[q],weights=wt[q]))}; r.update({f:float(np.average(pc[q,j],weights=wt[q])) for j,f in enumerate(FAMILIES)}); rows.append(r)
    return pd.DataFrame(rows)

def reverse_profiles(pred,micro,targets):
    wt=pd.to_numeric(micro.MishkalPratPUF,errors="coerce").fillna(0).to_numpy(float); pop=pd.to_numeric(micro.KvutzaUchlusiyaPUF,errors="coerce"); rel=pd.to_numeric(micro.DatiyutPUF,errors="coerce"); masks={"arab":pop.eq(2),"hiloni":pop.eq(1)&rel.eq(1),"masorti":pop.eq(1)&rel.isin([2,6]),"dati":pop.eq(1)&rel.eq(3),"haredi":pop.eq(1)&rel.eq(5)}; rows=[]
    for e,pt,pc in pred:
      z=targets[targets.election.eq(e)]; shares=z[FAMILIES].sum()/z[FAMILIES].sum().sum()
      for j,f in enumerate(FAMILIES):
        if shares[f]<.02: continue
        mass=wt*pt*pc[:,j]; den=mass.sum(); rows.append({"election":e,"date":multi.DATES[e],"family":f,"observed_training_share":shares[f],**{n:float(mass[m.to_numpy()].sum()/den) for n,m in masks.items()}})
    return pd.DataFrame(rows)

def pure_checks(model,cells,geos,targets,cols):
    sh=[]
    for gi,z in cells.groupby("g"):
      w=z.weight.to_numpy(); sh.append({"g":gi,**{n:np.average(z[n],weights=w) for n in ["arab","haredi","dati"]}})
    sh=pd.DataFrame(sh).set_index("g"); x,fsu,w,g,obs=setup(cells,geos,targets,cols); rows=[]
    with torch.no_grad():
      for ei,e in enumerate(ELECTIONS):
        o=obs[ei]; pt,pc=model.probs(x,fsu,ei); _,pf=aggregate(pt,pc,w,g,len(geos))
        for grp in ["arab","haredi","dati"]:
          for cutoff in [.5,.7]:
            gs=sh.index[sh[grp]>=cutoff]; m=o["z"].g.isin(gs).to_numpy(); ix=o["ix"].numpy()[m]; ww=o["z"].valid.to_numpy()[m]
            if not len(ix): continue
            actual=o["fam"].numpy()[m]
            for j,f in enumerate(FAMILIES): rows.append({"election":e,"date":multi.DATES[e],"group":grp,"cutoff":cutoff,"family":f,"observed":float(np.average(actual[:,j],weights=ww)),"predicted":float(np.average(pf.numpy()[ix,j],weights=ww))})
    return pd.DataFrame(rows)

def export(model,pred,micro,path):
    keep=["SmlYishuvPUF","TatRovaKtvtMegurimPUF","GilPUF","KvutzaUchlusiyaPUF","DatiyutPUF","HchnsAvgChdshPratPUF","MishkalPratPUF"]; out=micro[keep].copy(); joint=[]
    for e,pt,pc in pred:
      out[f"p_turnout_{e}"]=pt
      for j,f in enumerate(FAMILIES): out[f"p_{f}_{e}"]=pt*pc[:,j]
      joint.append(np.c_[(1-pt),pt[:,None]*pc])
    avg=np.mean(joint,axis=0); out["p_abstain_typical"]=avg[:,0]
    for j,f in enumerate(FAMILIES): out[f"p_{f}_typical"]=avg[:,j+1]
    out.to_csv(path,index=False,compression="gzip")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--puf",type=Path,default=multi.base.PUF_DEFAULT); ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--folds",type=int,default=3); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    targets,meta=multi.build_targets(); cells,geos,targets,_,_,_=multi.prepare(targets,args); locs=np.array(sorted(targets.locality.unique())); split=GroupKFold(args.folds); cv=[]
    for tier,cols in TIERS.items():
      for pool in [.05,.20]:
        for fold,(tr,te) in enumerate(split.split(locs,groups=locs),1):
          model,info=fit(cells,geos,targets,cols,tier=="C_targeted_fsu",set(locs[tr]),pool,args.epochs,100+fold)
          for r in evaluate(model,cells,geos,targets,cols,set(locs[te])): r.update({"tier":tier,"pool":pool,"fold":fold,**info}); cv.append(r)
    cv=pd.DataFrame(cv); cv.to_csv(OUT/"cv_metrics.csv",index=False); sel=cv.groupby(["tier","pool"],as_index=False).agg(family_mae_pp=("family_mae_pp","mean"),turnout_mae_pp=("turnout_mae_pp","mean"),fold_sd=("family_mae_pp","std")); sel["score"]=sel.family_mae_pp+.1*sel.fold_sd
    # Parsimony: C must improve B by at least 0.10pp at its best pooling.
    b=sel[sel.tier.eq("B_plus_income")].sort_values("score").iloc[0]; c=sel[sel.tier.eq("C_targeted_fsu")].sort_values("score").iloc[0]
    best=c if c.family_mae_pp<=b.family_mae_pp-.10 else b; sel.to_csv(OUT/"model_selection.csv",index=False)
    cols=TIERS[best.tier]; model,info=fit(cells,geos,targets,cols,best.tier=="C_targeted_fsu",set(locs),float(best.pool),args.epochs,42); micro=multi.load_all_adults(args.puf); pred=micro_predict(model,micro,cols)
    prof=profiles(pred,micro); prof.to_csv(OUT/"demographic_profiles.csv",index=False); reverse_profiles(pred,micro,targets).to_csv(OUT/"party_voter_profiles.csv",index=False); pure_checks(model,cells,geos,targets,cols).to_csv(OUT/"pure_area_checks.csv",index=False); export(model,pred,micro,OUT/"individual_probabilities_party_family.csv.gz")
    # Cross-election stability of each segment-family probability.
    long=prof.melt(id_vars=["election","date","segment","turnout"],value_vars=FAMILIES,var_name="family",value_name="probability"); stab=long.groupby(["segment","family"],as_index=False).probability.agg(["mean","std","min","max"]).reset_index(); stab.to_csv(OUT/"cross_election_stability.csv",index=False)
    audit={**meta,"model":"party-family only","clean_bloc_head":False,"anchors":False,"locality_effects":False,"selected_tier":str(best.tier),"selected_pool":float(best.pool),"selection":sel.to_dict("records"),"suppressed_reverse_profile_below_share":.02,"dashboard_updated":False,"fit_info":info}
    (OUT/"model_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8"); print(sel.sort_values("score").to_string(index=False)); print("SELECTED",best.tier,best.pool)

if __name__=="__main__": main()
