"""Fixed Census donor distributions for population cells absent from the PUF.

This supplies demographic support, with no fitted parameters or election
outcomes. Existing records remain unchanged; imputed records are marked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKER="population_donor_imputed"
KEYS=["locality_code","age_band","sex","broad_group"]
POPULATION={"jewish":1,"arab":2,"other":3}
SEX={"male":1,"female":2}
PROFILE=["DatiyutPUF","HchnsAvgChdshPratPUF","_fsu_proxy"]


def expand_missing_cohorts(micro,controls,locality_geography=None):
    """Return expanded raw-feature rows and a disclosure-safe imputation audit.

    A union of election periods is accepted. Placeholder weight uses the2022
    mean where positive, otherwise the positive-period mean. The consumer must
    rescale each cohort to its actual dated control after expansion.
    """
    required=["SmlYishuvPUF","GilPUF","MinPUF","KvutzaUchlusiyaPUF","MishkalPratPUF","DatiyutPUF",
              "HchnsAvgChdshPratPUF","MachozPUF","TatRovaKtvtMegurimPUF","YabeshetMotzaMchlkPUF","OleShnot90PUF"]
    if not set(required)<=set(micro) or not set(KEYS+["adult_population"])<=set(controls):
        raise ValueError("Missing Census or dated-control dimensions")
    selected=controls.loc[pd.to_numeric(controls.adult_population,errors="raise").gt(0)].copy()
    if not selected.sex.isin(SEX).all() or not selected.broad_group.isin(POPULATION).all():
        raise ValueError("Unknown control sex or population group")
    if not pd.to_numeric(selected.age_band).isin(range(2,8)).all():
        raise ValueError("Donors require adult PUFage bands2..7")
    values=pd.to_numeric(controls.adult_population,errors="raise")
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError("Invalid control population")
    targets=selected.groupby(KEYS,as_index=False).adult_population.mean()
    if "year" in selected:
        anchor=selected.loc[pd.to_numeric(selected.year).eq(2022)].groupby(KEYS).adult_population.mean()
        index=pd.MultiIndex.from_frame(targets[KEYS])
        preferred=anchor.reindex(index).to_numpy()
        targets["adult_population"]=np.where(np.isfinite(preferred)&(preferred>0),preferred,targets.adult_population)
    source=micro[required+(["NafaPUF"] if "NafaPUF"in micro else [])+(["TzuratYishuvPUF"]if "TzuratYishuvPUF"in micro else [])].copy()
    for column in source:
        source[column]=pd.to_numeric(source[column],errors="coerce")
    source=source.loc[source.MishkalPratPUF.gt(0)].copy()
    source["_fsu_proxy"]=(source.YabeshetMotzaMchlkPUF.eq(1)|source.OleShnot90PUF.gt(0)).astype(int)
    source.HchnsAvgChdshPratPUF=source.HchnsAvgChdshPratPUF.fillna(0).clip(0,20)
    #9999 masks several settlement forms, including small urban localities.
    source["_masked"]=source.SmlYishuvPUF.eq(9999)
    source["_source_type"]=source.TzuratYishuvPUF if "TzuratYishuvPUF"in source else np.nan
    dimensions=["GilPUF","MinPUF","KvutzaUchlusiyaPUF","MachozPUF","_masked","_source_type"]
    profiles=source.groupby(dimensions+PROFILE,as_index=False,dropna=False).MishkalPratPUF.sum()
    support=set(zip(source.SmlYishuvPUF.astype(int),source.GilPUF.astype(int),source.MinPUF.astype(int),source.KvutzaUchlusiyaPUF.astype(int)))
    quarters={}
    districts={}
    subdistricts={}
    types={}
    for code,frame in source.groupby("SmlYishuvPUF"):
        code=int(code)
        q=frame.assign(_quarter=frame.TatRovaKtvtMegurimPUF.fillna(0)).groupby("_quarter").MishkalPratPUF.sum()
        quarters[code]=(q/q.sum()).to_dict()
        mode=frame.MachozPUF.dropna().mode()
        districts[code]=float(mode.iloc[0]) if len(mode) else np.nan
        if "NafaPUF" in frame:
            mode=frame.NafaPUF.dropna().mode()
            subdistricts[code]=float(mode.iloc[0]) if len(mode) else np.nan
    if "TzuratYishuvPUF" in micro:
        for code,frame in micro.groupby("SmlYishuvPUF"):
            mode=pd.to_numeric(frame.TzuratYishuvPUF,errors="coerce").dropna().mode()
            types[int(code)]=float(mode.iloc[0]) if len(mode) else np.nan
    if locality_geography is not None:
        if not isinstance(locality_geography,pd.DataFrame):
            raise ValueError("Locality geography must be a normalized public DataFrame")
        for row in locality_geography.to_dict("records"):
            code=int(row["locality_code"])
            if code not in districts:
                districts[code]=pd.to_numeric(row.get("district_code"),errors="coerce")
            if code not in subdistricts:
                subdistricts[code]=pd.to_numeric(row.get("subdistrict_code"),errors="coerce")
            if code not in types:
                value=pd.to_numeric(row.get("puf_locality_type",row.get("TzuratYishuvPUF")),errors="coerce")
                if pd.notna(value):
                    types[code]=float(value)
    template=micro.iloc[0].to_dict()
    rows=[]
    audits=[]
    for target in targets.itertuples(index=False):
        code,age,sex,group=int(target.locality_code),int(target.age_band),SEX[target.sex],POPULATION[target.broad_group]
        if (code,age,sex,group) in support:
            continue
        pool=profiles.loc[profiles.GilPUF.eq(age)&profiles.MinPUF.eq(sex)&profiles.KvutzaUchlusiyaPUF.eq(group)]
        masked=code not in quarters
        if masked:
            pool=pool.loc[pool._masked]
        district=districts.get(code,np.nan)
        preferred=pool.loc[pool.MachozPUF.eq(district)] if pd.notna(district) else pool.iloc[:0]
        stratum=types.get(code,np.nan)
        matched_stratum=False
        if masked and pd.notna(stratum):
            same_form=preferred.loc[preferred._source_type.eq(stratum)]
            if len(same_form):
                preferred=same_form
                matched_stratum=True
        fallback="same_district" if len(preferred) else "national_same_cohort"
        chosen=preferred if len(preferred) else pool
        if chosen.empty:
            raise ValueError(f"No{' masked' if masked else ''} Census donor support for locality{code},age{age},sex{sex},group{group}")
        distribution=chosen.groupby(PROFILE,as_index=False,dropna=False).MishkalPratPUF.sum()
        distribution["fraction"]=distribution.MishkalPratPUF/distribution.MishkalPratPUF.sum()
        quarter_distribution=quarters.get(code,{0:1.})
        if masked and pd.isna(district):
            raise ValueError("Masked donor location requires an official district code")
        for record in distribution.to_dict("records"):
            for quarter,quarter_fraction in quarter_distribution.items():
                row=dict(template)
                row.update({"SmlYishuvPUF":code,"GilPUF":age,"MinPUF":sex,"KvutzaUchlusiyaPUF":group,
                            "DatiyutPUF":record["DatiyutPUF"],"HchnsAvgChdshPratPUF":record["HchnsAvgChdshPratPUF"],
                            "YabeshetMotzaMchlkPUF":1 if record["_fsu_proxy"] else 0,"OleShnot90PUF":0,
                            "MachozPUF":district,"TatRovaKtvtMegurimPUF":quarter,
                            "MishkalPratPUF":float(target.adult_population)*record["fraction"]*quarter_fraction,MARKER:True})
                if "NafaPUF"in micro:
                    row["NafaPUF"]=subdistricts.get(code,np.nan)
                if "TzuratYishuvPUF"in micro:
                    row["TzuratYishuvPUF"]=types.get(code,np.nan)
                rows.append(row)
        audits.append({"locality_code":code,"age_band":age,"sex":target.sex,"broad_group":target.broad_group,
                       "placeholder_population":float(target.adult_population),"donor_scope":fallback,
                       "donor_masked_only":masked,"donor_rural_only":bool(masked and stratum==300 and matched_stratum),
                       "target_PUF_form":float(stratum)if pd.notna(stratum)else None,"matched_PUF_form":matched_stratum,
                       "donor_profiles":len(distribution),"target_quarters":len(quarter_distribution)})
    result=micro.copy()
    if MARKER not in result:
        result[MARKER]=False
    if rows:
        result=pd.concat([result,pd.DataFrame(rows)],ignore_index=True)
    added=result.loc[result[MARKER].fillna(False)]
    audit={"missing_cohorts_filled":len(audits),"added_rows":len(rows),"placeholder_imputed_population":float(added.MishkalPratPUF.sum()),
           "cohorts":audits,"estimated_new_model_parameters":0,
           "source_features_preserved":"Joint household religiosity×income×FSUproxy within age×sex×broadgroup; same district preferred then national",
           "quarter_allocation":"Existing target-locality quarter shares across all original adult Census population; unrepresented localities use quarter0",
           "masked_geography":"PUF9999 includes regional-council and small urban settlements; unknown target settlement form remains unknown, never automatically rural300",
           "unused_raw_features":"Representative original row values; these are synthetic demographic profiles, not observed target-locality people. Rebuild if the feature design expands.",
           "actual_imputed_population":"Consumer must sum dated weights where population_donor_imputed after actual cohort scaling; placeholder is not actual dated mass"}
    return result,audit
