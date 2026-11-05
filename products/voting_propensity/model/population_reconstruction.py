"""Reconstruct dated population cells independently of election outcomes.

Raw PUF records never leave the local project. The retained Census anchors,
dated administrative controls and survey trends have separate provenance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import pandas as pd


GROUPS = {1: "jewish", 2: "arab", 3: "other"}
SEXES = {1: "male", 2: "female"}
AGE_LIMITS = {2: (18, 24), 3: (25, 34), 4: (35, 44), 5: (45, 54), 6: (55, 64), 7: (65, 120)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def age_band(age):
    if age < 18:
        return 0
    return next(code for code, (lower, upper) in AGE_LIMITS.items() if lower <= age <= upper)


def census2022_national(path):
    """Parse actual single-year age×sex×religion counts; exclude foreigners separately."""
    raw = pd.read_excel(path, header=None)
    rows, religion_rows, sex = [], [], None
    religious_columns = {"jewish": 3, "non_arab_christian": 4, "unclassified": 5,
                         "muslim": 7, "arab_christian": 8, "druze": 9, "foreign": 10}
    for _, record in raw.iterrows():
        label = record.iloc[0]
        if isinstance(label, str) and label.strip() in ["זכרים", "נקבות"]:
            sex = "male" if label.strip() == "זכרים" else "female"
            continue
        if sex is None:
            continue
        if isinstance(label, str) and label.strip().replace(" ", "") in ["+85", "85+"]:
            age, open_age = 85, True
        else:
            age = pd.to_numeric(label, errors="coerce")
            if pd.isna(age) or not float(age).is_integer() or not 0 <= age <= 84:
                continue
            age, open_age = int(age), False
        values = pd.to_numeric(record.iloc[1:], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError("Missing/negative official national Census cell")
        base = {"year": 2022, "age": age, "open_age": open_age, "age_band": age_band(age), "sex": sex,
                "reference_basis": "2022_Census_estimate", "status": "observed", "source_id": Path(path).name}
        groups = {"jewish": float(record.iloc[3]), "arab": float(record.iloc[6]),
                  "other": float(record.iloc[4])+float(record.iloc[5])}
        for group, count in groups.items():
            rows.append({**base, "broad_group": group, "population": count})
        for religion, column in religious_columns.items():
            religion_rows.append({**base, "religion": religion, "population": float(record.iloc[column])})
    national, religions = pd.DataFrame(rows), pd.DataFrame(religion_rows)
    if len(national) != 2*86*3 or national.duplicated(["sex", "age", "broad_group"]).any():
        raise ValueError("Official Census must contain every age0..85+×sex×three broad groups exactly once")
    return national, religions


def puf_anchor(path):
    """Aggregate observed adult PUF joint cells, retaining masked code9999."""
    required = ["SmlYishuvPUF", "MachozPUF", "NafaPUF", "TzuratYishuvPUF", "GilPUF", "MinPUF",
                "KvutzaUchlusiyaPUF", "DatiyutPUF", "MishkalPratPUF"]
    header = pd.read_csv(path, nrows=0)
    missing = set(required)-set(header.columns)
    if missing:
        raise ValueError("Required PUF population dimensions absent: " + str(sorted(missing)))
    parts = []
    for chunk in pd.read_csv(path, usecols=required, chunksize=250_000, low_memory=False):
        for name in required:
            chunk[name] = pd.to_numeric(chunk[name], errors="coerce")
        chunk = chunk.loc[chunk.GilPUF.ge(2) & chunk.MishkalPratPUF.gt(0)].copy()
        if not len(chunk):
            continue
        if not chunk.GilPUF.isin(AGE_LIMITS).all() or not chunk.MinPUF.isin(SEXES).all() or not chunk.KvutzaUchlusiyaPUF.isin(GROUPS).all():
            raise ValueError("Unknown adult PUF age/sex/population-group category")
        chunk = chunk.rename(columns={"SmlYishuvPUF": "locality_code", "MachozPUF": "district", "NafaPUF": "subdistrict",
                                      "TzuratYishuvPUF": "locality_type", "GilPUF": "age_band", "MinPUF": "sex",
                                      "KvutzaUchlusiyaPUF": "broad_group", "DatiyutPUF": "religiosity", "MishkalPratPUF": "population"})
        chunk.sex = chunk.sex.map(SEXES)
        chunk.broad_group = chunk.broad_group.map(GROUPS)
        keys = ["locality_code", "district", "subdistrict", "locality_type", "age_band", "sex", "broad_group", "religiosity"]
        parts.append(chunk.groupby(keys, as_index=False, dropna=False).population.sum())
    result = pd.concat(parts, ignore_index=True).groupby(keys, as_index=False, dropna=False).population.sum()
    result["year"] = 2022
    result["geography_status"] = np.where(result.locality_code.eq(9999), "masked_locality_pool", "identified_locality")
    result["source_id"] = Path(path).name
    result["status"] = "weighted_2022_Census_PUF"
    return result


def census2022_local_margins(path):
    """Independent published locality-age/sex margins; never classify people by dominant religion."""
    data = pd.read_excel(path, header=2)
    data["locality_code"] = pd.to_numeric(data.LocalityCode, errors="coerce")
    data = data.loc[data.locality_code.gt(0) & data.StatArea.isna()].copy()
    if data.locality_code.duplicated().any():
        raise ValueError("Duplicate Census locality summary")
    rows = []
    for row in data.to_dict("records"):
        population = pd.to_numeric(row.get("pop_approx"), errors="coerce")
        if not np.isfinite(population) or population <= 0:
            continue
        common = {"year": 2022, "locality_code": int(row["locality_code"]), "locality_name": row["LocNameHeb"],
                  "population_basis": "Census_all_population_including_foreigners_and_institutions",
                  "source_id": Path(path).name, "status": "published_rounded_aggregate"}
        for dimension, category, column in [("age", "age0_19", "age0_19_pcnt"), ("age", "age20_64", "age20_64_pcnt"),
                                             ("age", "age65_plus", "age65_pcnt")]:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                rows.append({**common, "dimension": dimension, "category": category, "value": population*value/100,
                             "share": value/100, "total_population": population})
        ratio = pd.to_numeric(row.get("sexRatio"), errors="coerce")
        if pd.notna(ratio) and ratio > 0:
            for sex, fraction in [("male", ratio/(100+ratio)), ("female", 100/(100+ratio))]:
                rows.append({**common, "dimension": "sex", "category": sex, "value": population*fraction,
                             "share": fraction, "total_population": population})
    return pd.DataFrame(rows)


def rake_joint(prior, age=None, sex=None, group=None, iterations=1000):
    """IPF on all-age locality×age×sex×group cells; exact structural zeros stay zero."""
    result = np.asarray(prior, float).copy()
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Invalid demographic prior")
    targets = [(age, (2, 3)), (sex, (1, 3)), (group, (1, 2))]
    for _ in range(iterations):
        maximum = 0.
        for target, axes in targets:
            if target is None:
                continue
            target = np.asarray(target, float)
            actual = result.sum(axis=axes)
            if (target < 0).any() or not np.isfinite(target).all() or ((actual <= 0)&(target > 1e-7)).any():
                raise ValueError("Demographic margin lacks prior support")
            ratio = np.divide(target, actual, out=np.zeros_like(actual), where=actual > 0)
            shape = [len(result), 1, 1, 1]
            shape[next(axis for axis in [1, 2, 3] if axis not in axes)] = target.shape[1]
            result *= ratio.reshape(shape)
        for target, axes in targets:
            if target is not None:
                maximum = max(maximum, float(np.max(abs(result.sum(axis=axes)-target)/np.maximum(np.asarray(target), 1))))
        if maximum < 1e-8:
            break
    if maximum > 1e-5:
        raise ValueError("Incompatible demographic margins; maximum_relative_error="+str(maximum))
    return result


def _national_array(national, year):
    z = national.loc[national.year.eq(year)].copy()
    if z.empty or z.duplicated(["age", "sex", "broad_group"]).any():
        raise ValueError("Missing or duplicated annual national controls")
    index = pd.MultiIndex.from_product([range(86), ["male", "female"], ["jewish", "arab", "other"]])
    values = z.set_index(["age", "sex", "broad_group"]).population.reindex(index)
    if values.isna().any() or (values < 0).any():
        raise ValueError("National trajectory must contain every0..85+ age×sex×group cell")
    return values.to_numpy(float).reshape(86, 2, 3)


def _coarse(array):
    return np.stack([array[:18].sum(0)]+[array[lower:min(upper+1, 86)].sum(0) for lower, upper in AGE_LIMITS.values()])


def national_at_date(national, date):
    """Interpolate observed July1 anchors; max1.5yr projection,0.5yr backcast."""
    observed=national.loc[national.status.eq("observed")].copy()
    years=sorted(observed.year.unique())
    target=pd.Timestamp(date)
    last=pd.Timestamp(f"{years[-1]}-07-01")
    first=pd.Timestamp(f"{years[0]}-07-01")
    if (target-last).days>550 or (first-target).days>184:
        raise ValueError("Unsupported dated national projection horizon")
    after=[year for year in years if pd.Timestamp(f"{year}-07-01")>=target]
    if target>=last:
        high=years[-1]
        low=high-2 if high-2 in years else years[-2]
    elif not after or target<first:
        low,high=years[:2]
    else:
        high=after[0]
        if target==pd.Timestamp(f"{high}-07-01"):
            return _national_array(observed,high)
        low=max(year for year in years if year<high)
    left,right=_national_array(observed,low),_national_array(observed,high)
    fraction=(target-pd.Timestamp(f"{low}-07-01")).days/(pd.Timestamp(f"{high}-07-01")-pd.Timestamp(f"{low}-07-01")).days
    result=np.maximum(0,left+fraction*(right-left))
    positive=(left>0)&(right>0)
    result[positive]=np.exp(np.log(left[positive])+fraction*np.log(right[positive]/left[positive]))
    return result


def rake_national_absolute(prior,national_target,iterations=1000):
    """Exact national age×sex×group; preserve relative locality mass within group.

    Local levels are reconciled estimates. Every raw group sum and scaling
    factor is returned; this is never presented as unchanged local observations.
    """
    result=np.asarray(prior,float).copy()
    target=np.asarray(national_target,float)
    raw=result.sum(axis=(0,1,2))
    requested=target.sum(axis=(0,1))
    if ((raw<=0)&(requested>0)).any():
        raise ValueError("National group has no located population support")
    factor=np.divide(requested,raw,out=np.zeros_like(raw),where=raw>0)
    local_target=result.sum(axis=(1,2))*factor[None,:]
    support=target/target.sum()
    result+=local_target.sum(1)[:,None,None,None]*support[None,:,:,:]*1e-6
    for iteration in range(iterations):
        total=result.sum(axis=0)
        result*=np.divide(target,total,out=np.zeros_like(target),where=total>0)[None,:,:,:]
        local=result.sum(axis=(1,2))
        result*=np.divide(local_target,local,out=np.zeros_like(local),where=local>0)[:,None,None,:]
        error=float(np.max(abs(result.sum(axis=0)-target)/np.maximum(target,1)))
        if error<1e-9:
            break
    if error>1e-6:
        raise ValueError("National population calibration did not converge: "+str(error))
    return result,{"raw_group_population":raw.tolist(),"national_group_population":requested.tolist(),
                   "group_level_rescale_factor":factor.tolist(),"max_national_relative_error":error,"iterations":iteration+1}


def _local_tables(annual, codes, year):
    z = annual.loc[annual.year.eq(year)&annual.dimension.eq("population_group")&annual.unit.eq("persons")].copy()
    z = z.loc[z.locality_code.isin(codes)]
    if z.duplicated(["locality_code", "category"]).any():
        raise ValueError("Duplicate annual locality control")
    return z.pivot(index="locality_code", columns="category", values="value").reindex(codes)


def _group_proportions(table, fallback, national_group=None):
    result = fallback.copy()
    for i, (_, row) in enumerate(table.iterrows()):
        arab, combined, jewish, other = (pd.to_numeric(row.get(name), errors="coerce") for name in ["arab", "jewish_and_other", "jewish", "other"])
        if np.isfinite(arab) and np.isfinite(combined) and arab+combined > 0:
            if not np.isfinite(jewish) or not np.isfinite(other):
                conditional = result[i, [0, 2]]
                if conditional.sum()>0:
                    conditional=conditional/conditional.sum()
                else:
                    conditional=np.asarray(national_group,float)[[0,2]] if national_group is not None else fallback[:,[0,2]].sum(axis=0)
                    if conditional.sum()<=0:
                        raise ValueError("Jewish/Other split needs an explicit demographic prior")
                    conditional=conditional/conditional.sum()
                jewish, other = combined*conditional
            result[i] = [jewish, arab, other]
    return result/np.maximum(result.sum(1, keepdims=True), 1e-30)


def _support_new_groups(joint, target, national_coarse):
    """Observed migration/new groups get an explicit national demographic prior."""
    result = joint.copy()
    absent = (result.sum(axis=(1,2))<=0)&(target>0)
    for locality,group in np.argwhere(absent):
        template = national_coarse[:,:,group]
        result[locality,:,:,group] = template/template.sum()*target[locality,group]
    return result


def active_localities(founding_year,year):
    founding_year=np.asarray(founding_year,float)
    return ~np.isfinite(founding_year)|(founding_year<=year)


def published_age65_targets(raw,codes):
    """Only complete published constituent-SA cells qualify as heldout truth."""
    columns=["age_65_69","age_70_74","age_75_79","age_80_84","age_85_up"]
    data=raw.loc[raw.Pop_Total.notna()].copy()
    complete=data[columns].notna().all(axis=1)
    sex_complete=data[["Male_Total","Female_Total"]].notna().all(axis=1)
    flags=pd.DataFrame({"code":data.SEMEL_YISHUV,"age":complete,"sex":sex_complete}).groupby("code").all().reindex(codes,fill_value=False)
    sums=data.groupby("SEMEL_YISHUV")[["Pop_Total"]+columns].sum(min_count=1).reindex(codes)
    age=sums[columns].sum(axis=1,min_count=len(columns))
    return flags.age.to_numpy(bool),flags.sex.to_numpy(bool),age.to_numpy(float)/sums.Pop_Total.to_numpy(float)


def local_age_backcast_prior(margins,codes,national_old,national2022,fallback):
    """Prefer published locality2022 broad ages, backcast by national count ratios.

    Output is a prior, not an observed2020 margin. Missing2022 categories retain
    only the unreported share, with the older demographic prior allocating it.
    """
    categories=["age0_19","age20_64","age65_plus"]
    source=margins.loc[margins.dimension.eq("age")&margins.category.isin(categories)]
    if source.duplicated(["locality_code","category"]).any():
        raise ValueError("Duplicate published locality-age margin")
    table=source.pivot(index="locality_code",columns="category",values="share").reindex(index=codes,columns=categories)
    ranges=[slice(0,4),slice(4,13),slice(13,18)]
    national_old_age=national_old.sum(axis=(1,2))
    national2022_age=national2022.sum(axis=(1,2))
    old_broad=np.array([national_old_age[:20].sum(),national_old_age[20:65].sum(),national_old_age[65:].sum()])
    new_broad=np.array([national2022_age[:20].sum(),national2022_age[20:65].sum(),national2022_age[65:].sum()])
    result=np.asarray(fallback,float).copy()
    used=np.zeros(len(codes),bool)
    complete=np.zeros(len(codes),bool)
    for i,values in enumerate(table.to_numpy(float)):
        known=np.isfinite(values)
        if not known.any():
            continue
        broad=np.array([result[i,band].sum() for band in ranges])
        broad=broad*np.divide(new_broad,old_broad,out=np.ones(3),where=old_broad>0)
        broad/=broad.sum()
        if known.all():
            broad=values/values.sum()
        else:
            residual=max(1-float(values[known].sum()),0.)
            unknown=broad[~known]
            broad[known]=values[known]
            broad[~known]=residual*unknown/unknown.sum() if unknown.sum()>0 else 0
            broad/=broad.sum()
        broad*=np.divide(old_broad,new_broad,out=np.ones(3),where=new_broad>0)
        broad/=broad.sum()
        for j,band in enumerate(ranges):
            within=result[i,band]
            result[i,band]=broad[j]*(within/within.sum()) if within.sum()>0 else broad[j]/len(within)
        used[i],complete[i]=True,bool(known.all())
    return result,used,complete


def impute_age_residuals(known,total,prior,missing):
    """Allocate only unobserved five-year components; leave known counts intact."""
    result=np.nan_to_num(np.asarray(known,float),nan=0.).copy()
    total=np.asarray(total,float)
    residual=np.maximum(total-result.sum(axis=1),0.)
    unallocated=np.zeros(len(result))
    for i in range(len(result)):
        available=np.asarray(missing[i],bool)
        distribution=np.asarray(prior[i],float)*available
        if distribution.sum()>0:
            result[i]+=residual[i]*distribution/distribution.sum()
        else:
            # Published rounding/disclosure residual is retained separately;
            # never distribute it over already observed age cells.
            unallocated[i]=residual[i]
    return result,unallocated


def dated_lifecycle_prior(joint,codes,date,registry_current):
    """Select one historical backbone; never add overlapping old/new estimates."""
    import locality_lifecycle as life
    result=joint.copy()
    index={int(code):i for i,code in enumerate(codes)}
    audit={"discarded_synthetic_overlap":0.,"events":[]}
    when=pd.Timestamp(date)
    for event in life.EVENTS:
        if event["kind"]=="merge":
            old=event["predecessors"]
            new=event["successors"][0]
            if not all(code in index for code in (*old,new)):
                continue
            if when<pd.Timestamp(event["earliest"]):
                audit["discarded_synthetic_overlap"]+=float(result[index[new]].sum())
                result[index[new]]=0
            else:
                audit["discarded_synthetic_overlap"]+=float(result[index[new]].sum())
                result[index[new]]=sum(result[index[code]]for code in old)
                for code in old:
                    result[index[code]]=0
            audit["events"].append(event["event_id"])
        elif event["kind"]=="split":
            parent=event["predecessors"][0]
            child=next(code for code in event["successors"]if code!=parent)
            if parent not in index or child not in index:
                continue
            audit["discarded_synthetic_overlap"]+=float(result[index[child]].sum())
            result[index[child]]=0
            # No invented day within the documented uncertain2023 code year.
            if when>pd.Timestamp(event["latest"]):
                weights={code:float(registry_current.get(code,0.))for code in event["successors"]}
                split=life.allocate_split(result[index[parent]],weights)
                for code,value in split.items():
                    result[index[code]]=value
                audit["split_weights_source"]="observed2026PIBAfamilytotalshares; common share applied to each prior demographic cell"
            audit["events"].append(event["event_id"])
    return result,audit


def local_register_growth_prior(codes,resident2020,register2020,register2026,fallback,date):
    """Same-series local registry growth, with reviewed merge/split denominators."""
    import locality_lifecycle as life
    result=np.asarray(fallback,float).copy()
    sources=np.full(len(codes),"latest_CBS_or_national_growth_fallback",dtype=object)
    for i,code in enumerate(codes):
        predecessors=life.predecessor_group(int(code),date)
        old=sum(float(register2020.get(c,0.))for c in predecessors)
        base=sum(float(resident2020.get(c,0.))for c in predecessors)
        current=float(register2026.get(int(code),0.))
        if old>0 and base>0 and current>0:
            result[i]=base*current/old
            sources[i]="CBS2020_local_total_times_PIBA2026_over2020_same_series"
        elif current>0:
            result[i]=current
            sources[i]="PIBA2026_absolute_geographic_prior_no_observed2020_base"
    return result,sources


def _regional2008(path, geography, national2008):
    """Actual observed subdistrict age×sex; within5-year ages follow national2008."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [item["attributes"] for item in raw["features"]]
    regional = {}
    bands = [(0,4),(5,9),(10,14),(15,19),(20,24),(25,29),(30,34),(35,39),(40,44),
             (45,49),(50,54),(55,59),(60,64),(65,69),(70,74),(75,79),(80,84),(85,85)]
    for row in rows:
        if float(row.get("pop_thou",0))<=0 or float(row.get("men_thou",0))+float(row.get("women_thou",0))<=0:
            continue
        joint = np.zeros((86, 2, 3))
        for s, prefix, count_name in [(0, "m_", "men_thou"), (1, "w_", "women_thou")]:
            columns = [name for name in row if name.startswith(prefix+"age") and name not in ["m_age_medi", "w_age_medi"]]
            if len(columns) != 18:
                raise ValueError("2008 subdistrict requires18 sex-specific age bins")
            for (lower, upper), column in zip(bands, columns):
                count = float(row[count_name])*1000*float(row[column])/100
                prior = national2008[lower:upper+1, s, :]
                joint[lower:upper+1, s, :] = prior/prior.sum()*count
        if int(row["CODE"]) in regional:
            raise ValueError("Duplicate inhabited2008subdistrict")
        regional[int(row["CODE"])] = joint
    associations = []
    nat = _coarse(national2008).sum(axis=2)
    nat /= nat.sum()
    for code in geography.subdistrict_code:
        joint = regional.get(int(code)) if pd.notna(code) else None
        q = _coarse(joint).sum(axis=2) if joint is not None else nat.copy()
        if q.sum()<=0:
            raise ValueError("Empty inhabited2008regional population")
        q /= q.sum()
        associations.append(q/np.maximum(nat, 1e-30))
    return np.asarray(associations), len(regional)


def reconstruct(args):
    """Run dated, all-locality reconstruction, then export adult consumer cells."""
    started = time.perf_counter()
    import locality_lifecycle as life
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    national = pd.read_csv(args.national_controls)
    source_paths={key:getattr(args,key) for key in ["national_controls","annual_targets","gdb_attributes","subdistrict2008","geography","registry_age"]}
    source_paths["PUF_joint_anchor"]=folder/"census2022_PUF_joint_anchor.csv"
    source_paths["local2022_observed_margins"]=folder/"census2022_local_observed_margins.csv"
    source_paths["locality_lifecycle_module"]=Path(life.__file__)
    source_paths["locality_lifecycle_evidence"]=folder/"locality_lifecycle_evidence.json"
    source_paths["latest_locality_catalog"]=args.latest_locality_catalog
    input_hashes={key:sha256(path) for key,path in source_paths.items()}
    code_hash=sha256(Path(__file__))
    annual = pd.read_csv(args.annual_targets, low_memory=False)
    raw = pd.read_csv(args.gdb_attributes)
    fields = [f"age_{i}_{i+4}" for i in range(0, 85, 5)]+["age_85_up"]
    populated = raw.loc[raw.Pop_Total.notna()].copy()
    geo2020 = populated.groupby("SEMEL_YISHUV")[["Pop_Total", "Male_Total", "Female_Total"]+fields].sum(min_count=1)
    positive_annual = annual.loc[annual.dimension.eq("population_group")&annual.category.isin(["total_population", "total_israeli"])&annual.value.gt(0)]
    registry = pd.read_csv(args.registry_age, low_memory=False)
    codes = np.array(sorted((set(geo2020.index)|set(positive_annual.locality_code)|set(registry.locality_code))-{0,9999}), dtype=int)
    geo2020 = geo2020.reindex(codes)
    geog = pd.read_csv(args.geography).set_index("locality_code").reindex(codes).reset_index()
    n2020 = _national_array(national, 2020)
    n2008 = _national_array(national, 2008)
    coarse2020 = _coarse(n2020)
    regional_association, regions = _regional2008(args.subdistrict2008, geog, n2008)
    prior = coarse2020[None, :, :, :]*regional_association[:, :, :, None]
    group_prior = np.broadcast_to(n2020.sum(axis=(0,1))/n2020.sum(), (len(codes),3)).copy()
    group2020 = _group_proportions(_local_tables(annual, codes, 2020), group_prior,n2020.sum(axis=(0,1)))
    total2020 = geo2020.Pop_Total.to_numpy(float)
    observed_total2020 = _local_tables(annual, codes, 2020).get("total_population", pd.Series(index=codes, dtype=float)).to_numpy(float)
    total2020 = np.where(np.isfinite(total2020)&(total2020>0), total2020, observed_total2020)
    # New/missing2020 localities use the earliest official total and national growth; never votes.
    missing = ~np.isfinite(total2020)|(total2020<=0)
    for year in range(2003, 2027):
        table = _local_tables(annual, codes, year)
        value = table.get("total_israeli", table.get("total_population", pd.Series(index=codes, dtype=float))).to_numpy(float)
        usable = missing&np.isfinite(value)&(value>0)
        total2020[usable] = value[usable]*n2020.sum()/_national_array(national,year).sum()
        missing[usable] = False
    # Registry-only new localities have an explicitly registered-person universe.
    registry_totals = registry.loc[registry.category.eq("all_ages")].groupby(["year","reference_date","locality_code"]).value.sum().reset_index()
    registry_only = np.zeros(len(codes), bool)
    for _, row in registry_totals.sort_values("reference_date").iterrows():
        ix = np.searchsorted(codes, int(row.locality_code))
        if ix<len(codes) and codes[ix]==row.locality_code and missing[ix] and row.value>0:
            total2020[ix] = row.value*n2020.sum()/_national_array(national,int(row.year)).sum()
            missing[ix], registry_only[ix] = False, True
    keep = ~missing
    codes, total2020, prior, group2020, regional_association, registry_only = (a[keep] for a in [codes,total2020,prior,group2020,regional_association,registry_only])
    geo2020, geog = geo2020.loc[codes], geog.loc[keep].reset_index(drop=True)
    prior *= total2020[:,None,None,None]/prior.sum(axis=(1,2,3))[:,None,None,None]
    age2020 = np.zeros((len(codes),7))
    fraction18 = n2020[18:20].sum()/n2020[15:20].sum()
    age_coarse_index=np.array([0]*18+[1]*7+[2]*10+[3]*10+[4]*10+[5]*10+[6]*21)
    fine_single=prior[:,age_coarse_index,:,:]*np.divide(n2020,coarse2020[age_coarse_index],out=np.zeros_like(n2020),where=coarse2020[age_coarse_index]>0)[None,:,:,:]
    five_year_ranges=[(i,min(i+4,85))for i in range(0,85,5)]+[(85,85)]
    regional_fine=np.stack([fine_single[:,lo:hi+1].sum(axis=(1,2,3))for lo,hi in five_year_ranges],axis=1)
    local2022_margins=pd.read_csv(source_paths["local2022_observed_margins"])
    fine_prior,local2022_used,local2022_complete=local_age_backcast_prior(local2022_margins,codes,n2020,_national_array(national,2022),regional_fine)
    raw_age=geo2020[fields].to_numpy(float)
    no_age=~np.isfinite(raw_age).any(axis=1)
    raw_age_known=np.nansum(raw_age,axis=1)
    missing_components=populated[fields].isna().groupby(populated.SEMEL_YISHUV).any().reindex(codes,fill_value=True).to_numpy(bool)
    filled_age,rounding_residual=impute_age_residuals(raw_age,total2020,fine_prior,missing_components)
    age2020[:,0]=filled_age[:,:3].sum(axis=1)+filled_age[:,3]*(1-fraction18)
    age2020[:,1]=filled_age[:,4]+filled_age[:,3]*fraction18
    for i,(start,stop)in enumerate([(5,7),(7,9),(9,11),(11,13),(13,18)],2):
        age2020[:,i]=filled_age[:,start:stop].sum(axis=1)
    # Exact published age components remain unchanged. Any all-known-bin
    # disclosure/rounding residual is not invented in an observed age cell.
    anchor_total2020=age2020.sum(axis=1)
    sex2020 = geo2020[["Male_Total","Female_Total"]].to_numpy(float,copy=True)
    for i in range(len(codes)):
        if not np.isfinite(sex2020[i]).all() or sex2020[i].sum()<=0:
            sex2020[i] = prior[i].sum(axis=(0,2))
    sex_fraction2020=sex2020/sex2020.sum(axis=1,keepdims=True)
    sex2020=sex_fraction2020*anchor_total2020[:,None]
    base = rake_joint(prior, age2020, sex2020, group2020*anchor_total2020[:,None])
    # Independent age holdout: NO2020local ages or2020-derived regional totals enter this prior.
    observed = national.loc[national.status.eq("observed")&national.year.ne(2020)].copy()
    import national_population_controls as npc
    blind_n2020 = _national_array(npc.annualize(observed,start=2003,end=2026),2020)
    blind = _coarse(blind_n2020)[None,:,:,:]*regional_association[:,:,:,None]
    blind *= total2020[:,None,None,None]/blind.sum(axis=(1,2,3))[:,None,None,None]
    blind = rake_joint(blind, sex=sex_fraction2020*total2020[:,None], group=group2020*total2020[:,None])
    # A target is observed only when ALL five published65+ components exist.
    observed65,observed_sex,true65_fraction=published_age65_targets(raw,codes)
    holdout = pd.DataFrame({"locality_code":codes,"population2020":total2020,"age65_observed":true65_fraction,
                            "age65_reconstructed_without2020age":blind[:,6].sum(axis=(1,2))/total2020,
                            "prediction_method":"2008_regional_fallback_diagnostic_not_full_pipeline",
                            "age20_64_source_count":geo2020[fields[4:13]].sum(axis=1,min_count=1).to_numpy(),
                            "age2020_observed_available":observed65,
                            "sex2020_observed_available":observed_sex})
    holdout.to_csv(folder/"local2020_population_age_holdout.csv",index=False)
    # PUF2022 observed joint association supplements actual2020local age counts for named towns.
    puf = pd.read_csv(folder/"census2022_PUF_joint_anchor.csv")
    puf2022 = np.zeros_like(base)
    ix = {int(code):i for i,code in enumerate(codes)}
    for row in puf.loc[puf.locality_code.isin(codes)].itertuples():
        puf2022[ix[int(row.locality_code)],int(row.age_band)-1,SEXES_REVERSE[row.sex],GROUPS_REVERSE[row.broad_group]] += row.population
    ng2022 = _coarse(_national_array(national,2022))/np.maximum(coarse2020,1e-30)
    anchor2022 = base*ng2022[None,:,:,:]
    named = puf2022.sum(axis=(1,2,3))>0
    # Children are absent from the adult PUFexport; keep projected observed2020child cells.
    anchor2022[named,1:,:,:] = puf2022[named,1:,:,:]
    table2022 = _local_tables(annual,codes,2022)
    total2022 = table2022.get("total_population",pd.Series(index=codes,dtype=float)).to_numpy(float)
    fallback2022 = total2020*_national_array(national,2022).sum()/n2020.sum()
    total2022 = np.where(np.isfinite(total2022)&(total2022>0),total2022,fallback2022)
    gp2022 = _group_proportions(table2022,anchor2022.sum(axis=(1,2))/anchor2022.sum(axis=(1,2,3))[:,None],_national_array(national,2022).sum(axis=(0,1)))
    anchor2022 = _support_new_groups(anchor2022,gp2022*total2022[:,None],_coarse(_national_array(national,2022)))
    anchor2022 = rake_joint(anchor2022,group=gp2022*total2022[:,None])
    age_coarse = np.array([0]*18+[1]*7+[2]*10+[3]*10+[4]*10+[5]*10+[6]*21)
    registry_age_intervals = {"youngest_source_bin":(0,5),"age6_18":(6,18),"age19_45":(19,45),"age46_55":(46,55),"age56_64":(56,64),"age65_plus":(65,85)}
    registry_frames = {(int(year),str(date)):z.pivot(index="locality_code",columns="category",values="value").reindex(codes)
                       for (year,date),z in registry.groupby(["year","reference_date"])}
    catalog_path=getattr(args,"locality_catalog",None)
    if catalog_path is None and "founding_year" not in geog:
        raise ValueError("Historical reconstruction requires an official locality founding-year catalog")
    founding=pd.to_numeric(geog.founding_year,errors="coerce").to_numpy(float,copy=True) if "founding_year" in geog else np.full(len(codes),np.nan)
    if catalog_path is not None and "founding_year" not in geog:
        catalog=json.loads(catalog_path.read_text(encoding="utf-8"))["result"]["records"]
        lookup={int(record["סמל יישוב"]):record for record in catalog if str(record.get("סמל יישוב","")).isdigit()}
        for i,code in enumerate(codes):
            text=str(lookup.get(int(code),{}).get("שנת ייסוד",""))
            match=re.search(r"(?:1[0-9]{3}|20[0-9]{2})",text)
            if match:
                founding[i]=int(match.group())
        input_hashes["locality_catalog"]=sha256(catalog_path)
        source_paths["locality_catalog"]=catalog_path
    latest_catalog=json.loads(args.latest_locality_catalog.read_text(encoding="utf-8"))["result"]["records"]
    latest_by_code={int(row["סמל יישוב"]):row for row in latest_catalog if str(row.get("סמל יישוב","")).isdigit()}
    for i,code in enumerate(codes):
        if not np.isfinite(founding[i]):
            match=re.search(r"(?:1[0-9]{3}|20[0-9]{2})",str(latest_by_code.get(int(code),{}).get("שנת ייסוד","")))
            if match:
                founding[i]=int(match.group())
    raw_resident_base=geo2020.Pop_Total.copy()
    missing_base=raw_resident_base.isna()|raw_resident_base.le(0)
    raw_resident_base.loc[missing_base]=observed_total2020[keep][missing_base.to_numpy()]
    resident_base_mapping={int(code):float(value) for code,value in raw_resident_base.items() if pd.notna(value)and value>0}
    registry_mapping={year:{int(row.locality_code):float(row.value)for row in registry_totals.loc[registry_totals.year.eq(year)].sort_values("reference_date").itertuples()}
                      for year in [2020,2026]}
    registry_current=registry_mapping[2026]
    latest_table=_local_tables(annual,codes,2023)
    latest_total=latest_table.get("total_israeli",latest_table.get("total_population",pd.Series(index=codes,dtype=float))).to_numpy(float)
    dates = {str(year):(year,f"{year}-07-01") for year in range(2003,2027)}
    dates.update({"K16":(2003,"2003-01-28"),"K18":(2009,"2009-02-10"),"K19":(2013,"2013-01-22"),"K20":(2015,"2015-03-17"),
                  "K21":(2019,"2019-04-09"),"K22":(2019,"2019-09-17"),"K23":(2020,"2020-03-02"),"K24":(2021,"2021-03-23"),"K25":(2022,"2022-11-01"),"K26":(2026,"2026-10-27")})
    priority=["2022","2026","K21","K22","K23","K24","K25","K26"]
    dates={key:dates[key] for key in priority+[key for key in dates if key not in priority]}
    coverage=[]
    reconciliation=[]
    resident_priors=[]
    temporary=folder/"adult_controls.building.csv.gz"
    with gzip.open(temporary,"wt",encoding="utf-8",newline="") as stream:
        for period,(year,date) in dates.items():
            nat = national_at_date(national,date) if period.startswith("K") else _national_array(national,year)
            ncoarse = _coarse(nat)
            if year<=2020:
                joint = base*(ncoarse/np.maximum(coarse2020,1e-30))[None,:,:,:]
            elif year>=2022:
                joint = anchor2022*(ncoarse/np.maximum(_coarse(_national_array(national,2022)),1e-30))[None,:,:,:]
            else:
                joint = (base+anchor2022)/2
            joint,lifecycle_prior_audit=dated_lifecycle_prior(joint,codes,date,registry_current)
            table = _local_tables(annual,codes,year)
            total = table.get("total_israeli",table.get("total_population",pd.Series(index=codes,dtype=float))).to_numpy(float)
            fallback = total2020*nat.sum()/n2020.sum() if year<2020 else total2022*nat.sum()/_national_array(national,2022).sum()
            if year>2023:
                fallback=np.where(np.isfinite(latest_total)&(latest_total>0),latest_total*nat.sum()/_national_array(national,2023).sum(),fallback)
            total = np.where(np.isfinite(total)&(total>0),total,fallback)
            growth_sources=np.full(len(codes),"actual_CBS_year_or_national_growth_fallback",dtype=object)
            if year==2026:
                total,growth_sources=local_register_growth_prior(codes,resident_base_mapping,registry_mapping[2020],registry_current,total,date)
            #2023code timing is unresolved: retain the physical family in its
            #pooled parent, including both separately published endyear totals.
            for event in life.EVENTS:
                if event["kind"]=="split" and pd.Timestamp(event["earliest"])<=pd.Timestamp(date)<=pd.Timestamp(event["latest"]):
                    parent=event["predecessors"][0]
                    child=next(code for code in event["successors"]if code!=parent)
                    if parent in codes and child in codes:
                        total[codes==parent]+=float(total[codes==child].sum())
                        total[codes==child]=0
            period_active=active_localities(founding,year)
            for event in life.EVENTS:
                if event["kind"]=="merge":
                    for code in (*event["predecessors"],*event["successors"]):
                        period_active[codes==code]&=life.active_status(code,date)!="inactive"
                elif event["kind"]=="split":
                    child=next(code for code in event["successors"]if code not in event["predecessors"])
                    if pd.Timestamp(date)<=pd.Timestamp(event["latest"]):
                        period_active[codes==child]=False
            total[~period_active]=0
            joint[~period_active]=0
            fallback_group=np.broadcast_to(nat.sum(axis=(0,1))/nat.sum(),(len(codes),3)).copy()
            fallback_group=np.divide(joint.sum(axis=(1,2)),joint.sum(axis=(1,2,3))[:,None],out=fallback_group,where=joint.sum(axis=(1,2,3))[:,None]>0)
            group_table=table if year<=2023 else latest_table
            gp = _group_proportions(group_table,fallback_group,nat.sum(axis=(0,1)))
            joint = _support_new_groups(joint,gp*total[:,None],ncoarse)
            joint = rake_joint(joint,group=gp*total[:,None])
            nii_count=0
            if year in [2003,2006]:
                nii=annual.loc[annual.year.eq(year)&annual.dimension.eq("age")&annual.category.isin(["age_0_17","adult18_plus"])].pivot(index="locality_code",columns="category",values="value").reindex(codes)
                if {"age_0_17","adult18_plus"}<=set(nii):
                    nii_valid=nii.notna().all(axis=1).to_numpy()&nii.sum(axis=1).gt(0).to_numpy()
                    old_child=joint[:,0].sum(axis=(1,2))
                    old_adult=joint[:,1:].sum(axis=(1,2,3))
                    for cell,old,column in [(slice(0,1),old_child,"age_0_17"),(slice(1,7),old_adult,"adult18_plus")]:
                        ratio=np.divide(nii[column].to_numpy(float),old,out=np.ones_like(old),where=nii_valid&(old>0))
                        joint[:,cell,:,:]*=ratio[:,None,None,None]
                    total=joint.sum(axis=(1,2,3))
                    gp=np.divide(joint.sum(axis=(1,2)),total[:,None],out=gp.copy(),where=total[:,None]>0)
                    nii_count=int(nii_valid.sum())
            matching = [(key,value) for key,value in registry_frames.items() if key[0]==year]
            registry_count=0
            source_date=""
            if matching:
                (key,reg) = min(matching,key=lambda item:abs((pd.Timestamp(item[0][1])-pd.Timestamp(date)).days))
                source_date=key[1]
                fine = joint[:,age_coarse,:,:]*np.divide(nat,ncoarse[age_coarse],out=np.zeros_like(nat),where=ncoarse[age_coarse]>0)[None,:,:,:]
                fine += total[:,None,None,None]*(nat/nat.sum())[None,:,:,:]*1e-6
                fine_total=fine.sum(axis=(1,2,3))
                fine *= np.divide(total,fine_total,out=np.zeros_like(total),where=fine_total>0)[:,None,None,None]
                regtotal = reg["all_ages"].to_numpy(float)
                valid = np.isfinite(regtotal)&(regtotal>0)
                for category,(lo,hi) in registry_age_intervals.items():
                    if category not in reg:
                        raise ValueError("Incomplete administrative agepartition")
                    target = reg[category].to_numpy(float)/regtotal*total
                    actual = fine[:,lo:hi+1,:,:].sum(axis=(1,2,3))
                    ratio = np.divide(target,actual,out=np.ones_like(actual),where=valid&(actual>0))
                    fine[:,lo:hi+1,:,:] *= ratio[:,None,None,None]
                joint = np.stack([fine[:,age_coarse==i,:,:].sum(axis=1) for i in range(7)],axis=1)
                # Preserve actual broadgroup fractions while calibrating administrative agefractions.
                age_targets = joint.sum(axis=(2,3))
                # PUF sample zeros are not population structural zeros. Supply a
                # tiny, explicit national demographic support prior for raking;
                # observed zero target margins still remain exactly zero.
                support=ncoarse/ncoarse.sum()
                joint += total[:,None,None,None]*support[None,:,:,:]*1e-6
                joint = rake_joint(joint,age=age_targets,group=gp*total[:,None])
                registry_count=int((valid&period_active).sum())
            # Known not-yet-founded places cannot acquire historical residents.
            inactive=~period_active
            joint[inactive]=0
            outside=np.zeros(joint.shape[1:])
            for event in life.EVENTS:
                move_outside=event["kind"]=="outside_localities"and pd.Timestamp(date)>pd.Timestamp(event["latest"])
                if move_outside:
                    for code in event["predecessors"]:
                        if code in codes:
                            outside+=joint[codes==code].sum(axis=0)
                            joint[codes==code]=0
            output_codes=codes
            if outside.sum()>0:
                output_codes=np.r_[codes,life.OUTSIDE_LOCALITIES]
                joint=np.concatenate([joint,outside[None,:,:,:]],axis=0)
            joint,level_audit=rake_national_absolute(joint,ncoarse)
            reconciliation.append({"period_id":period,"reference_date":date,"known_pre_founding_localities_excluded":int(inactive.sum()),
                                   "historical_NII_localities":nii_count,"lifecycle_prior":lifecycle_prior_audit,
                                   "outside_localities_prior_population":float(outside.sum()),
                                   "local_total_prior_sources":pd.Series(growth_sources[period_active]).value_counts().to_dict(),
                                   "lifecycle_uncertain_codes":life.lifecycle_plan(codes,date)["uncertain_codes"],**level_audit})
            rows=[]
            for local,iage,isex,igroup in np.argwhere(joint[:,1:,:,:]>0):
                rows.append((period,year,date,int(output_codes[local]),int(iage+2),["male","female"][isex],["jewish","arab","other"][igroup],
                             float(joint[local,iage+1,isex,igroup]),"observed_margins_estimated_joint","2020_local;2022_PUF;annual_national;dated_registry",source_date))
            frame = pd.DataFrame(rows,columns=["period_id","year","reference_date","locality_code","age_band","sex","broad_group","adult_population","status","source_id","registry_reference_date"])
            frame.to_csv(stream,index=False,header=not coverage)
            if period in ["K25","K26"]:
                citizen_input = frame.groupby(["year","locality_code","broad_group"],as_index=False).adult_population.sum()
                citizen_input["period_id"]=period
                citizen_input["reference_date"]=date
                citizen_input["population_basis"]="reconstructed_resident_adults_not_citizens"
                resident_priors.append(citizen_input)
                pd.concat(resident_priors,ignore_index=True).to_csv(folder/"resident_adult_priors.csv",index=False)
            coverage.append({"period_id":period,"year":year,"localities":int(frame.locality_code.nunique()),"adult_population":float(frame.adult_population.sum()),"registry_age_fraction_localities":registry_count,"registry_reference_date":source_date})
            print(json.dumps({"phase":"population_period",**coverage[-1]}),flush=True)
    temporary.replace(folder/"adult_controls.csv.gz")
    for key,path in source_paths.items():
        if sha256(path)!=input_hashes[key]:
            raise ValueError("Population source changed during reconstruction: "+key)
    audit={"population_reconstruction_complete":True,"runtime_seconds":time.perf_counter()-started,"python":platform.python_version(),
           "localities":len(codes),"observed2020_local_age_localities":int((~no_age).sum()),"observed2008_subdistricts":regions,
           "excluded_zero_population2008_water_features":len(json.loads(args.subdistrict2008.read_text(encoding="utf-8"))["features"])-regions,
           "periods":coverage,"registry_only_population_universe_localities":[int(code) for code in codes[registry_only]],
           "adult_controls_sha256":sha256(folder/"adult_controls.csv.gz"),
           "input_sha256_at_start":input_hashes,"reconstruction_source_sha256_at_start":code_hash,
           "known_founding_year_localities":int(np.isfinite(founding).sum()),
           "unknown_founding_year_localities":int((~np.isfinite(founding)).sum()),
           "national_reconciliation":reconciliation,
           "geographic_coverage_assumption":"National resident joint totals allocated across supported active localitycodes; no authoritative CBS unlocated-person total supplied. PIBAcode0 is separately retained register information, not equated to CBS residents.",
           "raw2020_unknown_age_residual_total":float(np.maximum(total2020-raw_age_known,0).sum()),
           "local2022_age_prior_localities":int(local2022_used.sum()),
           "local2022_age_prior_complete3bin_localities":int(local2022_complete.sum()),
           "no2020_age_localities_using_local2022_prior":int((no_age&local2022_used).sum()),
           "unallocated2020_published_age_rounding_residual":float(rounding_residual.sum()),
           "age_imputation_priority":"Known2020five-year components unchanged; missingcomponents use published2022local0–19/20–64/65+ shares backcast by national2020/22count ratios, then2008regional/nationalprior where unavailable",
           "local2026_total_prior":"CBS2020localresidentlevel times actual same-seriesPIBA2026/2020 localtotalgrowth, using merged/split predecessor denominators. Missing2020bases use explicitly registered-person geography priors; national CBS reconciliation remains separate.",
           "lifecycle_scope":"Reviewed municipal merge, split, recognition and outside-locality events applied before national reconciliation; code birth does not imply physical population birth. Uncertain2023 split remains pooled until2024.",
           "outside_locality_code":life.OUTSIDE_LOCALITIES,
           "quality_scope":"Regional-fallback diagnostic is not the full pipeline or a claim of improvement over frozen locality2022. Separate local2022-backcast diagnostic measures the preferred imputation prior on strict observed targets.",
           "national_annual_universe":"CBSresidentpopulation; separately retainedfromPIBAregistryincludingabroad/noncitizens",
           "registry_use":"Observed geographic ageprofile association prior, subsequently reconciled to exact dated CBS national resident age×sex×group; raw register fractions are NOT asserted resident fractions",
           "age18_method":"National singleageprior allocates18 within observed6–18 registrybin; other19+registrybinsobserved",
           "older_years":"Actual2020localagepatterns backcast by dated national age×sex×group trajectories;2008regionalage×sex supplies associationprior, no claimobserved historical localjoint",
           "PUF2022_use":"Observed adult age×sex×group association for200namedtowns; conditionalincome/religiosity handled byfitconsumer",
           "national_date_alignment":"Calendarperiods useJuly1annual averages; Kperiods interpolate observed national anchors toactualelectiondate. K26uses2023–25 log-growth projected1.33years afterJuly2025; notobserved2026nationalcounts.",
           "sample_zero_support":"Nationalage×sex×group prior mass1e-6 prevents PUFsample-zero cells from becoming false population structural zeros; observed zero targetmargins remainzero.",
           "quality_holdout":"Entire2020localagecells withheldglobally, including2020nationalanchor; actual2008regionalprior+interpolated2019/2021national+observedsex/groupcontrols predictheldoutage65",
           "published_parameters_written":False}
    (folder/"reconstruction_readiness.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return audit


SEXES_REVERSE = {value:key-1 for key,value in SEXES.items()}
GROUPS_REVERSE = {value:key-1 for key,value in GROUPS.items()}


def build(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    national, religions = census2022_national(args.national2022)
    puf = puf_anchor(args.puf)
    local_margins = census2022_local_margins(args.local2022)
    national.to_csv(args.output/"census2022_national_singleage.csv", index=False)
    religions.to_csv(args.output/"census2022_national_religion_singleage.csv", index=False)
    puf.to_csv(args.output/"census2022_PUF_joint_anchor.csv", index=False)
    local_margins.to_csv(args.output/"census2022_local_observed_margins.csv", index=False)
    adult = national.loc[national.age.ge(18)].groupby(["age_band", "sex", "broad_group"], as_index=False).population.sum()
    puf_national = puf.groupby(["age_band", "sex", "broad_group"], as_index=False).population.sum()
    comparison = adult.merge(puf_national, on=["age_band", "sex", "broad_group"], suffixes=("_Census", "_PUF"), validate="one_to_one")
    comparison["difference"] = comparison.population_PUF-comparison.population_Census
    comparison.to_csv(args.output/"census2022_anchor_basis_comparison.csv", index=False)
    audit = {"phase": "observed_anchors_built", "python": platform.python_version(), "runtime_seconds": time.perf_counter()-started,
             "source_sha256": {"national2022": sha256(args.national2022), "local2022": sha256(args.local2022), "puf": sha256(args.puf)},
             "national_singleage_cells": len(national), "PUF_joint_cells": len(puf),
             "identified_PUF_localities": int(puf.loc[puf.geography_status.eq("identified_locality"), "locality_code"].nunique()),
             "PUF_masked_locality_adults": float(puf.loc[puf.locality_code.eq(9999), "population"].sum()),
             "Census_resident_adults_excluding_foreigners": float(adult.population.sum()), "PUF_weighted_adults": float(puf.population.sum()),
             "dominant_locality_religion_used_to_label_individuals": False,
             "population_basis_reconciliation_required": "PUF and all-resident Census anchors retain separate denominators; institution/universe differences must be reconciled before electorate interpretation"}
    (args.output/"anchor_readiness.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["anchors","reconstruct"],default="anchors")
    parser.add_argument("--puf", type=Path)
    parser.add_argument("--national2022", type=Path)
    parser.add_argument("--local2022", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name in ["national-controls","annual-targets","gdb-attributes","subdistrict2008","geography","registry-age","locality-catalog","latest-locality-catalog"]:
        parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    required=["puf","national2022","local2022"] if args.mode=="anchors" else ["national_controls","annual_targets","gdb_attributes","subdistrict2008","geography","registry_age","latest_locality_catalog"]
    if any(getattr(args,key) is None for key in required):
        parser.error("Required for mode: "+", ".join(required))
    build(args) if args.mode=="anchors" else reconstruct(args)


if __name__ == "__main__":
    main()
