"""Auditable Census 2022 adult population cells and separately sourced annual targets.

No election outcome enters this builder. Population is not the electoral register.
PUF household religiosity is retained, including mixed/other/unknown categories;
Arab people are not assigned a religion from their locality of residence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd


FEATURES = ["SmlYishuvPUF", "TatRovaKtvtMegurimPUF", "KvutzaUchlusiyaPUF",
            "DatiyutPUF", "MinPUF", "GilPUF", "MishkalPratPUF"]
AGE = {1: "0_17", 2: "18_24", 3: "25_34", 4: "35_44", 5: "45_54", 6: "55_64", 7: "65_plus"}
POPULATION = {1: "jewish", 2: "arab", 3: "other"}
RELIGIOSITY = {1: "secular", 2: "traditional", 3: "religious", 5: "haredi", 6: "mixed", 7: "other"}
KEYS = ["locality_code", "subquarter_code", "population_group", "household_religiosity", "sex", "age_band", "geography_status"]
CATALOG_URL = "https://data.gov.il/api/3/action/package_show?id=localities-in-israel"

# Exact primary URLs and immutable content pins. URL folder labels for2024 are
# misleading: the YearAvg file itself says December2024 and is the dated anchor.
NII_SOURCES = [
    {
        "filename": "2003_locality_population.xls",
        "year": 2003,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/year2003/Documents/2003-1.xls",
        "sha256": "c3a8b8872e75b07c3c6705225bfd612c9a1777c78bed18144bbb772350dd1082",
        "bytes": 115712,
        "publisher": "National Insurance Institute",
        "landing_url": "https://www.btl.gov.il/Publications/yeshuvim/year2003/Pages/default.aspx",
        "reference_period": "December; first-of-month population, verified workbook title and parent methodology"
    },
    {
        "filename": "2006_locality_population.xls",
        "year": 2006,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/year2006/Documents/2006-1.xls",
        "sha256": "32257d0c56b3f39ebcb0cee7130ef6e23d512bd3aa01f589deebf613d6b81b05",
        "bytes": 118784,
        "publisher": "National Insurance Institute",
        "landing_url": "https://www.btl.gov.il/Publications/yeshuvim/year2006/Pages/default.aspx",
        "reference_period": "December; first-of-month population, verified workbook title and parent methodology"
    },
    {
        "filename": "2009_locality_population.xls",
        "year": 2009,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2009/Documents/2009-Y1.xls",
        "sha256": "f9fc1c1daed4a57cf73f122da3fdbacc85b4ea47cece2eca7a816e2592341b22",
        "bytes": 125952,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2010_locality_population.xlsx",
        "year": 2010,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2010/Documents/2010-y1.xlsx",
        "sha256": "9b7d5e2af0a8afc01e9135f63552d14d12e91d37875e91bdffa08dadbea8abcb",
        "bytes": 66231,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2011_locality_population.xls",
        "year": 2011,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2011/Documents/2011-Y1.xls",
        "sha256": "e3d3d69286b6bbd28a6f72c85b1679da5eac92d9c5bfc34b7c7ea2b654605249",
        "bytes": 117760,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2012_locality_population.xls",
        "year": 2012,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2012/Documents/2012-Y1.xls",
        "sha256": "964b230cc7d4867f919cfe3f1911d88376640121f0cea1556374b48ba8af0d53",
        "bytes": 112640,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2013_locality_population.xls",
        "year": 2013,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2013/Documents/2013-Y1.xls",
        "sha256": "0389c0cb1997e0179d200eb8ed009d3bb96f00611634f8bbaeeb0ed41fbd9768",
        "bytes": 123392,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2014_locality_population.xls",
        "year": 2014,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2014/Documents/2014-Y1.xls",
        "sha256": "8f0d173b338b8dafddb270c35cbf6f466b80f9cbfa2872b5f7fb2c9bc2182117",
        "bytes": 123904,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2015_locality_population.xls",
        "year": 2015,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2015/Documents/2015-Y1.xls",
        "sha256": "6fc001c9c3f7095a3fb70ad1159fdc994da05046258a89a708b3d0396c029149",
        "bytes": 124928,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2016_locality_population.xls",
        "year": 2016,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2016/Documents/2016-Y1.xls",
        "sha256": "8b56e022771f914f8253c3c259382bfc1db01b72ca8a730705f8ea489e837c5d",
        "bytes": 133120,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2017_locality_population.xls",
        "year": 2017,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2017/Documents/2017-Y1.xls",
        "sha256": "1d920636e0448c683d359be34a0ca1765b6af3cbc5e7d601385b931de09798e2",
        "bytes": 128000,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2018_locality_population.xls",
        "year": 2018,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2018/Documents/2018-Y1.xls",
        "sha256": "6ee994f5004ecff9602a09c6b1bfe7a977c5d148636d46a0d5b57b16378f688d",
        "bytes": 137216,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2019_locality_population.xls",
        "year": 2019,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2019/Documents/2019-Y1.xls",
        "sha256": "e0ec002f5d4d6c32a7d005c82be0a310f73257d9cecec26b0554e202b6ebd7a5",
        "bytes": 133120,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2020_locality_population.xls",
        "year": 2020,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2020/Documents/2020-Y1.xls",
        "sha256": "4ccc9bc114a8047afe033a071140e2bec93abf45289d97edc43454f61142eed0",
        "bytes": 128512,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2021_locality_population.xls",
        "year": 2021,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2021/Documents/2021-Y1.xls",
        "sha256": "235cf80ad2919f9563a1ceed6348a7a1921644d999b65d8128dc74ba87597dd6",
        "bytes": 120832,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2022_locality_population.xls",
        "year": 2022,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2022/Documents/2022-Y1.xls",
        "sha256": "2c0e56fa59fc78d81f3036a795c1218bbdd4231ef90b6b0690889f2e9c3aac2d",
        "bytes": 128000,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2023_locality_population.xls",
        "year": 2023,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2023/Documents/2023-Y1.xls",
        "sha256": "01ced512e01c75214071945ddbf72f83ebfa90e3ba04bf3fd52738021955846d",
        "bytes": 129024,
        "publisher": "National Insurance Institute"
    },
    {
        "filename": "2024_YearAvg_population.xlsx",
        "year": 2024,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2024/YearAvg/Documents/%D7%9E%D7%A7%D7%91%D7%9C%D7%99%20%D7%92%D7%9E%D7%9C%D7%90%D7%95%D7%AA%20%D7%9C%D7%A4%D7%99%20%D7%99%D7%99%D7%A9%D7%95%D7%91%D7%99%D7%9D%20%D7%91%20-%202024-12.xlsx",
        "sha256": "fcd3b73261c28c5a7eb039d627f5ee289e216bd882160b1af04cc6c4276ae4b4",
        "bytes": 63430,
        "publisher": "National Insurance Institute",
        "landing_url": "https://www.btl.gov.il/Publications/yeshuvim/2024/YearAvg/Pages/default.aspx",
        "reference_period": "December; first-of-month population, verified workbook title and parent methodology"
    },
    {
        "filename": "2025_monthly_population.xlsx",
        "year": 2025,
        "source_url": "https://www.btl.gov.il/Publications/yeshuvim/2025/lastmonth/Documents/%D7%9E%D7%A7%D7%91%D7%9C%D7%99%20%D7%92%D7%9E%D7%9C%D7%90%D7%95%D7%AA%20%D7%9C%D7%A4%D7%99%20%D7%99%D7%99%D7%A9%D7%95%D7%91%D7%99%D7%9D%20%D7%91%20-%202025-12.xlsx",
        "sha256": "6740c1ab333e828178687c92e91af6c3e9067b4f9e30ee3a02325fd90b56a951",
        "bytes": 65730,
        "publisher": "National Insurance Institute",
        "landing_url": "https://www.btl.gov.il/Publications/yeshuvim/2025/lastmonth/Pages/default.aspx",
        "reference_period": "December; first-of-month population, verified workbook title and parent methodology"
    }
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(chunk):
    """Codebook categories with no population redistribution or religion imputation."""
    numeric = chunk[FEATURES].apply(pd.to_numeric, errors="coerce")
    weight = numeric.MishkalPratPUF
    if (weight.dropna() < 0).any():
        raise ValueError("Negative PUF person weights")
    out = pd.DataFrame(index=chunk.index)
    out["locality_code"] = numeric.SmlYishuvPUF.fillna(-1).astype(int)
    out["subquarter_code"] = numeric.TatRovaKtvtMegurimPUF.fillna(-1).astype(int)
    out["population_group"] = numeric.KvutzaUchlusiyaPUF.map(POPULATION).fillna("unknown")
    out["household_religiosity"] = numeric.DatiyutPUF.map(RELIGIOSITY).fillna("unknown")
    out["sex"] = numeric.MinPUF.map({1: "male", 2: "female"}).fillna("unknown")
    out["age_band"] = numeric.GilPUF.map(AGE).fillna("unknown")
    out["geography_status"] = np.where(out.locality_code.eq(9999), "masked_locality_pool",
                                         np.where(out.locality_code.gt(0), "identified_locality", "unknown"))
    out["person_weight"] = weight.fillna(0)
    out["puf_records"] = 1
    out["is_adult"] = numeric.GilPUF.isin([2, 3, 4, 5, 6, 7])
    return out


def collapse_puf(path, chunksize=250_000):
    pieces, stats = [], {"raw_records": 0, "positive_weight_records": 0, "zero_or_missing_weight_records": 0}
    for chunk in pd.read_csv(path, usecols=FEATURES, chunksize=chunksize, low_memory=False):
        frame = classify(chunk)
        stats["raw_records"] += len(frame)
        stats["positive_weight_records"] += int(frame.person_weight.gt(0).sum())
        stats["zero_or_missing_weight_records"] += int(frame.person_weight.le(0).sum())
        frame = frame.loc[frame.person_weight.gt(0)]
        pieces.append(frame.groupby(KEYS+["is_adult"], dropna=False, observed=True)[["person_weight", "puf_records"]].sum().reset_index())
    cells = pd.concat(pieces, ignore_index=True).groupby(KEYS+["is_adult"], dropna=False, observed=True)[["person_weight", "puf_records"]].sum().reset_index()
    stats.update({"weighted_population_all_ages": float(cells.person_weight.sum()),
                  "weighted_adults_18_plus": float(cells.loc[cells.is_adult, "person_weight"].sum()),
                  "weighted_unknown_age": float(cells.loc[cells.age_band.eq("unknown"), "person_weight"].sum()),
                  "adult_puf_records": int(cells.loc[cells.is_adult, "puf_records"].sum())})
    return cells, stats


def population_mix(cells):
    """Explicit proxy categories; unknown Jewish religiosity never becomes secular."""
    frame = cells.copy()
    frame["group"] = frame.population_group
    jewish = frame.population_group.eq("jewish")
    frame.loc[jewish, "group"] = "jewish_"+frame.loc[jewish, "household_religiosity"]
    result = frame.groupby(["locality_code", "geography_status", "group"], observed=True).person_weight.sum().reset_index()
    result["population_total"] = result.groupby("locality_code").person_weight.transform("sum")
    result["share"] = result.person_weight/result.population_total
    return result


def compare_bridge(adult_cells, all_cells, bridge_path):
    """Compare only a transparent coarse crosswalk, retaining unallocated mass."""
    groups = ["haredi", "dati", "masorti", "hiloni", "arab_including_druze", "unallocated"]
    def coarse(cells):
        f = cells.copy()
        f["group"] = "unallocated"
        f.loc[f.population_group.eq("arab"), "group"] = "arab_including_druze"
        for rel, group in {"haredi": "haredi", "religious": "dati", "traditional": "masorti", "mixed": "masorti", "secular": "hiloni"}.items():
            f.loc[f.population_group.eq("jewish") & f.household_religiosity.eq(rel), "group"] = group
        tab = f.pivot_table(index="locality_code", columns="group", values="person_weight", aggfunc="sum", fill_value=0).reindex(columns=groups, fill_value=0)
        return tab.div(tab.sum(axis=1), axis=0)
    adult, all_ages = coarse(adult_cells), coarse(all_cells)
    old = pd.read_csv(bridge_path, usecols=["election", "locality_code", "eligible_voters", "is_envelope", "demog_source"]+["share_"+g for g in groups[:4]]+["share_arab", "share_druze"])
    keep = old.election.eq("K25") & ~old.is_envelope.astype(str).str.lower().isin(["true", "1"]) & old.demog_source.ne("none") & old.eligible_voters.gt(0)
    old = old.loc[keep].dropna()
    old["share_arab_including_druze"] = old.share_arab+old.share_druze
    weights = old.groupby("locality_code").eligible_voters.sum()
    table = old[["share_"+g for g in groups[:5]]].multiply(old.eligible_voters, axis=0)
    table["locality_code"] = old.locality_code
    prior = table.groupby("locality_code").sum().div(weights, axis=0)
    prior.columns = groups[:5]
    prior["unallocated"] = 0.
    rows = []
    for code in adult.index.intersection(prior.index):
        for group in groups:
            rows.append({"locality_code": code, "group": group, "old_bridge_share": prior.loc[code, group],
                         "all_age_puf_share": all_ages.loc[code, group], "adult18_puf_share": adult.loc[code, group],
                         "adult_minus_all_age_pp": 100*(adult.loc[code, group]-all_ages.loc[code, group]),
                         "adult_minus_old_bridge_pp": 100*(adult.loc[code, group]-prior.loc[code, group]),
                         "k25_mapped_eligible": weights.loc[code]})
    return pd.DataFrame(rows)


def archive_url(url, destination, *, metadata_path=None, expected_sha256=None):
    """Immutable raw bytes plus URL/date/SHA; never overwrite a changed source."""
    destination = Path(destination)
    metadata = Path(metadata_path) if metadata_path is not None else destination.with_suffix(destination.suffix+".json")
    if destination.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["source_url"] != url or saved["sha256"] != sha256(destination):
            raise ValueError("Archived source URL/hash mismatch: "+destination.name)
        if expected_sha256 is not None and saved["sha256"] != expected_sha256:
            raise ValueError("Archived source differs from pinned SHA256: "+destination.name)
        return saved
    request = urllib.request.Request(url, headers={"User-Agent": "Israel-demographic-foundation-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    if not body:
        raise ValueError("Empty source response")
    signatures = {".xlsx": b"PK", ".xls": bytes.fromhex("d0cf11e0"), ".pdf": b"%PDF"}
    if destination.suffix in signatures and not body.startswith(signatures[destination.suffix]):
        raise ValueError("Expected workbook/PDF, received another response")
    if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("Downloaded source differs from pinned SHA256: "+destination.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    saved = {"source_url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
             "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "content_type": content_type}
    metadata.write_text(json.dumps(saved, indent=2)+"\n", encoding="utf-8")
    return saved


def download_nii(raw):
    """Fetch all pinned primary workbooks needed by the annual/projection build."""
    folder = Path(raw)/"nii_annual"
    result = []
    for source in NII_SOURCES:
        filename = source["filename"]
        if Path(filename).name != filename:
            raise ValueError("Source filename must be a single path component")
        path = folder/filename
        metadata = path.with_suffix(".metadata.json")
        existing = path.exists()
        saved = archive_url(source["source_url"], path, metadata_path=metadata, expected_sha256=source["sha256"])
        if path.stat().st_size != source["bytes"]:
            raise ValueError("Pinned NII source byte count differs: "+filename)
        if not existing:
            saved = {**{key: value for key, value in source.items() if key != "filename"}, **saved}
            metadata.write_text(json.dumps(saved, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        result.append({"file": filename, **saved})
    return result


def download_catalog(raw):
    archive_url(CATALOG_URL, raw/"localities_catalog.json")
    catalog = json.loads((raw/"localities_catalog.json").read_text(encoding="utf-8"))["result"]
    records = []
    for resource in catalog["resources"]:
        year = int(resource["url"].split("bycode")[-1][:4])
        name = "localities_"+str(year)+".json"
        url = "https://data.gov.il/api/3/action/datastore_search?resource_id="+resource["id"]+"&limit=10000"
        saved = archive_url(url, raw/name)
        payload = json.loads((raw/name).read_text(encoding="utf-8"))
        if not payload.get("success") or payload["result"]["resource_id"] != resource["id"] or len(payload["result"]["records"]) != payload["result"]["total"]:
            raise ValueError("Incomplete/mismatched official locality resource: "+name)
        records.append({"year": year, "source_id": resource["id"], "title": resource["name"], "file": name,
                        "records": len(payload["result"]["records"]), "original_workbook_url": resource["url"], **saved})
    return records


def annual_catalog_targets(raw, sources):
    rows = []
    for source in sources:
        if sha256(raw/source["file"]) != source["sha256"]:
            raise ValueError("Archived annual source hash mismatch")
        payload = json.loads((raw/source["file"]).read_text(encoding="utf-8"))
        result = payload["result"]
        if not payload.get("success") or result["resource_id"] != source["source_id"] or len(result["records"]) != result["total"] or len(result["records"]) != source["records"]:
            raise ValueError("Incomplete/mismatched preserved annual resource")
        records = result["records"]
        for record in records:
            code = record.get("סמל יישוב", record.get("סמל"))
            if not isinstance(code, (int, float)) or code <= 0:
                continue
            for field, value in record.items():
                category = None
                if field.startswith("יהודים ואחרים"):
                    category = "jewish_and_other"
                elif field in ["מזה: יהודים", "יהודים"]:
                    category = "jewish"
                elif field.startswith("ערבים"):
                    category = "arab"
                elif field.startswith("סך הכל ישראלים"):
                    category = "total_israeli"
                elif field.startswith("סך הכל אוכלוסייה") or field.startswith("סהכ אוכלוסייה"):
                    category = "total_population"
                if category is None or value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                    continue
                if source["year"] == 2017 and category != "total_population":
                    # Raw ethnic fields use a different scale from the total.
                    # Preserve raw source; do not silently infer a conversion.
                    continue
                rows.append({"year": source["year"], "locality_code": int(code), "locality_name": record.get("שם יישוב", record.get("שם יישוב -A", "")),
                             "dimension": "population_group", "category": category, "value": float(value), "unit": "persons",
                             "age_scope": "all_ages", "status": "provisional" if "ארעי" in field else "source_estimate",
                             "source_id": source["source_id"], "evidence_grade": "official_published_aggregate",
                             "source_field": field})
            combined = record.get("יהודים ואחרים")
            jewish = record.get("מזה: יהודים", record.get("יהודים"))
            if source["year"] >= 2018 and isinstance(combined, (int, float)) and isinstance(jewish, (int, float)) and np.isfinite(combined) and np.isfinite(jewish):
                if combined < jewish:
                    raise ValueError("Jewish count exceeds Jewish+Other")
                rows.append({"year": source["year"], "locality_code": int(code), "locality_name": record.get("שם יישוב", record.get("שם יישוב -A", "")),
                             "dimension": "population_group", "category": "other", "value": float(combined-jewish), "unit": "persons",
                             "age_scope": "all_ages", "status": "arithmetic_derived", "source_id": source["source_id"],
                             "evidence_grade": "official_aggregates_difference", "source_field": "Jewish+Other minus Jewish (same record)"})
            if source["year"] >= 2018:
                total = next((v for k, v in record.items() if k.startswith("סך הכל ישראלים" if source["year"] == 2023 else "סך הכל אוכלוסייה")), None)
                components = {"jewish_and_other": next((v for k, v in record.items() if k.startswith("יהודים ואחרים")), None),
                              "arab": next((v for k, v in record.items() if k.startswith("ערבים")), None)}
                for category, present in components.items():
                    other = components["arab" if category == "jewish_and_other" else "jewish_and_other"]
                    if present is None and all(isinstance(v, (int, float)) and np.isfinite(v) for v in [total, other]) and 0 <= other <= total:
                        rows.append({"year": source["year"], "locality_code": int(code), "locality_name": record.get("שם יישוב", ""),
                                     "dimension": "population_group", "category": category, "value": float(total-other), "unit": "persons",
                                     "age_scope": "all_ages", "status": "arithmetic_derived", "source_id": source["source_id"],
                                     "evidence_grade": "official_aggregates_difference", "source_field": "Same-record Israeli total minus other exhaustive population-group component; null not treated as zero"})
    result = pd.DataFrame(rows)
    if not result.empty and result.duplicated(["year", "locality_code", "dimension", "category"]).any():
        raise ValueError("Duplicate annual target")
    return result


def annual_availability(targets):
    rows = []
    for year in range(2003, 2027):
        table = targets.loc[targets.year.eq(year)] if not targets.empty else targets
        rows.append({"year": year, "downloaded_target_rows": len(table),
                     "localities": int(table.locality_code.nunique()) if not table.empty else 0,
                     "categories": ";".join(sorted(table.category.unique())) if not table.empty else "",
                     "age_scope": ";".join(sorted(table.age_scope.unique())) if len(table) else "not_acquired",
                     "status": "partial_official_margins" if len(table) else "not_acquired",
                     "religiosity_status": "not_observed_in_annual_targets"})
    return pd.DataFrame(rows)


def normalize_name(value):
    return re.sub(r"[\s\-־'\"׳״]+", "", unicodedata.normalize("NFKC", str(value)).strip())


def validate_nii_headers(table, year):
    if table.shape[0] < 5 or table.shape[1] < 5 or str(year) not in str(table.iloc[0, 0]):
        raise ValueError("NII title/year schema changed")
    clean = lambda value: re.sub(r"\s+", "", str(value))
    if (clean(table.iloc[2, 2]) != "17-0" or clean(table.iloc[2, 3]) != "נשים59-18גברים64-18"
            or clean(table.iloc[2, 4]) != "נשים+60גברים+65"):
        raise ValueError("NII age header changed")


def nii_age_targets(raw, catalog):
    """Unique exact normalized-name matches only; unmatched rows retained in audit."""
    names = {}
    for source in catalog:
        records = json.loads((raw/source["file"]).read_text(encoding="utf-8"))["result"]["records"]
        for record in records:
            code = record.get("סמל יישוב", record.get("סמל"))
            name = record.get("שם יישוב", record.get("שם יישוב -A"))
            if isinstance(code, (int, float)) and code > 0 and name:
                names.setdefault(normalize_name(name), set()).add(int(code))
    rows, matches, sources = [], [], []
    for path in sorted((raw/"nii_annual").glob("*_locality_population.xls*")):
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if sha256(path) != metadata["sha256"]:
            raise ValueError("NII source hash mismatch")
        year = int(metadata.get("year", metadata.get("publication_year")))
        source_id = "nii_locality_population_"+str(year)
        table = pd.read_excel(path, sheet_name=1, header=None)
        validate_nii_headers(table, year)
        used = set()
        for index, row in table.iloc[4:].iterrows():
            counts = pd.to_numeric(row.iloc[1:5], errors="coerce")
            if counts.isna().any():
                continue
            total, child, working, retirement = counts.astype(float)
            key = normalize_name(row.iloc[0])
            codes = names.get(key, set())
            status = "unique_normalized_exact" if len(codes) == 1 else "ambiguous_name" if len(codes) > 1 else "unmatched_name_or_aggregate"
            if len(codes) == 1 and next(iter(codes)) in used:
                status = "duplicate_locality_row"
            if min(counts) < 0 or abs(total-child-working-retirement) > 200:
                status = "incoherent_counts"
            code = next(iter(codes)) if len(codes) == 1 else None
            matches.append({"year": year, "source_row": int(index), "source_name": str(row.iloc[0]),
                            "locality_code": code, "status": status, "population": total,
                            "rounding_residual": total-child-working-retirement})
            if status != "unique_normalized_exact":
                continue
            used.add(code)
            for dimension, category, value, scope in [("age", "age_0_17", child, "all_ages"),
                    ("age", "adult18_plus", total-child, "adult18_plus"),
                    ("retirement_age", "working_age_adult", working, "adult18_plus"),
                    ("retirement_age", "retirement_age", retirement, "adult18_plus")]:
                rows.append({"year": year, "locality_code": code, "locality_name": str(row.iloc[0]),
                             "dimension": dimension, "category": category, "value": value, "unit": "persons",
                             "age_scope": scope, "status": "source_estimate_rounded", "source_id": source_id,
                             "year_basis": "source_publication_year", "population_reference_date_status": "unverified",
                             "historical_boundary_status": "unverified; unique normalized name linked to2016-2023 CBS locality code",
                             "evidence_grade": "official_published_aggregate", "source_field": "NII sheet1 columns1-4; women18-59/60+,men18-64/65+"})
        sources.append({"source_id": source_id, "file": path.name, **metadata, "matched_localities": len(used),
                        "population_reference_date_status": "Unverified: workbook title gives publication year, not a population reference date. Do not equate it to year-end counts without methodology evidence.",
                        "age_definition": "Children0-17; working women18-59 men18-64; retirement women60+ men65+. Rounded source counts; adult18+ is total minus children."})
    return pd.DataFrame(rows), pd.DataFrame(matches), sources


def nii_coded_age_targets(raw):
    """New NII December series: official codes and observed true 65+ ages."""
    rows, sources = [], []
    paths = list((raw/"nii_annual").glob("*_monthly_population.xlsx"))+list((raw/"nii_annual").glob("*_YearAvg_population.xlsx"))
    # URL folder names are unreliable: every anchor must actually say December
    # and have the same coded age schema. Retained LastMonth2024 is an average.
    for path in sorted(paths):
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if sha256(path) != metadata["sha256"]:
            raise ValueError("NII coded-age source hash mismatch")
        year = int(metadata.get("year", metadata.get("publication_year")))
        table = pd.read_excel(path, sheet_name=0, header=None)
        clean = lambda value: re.sub(r"\s+", "", str(value))
        if (str(year) not in str(table.iloc[0, 0]) or "דצמבר" not in str(table.iloc[0, 0])
                or [clean(v) for v in table.iloc[2, 4:7]] != ["17-0", "64-18", "65+"]
                or "סמל" not in str(table.iloc[1, 1])):
            raise ValueError("NII coded age/date header changed")
        codes = set()
        for _, row in table.iloc[5:].iterrows():
            code = pd.to_numeric(row.iloc[1], errors="coerce")
            if not np.isfinite(code) or code <= 0 or code != int(code):
                continue
            code = int(code)
            if code in codes:
                raise ValueError("Duplicate official coded NII locality")
            codes.add(code)
            values = pd.to_numeric(row.iloc[3:7].astype(str).str.replace(",", "", regex=False), errors="coerce")
            total, child, working, elder = values
            if values.notna().all() and (min(values) < 0 or total != child+working+elder):
                raise ValueError("NII coded age counts do not reconcile")
            common = {"year": year, "locality_code": code, "locality_name": str(row.iloc[0]), "unit": "persons",
                      "status": "official_administrative_count", "source_id": "nii_december_population_"+str(year),
                      "evidence_grade": "official_coded_administrative_aggregate", "year_basis": "December_first_of_month",
                      "population_reference_date_status": "verified", "population_reference_date": f"{year}-12-01",
                      "population_basis": "NII population; differs from CBS resident-population estimates",
                      "historical_boundary_status": "official_locality_code_in_source",
                      "source_field": "NII December table columns3-6: total/0-17/18-64/65+; suppressed*** remain missing"}
            for dimension, category, value, scope in [("age", "age_0_17", child, "all_ages"),
                    ("adult_age", "age18_64", working, "adult18_plus"), ("adult_age", "age65_plus", elder, "adult18_plus")]:
                if np.isfinite(value):
                    rows.append({**common, "dimension": dimension, "category": category, "value": float(value), "age_scope": scope})
            if np.isfinite(working) and np.isfinite(elder):
                rows.append({**common, "dimension": "age", "category": "adult18_plus", "value": float(working+elder), "age_scope": "adult18_plus"})
        sources.append({"source_id": "nii_december_population_"+str(year), "file": path.name, **metadata,
                        "coded_localities": len(codes), "schema": "True0-17/18-64/65+, firstDecember; suppressed counts not reconstructed"})
    return pd.DataFrame(rows), sources


def project_latest_pair(targets, year=2026):
    """Transparent two-point extrapolation, never an observed annual distribution."""
    rows = []
    accepted = ((targets.dimension.eq("population_group") & targets.category.isin(["jewish_and_other", "arab"])) |
                (targets.dimension.isin(["retirement_age", "adult_age"])))
    for _, frame in targets.loc[accepted].groupby(["locality_code", "dimension", "category"]):
        frame = frame.sort_values("year")
        if len(frame) < 2:
            continue
        before, latest = frame.iloc[-2], frame.iloc[-1]
        if latest.year < 2023 or latest.year >= year or latest.year-before.year != 1:
            continue
        both_zero = before.value == 0 and latest.value == 0
        if not both_zero and (before.value <= 0 or latest.value <= 0):
            continue
        row = latest.to_dict()
        row.update({"year": year, "value": 0. if both_zero else float(latest.value*(latest.value/before.value)**(year-latest.year)),
                    "status": "projected", "evidence_grade": "two_point_compound_extrapolation",
                    "source_id": str(before.source_id)+"|"+str(latest.source_id),
                    "source_field": f"{int(before.year)}->{int(latest.year)} ratio compounded to{year}; no migration/fertility scenario or uncertainty interval"})
        if both_zero:
            row["evidence_grade"] = "two_explicit_zero_anchors_carried_forward"
            row["source_field"] = f"Explicit zeros in{int(before.year)}and{int(latest.year)}carried forward to{year}; assumes no entrants in this category; not an observed structural zero"
        if latest.get("population_reference_date_status") == "unverified":
            row["status"] = "projected_source_year_proxy"
            row["source_field"] += "; publication years used as approximate time index; population reference dates unverified"
        elif latest.get("population_reference_date_status") == "verified":
            row["projection_anchor_reference_date"] = latest.get("population_reference_date")
            row["population_reference_date"] = f"{year}-12-01"
            row["population_reference_date_status"] = "projected"
        rows.append(row)
    return pd.DataFrame(rows)


def build(puf, bridge, output, raw, download=False):
    output.mkdir(parents=True, exist_ok=True)
    cells, stats = collapse_puf(puf)
    adult = cells.loc[cells.is_adult].drop(columns="is_adult")
    adult.to_csv(output/"adult2022_cells.csv", index=False, encoding="utf-8-sig")
    population_mix(adult).to_csv(output/"locality2022_adult_mix.csv", index=False, encoding="utf-8-sig")
    population_mix(cells).to_csv(output/"locality2022_all_age_mix.csv", index=False, encoding="utf-8-sig")
    national = []
    for scope, subset in [("all_ages", cells), ("adult18_plus", adult)]:
        frame = population_mix(subset)
        summed = frame.groupby("group").person_weight.sum()
        for group, value in summed.items():
            national.append({"age_scope": scope, "group": group, "person_weight": value, "share": value/summed.sum()})
    pd.DataFrame(national).to_csv(output/"national2022_mix.csv", index=False, encoding="utf-8-sig")
    comparison = compare_bridge(adult, cells, bridge)
    comparison.to_csv(output/"bridge_comparison2022.csv", index=False, encoding="utf-8-sig")
    if download:
        catalog = download_catalog(raw)
        download_nii(raw)
        (raw/"sources.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    else:
        catalog = json.loads((raw/"sources.json").read_text(encoding="utf-8")) if (raw/"sources.json").exists() else []
        for source in catalog:
            if sha256(raw/source["file"]) != source["sha256"]:
                raise ValueError("Archived annual source hash mismatch")
    annual = annual_catalog_targets(raw, catalog)
    nii, matches, nii_sources = nii_age_targets(raw, catalog)
    if not nii.empty:
        annual = pd.concat([annual, nii], ignore_index=True)
    coded_nii, coded_nii_sources = nii_coded_age_targets(raw)
    if not coded_nii.empty:
        annual = pd.concat([annual, coded_nii], ignore_index=True)
    matches.to_csv(output/"nii_name_matches.csv", index=False, encoding="utf-8-sig")
    projections = project_latest_pair(annual) if not annual.empty else pd.DataFrame()
    projections.to_csv(output/"annual_projections.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output/"annual_targets.csv", index=False, encoding="utf-8-sig")
    annual_availability(annual).to_csv(output/"annual_availability.csv", index=False, encoding="utf-8-sig")
    audit = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
             "puf_file": puf.name, "puf_sha256": sha256(puf), "puf_bytes": puf.stat().st_size,
             "bridge_file": bridge.name, "bridge_sha256": sha256(bridge), "builder_sha256": sha256(__file__),
             "statistics": stats, "adult_cells": len(adult), "adult_locality_codes": int(adult.locality_code.nunique()),
             "identified_adult_localities": int(adult.loc[adult.geography_status.eq("identified_locality"), "locality_code"].nunique()),
             "compared_bridge_localities": int(comparison.locality_code.nunique()), "official_sources": catalog,
             "nii_sources": nii_sources,
             "coded_nii_sources": coded_nii_sources,
             "projection_rows": len(projections),
             "annual_target_years": sorted(annual.year.unique().tolist()) if not annual.empty else [],
             "annual_target_limitations": "Official locality catalog counts are all ages, not adult counts. Jewish+Other stays combined in2023;2018-2022 separately identify Jewish and derive Other by subtraction. No annual religiosity observed. Missing counts are never zero-imputed; only explicit arithmetic differences of known exhaustive components are used.",
             "denominator": "Resident population age18+ by Census PUF person weights; not citizen eligibility or election-register counts",
             "age_rule": "GilPUF codes2 through7, unknown age excluded from adults and reported separately",
             "population_groups": POPULATION, "household_religiosity": RELIGIOSITY,
             "classification": "Other population, mixed/other/unknown religiosity remain explicit; Arab unsplit because PUF has no individual religion",
             "bridge_comparison_crosswalk": "Only for comparison: Jewish mixed joins traditional; secular alone hiloni; Jewish other/unknown plus other-population retained unallocated; Arab+Druze combined",
             "limits": ["PUF household religiosity is not individual identification", "Locality9999 is a masked locality pool including small urban and regional-council localities and must not join election envelopes", "Subquarter cells are not observed statistical-area cells", "No annual six-way religiosity is observed", "Annual population targets must retain year, geography, denominator and provisional/final status", "2016 catalog contains names only; 2017 ethnic fields excluded because their raw scale differs from total and official unit metadata is unavailable", "NII reference population dates and historical locality boundaries remain unverified; its year index is publication-year proxy", "NII source counts are rounded; component sums may differ by up to200persons; exact discrepancy retained per source row"]}
    (output/"audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puf", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    audit = build(args.puf, args.bridge, args.output, args.raw, args.download)
    print(json.dumps(audit["statistics"]))


if __name__ == "__main__":
    main()
