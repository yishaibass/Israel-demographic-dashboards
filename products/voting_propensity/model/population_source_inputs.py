"""Reproduce population source conversions from immutable public aggregate data.

The retained CBS2020 geodatabase is read without geometry. CBS2008 subdistricts
and the CBS2022 locality catalog retain their exact raw responses and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import pandas as pd


GDB_SHA256 = "ef0bb63fc504dd58d6bd088d4b79d93a8211997278d9a3fe3e3a8cf998b94b62"
REGIONAL_URL = "https://services2.arcgis.com/xMRYm7cNgdR5RN6F/arcgis/rest/services/Pop_Sex_Age_Relig_SD/FeatureServer/0/query?where=1%3D1&outFields=%2A&returnGeometry=false&f=json&resultRecordCount=1000"
REGIONAL_SHA256 = "f2044d7aa6579b1918b94f5fb3c3d89f4201e1ac10843ff465fd948ae40d5729"
CATALOG_SHA256 = "2ce956f537a4d9bc6ff434a86f4d6bb02ac1814e8d41e61764b9f1d93de2d141"


def verify(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Population source content does not match reviewed identity: " + Path(path).name)
    return data


def convert_gdb(path, output):
    import pyogrio
    from pyogrio.raw import read
    verify(path, GDB_SHA256)
    with zipfile.ZipFile(path) as archive:
        directories = sorted({name.split(".gdb/")[0] + ".gdb" for name in archive.namelist() if ".gdb/" in name})
        if len(directories) != 1 or archive.testzip() is not None:
            raise ValueError("Expected one intact CBS geodatabase")
    virtual = "/vsizip/" + Path(path).resolve().as_posix() + "/" + directories[0]
    layers = pyogrio.list_layers(virtual)
    if len(layers) != 1 or layers[0, 0] != "statisticalareas_2020_demography":
        raise ValueError("CBS2020 demographic layer changed")
    metadata, _, geometry, columns = read(virtual, layer=layers[0, 0], read_geometry=False)
    if geometry is not None:
        raise ValueError("Population conversion unexpectedly read geometry")
    frame = pd.DataFrame(dict(zip(metadata["fields"], columns)))
    required = ["SEMEL_YISHUV", "STAT11", "Pop_Total", "Male_Total", "Female_Total"] + [f"age_{a}_{a+4}" for a in range(0,85,5)] + ["age_85_up"]
    if set(required) - set(frame) or len(frame) != 3187:
        raise ValueError("Unexpected CBS2020 demographic schema")
    path_out = output / "CBS2020_demography_GDB_attributes.csv"
    frame.to_csv(path_out, index=False)
    return dict(source_id="CBS2020_demography_GDB", raw_sha256=GDB_SHA256,
                extracted_sha256=hashlib.sha256(path_out.read_bytes()).hexdigest(),
                rows=len(frame), geometry_extracted=False, reader="pyogrio" + pyogrio.__version__)


def convert_geography(catalog_path, output):
    data = json.loads(verify(catalog_path, CATALOG_SHA256))
    result = data.get("result", {})
    records = result.get("records", [])
    if data.get("success") is not True or len(records) != result.get("total"):
        raise ValueError("Incomplete CBS2022 locality catalog")
    rows = []
    for record in records:
        code = pd.to_numeric(record.get("סמל יישוב"), errors="coerce")
        if pd.isna(code) or code <= 0:
            continue
        founding = re.search(r"(?:1[0-9]{3}|20[0-9]{2})", str(record.get("שנת ייסוד", "")))
        coordinate = str(record.get("קואורדינטות", ""))
        coordinate = coordinate.split(".")[0]
        coordinate = coordinate.zfill(12) if coordinate.isdigit() else ""
        rows.append(dict(locality_code=int(code), locality_name=record.get("שם יישוב"),
                         district_code=record.get("סמל מחוז"), subdistrict_code=record.get("סמל נפה"),
                         natural_area_code=record.get("אזור טבעי"),
                         easting=int(coordinate[:6]) if coordinate else None,
                         northing=int(coordinate[6:]) if coordinate else None,
                         founding_year=int(founding.group()) if founding else None,
                         founding_year_source_text=record.get("שנת ייסוד", "")))
    frame = pd.DataFrame(rows)
    if frame.locality_code.duplicated().any():
        raise ValueError("CBS locality catalog repeats locality identities")
    frame.to_csv(output / "locality_geography.csv", index=False)
    return dict(source_id="CBS2022_locality_catalog", raw_sha256=CATALOG_SHA256,
                localities=len(frame), founding_year_localities=int(frame.founding_year.notna().sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdb-archive", type=Path, required=True)
    parser.add_argument("--locality-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    regional = args.output / "CBS2008_subdistrict_attributes.json"
    if not regional.exists() and args.download:
        with urllib.request.urlopen(REGIONAL_URL, timeout=60) as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != REGIONAL_SHA256:
            raise ValueError("CBS2008 service response differs from retained reviewed source")
        regional.write_bytes(raw)
    payload = json.loads(verify(regional, REGIONAL_SHA256))
    if payload.get("exceededTransferLimit") or len(payload.get("features", [])) != 18:
        raise ValueError("Incomplete CBS2008 subdistrict response")
    audit = dict(gdb=convert_gdb(args.gdb_archive, args.output),
                 geography=convert_geography(args.locality_catalog, args.output),
                 regional=dict(source_url=REGIONAL_URL, sha256=REGIONAL_SHA256, features=18,
                               scope="2008 subdistrict age-sex aggregates; three zero-population water features are not inhabited regions"))
    (args.output / "population_source_inputs_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
