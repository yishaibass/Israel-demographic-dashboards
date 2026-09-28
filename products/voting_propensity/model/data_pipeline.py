"""Census PUF loading and demographic feature construction.

Raw microdata are intentionally not distributed with this repository. Set
``ISRAEL_DASHBOARD_DATA_DIR`` or pass ``--puf`` to the model entry point.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ISRAEL_DASHBOARD_DATA_DIR", REPO_ROOT / "data" / "raw"))
PUF_DEFAULT = DATA_DIR / "census" / "census_2022_puf.csv"

RAW_FEATURES = [
    "KvutzaUchlusiyaPUF", "MinPUF", "GilPUF", "DatiyutPUF",
    "MatzavMishpachtiDeYurePUF", "MakomLeidaPUF", "YabeshetMotzaMchlkPUF",
    "OleShnot90PUF", "OleHadashPUF", "MspShnotLimudPUF",
    "LimudToarAcademiPUF", "ShayachutKoahAvdShavuaPUF", "HchnsAvgChdshPratPUF",
    "MachozPUF", "TzuratYishuvPUF", "SmlYishuvPUF", "TatRovaKtvtMegurimPUF",
    "MishkalPratPUF",
]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create the pre-specified demographic design matrix."""
    age_code = pd.to_numeric(df.GilPUF, errors="coerce").fillna(4).clip(2, 7)
    age = age_code.map({2: 21, 3: 30, 4: 40, 5: 50, 6: 60, 7: 72}).fillna(40)
    income = pd.to_numeric(df.HchnsAvgChdshPratPUF, errors="coerce").fillna(0).clip(0, 20)
    years_code = pd.to_numeric(df.MspShnotLimudPUF, errors="coerce").fillna(0).clip(0, 6)
    years = years_code.map({0: 0, 1: 2.5, 2: 6.5, 3: 9.5, 4: 11.5, 5: 14, 6: 17}).fillna(0)
    pop = pd.to_numeric(df.KvutzaUchlusiyaPUF, errors="coerce").fillna(0)
    rel = pd.to_numeric(df.DatiyutPUF, errors="coerce").fillna(0)
    sex = pd.to_numeric(df.MinPUF, errors="coerce").fillna(0)
    origin = pd.to_numeric(df.YabeshetMotzaMchlkPUF, errors="coerce").fillna(0)
    acad = pd.to_numeric(df.LimudToarAcademiPUF, errors="coerce").fillna(0)
    employed = pd.to_numeric(df.ShayachutKoahAvdShavuaPUF, errors="coerce").fillna(0)
    ole90 = pd.to_numeric(df.OleShnot90PUF, errors="coerce").fillna(0)
    out = pd.DataFrame(index=df.index)
    out["intercept"] = 1.0
    out["age_z"] = (age - 45) / 18
    out["age_z2"] = out.age_z ** 2
    out["income_log_z"] = (income - 10) / 6.0
    out["education_z"] = (years - 12) / 5
    out["female"] = sex.eq(2).astype(float)
    out["arab"] = pop.eq(2).astype(float)
    out["other_pop"] = pop.eq(3).astype(float)
    out["haredi"] = ((pop == 1) & (rel == 5)).astype(float)
    out["dati"] = ((pop == 1) & (rel == 3)).astype(float)
    out["masorti"] = ((pop == 1) & rel.isin([2, 6])).astype(float)
    out["academic"] = acad.gt(0).astype(float)
    out["employed"] = employed.isin([1, 2]).astype(float)
    out["fsu_proxy"] = ((origin == 1) | (ole90 > 0)).astype(float)
    out["haredi_x_income"] = out.haredi * out.income_log_z
    out["dati_x_income"] = out.dati * out.income_log_z
    out["arab_x_income"] = out.arab * out.income_log_z
    out["haredi_x_age"] = out.haredi * out.age_z
    out["dati_x_age"] = out.dati * out.age_z
    out["arab_x_age"] = out.arab * out.age_z
    out["fsu_x_age"] = out.fsu_proxy * out.age_z
    out["academic_x_income"] = out.academic * out.income_log_z
    return out.astype("float32")


def load_cells(puf_path: Path, targets: pd.DataFrame, chunksize: int = 250_000):
    """Load adult PUF records and collapse identical design rows by geography."""
    keys = set(targets.geo_key)
    parts = []
    for chunk in pd.read_csv(puf_path, usecols=RAW_FEATURES, chunksize=chunksize, low_memory=False):
        age = pd.to_numeric(chunk.GilPUF, errors="coerce")
        weight = pd.to_numeric(chunk.MishkalPratPUF, errors="coerce")
        chunk = chunk[(age >= 2) & (weight > 0)].copy()
        chunk["locality"] = pd.to_numeric(chunk.SmlYishuvPUF, errors="coerce").astype("Int64")
        chunk["rova"] = pd.to_numeric(chunk.TatRovaKtvtMegurimPUF, errors="coerce").fillna(0).astype("Int64")
        chunk["geo_key"] = chunk.locality.astype(str) + "_" + chunk.rova.astype(str)
        chunk = chunk[chunk.geo_key.isin(keys)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise ValueError("No PUF records matched the election geography crosswalk")
    micro = pd.concat(parts, ignore_index=True)
    features = feature_frame(micro)
    tmp = pd.concat([
        micro[["geo_key", "locality", "MishkalPratPUF"]].reset_index(drop=True), features
    ], axis=1)
    fcols = list(features.columns)
    cells = tmp.groupby(["geo_key", "locality"] + fcols, as_index=False, dropna=False).MishkalPratPUF.sum()
    covered = set(cells.geo_key)
    geographies = targets[targets.geo_key.isin(covered)][["geo_key", "locality", "rova"]].drop_duplicates("geo_key")
    geographies = geographies.sort_values("geo_key").reset_index(drop=True)
    geo_index = {key: i for i, key in enumerate(geographies.geo_key)}
    cells = cells[cells.geo_key.isin(geo_index)].copy()
    cells["g"] = cells.geo_key.map(geo_index)
    eligible = targets.groupby("geo_key").eligible.mean().reindex(geographies.geo_key).to_numpy()
    adult = cells.groupby("g").MishkalPratPUF.sum().reindex(range(len(geographies))).to_numpy()
    scale = pd.Series(eligible / adult, index=range(len(geographies)))
    cells["weight"] = cells.MishkalPratPUF * cells.g.map(scale)
    return cells, geographies, fcols, micro
