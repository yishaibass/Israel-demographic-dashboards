"""Geographic context features derived from Census variables only."""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_party_family_final as model
import run_multi_election_average as multi


BLOCKS = {
    "own_group_concentration": ["geo_arab_share", "geo_dati_share", "geo_haredi_share",
                                "arab_x_geo_arab", "dati_x_geo_dati", "haredi_x_geo_haredi"],
    "municipality_size": ["locality_logpop_z", "arab_x_logpop", "dati_x_logpop", "haredi_x_logpop"],
    "metro_periphery": ["periphery", "arab_x_periphery", "dati_x_periphery", "haredi_x_periphery"],
}
ALL_CONTEXT = BLOCKS["own_group_concentration"] + BLOCKS["municipality_size"] + BLOCKS["metro_periphery"]
MODEL_COLUMNS = model.INCOME + ALL_CONTEXT


def add_context(cells: pd.DataFrame, micro: pd.DataFrame):
    """Add context variables to collapsed model cells."""
    cells = cells.copy()
    micro = micro.copy()
    for name in ["arab", "dati", "haredi"]:
        share = cells.groupby("g").apply(
            lambda z: np.average(z[name], weights=z.weight), include_groups=False)
        cells[f"geo_{name}_share"] = cells.g.map(share)
    cells["arab_x_geo_arab"] = cells.arab * cells.geo_arab_share
    cells["dati_x_geo_dati"] = cells.dati * cells.geo_dati_share
    cells["haredi_x_geo_haredi"] = cells.haredi * cells.geo_haredi_share
    locality_population = micro.groupby("locality").MishkalPratPUF.sum()
    log_population = np.log1p(locality_population)
    locality_z = (log_population - log_population.mean()) / log_population.std()
    cells["locality_logpop_z"] = cells.locality.map(locality_z).fillna(0)
    for name in ["arab", "dati", "haredi"]:
        cells[f"{name}_x_logpop"] = cells[name] * cells.locality_logpop_z
    district = (micro.assign(district=pd.to_numeric(micro.MachozPUF, errors="coerce"))
                .groupby("geo_key").district.agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan))
    cells["periphery"] = cells.geo_key.map(district).isin([2, 6, 7]).astype(float)
    for name in ["arab", "dati", "haredi"]:
        cells[f"{name}_x_periphery"] = cells[name] * cells.periphery
    return cells, locality_z


def add_micro_context(micro: pd.DataFrame, locality_z: pd.Series) -> pd.DataFrame:
    """Build the same design matrix for every adult PUF record."""
    m = micro.copy()
    ff = multi.base.feature_frame(m)
    weight = pd.to_numeric(m.MishkalPratPUF, errors="coerce").fillna(0)
    m["locality"] = pd.to_numeric(m.SmlYishuvPUF, errors="coerce")
    m["rova"] = pd.to_numeric(m.TatRovaKtvtMegurimPUF, errors="coerce").fillna(0)
    m["geo_key"] = m.locality.astype("Int64").astype(str) + "_" + m.rova.astype("Int64").astype(str)
    for name in ["arab", "dati", "haredi"]:
        share = pd.DataFrame({"geo": m.geo_key, "value": ff[name], "weight": weight}).groupby("geo").apply(
            lambda z: np.average(z.value, weights=z.weight), include_groups=False)
        ff[f"geo_{name}_share"] = m.geo_key.map(share).fillna(np.average(ff[name], weights=weight))
        ff[f"{name}_x_geo_{name}"] = ff[name] * ff[f"geo_{name}_share"]
    ff["locality_logpop_z"] = m.locality.map(locality_z).fillna(0)
    for name in ["arab", "dati", "haredi"]:
        ff[f"{name}_x_logpop"] = ff[name] * ff.locality_logpop_z
    ff["periphery"] = pd.to_numeric(m.MachozPUF, errors="coerce").isin([2, 6, 7]).astype(float)
    for name in ["arab", "dati", "haredi"]:
        ff[f"{name}_x_periphery"] = ff[name] * ff.periphery
    return ff
