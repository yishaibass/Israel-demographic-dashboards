"""Optional transaction-cell prior for residual primary-home imputation.

The module deliberately accepts only already-aggregated cells. It never accepts or
joins household, address, locality, block, or parcel identifiers. The production
pipeline keeps this feature disabled unless an explicit configuration and a valid
cell file are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


WAVE_YEAR = {1: 2012, 2: 2013, 3: 2014, 4: 2016, 5: 2017,
             6: 2018, 7: 2019, 8: 2020, 9: 2021, 10: 2023}
CELL_KEYS = ["survey_year", "socioeconomic_cluster", "dwelling_type_group", "size_band"]
REQUIRED_CELL_COLUMNS = CELL_KEYS + [
    "median_price_per_sqm", "transaction_count", "source_version"
]
ALLOWED_DWELLING_TYPES = {"ordinary_apartment", "special_apartment", "house"}
ALLOWED_SIZE_BANDS = {"under_60", "60_89", "90_119", "120_plus"}
FORBIDDEN_IDENTIFIER_TOKENS = (
    "address", "street", "house_number", "locality", "settlement", "parcel",
    "gush", "helka", "postcode", "zip", "latitude", "longitude", "household_id"
)


@dataclass(frozen=True)
class HomeTransactionPriorConfig:
    enabled: bool = False
    cell_path: str | None = None
    blend_weight: float = 0.25
    min_transactions: int = 30
    supported_waves: tuple[int, ...] = (6, 7, 8, 9, 10)

    def validate(self) -> None:
        if not 0 <= self.blend_weight <= 1:
            raise ValueError("blend_weight must be between 0 and 1")
        if self.min_transactions < 1:
            raise ValueError("min_transactions must be positive")
        if self.enabled and not self.cell_path:
            raise ValueError("enabled home transaction prior requires cell_path")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "HomeTransactionPriorConfig":
        values = os.environ if env is None else env
        enabled = values.get("HH_HOME_PRIOR_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
        config = cls(
            enabled=enabled,
            cell_path=values.get("HH_HOME_PRIOR_CELLS") or None,
            blend_weight=float(values.get("HH_HOME_PRIOR_BLEND_WEIGHT", "0.25")),
            min_transactions=int(values.get("HH_HOME_PRIOR_MIN_TRANSACTIONS", "30")),
        )
        config.validate()
        return config


def _validate_transaction_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Return canonical aggregate cells, failing closed on any other schema."""
    normalized = {str(c).strip().lower() for c in cells.columns}
    forbidden = sorted(c for c in normalized if any(t in c for t in FORBIDDEN_IDENTIFIER_TOKENS))
    if forbidden:
        raise ValueError(f"transaction-cell file contains forbidden identifier columns: {forbidden}")
    missing = [c for c in REQUIRED_CELL_COLUMNS if c not in cells.columns]
    if missing:
        raise ValueError(f"transaction-cell file missing required columns: {missing}")
    unexpected = sorted(str(c) for c in cells.columns if c not in REQUIRED_CELL_COLUMNS)
    if unexpected:
        raise ValueError(
            "transaction-cell file contains unexpected columns; keep provenance in a sidecar: "
            f"{unexpected}"
        )

    cells = cells[REQUIRED_CELL_COLUMNS].copy()
    for c in ["survey_year", "socioeconomic_cluster", "median_price_per_sqm", "transaction_count"]:
        cells[c] = pd.to_numeric(cells[c], errors="coerce")
    if cells[["survey_year", "socioeconomic_cluster", "median_price_per_sqm",
              "transaction_count"]].isna().any().any():
        raise ValueError("transaction-cell numeric fields must be populated")
    numeric = cells[["survey_year", "socioeconomic_cluster", "median_price_per_sqm",
                     "transaction_count"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("transaction-cell numeric fields must be finite")
    integral = ["survey_year", "socioeconomic_cluster", "transaction_count"]
    if any(not np.equal(cells[c], np.floor(cells[c])).all() for c in integral):
        raise ValueError("transaction-cell year, cluster, and count must be integers")
    if (cells["median_price_per_sqm"] <= 0).any() or (cells["transaction_count"] < 0).any():
        raise ValueError("transaction price must be positive and count non-negative")
    cells["survey_year"] = cells["survey_year"].astype(int)
    cells["socioeconomic_cluster"] = cells["socioeconomic_cluster"].astype(int)
    cells["transaction_count"] = cells["transaction_count"].astype(int)
    if not cells["survey_year"].isin(WAVE_YEAR.values()).all():
        raise ValueError("transaction-cell survey_year is not a supported survey year")
    if not cells["socioeconomic_cluster"].between(1, 10).all():
        raise ValueError("transaction-cell socioeconomic_cluster must be in 1..10")
    if not cells["dwelling_type_group"].isin(ALLOWED_DWELLING_TYPES).all():
        raise ValueError("transaction-cell dwelling_type_group contains an unknown category")
    if not cells["size_band"].isin(ALLOWED_SIZE_BANDS).all():
        raise ValueError("transaction-cell size_band contains an unknown category")
    source_version = cells["source_version"].astype("string").str.strip()
    if source_version.isna().any() or source_version.eq("").any():
        raise ValueError("transaction-cell source_version must be populated")
    cells["source_version"] = source_version.astype(str)
    if cells.duplicated(CELL_KEYS).any():
        raise ValueError("transaction-cell keys must be unique")
    return cells.sort_values(CELL_KEYS, kind="stable").reset_index(drop=True)


def load_transaction_cells(path: str | Path) -> pd.DataFrame:
    """Load and validate aggregate cells; reject record-level identifiers."""
    return _validate_transaction_cells(pd.read_csv(path))


def dwelling_type_group(raw: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw, errors="coerce")
    grouped = pd.Series(np.nan, index=raw.index, dtype="object")
    grouped.loc[numeric.eq(4)] = "ordinary_apartment"
    grouped.loc[numeric.isin([1, 2, 3])] = "special_apartment"
    grouped.loc[numeric.isin([5, 6, 7, 8])] = "house"
    return grouped


def size_band(sqm: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(sqm, errors="coerce")
    return pd.cut(
        numeric, bins=[-np.inf, 60, 90, 120, np.inf], right=False,
        labels=["under_60", "60_89", "90_119", "120_plus"]
    ).astype("object")


def transaction_cell_match_mask(
    df: pd.DataFrame, cells: pd.DataFrame, config: HomeTransactionPriorConfig
) -> pd.Series:
    """Identify residual owners with an exact usable cell (no geographic fallback)."""
    cells = _validate_transaction_cells(cells)
    keys = pd.DataFrame(index=df.index)
    keys["survey_year"] = pd.to_numeric(df["wave"], errors="coerce").map(WAVE_YEAR)
    cluster = pd.to_numeric(df["socioeconomic_cluster"], errors="coerce")
    keys["socioeconomic_cluster"] = cluster.where(cluster.eq(np.floor(cluster)))
    keys["dwelling_type_group"] = dwelling_type_group(df["apt_type"])
    keys["size_band"] = size_band(df["apt_size_sqm"])
    joined = keys.reset_index(names="_row").merge(
        cells[CELL_KEYS + ["transaction_count"]], on=CELL_KEYS, how="left", sort=False
    ).set_index("_row").reindex(df.index)
    owner = (df["tenure_type"] == 1) | (df["owns_primary_apt"] == 1)
    residual = owner & df["apt_value_subjective"].isna() & df["apt_value_estimated"].isna()
    supported = pd.to_numeric(df["wave"], errors="coerce").isin(config.supported_waves)
    return (residual & supported & joined["transaction_count"].ge(config.min_transactions)
            & pd.to_numeric(df["apt_size_sqm"], errors="coerce").gt(0))


def apply_transaction_cell_prior(
    df: pd.DataFrame,
    base_prediction: pd.Series,
    cells: pd.DataFrame | None,
    config: HomeTransactionPriorConfig,
    cap: float | None = None,
) -> tuple[pd.Series, dict]:
    """Blend a cell prior into residual predictions only; otherwise return base."""
    config.validate()
    out = pd.Series(base_prediction, index=df.index, dtype=float).copy()
    base_diag = {"enabled": config.enabled, "status": "disabled", "n_applied": 0}
    if not config.enabled:
        return out, base_diag
    if cells is None:
        cells = load_transaction_cells(config.cell_path)  # type: ignore[arg-type]
    else:
        # Callers cannot bypass the identifier/schema boundary by passing a frame.
        cells = _validate_transaction_cells(cells)

    missing_df = [c for c in ["wave", "socioeconomic_cluster", "apt_type", "apt_size_sqm",
                               "apt_value_subjective", "apt_value_estimated", "tenure_type",
                               "owns_primary_apt"] if c not in df.columns]
    if missing_df:
        raise ValueError(f"household frame missing home-prior fields: {missing_df}")

    keys = pd.DataFrame(index=df.index)
    keys["survey_year"] = pd.to_numeric(df["wave"], errors="coerce").map(WAVE_YEAR)
    cluster = pd.to_numeric(df["socioeconomic_cluster"], errors="coerce")
    keys["socioeconomic_cluster"] = cluster.where(cluster.eq(np.floor(cluster)))
    keys["dwelling_type_group"] = dwelling_type_group(df["apt_type"])
    keys["size_band"] = size_band(df["apt_size_sqm"])
    joined = keys.reset_index(names="_row").merge(cells, on=CELL_KEYS, how="left", sort=False).set_index("_row")
    joined = joined.reindex(df.index)

    owner = (df["tenure_type"] == 1) | (df["owns_primary_apt"] == 1)
    residual = owner & df["apt_value_subjective"].isna() & df["apt_value_estimated"].isna()
    supported = pd.to_numeric(df["wave"], errors="coerce").isin(config.supported_waves)
    usable = (residual & supported & joined["median_price_per_sqm"].notna()
              & (joined["transaction_count"] >= config.min_transactions)
              & pd.to_numeric(df["apt_size_sqm"], errors="coerce").gt(0)
              & out.gt(0))

    prior_value = joined["median_price_per_sqm"] * pd.to_numeric(df["apt_size_sqm"], errors="coerce")
    blended = np.exp((1 - config.blend_weight) * np.log(out.clip(lower=1))
                     + config.blend_weight * np.log(prior_value.clip(lower=1)))
    if cap is not None and np.isfinite(cap):
        blended = blended.clip(upper=cap)
    out.loc[usable] = blended.loc[usable]

    versions = sorted(str(x) for x in joined.loc[usable, "source_version"].dropna().unique())
    return out, {
        "enabled": True,
        "status": "applied" if usable.any() else "no_matching_cells",
        "n_residual_eligible": int((residual & supported).sum()),
        "n_applied": int(usable.sum()),
        "coverage": float(usable.sum() / max(1, (residual & supported).sum())),
        "blend_weight": config.blend_weight,
        "min_transactions": config.min_transactions,
        "source_versions": versions,
    }


def _weighted_metrics(target: pd.Series, prediction: pd.Series, weight: pd.Series) -> dict:
    y = pd.to_numeric(target, errors="coerce")
    p = pd.to_numeric(prediction, errors="coerce")
    w = pd.to_numeric(weight, errors="coerce")
    mask = y.gt(0) & p.gt(0) & w.gt(0) & y.notna() & p.notna() & w.notna()
    if not mask.any():
        return {"n": 0, "weight_sum": 0.0, "mae": None, "rmse": None,
                "log_rmse": None, "weighted_log_r2": None}
    y, p, w = y[mask], p[mask], w[mask]
    err = p - y
    log_y, log_p = np.log(y), np.log(p)
    log_mean = float(np.average(log_y, weights=w))
    sse = float(np.sum(w * (log_y - log_p) ** 2))
    sst = float(np.sum(w * (log_y - log_mean) ** 2))
    return {
        "n": int(mask.sum()), "weight_sum": float(w.sum()),
        "mae": float(np.average(np.abs(err), weights=w)),
        "rmse": float(np.sqrt(np.average(err ** 2, weights=w))),
        "log_rmse": float(np.sqrt(np.average((log_y - log_p) ** 2, weights=w))),
        "weighted_log_r2": float(1 - sse / sst) if sst > 0 else None,
    }


def evaluate_heldout_home_predictions(
    heldout: pd.DataFrame,
    base_oof_col: str,
    candidate_oof_col: str,
    subjective_col: str = "apt_value_subjective",
    cbs_estimate_col: str = "apt_value_estimated",
    weight_col: str = "weight_hh",
    group_col: str = "group",
    year_col: str = "survey_year",
) -> dict:
    """Score already out-of-fold predictions; caller must refit each time block.

    Subjective owner reports are the primary target. CBS estimates are reported
    separately as secondary concordance and are never mixed into the primary score.
    """
    required = [base_oof_col, candidate_oof_col, subjective_col, cbs_estimate_col,
                weight_col, group_col, year_col]
    missing = [c for c in required if c not in heldout.columns]
    if missing:
        raise ValueError(f"heldout frame missing validation columns: {missing}")

    def comparison(frame: pd.DataFrame, target: str) -> dict:
        return {
            "base": _weighted_metrics(frame[target], frame[base_oof_col], frame[weight_col]),
            "candidate": _weighted_metrics(frame[target], frame[candidate_oof_col], frame[weight_col]),
        }

    primary = heldout[heldout[subjective_col].notna()].copy()
    report = {
        "primary_subjective": {
            "overall": comparison(primary, subjective_col),
            "by_time_block": {str(k): comparison(g, subjective_col)
                              for k, g in primary.groupby(year_col, sort=True)},
            "by_social_group": {str(k): comparison(g, subjective_col)
                                for k, g in primary.groupby(group_col, sort=True)},
        }
    }
    secondary = heldout[heldout[subjective_col].isna() & heldout[cbs_estimate_col].notna()].copy()
    report["secondary_cbs_concordance"] = {
        "overall": comparison(secondary, cbs_estimate_col),
        "note": "Secondary concordance only; CBS estimates are not an independent validation target.",
    }
    return report


def expanding_time_blocks(years: pd.Series, min_train_periods: int = 2) -> list[dict]:
    """Return deterministic expanding-window train/test year definitions."""
    unique = sorted(int(x) for x in pd.Series(years).dropna().unique())
    return [{"train_years": unique[:i], "test_year": unique[i]}
            for i in range(min_train_periods, len(unique))]
