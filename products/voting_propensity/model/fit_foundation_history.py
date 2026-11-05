"""Research-only controlled comparisons of demographic weights and history.

The published 24-column context design, logistic/softmax parameterization and
penalties are reused. Each election has its own cell weights and context values.
Annual all-age group margins are explicitly an adult-composition proxy, not an
observed electoral roll. Outputs are aggregates; production files are untouched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import fit_k26 as common

FAMILIES = list(common.FAMILIES)
FEATURES = list(common.context.MODEL_COLUMNS)


def identity(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fitted_identity(fitted, datasets):
    digest = hashlib.sha256(json.dumps({"features": FEATURES, "elections": [data.election for data in datasets],
                                      "partitions": [data.partition for data in datasets],
                                      "available": [data.available for data in datasets]}, sort_keys=True).encode())
    for name, value in sorted(fitted.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


@contextmanager
def election_scope(elections):
    previous = common.model.ELECTIONS
    common.model.ELECTIONS = list(elections)
    try:
        yield
    finally:
        common.model.ELECTIONS = previous


def validate_partition(partition, available):
    """A disjoint partition exhausts all and only historically available families."""
    available = list(available)
    flat = [family for item in partition for family in item["families"]]
    if not available or len(set(available)) != len(available) or not set(available) <= set(FAMILIES):
        raise ValueError("Invalid family availability")
    if (any(not item["families"] for item in partition)
            or len(flat) != len(set(flat)) or set(flat) != set(available)
            or len({item["id"] for item in partition}) != len(partition)):
        raise ValueError("Observed family partitions must be disjoint and exhaustive")
    return np.asarray([[float(f in item["families"]) for item in partition] for f in FAMILIES], np.float32)


def normalized_choice(choice, available):
    if set(available) == set(FAMILIES):
        return choice
    mask = torch.tensor([f in available for f in FAMILIES], dtype=choice.dtype)
    masked = choice * mask
    return masked / masked.sum(1, keepdim=True).clamp_min(1e-30)


def calibrate_annual_weights(micro, margins, year, prefer_exact_age=True):
    """Reweight only Arab vs Jewish+Other; preserve 2022 structure within groups.

    All-age source ratios proxy adult group ratios. Unknown population codes and
    positive target cells with no PUF support are ineligible; no imputation.
    """
    source = margins.loc[(pd.to_numeric(margins.year) == year)
                         & margins.dimension.eq("population_group")
                         & margins.category.isin(["arab", "jewish_and_other"])].copy()
    if "unit" in source:
        source = source.loc[source.unit.eq("persons")]
    source["value"] = pd.to_numeric(source.value, errors="raise")
    source = source.loc[source.value.notna()]
    if source.empty:
        return micro.iloc[:0].copy(), {"year": year, "status": "unsupported_year"}
    if source.duplicated(["locality_code", "category"]).any():
        raise ValueError("Duplicate annual demographic margins")
    source["value"] = pd.to_numeric(source.value, errors="raise")
    if not np.isfinite(source.value).all() or (source.value < 0).any():
        raise ValueError("Invalid annual population count")
    table = source.pivot(index="locality_code", columns="category", values="value")
    table = table.reindex(columns=["arab", "jewish_and_other"]).dropna()
    table = table.loc[table.sum(axis=1) > 0]
    out = micro.copy()
    locality = pd.to_numeric(out.SmlYishuvPUF, errors="coerce")
    pop = pd.to_numeric(out.KvutzaUchlusiyaPUF, errors="coerce")
    weight = pd.to_numeric(out.MishkalPratPUF, errors="coerce")
    category = pd.Series(np.select([pop.eq(2), pop.isin([1, 3])],
                                  ["arab", "jewish_and_other"], default="unknown"), index=out.index)
    adult = pd.DataFrame({"locality": locality, "category": category, "weight": weight})
    support = adult.pivot_table(index="locality", columns="category", values="weight", aggfunc="sum", fill_value=0)
    support = support.reindex(columns=["arab", "jewish_and_other", "unknown"], fill_value=0)
    candidate = table.index.intersection(support.index)
    bad = support.loc[candidate, "unknown"].gt(0)
    for cat in ["arab", "jewish_and_other"]:
        bad |= table.loc[candidate, cat].gt(0) & support.loc[candidate, cat].le(0)
    good = candidate[~bad]
    ratios = table.loc[good].div(support.loc[good, ["arab", "jewish_and_other"]].replace(0, np.nan)).fillna(0)
    # Retain each locality's 2022 adult total; only its within-locality mix moves.
    ratios = ratios.mul(support.loc[good].sum(axis=1) / table.loc[good].sum(axis=1), axis=0)
    scale = pd.MultiIndex.from_arrays([locality, category]).map(ratios.stack())
    keep = locality.isin(good) & category.ne("unknown")
    out = out.loc[keep].copy()
    out["MishkalPratPUF"] = weight.loc[keep].to_numpy() * np.asarray(scale[keep], float)
    out = out.loc[out.MishkalPratPUF.gt(0)].copy()
    out, age_audit = calibrate_retirement(out, margins, year, prefer_exact_age)
    status_values = sorted(set(source.status.dropna())) if "status" in source else []
    return out, {"year": year, "status": "annual_all_age_group_proxy", "supported_localities": len(good),
                 "excluded_unsupported_or_unknown_localities": int(bad.sum()),
                 "margin_rows": len(source), "margin_sha256": identity(source),
                 "input_statuses": status_values, "includes_projection": any(s.startswith("projected") for s in status_values),
                 "age_calibration": age_audit,
                 "proxy": "All-age Arab versus Jewish+Other shares proxy adult composition; age margin uses a fractional female55–64 proxy; income, religiosity and Jewish:Other structure remain conditional2022"}


def retirement_score(micro):
    """NII mixed threshold: women60+/men65+; half of female55–64 is a proxy."""
    age = pd.to_numeric(micro.GilPUF, errors="coerce")
    female = pd.to_numeric(micro.MinPUF, errors="coerce").eq(2)
    return np.select([age.ge(7), age.eq(6)&female], [1., .5], default=0.)


def calibrate_growth_weights(micro, margins, year):
    """Anchor adults in 2022, update only observed same-series group growth."""
    tables = {}
    for selected_year in {2022, year}:
        source = margins.loc[pd.to_numeric(margins.year).eq(selected_year)
                             & margins.dimension.eq("population_group")
                             & margins.category.isin(["arab", "jewish_and_other"])].copy()
        if "unit" in source:
            source = source.loc[source.unit.eq("persons")]
        source["value"] = pd.to_numeric(source.value, errors="raise")
        source = source.loc[source.value.notna()]
        if source.duplicated(["locality_code", "category"]).any() or (source.value < 0).any():
            raise ValueError("Invalid anchored-growth source margins")
        tables[selected_year] = source.pivot(index="locality_code", columns="category", values="value").reindex(columns=["arab", "jewish_and_other"]).dropna()
    base, current = tables[2022], tables[year]
    codes = base.index.intersection(current.index)
    if not len(codes):
        return micro.iloc[:0].copy(), {"year": year, "status": "unsupported_year"}
    positive_over_zero = (base.loc[codes].eq(0) & current.loc[codes].gt(0)).any(axis=1)
    unsupported = list(codes[positive_over_zero])
    codes = codes[~positive_over_zero]
    zero_zero = base.loc[codes].eq(0) & current.loc[codes].eq(0)
    factors = current.loc[codes].div(base.loc[codes].replace(0, np.nan)).mask(zero_zero, 1.)
    pop = pd.to_numeric(micro.KvutzaUchlusiyaPUF, errors="coerce")
    locality = pd.to_numeric(micro.SmlYishuvPUF, errors="coerce")
    category = pd.Series(np.select([pop.eq(2), pop.isin([1, 3])], ["arab", "jewish_and_other"], default="unknown"), index=micro.index)
    unknown_codes = set(locality.loc[category.eq("unknown")])
    keep = locality.isin(codes) & ~locality.isin(unknown_codes)
    key = pd.MultiIndex.from_arrays([locality, category])
    scale = pd.Series(key.map(factors.stack()).to_numpy(), index=micro.index)
    out = micro.loc[keep].copy()
    out["MishkalPratPUF"] = pd.to_numeric(out.MishkalPratPUF).to_numpy(float)*scale.loc[keep].to_numpy(float)
    out = out.loc[out.MishkalPratPUF.gt(0)].copy()
    # Only date-verified, exact65+ bins enter this preferred condition.
    verified_age = margins.loc[~margins.dimension.eq("retirement_age")].copy()
    if verified_age.dimension.eq("adult_age").any():
        if not {"population_reference_date_status", "population_reference_date"} <= set(verified_age.columns):
            raise ValueError("Preferred age calibration requires explicit verified/projected reference dates")
        allowed_age = verified_age.population_reference_date_status.isin(["verified", "projected"]) & verified_age.population_reference_date.notna()
        verified_age = verified_age.loc[~verified_age.dimension.eq("adult_age") | allowed_age]
    out, age_audit = calibrate_retirement(out, verified_age, year, prefer_exact_age=True)
    return out, {"year": year, "status": "anchored_2022_adults_same_series_group_growth",
                 "supported_localities": int(pd.to_numeric(out.SmlYishuvPUF).nunique()),
                 "positive_year_over_zero_anchor_localities": [int(code) for code in unsupported],
                 "zero_zero_growth_factor_one_cells": int(zero_zero.sum().sum()),
                 "zero_zero_policy": "Factor1 preserves known2022PUF adults; no observed growth, not observed futurezero",
                 "age_calibration": age_audit, "source_sha256": identity(margins.loc[pd.to_numeric(margins.year).isin([2022, year])]),
                 "proxy": "2022 adult PUF group weights times official all-age group growth ratio; only verified exact65+ age margins raked, where available. No unverified historical retirement margins."}


def calibrate_retirement(micro, margins, year, prefer_exact_age=True):
    """Entropy-rake known age and coarse ethnicity without manufacturing cells."""
    exact = margins.loc[(pd.to_numeric(margins.year) == year)
                        & margins.dimension.eq("adult_age")
                        & margins.category.isin(["age18_64", "age65_plus"])].copy()
    exact_age = prefer_exact_age and not exact.empty
    source = exact if exact_age else margins.loc[(pd.to_numeric(margins.year) == year)
                                                 & margins.dimension.eq("retirement_age")
                                                 & margins.category.isin(["retirement_age", "working_age_adult"])].copy()
    if source.empty or micro.empty:
        return micro, {"status": "unavailable", "localities": 0}
    if source.duplicated(["locality_code", "category"]).any():
        raise ValueError("Duplicate retirement-age margin")
    source["value"] = pd.to_numeric(source.value, errors="raise")
    source = source.loc[source.value.notna()]
    if exact_age:
        source["category"] = source.category.map({"age18_64": "working_age_adult", "age65_plus": "retirement_age"})
    table = source.pivot(index="locality_code", columns="category", values="value")
    table = table.reindex(columns=["working_age_adult", "retirement_age"]).dropna()
    if (table < 0).any().any() or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Invalid adult age margin")
    table = table.loc[table.sum(axis=1) > 0]
    frame = pd.DataFrame({"locality": pd.to_numeric(micro.SmlYishuvPUF),
                          "ethnic": pd.to_numeric(micro.KvutzaUchlusiyaPUF).eq(2).astype(int),
                          "age_score": (pd.to_numeric(micro.GilPUF).ge(7).astype(float).to_numpy() if exact_age else retirement_score(micro)),
                          "weight": micro.MishkalPratPUF.to_numpy(float)}, index=micro.index)
    columns = pd.MultiIndex.from_product([[0, 1], [0., .5, 1.]])
    aggregated = frame.pivot_table(index="locality", columns=["ethnic", "age_score"], values="weight", aggfunc="sum", fill_value=0).reindex(columns=columns, fill_value=0)
    codes = aggregated.index.intersection(table.index)
    if not len(codes):
        return micro, {"status": "unmatched", "localities": 0}
    initial = aggregated.loc[codes].to_numpy(float)
    total = table.loc[codes].sum(axis=1).to_numpy(float)
    age_target = table.loc[codes, "retirement_age"].to_numpy(float) / total
    group_target = np.stack([initial[:, :3].sum(1), initial[:, 3:].sum(1)], axis=1)
    group_target /= group_target.sum(1, keepdims=True)
    q = np.tile([0., .5, 1.], 2)
    values = initial / initial.sum(1, keepdims=True)
    error = np.full(len(codes), np.inf)
    for iteration in range(200):
        for group in range(2):
            cell = slice(group*3, (group+1)*3)
            values[:, cell] *= (group_target[:, group] / np.maximum(values[:, cell].sum(1), 1e-300))[:, None]
        lower, upper = np.full(len(codes), -60.), np.full(len(codes), 60.)
        for _ in range(45):
            middle = (lower+upper)/2
            tilted = values * np.exp(middle[:, None]*q)
            predicted_age = (tilted*q).sum(1) / tilted.sum(1)
            lower = np.where(predicted_age < age_target, middle, lower)
            upper = np.where(predicted_age >= age_target, middle, upper)
        values *= np.exp(((lower+upper)/2)[:, None]*q)
        values /= values.sum(1, keepdims=True)
        error = np.maximum(abs(values[:, :3].sum(1)-group_target[:, 0]), abs((values*q).sum(1)-age_target))
        if np.max(error) < 1e-7:
            break
    accepted = np.isfinite(error) & (error < 1e-6)
    calibrated = pd.DataFrame(np.divide(values*total[:, None], initial, out=np.zeros_like(values), where=initial>0), index=codes, columns=columns)
    factors = calibrated.stack(level=[0, 1])
    key = pd.MultiIndex.from_arrays([frame.locality, frame.ethnic, frame.age_score])
    scaling = pd.Series(key.map(factors).to_numpy(), index=frame.index).fillna(1.)
    rejected = set(codes[~accepted])
    result = micro.loc[~frame.locality.isin(rejected)].copy()
    result["MishkalPratPUF"] *= scaling.loc[result.index].to_numpy(float)
    return result, {"status": "observed_or_projected_exact_65plus_age_margin" if exact_age else "observed_or_projected_retirement_margin_with_fractional_PUF_proxy", "localities": int(accepted.sum()),
                    "localities_without_age_margin": int(len(aggregated.index.difference(table.index))),
                    "incompatible_localities_excluded": sorted(int(code) for code in rejected),
                    "max_accepted_margin_error": float(error[accepted].max()) if accepted.any() else None,
                    "iterations": iteration+1, "margin_sha256": identity(source),
                    "indicator": "Gil>=7:1; otherwise0 (exact65+PUFbin)." if exact_age else "Gil>=7:1; femaleGil6:0.5 under uniform55–64 assumption; otherwise0. NII women60+/men65+ is not age65+.",
                    "population_total": "NII retirement+working-age adults where matched; 2022 PUF adult total elsewhere",
                    "max_weight_multiplier": float(scaling.loc[result.index].max()) if len(result) else None,
                    "source_time_basis": sorted(set(source.year_basis.dropna())) if "year_basis" in source else [],
                    "population_reference_date_status": sorted(set(source.population_reference_date_status.dropna())) if "population_reference_date_status" in source else [],
                    "population_reference_dates": sorted(set(source.population_reference_date.dropna())) if "population_reference_date" in source else [],
                    "projection_anchor_reference_dates": sorted(set(source.projection_anchor_reference_date.dropna())) if "projection_anchor_reference_date" in source else [],
                    "input_statuses": sorted(set(source.status.dropna())) if "status" in source else []}


def prepare_cells(micro, targets, locality_scale=None):
    """Use the actual published context path with this election's population."""
    locality = pd.to_numeric(micro.SmlYishuvPUF, errors="coerce").astype("Int64")
    weights = pd.to_numeric(micro.MishkalPratPUF, errors="coerce").fillna(0)
    logpop = np.log1p(pd.DataFrame({"locality": locality, "weight": weights}).groupby("locality").weight.sum())
    mean, sd = locality_scale if locality_scale is not None else (logpop.mean(), logpop.std())
    locality_z = (logpop - mean) / sd
    features = common.context.add_micro_context(micro, locality_z)
    selected = features[FEATURES].copy()
    selected["fsu_proxy"] = features.fsu_proxy
    selected["locality"] = locality
    selected["population_weight"] = weights
    pop = pd.to_numeric(micro.KvutzaUchlusiyaPUF, errors="coerce")
    rel = pd.to_numeric(micro.DatiyutPUF, errors="coerce")
    selected["profile_group"] = np.select([pop.eq(2), pop.eq(1)&rel.eq(1), pop.eq(1)&rel.isin([2, 6]),
                                           pop.eq(1)&rel.eq(3), pop.eq(1)&rel.eq(5)],
                                          ["arab", "hiloni", "masorti", "dati", "haredi"], default="other_or_unknown")
    cells = selected.groupby(["locality"] + FEATURES + ["fsu_proxy", "profile_group"], as_index=False, dropna=False).population_weight.sum()
    present = set(cells.locality.dropna().astype(int))
    targets = targets.loc[targets.locality.isin(present)].copy().sort_values("locality").reset_index(drop=True)
    geos = targets[["locality"]].drop_duplicates().reset_index(drop=True)
    geos["geo_key"] = geos.locality.astype(str)
    gi = {int(code): i for i, code in enumerate(geos.locality)}
    cells = cells.loc[cells.locality.isin(gi)].copy()
    cells["g"] = cells.locality.map(gi).astype(int)
    eligible = targets.set_index("locality").eligible
    population = cells.groupby("locality").population_weight.sum()
    cells["weight"] = cells.population_weight * cells.locality.map(eligible / population)
    if not len(cells) or not np.isfinite(cells[FEATURES + ["weight"]].to_numpy(float)).all():
        raise ValueError("Empty or nonfinite model design")
    return cells.reset_index(drop=True), geos, targets


@dataclass
class ElectionData:
    election: str
    cells: pd.DataFrame
    geos: pd.DataFrame
    targets: pd.DataFrame
    partition: list
    available: list

    def tensors(self):
        z = self.targets
        matrix = validate_partition(self.partition, self.available)
        counts = z[[item["id"] for item in self.partition]].to_numpy(np.float32)
        if not np.allclose(counts.sum(1), z.valid) or (counts < 0).any():
            raise ValueError("Observed partitions do not exhaust valid votes")
        if (z.eligible <= 0).any() or (z.voted > z.eligible).any() or (z.valid > z.voted).any():
            raise ValueError("Invalid eligible, turnout or valid count")
        gi = {int(code): i for i, code in enumerate(self.geos.locality)}
        return {"x": torch.tensor(self.cells[FEATURES].to_numpy(np.float32)),
                "fsu": torch.tensor(self.cells.fsu_proxy.to_numpy(np.float32)),
                "w": torch.tensor(self.cells.weight.to_numpy(np.float32)),
                "g": torch.tensor(self.cells.g.to_numpy(np.int64)),
                "ix": torch.tensor(z.locality.map(gi).to_numpy(np.int64)),
                "turn": torch.tensor((z.voted / z.eligible).to_numpy(np.float32)),
                "family": torch.tensor(counts / counts.sum(1, keepdims=True)),
                "partition": torch.tensor(matrix)}


def predicted(fitted, data, tensors, election_index):
    pt, pc = fitted.probs(tensors["x"], tensors["fsu"], election_index)
    pc = normalized_choice(pc, data.available)
    return common.model.aggregate(pt, pc, tensors["w"], tensors["g"], len(data.geos))


def fit_history(datasets, train_localities, epochs, seed, pool=.05):
    """Same parameter architecture/objective; each year retains its own design."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    arrays = [data.tensors() for data in datasets]
    train = [torch.tensor(np.flatnonzero(data.targets.locality.isin(train_localities))) for data in datasets]
    if any(not len(ix) for ix in train):
        raise ValueError("Every election needs training localities")
    with election_scope([data.election for data in datasets]):
        fitted = common.model.FamilyModel(len(FEATURES), len(FAMILIES), False)
    optimizer = torch.optim.Adam(fitted.parameters(), lr=.04)
    best, state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        losses = []
        for ei, (data, tensors, ii) in enumerate(zip(datasets, arrays, train)):
            at, af = predicted(fitted, data, tensors, ei)
            ix = tensors["ix"][ii]
            gw = torch.sqrt(torch.tensor(data.targets.valid.to_numpy(np.float32))[ii])
            gw /= gw.mean()
            target = tensors["turn"][ii]
            tl = -(target * at[ix].clamp(1e-6, 1-1e-6).log() + (1-target) * (1-at[ix]).clamp_min(1e-6).log())
            observed_probs = af[ix] @ tensors["partition"]
            fl = -(tensors["family"][ii] * observed_probs.clamp_min(1e-8).log()).sum(1)
            losses.append(((tl + fl) * gw).mean())
        reg = .08 * (fitted.turn_shared[1:].square().mean() + fitted.choice_shared[1:].square().mean())
        pooling = pool * (fitted.turn_dev[:, 1:].square().mean() + fitted.choice_dev[:, 1:].square().mean() + fitted.fsu_dev.square().mean())
        loss = torch.stack(losses).mean() + reg + pooling + .08 * fitted.fsu_shared.square().mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best - 1e-6:
            best, state, stale = value, {key: val.detach().clone() for key, val in fitted.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 30:
            break
    fitted.load_state_dict(state)
    return fitted, {"epochs": epoch + 1, "objective": best, "model_fit_id": fitted_identity(fitted, datasets)}


def evaluate(fitted, datasets, test_localities):
    rows, predictions = [], []
    with torch.no_grad():
        for ei, data in enumerate(datasets):
            tensors = data.tensors()
            at, af = predicted(fitted, data, tensors, ei)
            ix = tensors["ix"].numpy()
            turn, family = at.numpy()[ix], (af @ tensors["partition"]).numpy()[ix]
            selected = data.targets.locality.isin(test_localities).to_numpy()
            if not selected.any():
                continue
            z = data.targets.loc[selected]
            turn_error = turn[selected] - tensors["turn"].numpy()[selected]
            family_error = family[selected] - tensors["family"].numpy()[selected]
            rows.append({"election": data.election, "turnout_mae_pp": float(np.average(abs(turn_error), weights=z.eligible)*100),
                         "turnout_rmse_pp": float(np.sqrt(np.average(turn_error**2, weights=z.eligible))*100),
                         "observed_partition_mae_pp": float(np.average(abs(family_error).mean(1), weights=z.valid)*100),
                         "partition_count": len(data.partition), "family_metric_comparable_only_with_identical_partition": True,
                         "localities": len(z), "eligible": int(z.eligible.sum()), "valid": int(z.valid.sum())})
            for original_index in np.flatnonzero(selected):
                target = data.targets.iloc[original_index]
                base = {"election": data.election, "locality": int(target.locality), "eligible": int(target.eligible),
                        "voted": int(target.voted), "valid": int(target.valid), "predicted_turnout": float(turn[original_index])}
                for j, part in enumerate(data.partition):
                    predictions.append({**base, "observation_id": part["id"], "families": json.dumps(part["families"]),
                                        "observed_votes": int(target[part["id"]]), "predicted_share_valid": float(family[original_index, j])})
    return rows, predictions


def subgroup_profiles(fitted, datasets, reference, estimand):
    """Ratios use each year's population, or an explicitly fixed reference."""
    rows = []
    with torch.no_grad():
        for ei, data in enumerate(datasets):
            cells = reference.cells if reference is not None else data.cells
            x = torch.tensor(cells[FEATURES].to_numpy(np.float32))
            fsu = torch.tensor(cells.fsu_proxy.to_numpy(np.float32))
            pt, pc = fitted.probs(x, fsu, ei)
            pc = normalized_choice(pc, data.available).numpy()
            pt = pt.numpy()
            w = cells.population_weight.to_numpy(float)
            masks = {group: cells.profile_group.eq(group).to_numpy()
                     for group in ["arab", "haredi", "dati", "masorti", "hiloni", "other_or_unknown"]}
            for group, mask in masks.items():
                if not mask.any() or w[mask].sum() <= 0:
                    continue
                base = {"election": data.election, "group": group, "estimand": estimand,
                        "population_scope": "matched_PUF_localities", "population_localities": int(cells.locality.nunique()),
                        "weighted_population": float(w[mask].sum())}
                rows.append({**base, "outcome": "turnout", "value": float(np.average(pt[mask], weights=w[mask]))})
                # Export only identifiable observed partitions, never fabricated joint-list components.
                for part in data.partition:
                    j = [FAMILIES.index(family) for family in part["families"]]
                    rows.append({**base, "outcome": part["id"], "value": float(np.average(pc[mask][:, j].sum(1), weights=w[mask]*pt[mask])),
                                 "family_components": json.dumps(part["families"]),
                                 "party_target_support": "single_family_aggregate_target" if len(j) == 1 else "joint_family_aggregate_target_only"})
    return rows


def history_sd(profiles):
    rows = []
    for keys, values in profiles.groupby(["variant", "estimand", "group", "outcome"]):
        expected_history = set(profiles.loc[profiles.variant.eq(keys[0]), "election"])-{"K26"}
        for label, selections in [("previous_5", ["K21", "K22", "K23", "K24", "K25"]),
                                  ("all_available_historical", sorted(expected_history))]:
            past = values.loc[values.election.isin(selections)]
            if len(past) < 2 or (label == "previous_5" and len(past) != 5):
                continue
            sd, mean = float(past.value.std(ddof=0)), float(past.value.mean())
            current = values.loc[values.election.eq("K26"), "value"]
            rows.append(dict(zip(["variant", "estimand", "group", "outcome"], keys), window=label,
                             elections=json.dumps(list(past.election)), n_elections=len(past), mean=mean, sd=sd,
                             excluded_elections_without_separate_target=json.dumps(sorted(set(selections)-set(past.election))),
                             k26=float(current.iloc[0]) if len(current) else None,
                             k26_historical_sd=float((current.iloc[0]-mean)/sd) if len(current) and sd > 0 else None,
                             interpretation="Descriptive fitted-election variation; no sampling uncertainty or future tail probability"))
    return pd.DataFrame(rows)


def read_older(folder):
    if folder is None:
        return {}, {}
    readiness = json.loads((folder / "readiness.json").read_text(encoding="utf-8"))
    table = pd.read_csv(folder / "targets.csv")
    entries = readiness.get("elections", readiness)
    datasets = {}
    for election, meta in entries.items():
        if not isinstance(meta, dict) or not meta.get("training_ready", False):
            continue
        z = table.loc[table.election.eq(election)].copy()
        z = z.loc[pd.to_numeric(z.eligible, errors="coerce").ge(100)
                  & pd.to_numeric(z.valid, errors="coerce").gt(0)].copy()
        if z.empty:
            continue
        z["locality"] = pd.to_numeric(z.locality, errors="raise").astype(int)
        z["geo_key"] = z.locality.astype(str)
        partition = []
        for item in meta["observed_partition"]:
            if item["families"]:
                partition.append(item)
            elif item["id"] in z and not pd.to_numeric(z[item["id"]], errors="raise").fillna(0).eq(0).all():
                raise ValueError("Unresolved historical party has positive vote mass")
        validate_partition(partition, meta["available_families"])
        datasets[election] = (z, partition, meta["available_families"])
    return datasets, readiness


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run(args):
    started = time.perf_counter()
    source_hashes = {name: file_identity(Path(__file__).parent/name) for name in
                     ["fit_foundation_history.py", "fit_k26.py", "context_features.py", "run_party_family_final.py", "data_pipeline.py"]}
    input_hashes = {name: file_identity(path) for name, path in
                    [("puf", args.puf), ("annual_targets", args.annual_targets), ("annual_projections", args.annual_projections)] if path is not None}
    torch.set_num_threads(args.threads)
    historical, _ = common.build_historical_targets(args.project)
    k26, k26audit = common.build_targets(args.current)
    targets = pd.concat([historical, k26], ignore_index=True)
    default_partition = [{"id": family, "families": [family]} for family in FAMILIES]
    older, older_audit = read_older(args.historical_targets)
    micro = common.elections.load_all_adults(args.puf).reset_index(drop=True)
    reference_population = pd.DataFrame({"locality": pd.to_numeric(micro.SmlYishuvPUF, errors="coerce"),
                                         "weight": pd.to_numeric(micro.MishkalPratPUF, errors="coerce")}).groupby("locality").weight.sum()
    reference_logpop = np.log1p(reference_population)
    locality_scale = (float(reference_logpop.mean()), float(reference_logpop.std()))
    base_cells, base_geos, _ = prepare_cells(micro, targets.loc[targets.election.eq("K25")])
    fixed = {}
    for election, z in targets.groupby("election", sort=False):
        # Same context features/PUF, but eligible totals are election-specific.
        cells = base_cells.copy()
        z = z.loc[z.locality.isin(set(base_geos.locality))].copy().sort_values("locality").reset_index(drop=True)
        geos = base_geos.copy()
        eligible = z.set_index("locality").eligible
        pop = cells.groupby("locality").population_weight.sum()
        cells["weight"] = cells.population_weight * cells.locality.map(eligible/pop)
        cells = cells.loc[cells.weight.notna()].copy()
        fixed[election] = ElectionData(election, cells, geos, z, default_partition, FAMILIES)
    variants = {}
    if "fixed2022_recent" in args.variants:
        variants["fixed2022_recent"] = list(fixed.values())
    calibration_audit = []
    if args.annual_targets is not None and any(v in args.variants for v in ["fixed2022_paired", "annual_group_proxy", "annual_group_age65_projection", "annual_anchored_growth"]):
        margins = pd.read_csv(args.annual_targets, low_memory=False)
        if args.annual_projections is not None:
            projections = pd.read_csv(args.annual_projections, low_memory=False)
            if "status" not in projections or not projections.status.str.startswith("projected").fillna(False).all():
                raise ValueError("Projection input must be explicitly labeled projected")
            margins = pd.concat([margins, projections], ignore_index=True)
        annual_collections = {}
        requested_annual = [name for name in args.variants if name.startswith("annual_")]
        if not requested_annual and "fixed2022_paired" in args.variants:
            requested_annual = ["annual_group_proxy"]
        for name, exact_age in [("annual_group_proxy", False), ("annual_group_age65_projection", True), ("annual_anchored_growth", True)]:
            if name not in requested_annual:
                continue
            annual, cache = [], {}
            for election, data in fixed.items():
                year = int(common.DATES[election][:4])
                if year not in cache:
                    calibrated, audit = (calibrate_growth_weights(micro, margins, year) if name == "annual_anchored_growth"
                                         else calibrate_annual_weights(micro, margins, year, prefer_exact_age=exact_age))
                    calibration_audit.append(dict(audit, variant=name))
                    if len(calibrated):
                        cache[year] = prepare_cells(calibrated, data.targets, locality_scale)[:2]
                    else:
                        cache[year] = None
                if cache[year] is None:
                    continue
                cells, geos = cache[year]
                z = data.targets.loc[data.targets.locality.isin(geos.locality)].copy()
                cells = cells.copy()
                eligible = z.set_index("locality").eligible
                pop = cells.groupby("locality").population_weight.sum()
                cells["weight"] = (cells.population_weight * cells.locality.map(eligible/pop)).fillna(0)
                annual.append(ElectionData(election, cells, geos, z, default_partition, FAMILIES))
            if annual:
                annual_collections[name] = annual
        if annual_collections:
            supported_elections = set.intersection(*(set(d.election for d in collection) for collection in annual_collections.values()))
            annual_collections = {name: [data for data in collection if data.election in supported_elections] for name, collection in annual_collections.items()}
            common_locs = set.intersection(*(set(d.targets.locality) for collection in annual_collections.values() for d in collection))
            paired_fixed = [data for election, data in fixed.items() if election in supported_elections]
            def subset(data):
                # Independent objects prevent changing the recent6 baseline.
                return ElectionData(data.election, data.cells.loc[data.cells.locality.isin(common_locs)].copy(), data.geos,
                                    data.targets.loc[data.targets.locality.isin(common_locs)].copy(), data.partition, data.available)
            if "fixed2022_paired" in args.variants:
                variants["fixed2022_paired"] = [subset(data) for data in paired_fixed]
            for name, collection in annual_collections.items():
                if name in args.variants:
                    variants[name] = [subset(data) for data in collection]
    if "extended_fixed2022" in args.variants and older:
        extended = []
        for election, (z, partition, available) in older.items():
            cells, geos, z = prepare_cells(micro, z)
            extended.append(ElectionData(election, cells, geos, z, partition, available))
        variants["extended_fixed2022"] = sorted(extended + list(fixed.values()), key=lambda data: int(data.election[1:]))
    if not variants:
        raise ValueError("No requested variants have supported data")
    # One immutable split assignment, independent of outcomes and variant coverage.
    model_localities = set.union(*(set(d.targets.locality) for ds in variants.values() for d in ds))
    if args.splits is not None:
        splits = pd.read_csv(args.splits).sort_values("locality").reset_index(drop=True)
        if splits.locality.duplicated().any() or not model_localities <= set(splits.locality) or set(splits.fold) != set(range(args.folds)):
            raise ValueError("Frozen splits must cover every model locality exactly once")
        localities = splits.locality.to_numpy(int)
        split = [(np.flatnonzero(splits.fold.ne(fold)), np.flatnonzero(splits.fold.eq(fold))) for fold in range(args.folds)]
    else:
        localities = np.asarray(sorted(model_localities))
        split = list(GroupKFold(args.folds).split(localities, groups=localities))
        splits = pd.DataFrame([{"locality": int(localities[i]), "fold": fold} for fold, (_, test) in enumerate(split) for i in test])
    args.output.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output / "splits.csv", index=False)
    metrics, predictions, profiles, fits = [], [], [], []
    for variant, datasets in variants.items():
        print(json.dumps({"phase": "prepared", "variant": variant, "elections": [d.election for d in datasets],
                          "cells": len(datasets[0].cells), "localities": len(localities)}), flush=True)
        for fold, (train, test) in enumerate(split):
            if args.fold_limit and fold >= args.fold_limit:
                break
            before = time.perf_counter()
            fitted, info = fit_history(datasets, set(localities[train]), args.epochs, 100+fold)
            rows, preds = evaluate(fitted, datasets, set(localities[test]))
            metrics.extend(dict(row, variant=variant, fold=fold, **info) for row in rows)
            predictions.extend(dict(row, variant=variant, fold=fold, model_fit_id=info["model_fit_id"]) for row in preds)
            fits.append({"variant": variant, "fold": fold, **info, "runtime_seconds": time.perf_counter()-before})
            pd.DataFrame(metrics).to_csv(args.output / "cross_validation.csv", index=False)
            print(json.dumps({"phase": "fold_complete", **fits[-1], "metrics": rows}), flush=True)
        if not args.skip_full_fit:
            fitted, info = fit_history(datasets, set(localities), args.epochs, 42)
            reference = fixed["K25"]
            profiles.extend(dict(row, variant=variant, model_fit_id=info["model_fit_id"]) for row in subgroup_profiles(fitted, datasets, reference, "fixed_2022_common_reference"))
            population_estimand = "year_specific_demographic_proxy" if variant.startswith("annual_") else "fixed2022_fit_population_support"
            profiles.extend(dict(row, variant=variant, model_fit_id=info["model_fit_id"]) for row in subgroup_profiles(fitted, datasets, None, population_estimand))
            fits.append({"variant": variant, "fold": "all", **info})
    pd.DataFrame(predictions).to_csv(args.output / "heldout_predictions.csv", index=False)
    if profiles:
        frame = pd.DataFrame(profiles)
        frame.to_csv(args.output / "demographic_profiles.csv", index=False)
        history_sd(frame).to_csv(args.output / "historical_sd.csv", index=False)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.groupby(["variant", "election"], as_index=False)[["turnout_mae_pp", "turnout_rmse_pp", "observed_partition_mae_pp"]].mean().to_csv(args.output / "comparison.csv", index=False)
    for name, path in [("puf", args.puf), ("annual_targets", args.annual_targets), ("annual_projections", args.annual_projections)]:
        if path is not None and file_identity(path) != input_hashes[name]:
            raise ValueError("Input changed during run: " + name)
    write_json(args.output / "readiness.json", {"research_only": True, "published_parameters_written": False,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "seconds": time.perf_counter()-started},
               "source_sha256_at_start": source_hashes, "input_sha256_at_start": input_hashes,
               "variant_coverage": {name: [{"election": data.election, "localities": int(data.targets.locality.nunique()),
                                            "eligible": int(data.targets.eligible.sum()), "valid": int(data.targets.valid.sum()),
                                            "target_sha256": identity(data.targets),
                                            "population_weight_sum": float(data.cells.population_weight.sum())}
                                           for data in collection] for name, collection in variants.items()},
               "feature_path": "context_features.add_micro_context", "features": FEATURES,
               "locality_logpop_scale": {"reference": "2022 PUF, fixed across variants", "mean": locality_scale[0], "sd": locality_scale[1]},
               "architecture": "Unmodified FamilyModel logistic/softmax coefficients; observed family availability and disjoint merged losses only for historical extensions",
               "objective": "Equal election losses; sqrt(valid)-weighted locality turnout+family crossentropy; published penalties/pooling=.05",
               "fold_definition": "Whole locality histories held out; identical assignment across variants; retrospective fit, not future-election validation",
               "score_aggregation": "Eligible-weighted turnout metrics and valid-weighted partition metrics within each fold/election; comparison.csv uses arithmetic mean across folds, separately for each election",
               "epochs": args.epochs, "folds_completed": args.fold_limit or args.folds,
               "snapshot_id": k26audit["snapshot_id"], "targets_sha256": identity(targets), "splits_sha256": identity(splits),
               "calibration": calibration_audit, "historical_readiness": older_audit, "fits": fits,
               "limitations": ["Annual all-age ethnicity margins proxy adult ethnicity; NII retirement margins adjust age using half-female55–64 proxy. Other within-cell PUF structure remains2022; NII reference-year qualifications are retained in the calibration audit.",
                               "Fixed2022 historical extensions are common-reference sensitivities, not time-matched demography.",
                               "Projected2026 age margins target December1,2026, rather than the exact October election date; source population is NII registry residents, not the CEC eligible-voter roll.",
                               "Population-weighted subgroup profiles cover matched PUF localities only; they are not observed national turnout, group shares or individual votes.",
                               "Family metrics compare only identical observed partitions; merged lists do not identify their constituent shares.",
                               "Subgroup estimates are ecological model outputs, not individual truth. Historical SD is descriptive, not a future-tail probability.",
                               "Annual all-age reweighting changes both population composition and the same-design Census-derived context features."]})
    print(json.dumps({"phase": "complete", "runtime_seconds": time.perf_counter()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--puf", type=Path, required=True)
    parser.add_argument("--annual-targets", type=Path)
    parser.add_argument("--annual-projections", type=Path)
    parser.add_argument("--historical-targets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["fixed2022_recent", "fixed2022_paired", "annual_group_proxy", "annual_group_age65_projection", "annual_anchored_growth", "extended_fixed2022"],
                        choices=["fixed2022_recent", "fixed2022_paired", "annual_group_proxy", "annual_group_age65_projection", "annual_anchored_growth", "extended_fixed2022"])
    parser.add_argument("--splits", type=Path, help="Reuse an existing immutable locality-fold assignment")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--skip-full-fit", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2 or not 0 <= args.fold_limit <= args.folds:
        parser.error("Positive epochs and valid fold count/limit required")
    run(args)


if __name__ == "__main__":
    main()
