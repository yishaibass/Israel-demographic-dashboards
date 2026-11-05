"""Controlled model comparison using dated population and eligible-voter priors.

Resident weights construct Census context. Separate citizen-adjusted weights
enter the electoral mixture. Neither step uses turnout or party results.
The existing 24-feature model, penalties, partitions and folds remain fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

import annual_religiosity as religion
import fit_foundation_history as history
import population_electorate as electorate


CELL_KEYS = ["locality_code", "age_band", "sex", "broad_group"]
AGE_NAMES = {2: "18_24", 3: "25_34", 4: "35_44", 5: "45_54", 6: "55_64", 7: "65_plus"}
REL_NAMES = {1: "secular", 2: "traditional", 3: "religious", 5: "haredi", 6: "mixed", 7: "other"}
VARIANTS = ["fixed2022", "reconstructed_residents", "reconstructed_electorate",
            "reconstructed_electorate_fullwindow", "reconstructed_electorate_jerusalem_scope",
            "reconstructed_electorate_historical_mix"]
ALLOCATION_SENSITIVITIES = {"reconstructed_electorate_jerusalem_scope", "reconstructed_electorate_historical_mix"}
STABLE_CATEGORIES = {
    "right_religious_families": {"likud", "national_religious_right", "shas", "utj", "secular_national_right"},
    "center_left_and_other": {"liberal_center", "zionist_left", "other"},
    "arab_parties": {"arab"},
}


def selected_population(controls, year, election):
    chosen = controls.loc[controls.year.eq(year)].copy()
    if "period_id" in chosen:
        period = election if election in set(chosen.period_id.astype(str)) else str(year)
        chosen = chosen.loc[chosen.period_id.astype(str).eq(period)]
    return chosen


def requested_variants(args):
    names = getattr(args, "variants", None)
    if names is None:
        names = VARIANTS[:3] + ([VARIANTS[3]] if args.sensitivity_factors else []) + ([VARIANTS[4]] if args.sensitivity_electorate_controls else [])
    names = list(dict.fromkeys(names))
    if not names or not set(names) <= set(VARIANTS):
        raise ValueError("Unknown or empty model variant selection")
    if any(name.startswith("reconstructed_electorate") for name in names) and not args.electorate_controls:
        raise ValueError("Electorate variants require calibrated --electorate-controls; select reconstructed_residents for a population-only fit")
    if VARIANTS[3] in names and not args.sensitivity_factors:
        raise ValueError("Full-window variant requires --sensitivity-factors")
    if ALLOCATION_SENSITIVITIES.intersection(names) and not args.sensitivity_electorate_controls:
        raise ValueError("Scope variant requires --sensitivity-electorate-controls")
    return names


def voter_composition(fitted, datasets, variant):
    """Estimated shares of votes cast, with comparable broad party-family totals.

    These are sample projections and net support estimates, not individual flows.
    Category sums are exported only when every observed merged list is contained
    within one category, so unidentified historical components are never split.
    """
    composition, preferences = [], []
    for ei, data in enumerate(datasets):
        for part in data.partition:
            if not any(set(part["families"]) <= families for families in STABLE_CATEGORIES.values()):
                raise ValueError("A historical merged list crosses the trend categories")
        with torch.no_grad():
            x = torch.tensor(data.cells[history.FEATURES].to_numpy(np.float32))
            fsu = torch.tensor(data.cells.fsu_proxy.to_numpy(np.float32))
            pt, pc = fitted.probs(x, fsu, ei)
            pt = pt.numpy()
            pc = history.normalized_choice(pc, data.available).numpy()
        weight = data.cells.weight.to_numpy(float)
        votes = weight * pt
        for group, indices in data.cells.groupby("profile_group").indices.items():
            eligible, voted = float(weight[indices].sum()), float(votes[indices].sum())
            if eligible <= 0:
                continue
            composition.append(dict(variant=variant, election=data.election, group=group,
                                    estimated_eligible=eligible, estimated_eligible_share=eligible/weight.sum(),
                                    turnout=voted/eligible, estimated_votes=voted, voter_share=voted/votes.sum(),
                                    population_localities=int(data.cells.locality.nunique()),
                                    scope="common matched PUF-locality sample; modeled votes cast"))
            for category, families in STABLE_CATEGORIES.items():
                columns = [i for i, family in enumerate(history.FAMILIES) if family in families]
                value = float(np.average(pc[indices][:, columns].sum(1), weights=votes[indices])) if voted > 0 else 0.
                preferences.append(dict(variant=variant, election=data.election, group=group,
                                        category=category, value=value,
                                        family_components=json.dumps(sorted(families)),
                                        interpretation="Stable party-family total; not current coalition or individual voter switching"))
    return composition, preferences


def model_record(fitted, datasets, info):
    return dict(features=history.FEATURES, families=history.FAMILIES,
                elections=[data.election for data in datasets], targeted_fsu=bool(fitted.targeted_fsu),
                model_fit_id=info["model_fit_id"],
                state={key: value.detach().cpu().tolist() for key, value in fitted.state_dict().items()})


def restore_model(record):
    if record["features"] != history.FEATURES or record["families"] != history.FAMILIES:
        raise ValueError("Exported model feature/family schema differs")
    if not record["elections"] or len(set(record["elections"])) != len(record["elections"]):
        raise ValueError("Exported model has invalid election identity")
    with history.election_scope(record["elections"]):
        restored = history.common.model.FamilyModel(len(history.FEATURES), len(history.FAMILIES), record["targeted_fsu"])
    state = {key: torch.tensor(value, dtype=torch.float32) for key, value in record["state"].items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Exported model contains nonfinite coefficients")
    restored.load_state_dict(state, strict=True)
    return restored


def micro_keys(micro):
    return pd.DataFrame({"locality_code": pd.to_numeric(micro.SmlYishuvPUF, errors="raise").astype(int),
                         "age_band": pd.to_numeric(micro.GilPUF, errors="raise").astype(int),
                         "sex": pd.to_numeric(micro.MinPUF, errors="raise").map({1: "male", 2: "female"}),
                         "broad_group": pd.to_numeric(micro.KvutzaUchlusiyaPUF, errors="raise").map({1: "jewish", 2: "arab", 3: "other"})})


def reconstructed_micro(micro, controls, factors, year, period_id=None):
    """Preserve within-cell2022 income/origin structure; update observed margins."""
    keys = micro_keys(micro)
    if keys.isna().any().any():
        raise ValueError("Unmapped PUF adult demographics")
    chosen = controls.loc[controls.year.eq(year)].copy()
    selected_period = str(year)
    if "period_id" in chosen:
        period = period_id if period_id in set(chosen.period_id.astype(str)) else str(year)
        chosen = chosen.loc[chosen.period_id.astype(str).eq(period)]
        selected_period = str(period)
    if chosen.empty or chosen.duplicated(CELL_KEYS).any():
        raise ValueError("Missing or duplicated dated population controls")
    weights = pd.to_numeric(micro.MishkalPratPUF, errors="raise")
    base = keys.assign(base_weight=weights).groupby(CELL_KEYS, as_index=False).base_weight.sum()
    scaled = base.merge(chosen[CELL_KEYS + ["adult_population"]], on=CELL_KEYS, how="inner", validate="one_to_one")
    if not np.isfinite(scaled.adult_population).all() or scaled.adult_population.lt(0).any():
        raise ValueError("Invalid reconstructed adult population")
    if ((scaled.base_weight <= 0) & (scaled.adult_population > 0)).any():
        raise ValueError("Population controls require unsupported demographic cells")
    scaled["factor"] = scaled.adult_population / scaled.base_weight
    applied = keys.merge(scaled[CELL_KEYS + ["factor"]], on=CELL_KEYS, how="left", validate="many_to_one", sort=False)
    supported = applied.factor.notna()
    updated = micro.loc[supported.to_numpy()].copy().reset_index(drop=True)
    updated["MishkalPratPUF"] = weights.loc[supported.to_numpy()].to_numpy() * applied.loc[supported, "factor"].to_numpy()
    # Annual household religiosity changes are pooled survey trends, centered on
    # the2022 household Census definition. Other/unknown mass stays unchanged.
    rkeys = micro_keys(updated)
    rkeys["age_band"] = rkeys.age_band.map(AGE_NAMES)
    rkeys["household_religiosity"] = pd.to_numeric(updated.DatiyutPUF, errors="coerce").map(REL_NAMES).fillna("unknown")
    trend = factors.loc[factors.year.eq(year) & factors.age_band.ne("all"),
                        ["age_band", "sex", "religiosity", "relative_factor"]].rename(columns={"religiosity": "household_religiosity"})
    if trend.duplicated(["age_band", "sex", "household_religiosity"]).any():
        raise ValueError("Duplicate household religiosity factors")
    rel = rkeys.merge(trend, on=["age_band", "sex", "household_religiosity"], how="left", validate="many_to_one", sort=False)
    mutable = rel.broad_group.eq("jewish") & ~rel.household_religiosity.isin(["other", "unknown"])
    if rel.loc[mutable, "relative_factor"].isna().any():
        raise ValueError("Incomplete religiosity trends")
    w = updated.MishkalPratPUF.copy()
    grouping = [rkeys.loc[mutable, key] for key in CELL_KEYS]
    prior_mass = w.loc[mutable].groupby(grouping).transform("sum")
    changed = w.loc[mutable] * rel.loc[mutable, "relative_factor"]
    after_mass = changed.groupby(grouping).transform("sum")
    updated.loc[mutable, "MishkalPratPUF"] = np.divide(
        changed * prior_mass, after_mass, out=np.zeros(len(changed)), where=after_mass.to_numpy() > 0)
    actual = micro_keys(updated).assign(actual=updated.MishkalPratPUF).groupby(CELL_KEYS).actual.sum()
    expected = scaled.set_index(CELL_KEYS).adult_population
    if not np.allclose(actual.sort_index(), expected.sort_index(), rtol=1e-10, atol=1e-6):
        raise ValueError("Religiosity update changed adult age/sex/group population mass")
    unsupported = chosen.merge(base[CELL_KEYS], on=CELL_KEYS, how="left", indicator=True)
    unrepresented = unsupported.loc[unsupported._merge.eq("left_only") & unsupported.locality_code.isin(set(keys.locality_code)), "adult_population"].sum()
    audit = dict(year=year, requested_period_id=period_id, selected_period_id=selected_period,
                 annual_period_fallback=period_id is not None and selected_period != str(period_id),
                 supported_localities=int(updated.SmlYishuvPUF.nunique()),
                 resident_adult_population=float(updated.MishkalPratPUF.sum()),
                 unrepresented_control_mass_within_PUF_localities=float(unrepresented),
                 imputed_demographic_population=float(updated.loc[updated.get("population_donor_imputed", pd.Series(False, index=updated.index)).fillna(False), "MishkalPratPUF"].sum()),
                 conditional_income_origin_structure="2022 Census PUF, fixed within locality-age-sex-population-religiosity cell")
    return updated.loc[updated.MishkalPratPUF.gt(0)].reset_index(drop=True), audit


def apply_eligible_mixture(cells, targets, adult_priors, year, election, citizen_adjustment=True, jerusalem_rate=None, allocated_controls=None):
    """Keep resident context features; use eligibility only in mixture weights."""
    cells = cells.copy()
    group = np.where(cells.arab.eq(1), "arab", np.where(cells.other_pop.eq(1), "other", "jewish"))
    cells["broad_group"] = group
    n = targets[["locality", "eligible"]].rename(columns={"locality": "locality_code", "eligible": "registered_eligible"})
    n = n.assign(year=year, election=election, source_id="CEC locality eligible total only")
    if allocated_controls is not None:
        allocation = allocated_controls.loc[allocated_controls.election.eq(election) & allocated_controls.locality_code.isin(set(n.locality_code))].copy()
        if allocation.empty or allocation.duplicated(["locality_code", "broad_group"]).any():
            raise ValueError("Missing or duplicated calibrated electorate controls")
        if not set(allocation.broad_group) <= {"jewish", "arab", "other"}:
            raise ValueError("Primary model requires three allocated electorate groups")
        allocated = allocation.set_index(["locality_code", "broad_group"]).estimated_eligible
    elif citizen_adjustment:
        allocation = electorate.allocate_electorate(n, adult_priors, jerusalem_rate=jerusalem_rate)
        unknown = allocation.loc[allocation.broad_group.str.startswith("unallocated"), "estimated_eligible"].sum()
        if unknown > 1e-6:
            raise ValueError("Cannot fit unsupported allocated electorate")
        allocated = allocation.set_index(["locality_code", "broad_group"]).estimated_eligible
    else:
        prior = cells.groupby(["locality", "broad_group"]).population_weight.sum()
        denominator = prior.groupby(level="locality").transform("sum")
        allocated = prior / denominator * prior.index.get_level_values("locality").map(targets.set_index("locality").eligible)
        allocated.index = allocated.index.set_names(["locality_code", "broad_group"])
    if not np.isfinite(allocated).all() or (allocated < 0).any():
        raise ValueError("Invalid allocated electorate counts")
    populations = cells.groupby(["locality", "broad_group"]).population_weight.sum()
    populations.index = populations.index.set_names(["locality_code", "broad_group"])
    multipliers = allocated / populations
    index = pd.MultiIndex.from_arrays([cells.locality, cells.broad_group], names=["locality_code", "broad_group"])
    cells["weight"] = cells.population_weight.to_numpy() * multipliers.reindex(index).to_numpy()
    sums = cells.groupby("locality").weight.sum()
    if not np.isfinite(cells.weight).all() or (cells.weight < 0).any() or not np.allclose(sums.sort_index(), targets.set_index("locality").eligible.sort_index(), rtol=1e-10, atol=1e-6):
        raise ValueError("Mixture weights do not close to official eligible denominators")
    return cells


def run(args):
    started = time.perf_counter()
    requested = requested_variants(args)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    input_paths = {name: getattr(args, name) for name in ["puf", "population_controls", "religiosity_factors", "sensitivity_factors", "electorate_controls", "sensitivity_electorate_controls", "splits"] if getattr(args, name) is not None}
    if args.historical_targets:
        input_paths.update(historical_targets=args.historical_targets / "targets.csv",
                           historical_readiness=args.historical_targets / "readiness.json")
    input_hashes = {name: history.file_identity(path) for name, path in input_paths.items()}
    source_paths = {name: Path(__file__).parent / name for name in ["fit_population_reconstruction.py", "fit_foundation_history.py", "fit_k26.py", "context_features.py", "data_pipeline.py", "run_party_family_final.py", "run_multi_election_average.py", "population_electorate.py", "annual_religiosity.py", "population_donors.py"]}
    source_hashes = {name: history.file_identity(path) for name, path in source_paths.items()}
    controls = pd.read_csv(args.population_controls, low_memory=False)
    factors = pd.read_csv(args.religiosity_factors)
    calibrated_electorate = pd.read_csv(args.electorate_controls) if args.electorate_controls else None
    alternate_electorate = pd.read_csv(args.sensitivity_electorate_controls) if args.sensitivity_electorate_controls else None
    common = history.common
    historical, _ = common.build_historical_targets(args.project)
    k26, snapshot = common.build_targets(args.current)
    targets = pd.concat([historical, k26], ignore_index=True)
    partition = [{"id": family, "families": [family]} for family in history.FAMILIES]
    target_sets = {e: (z, partition, history.FAMILIES) for e, z in targets.groupby("election", sort=False)}
    if args.historical_targets:
        older, _ = history.read_older(args.historical_targets)
        target_sets = {**older, **target_sets}
    for name, allocation in [("primary", calibrated_electorate), ("scope", alternate_electorate)]:
        if allocation is not None and any(v.startswith("reconstructed_electorate") for v in requested):
            missing = set(target_sets) - set(allocation.election)
            if missing:
                raise ValueError(f"{name} electorate controls lack requested elections: {sorted(missing)}")
    micro = common.elections.load_all_adults(args.puf).reset_index(drop=True)
    population = pd.to_numeric(micro.MishkalPratPUF).groupby(pd.to_numeric(micro.SmlYishuvPUF)).sum()
    logpop = np.log1p(population)
    scale = (float(logpop.mean()), float(logpop.std()))
    # Freeze paired geographical support before model fitting or examining errors.
    supported = set(micro.SmlYishuvPUF.astype(int)) - {9999}
    for election, (z, _, _) in target_sets.items():
        year = int(electorate.ELECTION_DATES[election][:4])
        dated = selected_population(controls, year, election)
        supported &= set(z.locality) & set(dated.loc[dated.adult_population.gt(0), "locality_code"])
    micro = micro.loc[micro.SmlYishuvPUF.astype(int).isin(supported)].copy().reset_index(drop=True)
    if len(supported) < 100:
        raise ValueError("Paired model support unexpectedly small")
    from population_donors import expand_missing_cohorts
    selected_controls = []
    for election in target_sets:
        year = int(electorate.ELECTION_DATES[election][:4])
        chosen = selected_population(controls, year, election)
        selected_controls.append(chosen.loc[chosen.locality_code.isin(supported)])
    reconstructed_source, donor_audit = expand_missing_cohorts(micro, pd.concat(selected_controls, ignore_index=True))
    splits = pd.read_csv(args.splits).sort_values("locality")
    if splits.locality.duplicated().any() or not supported <= set(splits.locality) or set(splits.fold) != set(range(args.folds)):
        raise ValueError("Frozen locality folds fail coverage")
    splits = splits.loc[splits.locality.isin(supported)]
    splits.to_csv(args.output / "splits.csv", index=False)
    variants = {name: [] for name in requested}
    if args.sensitivity_factors:
        sensitivity = pd.read_csv(args.sensitivity_factors)
    audits = []
    fixed_template, fixed_geos, _ = history.prepare_cells(micro, targets.loc[targets.election.eq("K25") & targets.locality.isin(supported)], scale)
    fixed_population = fixed_template.groupby("locality").population_weight.sum()
    fixed_reference = None
    for election, (z, partitions, families) in sorted(target_sets.items(), key=lambda x: int(x[0][1:])):
        z = z.loc[z.locality.isin(supported)].copy().sort_values("locality").reset_index(drop=True)
        year = int(electorate.ELECTION_DATES[election][:4])
        fixed = fixed_template.copy()
        fixed["weight"] = fixed.population_weight * fixed.locality.map(z.set_index("locality").eligible / fixed_population)
        geos = fixed_geos.copy()
        if election == "K25":
            fixed_reference = history.ElectionData(election, fixed, geos, z, partitions, families)
        if "fixed2022" in variants:
            variants["fixed2022"].append(history.ElectionData(election, fixed, geos, z, partitions, families))
        dated, audit = reconstructed_micro(reconstructed_source, controls, factors, year, election)
        audits.append(dict(audit, election=election, variant="primary"))
        resident, geos, z = history.prepare_cells(dated, z, scale)
        # Exactly the represented resident priors are used, never all-age shares.
        dated_keys = micro_keys(dated).assign(adult_population=dated.MishkalPratPUF)
        priors = dated_keys.groupby(["locality_code", "broad_group"], as_index=False).adult_population.sum().assign(year=year)
        if "reconstructed_residents" in variants:
            variants["reconstructed_residents"].append(history.ElectionData(election, resident, geos, z, partitions, families))
        if any(name.startswith("reconstructed_electorate") for name in variants):
            eligible_cells = apply_eligible_mixture(resident, z, priors, year, election, allocated_controls=calibrated_electorate)
            imputed = dated_keys.loc[dated.population_donor_imputed.fillna(False)].groupby(["locality_code", "broad_group"]).adult_population.sum()
            fractions = (imputed / priors.set_index(["locality_code", "broad_group"]).adult_population).fillna(0)
            eligible_groups = eligible_cells.groupby(["locality", "broad_group"]).weight.sum()
            eligible_groups.index = eligible_groups.index.set_names(["locality_code", "broad_group"])
            audits[-1]["imputed_registered_eligible_mass"] = float((fractions * eligible_groups).sum())
            if "reconstructed_electorate" in variants:
                variants["reconstructed_electorate"].append(history.ElectionData(election, eligible_cells, geos, z, partitions, families))
        if ALLOCATION_SENSITIVITIES.intersection(variants):
            alternate_cells = apply_eligible_mixture(resident, z, priors, year, election, allocated_controls=alternate_electorate)
            for name in ALLOCATION_SENSITIVITIES.intersection(variants):
                variants[name].append(history.ElectionData(election, alternate_cells, geos, z, partitions, families))
        if "reconstructed_electorate_fullwindow" in variants:
            other, audit = reconstructed_micro(reconstructed_source, controls, sensitivity, year, election)
            audits.append(dict(audit, election=election, variant="fullwindow_sensitivity"))
            cells, g, tz = history.prepare_cells(other, z, scale)
            cells = apply_eligible_mixture(cells, tz, priors, year, election, allocated_controls=calibrated_electorate)
            variants["reconstructed_electorate_fullwindow"].append(history.ElectionData(election, cells, g, tz, partitions, families))
        print(json.dumps(dict(phase="prepared", election=election, localities=len(z), cells=len(resident))), flush=True)
    metrics, predictions, profiles, fits, populations, saved_models = [], [], [], [], [], {}
    composition, preferences = [], []
    for variant, datasets in variants.items():
        for data in datasets:
            for group, cells in data.cells.groupby("profile_group"):
                populations.append(dict(variant=variant, election=data.election, group=group,
                                        resident_population=float(cells.population_weight.sum()),
                                        estimated_eligible=float(cells.weight.sum()),
                                        estimated_eligible_share=float(cells.weight.sum()/data.cells.weight.sum()),
                                        scope="identical matched named PUF localities"))
        for fold in range(args.folds):
            train = set(splits.loc[splits.fold.ne(fold), "locality"])
            test = set(splits.loc[splits.fold.eq(fold), "locality"])
            fitted, info = history.fit_history(datasets, train, args.epochs, 100 + fold)
            rows, preds = history.evaluate(fitted, datasets, test)
            metrics.extend(dict(row, variant=variant, fold=fold, **info) for row in rows)
            predictions.extend(dict(row, variant=variant, fold=fold) for row in preds)
            fits.append(dict(variant=variant, fold=fold, **info))
            print(json.dumps(dict(phase="fold_complete", variant=variant, fold=fold)), flush=True)
        fitted, info = history.fit_history(datasets, supported, args.epochs, 42)
        fits.append(dict(variant=variant, fold="all", **info))
        record = model_record(fitted, datasets, info)
        # Check the actual JSON representation, not a shared in-memory object.
        restored = restore_model(json.loads(json.dumps(record)))
        with torch.no_grad():
            for ei, data in enumerate(datasets):
                x = torch.tensor(data.cells[history.FEATURES].iloc[:3].to_numpy(np.float32))
                fsu = torch.tensor(data.cells.fsu_proxy.iloc[:3].to_numpy(np.float32))
                before, after = fitted.probs(x, fsu, ei), restored.probs(x, fsu, ei)
                if any(not torch.equal(a, b) for a, b in zip(before, after)):
                    raise ValueError("Exported model changes predicted probabilities")
        saved_models[variant] = record
        reference = fixed_reference
        profiles.extend(dict(row, variant=variant) for row in history.subgroup_profiles(fitted, datasets, reference, "fixed2022_matched_resident_reference"))
        # Profiles of the election electorate use eligibility weights, while
        # features continue to describe the separately reconstructed residents.
        yearly = []
        for data in datasets:
            copy = data.cells.copy()
            copy["population_weight"] = copy.weight
            yearly.append(history.ElectionData(data.election, copy, data.geos, data.targets, data.partition, data.available))
        profiles.extend(dict(row, variant=variant) for row in history.subgroup_profiles(fitted, yearly, None, "year_specific_matched_eligible_estimate"))
        group_rows, preference_rows = voter_composition(fitted, datasets, variant)
        composition.extend(group_rows)
        preferences.extend(preference_rows)
    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output / "cross_validation.csv", index=False)
    comparison = frame.groupby(["variant", "election"], as_index=False)[["turnout_mae_pp", "turnout_rmse_pp", "observed_partition_mae_pp"]].mean()
    comparison.to_csv(args.output / "comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "heldout_predictions.csv", index=False)
    pd.DataFrame(populations).to_csv(args.output / "group_population_comparison.csv", index=False)
    profile_frame = pd.DataFrame(profiles)
    profile_frame.to_csv(args.output / "demographic_profiles.csv", index=False)
    history.history_sd(profile_frame).to_csv(args.output / "historical_sd.csv", index=False)
    history.write_json(args.output / "fitted_models.json", saved_models)
    pd.DataFrame(composition).to_csv(args.output / "electoral_composition.csv", index=False)
    pd.DataFrame(preferences).to_csv(args.output / "stable_political_preferences.csv", index=False)
    for name, path in input_paths.items():
        if history.file_identity(path) != input_hashes[name]:
            raise ValueError("Input changed during model run: " + name)
    for name, path in source_paths.items():
        if history.file_identity(path) != source_hashes[name]:
            raise ValueError("Model source changed during run: " + name)
    audit = dict(research_only=True, published_files_written=False, python=platform.python_version(),
                 runtime_seconds=time.perf_counter() - started, features=history.FEATURES,
                 localities=len(supported), elections=list(target_sets), epochs=args.epochs, folds=args.folds,
                 architecture="Existing24-feature logistic/softmax; same regularization, observed partitions and immutable locality folds",
                 resident_context_and_electorate_weights_separate=True, calibration=audits, fits=fits,
                 sample_zero_demographic_donors=donor_audit,
                 electorate_controls_sha256=history.file_identity(args.electorate_controls) if args.electorate_controls else None,
                 inputs=input_hashes, sources=source_hashes, snapshot_id=snapshot["snapshot_id"],
                 limitations=["Updated joint population composition is reconstructed from observed margins and explicit donor/relative-trend assumptions.",
                              "Conditional income and origin remain from2022 PUF within demographic cells.",
                              "Citizen-adjusted eligible group counts are estimates calibrated to official locality totals; Jerusalem2024 rate is carried to other years.",
                              "Paired model validation covers named PUF localities; population coverage elsewhere does not establish model validity outside that sample.",
                              "Whole-locality holdout tests geographical reconstruction of observed elections, not future elections."])
    history.write_json(args.output / "readiness.json", audit)
    print(json.dumps(dict(phase="complete", runtime_seconds=audit["runtime_seconds"], localities=len(supported))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "current", "puf", "population-controls", "religiosity-factors", "splits", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sensitivity-factors", type=Path)
    parser.add_argument("--electorate-controls", type=Path, help="Reviewed primary registered group allocations; applied directly, without a second citizenship adjustment")
    parser.add_argument("--sensitivity-electorate-controls", type=Path, help="Alternative population allocation assumption, using the same resident features")
    parser.add_argument("--historical-targets", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        help="Select only the model variants needed for a historical or controlled run")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.folds < 2:
        parser.error("Positive epochs and at least two folds required")
    run(args)


if __name__ == "__main__":
    main()
