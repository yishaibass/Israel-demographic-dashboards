"""Paired locality-history and geographic-block turnout sensitivity.

The design conditions on the observed elections and fixed Census composition.
It estimates uncertainty in ecological turnout coefficients and their changes;
it does not create additional independent national election shocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

GROUPS = ["haredi", "dati", "masorti", "hiloni", "arab", "druze"]
SHARES = ["share_" + group for group in GROUPS]


def prepare_panel(frame):
    required = ["election", "locality_code", "eligible_voters", "total_voted"] + SHARES
    if frame[required].isna().any().any() or frame.duplicated(["election", "locality_code"]).any():
        raise ValueError("Missing values or duplicate election-locality keys")
    if not frame.eligible_voters.gt(0).all() or not frame.total_voted.between(0, frame.eligible_voters).all():
        raise ValueError("Invalid locality turnout denominator/counts")
    if not frame[SHARES].ge(0).all().all() or not frame[SHARES].le(1).all().all() or not frame[SHARES].sum(axis=1).sub(1).abs().le(.01).all():
        raise ValueError("Invalid demographic composition")
    variation = frame.groupby("locality_code")[SHARES].agg(lambda x: x.max()-x.min())
    if variation.to_numpy().max() > 1e-10:
        raise ValueError("This research design requires shared fixed locality composition across elections")
    elections = sorted(frame.election.unique(), key=lambda x: int(x[1:]))
    if "K26" not in elections:
        raise ValueError("K26 comparison election is required")
    localities = np.array(sorted(frame.locality_code.unique()))
    composition = frame.drop_duplicates("locality_code").set_index("locality_code").loc[localities, SHARES].to_numpy(float)
    weights = frame.pivot(index="election", columns="locality_code", values="eligible_voters").reindex(index=elections, columns=localities).fillna(0).to_numpy(float)
    voted = frame.pivot(index="election", columns="locality_code", values="total_voted").reindex(index=elections, columns=localities).fillna(0).to_numpy(float)
    turnout = np.divide(voted, weights, out=np.zeros_like(voted), where=weights > 0)
    return {"elections": elections, "localities": localities, "x": composition, "weights": weights, "turnout": turnout}


def subset_panel(panel, keep):
    return {"elections": panel["elections"], "localities": panel["localities"][keep],
            "x": panel["x"][keep], "weights": panel["weights"][:, keep], "turnout": panel["turnout"][:, keep]}


def draw_multiplicities(labels, rng):
    """Draw whole units; each member locality receives the same unit count."""
    units, membership = np.unique(labels, return_inverse=True)
    sampled = rng.integers(0, len(units), len(units))
    return np.bincount(sampled, minlength=len(units))[membership]


def fit_panel(panel, multiplicities=None):
    counts = np.ones(len(panel["localities"])) if multiplicities is None else np.asarray(multiplicities)
    if counts.shape != (len(panel["localities"]),) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid resampling multiplicities")
    results = []
    for index, election in enumerate(panel["elections"]):
        weights = panel["weights"][index] * counts
        keep = weights > 0
        x = panel["x"][keep]
        if keep.sum() < len(GROUPS) or np.linalg.matrix_rank(x) < len(GROUPS):
            raise ValueError(election + ": zero weight or rank-deficient six-group design")
        sw = np.sqrt(weights[keep]/weights[keep].sum())
        solved = lsq_linear(x*sw[:, None], panel["turnout"][index, keep]*sw, bounds=(.2, .95))
        if not solved.success or not np.isfinite(solved.x).all():
            raise ValueError(election + ": constrained fit failed")
        results.append(solved.x)
    return np.array(results)


def covariance_identity(draws, current_index, history_indices):
    current = draws[:, current_index]
    historical_mean = draws[:, history_indices].mean(axis=1)
    variance_current = current.var(axis=0, ddof=1)
    variance_mean = historical_mean.var(axis=0, ddof=1)
    covariance = ((current-current.mean(axis=0))*(historical_mean-historical_mean.mean(axis=0))).sum(axis=0)/(len(draws)-1)
    variance_change = (current-historical_mean).var(axis=0, ddof=1)
    if not np.allclose(variance_change, variance_current+variance_mean-2*covariance, atol=1e-12):
        raise ValueError("Paired covariance identity failed")
    return variance_current, variance_mean, covariance, variance_change


def bootstrap(panel, labels, draws, seed):
    rng = np.random.default_rng(seed)
    samples, failures = [], {}
    for _ in range(draws):
        counts = draw_multiplicities(labels, rng)
        try:
            samples.append(fit_panel(panel, counts))
        except ValueError as error:
            key = str(error)
            failures[key] = failures.get(key, 0)+1
    if len(samples) < 20:
        raise ValueError("Too few jointly identified bootstrap draws: " + str(len(samples)))
    return np.array(samples), {"attempted": draws, "accepted": len(samples), "rejected": draws-len(samples),
                               "rejection_reasons": failures, "seed": seed,
                               "conditioning": "Each failed draw is rejected for every election together; percentiles condition on full-rank draws, with no replacement of failed attempts"}


def summary_rows(panel, point, samples, design):
    rows = []
    current = panel["elections"].index("K26")
    references = {"all_available_pre_K26": [i for i, e in enumerate(panel["elections"]) if int(e[1:]) < 26],
                  "recent_K21_K25": [i for i, e in enumerate(panel["elections"]) if 21 <= int(e[1:]) <= 25]}
    for reference, history in references.items():
        if len(history) < 2:
            continue
        var26, varmean, covariance, vardelta = covariance_identity(samples, current, history)
        changes = samples[:, current]-samples[:, history].mean(axis=1)
        for group_index, group in enumerate(GROUPS):
            historical = point[history, group_index]
            mean, sd = historical.mean(), historical.std(ddof=0)
            sigma = np.cov(samples[:, history, group_index], rowvar=False, ddof=1)
            estimation_scatter = np.trace(sigma)/len(history)-sigma.sum()/len(history)**2
            rows.append({"design": design, "reference": reference, "group": group,
                         "history_elections": ",".join(panel["elections"][i] for i in history),
                         "k26_estimate_pct": point[current, group_index]*100, "historical_mean_pct": mean*100,
                         "change_pp": (point[current, group_index]-mean)*100, "observed_historical_sd_pp": sd*100,
                         "descriptive_historical_sd_change": (point[current, group_index]-mean)/sd if sd >= 1e-6 else None,
                         "bootstrap_sd_k26_pp": np.sqrt(var26[group_index])*100,
                         "bootstrap_sd_historical_mean_pp": np.sqrt(varmean[group_index])*100,
                         "covariance_k26_historical_mean_pp2": covariance[group_index]*10000,
                         "bootstrap_sd_change_pp": np.sqrt(vardelta[group_index])*100,
                         "paired_change_p025_pp": np.percentile(changes[:, group_index], 2.5)*100,
                         "paired_change_p975_pp": np.percentile(changes[:, group_index], 97.5)*100,
                         "estimation_error_contribution_to_historical_variance_pp2": estimation_scatter*10000,
                         "observed_historical_variance_pp2": sd**2*10000,
                         "estimation_error_fraction_of_observed_historical_variance": estimation_scatter/sd**2 if sd >= 1e-6 else None,
                         "historical_variance_minus_estimation_error_pp2": (sd**2-estimation_scatter)*10000,
                         "noise_corrected_historical_sd_sensitivity_pp": np.sqrt(max(0., sd**2-estimation_scatter))*100,
                         "interpretation": "Conditional geographic resampling sensitivity; not a predictive interval for a new national election"})
    return rows


def geographic_jackknife(panel, labels):
    records = []
    for block in np.unique(labels):
        try:
            fitted = fit_panel(panel, (labels != block).astype(int))
            for t, election in enumerate(panel["elections"]):
                for g, group in enumerate(GROUPS):
                    records.append({"omitted_block": str(block), "election": election, "group": group,
                                    "estimate": fitted[t, g], "status": "identified"})
        except ValueError as error:
            records.append({"omitted_block": str(block), "election": None, "group": None,
                            "estimate": None, "status": str(error)})
    return records


def comovement(panel, point, labels):
    """Summarize weighted pair correlations without fitting a full covariance."""
    complete = (panel["weights"] > 0).all(axis=0)
    residual = (panel["turnout"]-point @ panel["x"].T)[:, complete]
    weights = panel["weights"][:, complete]
    residual -= residual.mean(axis=0, keepdims=True)
    residual -= np.array([np.average(row, weights=w) for row, w in zip(residual, weights)])[:, None]
    residual -= residual.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(residual, axis=0)
    valid = norms > 1e-10
    vectors = (residual[:, valid]/norms[valid]).T
    population_weights = weights.mean(axis=0)[valid]
    block_labels = np.asarray(labels)[complete][valid]

    def pair_totals(vectors, weights):
        weighted_sum = np.sum(vectors*weights[:, None], axis=0)
        self_terms = np.sum(weights**2*np.sum(vectors**2, axis=1))
        return (np.sum(weighted_sum**2)-self_terms)/2, (weights.sum()**2-np.sum(weights**2))/2

    total_numerator, total_denominator = pair_totals(vectors, population_weights)
    within_num, within_den = 0., 0.
    for block in np.unique(block_labels):
        keep = block_labels == block
        numerator, denominator = pair_totals(vectors[keep], population_weights[keep])
        within_num += numerator
        within_den += denominator
    across_den = total_denominator-within_den
    return {"complete_history_localities": int(valid.sum()), "elections": len(panel["elections"]),
            "weighted_mean_within_block_residual_pair_correlation": float(within_num/within_den) if within_den > 0 else None,
            "weighted_mean_across_block_residual_pair_correlation": float((total_numerator-within_num)/across_den) if across_den > 0 else None,
            "method": "Fit demographic turnout per election, remove persistent locality residual means and each election's eligible-weighted common residual, then summarize all pair correlations weighted by mean-locality eligibility",
            "limitation": "Short time series; descriptive residual co-movement only, not an estimated unrestricted population covariance or evidence of independent national shocks"}


def group_exposure(panel, labels):
    current = panel["elections"].index("K26")
    result = {}
    for g, group in enumerate(GROUPS):
        mass = panel["weights"][current]*panel["x"][:, g]
        by_block = np.array([mass[labels == block].sum() for block in np.unique(labels)])
        shares = by_block/by_block.sum()
        result[group] = {"blocks_with_positive_mass": int((by_block > 0).sum()),
                         "blocks_with_at_least_one_percent": int((shares >= .01).sum()),
                         "largest_block_share": float(shares.max()), "effective_blocks_inverse_herfindahl": float(1/np.sum(shares**2))}
    return result


def run(args):
    started = time.perf_counter()
    frame = pd.read_csv(args.panel)
    panel = prepare_panel(frame)
    original_panel = panel
    support = {"mode": "all_available_locality_histories", "original_localities": len(panel["localities"])}
    if args.balanced:
        complete = (panel["weights"] > 0).all(axis=0)
        panel = subset_panel(panel, complete)
        support["mode"] = "balanced_common_localities_across_all_available_elections"
    support["retained_localities"] = len(panel["localities"])
    support["eligible_coverage_of_input_by_election"] = {
        election: float(panel["weights"][i].sum()/original_panel["weights"][i].sum()) for i,election in enumerate(panel["elections"])}
    point = fit_panel(panel)
    designs = [("paired_locality", panel, np.arange(len(panel["localities"])))]
    geography_audit = {"status": "not_supplied"}
    jackknife = []
    if args.geography:
        geography = pd.read_csv(args.geography)
        if geography.locality_code.duplicated().any():
            raise ValueError("Geographic lookup must be unique per locality")
        lookup = geography.set_index("locality_code")[args.block_column].reindex(panel["localities"])
        keep = lookup.notna().to_numpy()
        geographic_panel = subset_panel(panel, keep)
        labels = lookup.loc[lookup.notna()].astype(str).to_numpy()
        units = np.unique(labels)
        if len(units) < 3:
            raise ValueError("Fewer than three validated geographic blocks")
        if not keep.all():
            designs.append(("paired_locality_geography_matched", geographic_panel, np.arange(keep.sum())))
        designs.append(("paired_" + args.block_column.replace("_code", "") + "_blocks", geographic_panel, labels))
        geographic_point = fit_panel(geographic_panel)
        geography_audit = {"status": "supplied", "path": str(args.geography), "block_column": args.block_column,
                           "source_provenance": args.geography_provenance,
                           "sha256": hashlib.sha256(args.geography.read_bytes()).hexdigest(),
                           "blocks": len(units), "block_codes": [str(x) for x in units],
                           "matched_localities": int(keep.sum()), "unmatched_localities": int((~keep).sum()),
                           "eligible_coverage_by_election": {e: float(geographic_panel["weights"][t].sum()/panel["weights"][t].sum()) for t,e in enumerate(panel["elections"])},
                           "few_cluster_warning": "Geographic bootstrap and deletion results are sensitivity diagnostics; few regional clusters do not establish calibrated 95% intervals",
                           "residual_comovement": comovement(geographic_panel, geographic_point, labels)}
        geography_audit["group_block_exposure"] = group_exposure(geographic_panel, labels)
        jackknife = [{**row, "block_type": args.block_column} for row in geographic_jackknife(geographic_panel, labels)]
        if args.stress_block_column:
            stress_lookup = geography.set_index("locality_code")[args.stress_block_column].reindex(geographic_panel["localities"])
            if stress_lookup.isna().any():
                raise ValueError("Stress block mapping must cover the same geographic comparison frame")
            stress_labels = stress_lookup.astype(str).to_numpy()
            designs.append(("paired_"+args.stress_block_column.replace("_code", "")+"_blocks", geographic_panel, stress_labels))
            geography_audit["stress_design"] = {"block_column": args.stress_block_column,
                 "blocks": int(len(np.unique(stress_labels))), "group_block_exposure": group_exposure(geographic_panel, stress_labels),
                 "residual_comovement": comovement(geographic_panel, geographic_point, stress_labels)}
            jackknife.extend({**row, "block_type": args.stress_block_column} for row in geographic_jackknife(geographic_panel, stress_labels))
    rows, draw_frames, design_audits, covariance = [], [], {}, {}
    for offset, (name, design_panel, labels) in enumerate(designs):
        estimate = fit_panel(design_panel)
        samples, audit = bootstrap(design_panel, labels, args.draws, args.seed+offset)
        audit.update({"sampling_units": int(len(np.unique(labels))), "localities": len(labels)})
        design_audits[name] = audit
        rows.extend(summary_rows(design_panel, estimate, samples, name))
        columns = [e+"_"+g for e in panel["elections"] for g in GROUPS]
        draws = pd.DataFrame(samples.reshape(len(samples), -1), columns=columns)
        draws.insert(0, "accepted_draw", np.arange(len(samples)))
        draws.insert(0, "design", name)
        draw_frames.append(draws)
        covariance[name] = {group: np.cov(samples[:, :, g], rowvar=False, ddof=1).tolist() for g,group in enumerate(GROUPS)}
        print(json.dumps({"design": name, "accepted": audit["accepted"], "rejected": audit["rejected"]}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    pd.concat(draw_frames, ignore_index=True).to_csv(args.output / "bootstrap_draws.csv", index=False)
    if jackknife:
        pd.DataFrame(jackknife).to_csv(args.output / "regional_jackknife.csv", index=False)
    point_records = [{"election": e, "group": g, "estimate": point[t,k]} for t,e in enumerate(panel["elections"]) for k,g in enumerate(GROUPS)]
    pd.DataFrame(point_records).to_csv(args.output / "point_estimates.csv", index=False)
    report = {"panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(), "panel_path": str(args.panel),
              "elections": panel["elections"], "groups": GROUPS, "point_estimates": point_records,
              "support": support,
              "designs": design_audits, "geography": geography_audit, "summary": rows,
              "bootstrap_coefficient_covariance": covariance,
              "decomposition": "Var(K26-historical_mean)=Var(K26)+Var(historical_mean)-2Cov(K26,historical_mean); historical estimation-scatter contribution = trace(Sigma_history)/T - sum(Sigma_history)/T^2",
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "runtime_seconds": time.perf_counter()-started,
              "limitations": ["All results condition on these observed national elections and fixed K25 locality demographic composition.",
                              "Older elections use a retrospective demographic composition bridge; error in that bridge is not captured by this bootstrap.",
                              "All years of a sampled locality or geographic block share a multiplicity, preserving observed temporal and within-block population dependence.",
                              "Locality-only resampling assumes different localities are independent; regional resampling relaxes that assumption within supplied blocks, not between them.",
                              "Neither procedure identifies the frequency or variance of future national election shocks from the number of voters or localities.",
                              "Percentiles are conditional resampling sensitivity ranges; regional few-cluster coverage is not validated.",
                              "The variance-minus-estimation-error column is a covariance-aware descriptive correction, not an identified latent temporal shock variance; it may be negative.",
                              "Individual Bernoulli variance and cross-locality turnout dispersion are not added to the historical SD denominator.",
                              "Geographic co-movement is a short-panel diagnostic; no unrestricted covariance matrix or temporal process was fitted."]}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"runtime_seconds": report["runtime_seconds"], "arab": [r for r in rows if r["group"] == "arab"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geography", type=Path)
    parser.add_argument("--block-column", default="district_code")
    parser.add_argument("--stress-block-column")
    parser.add_argument("--balanced", action="store_true", help="Use the same locality support in every election and both historical references")
    parser.add_argument("--geography-provenance", default="")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.draws < 20:
        parser.error("At least twenty attempted draws are required")
    run(args)


if __name__ == "__main__":
    main()
