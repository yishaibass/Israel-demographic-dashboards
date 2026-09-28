"""Reproducible acceptance-gate experiment for the optional home transaction prior.

The current Stage-1 cells may be supplied with ``--plumbing-only`` to exercise the
complete path. That flag forces NON-ACCEPTANCE irrespective of measured results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline  # noqa: E402
from home_transaction_prior import (  # noqa: E402
    HomeTransactionPriorConfig, load_transaction_cells, transaction_cell_match_mask,
)

FOLDS = [([2018, 2019], 2020), ([2018, 2019, 2020], 2021),
         ([2018, 2019, 2020, 2021], 2023)]
GROUPS = pipeline.GROUP_LABELS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric(y, p, w) -> dict:
    y, p, w = map(lambda x: pd.to_numeric(x, errors="coerce"), (y, p, w))
    m = y.gt(0) & p.gt(0) & w.gt(0) & y.notna() & p.notna() & w.notna()
    y, p, w = y[m].to_numpy(), p[m].to_numpy(), w[m].to_numpy()
    if not len(y):
        return {"n": 0, "weight_sum": 0.0, "effective_n": 0.0, "mae": None,
                "rmse": None, "log_rmse": None, "geometric_prediction_ratio": None}
    e, le = p - y, np.log(p) - np.log(y)
    return {"n": int(len(y)), "weight_sum": float(w.sum()),
            "effective_n": float(w.sum() ** 2 / np.square(w).sum()),
            "mae": float(np.average(np.abs(e), weights=w)),
            "rmse": float(np.sqrt(np.average(e * e, weights=w))),
            "log_rmse": float(np.sqrt(np.average(le * le, weights=w))),
            "geometric_prediction_ratio": float(np.exp(np.average(le, weights=w)))}


def compare(frame: pd.DataFrame, target="target") -> dict:
    b, c = metric(frame[target], frame["base"], frame["weight_hh"]), metric(
        frame[target], frame["candidate"], frame["weight_hh"])
    improve = {k: (b[k] - c[k]) / b[k] for k in ("mae", "rmse", "log_rmse")
               if b[k] not in (None, 0) and c[k] is not None}
    return {"base": b, "candidate": c, "fractional_improvement": improve}


def bootstrap_ci(frame: pd.DataFrame, reps: int, seed: int) -> dict:
    """Paired percentile CIs by resampling households, retaining all their waves."""
    f = frame.dropna(subset=["mezaheprat", "target", "base", "candidate", "weight_hh"]).copy()
    f = f[(f.target > 0) & (f.base > 0) & (f.candidate > 0) & (f.weight_hh > 0)]
    if f.empty:
        return {"available": False, "reason": "no valid household identifiers"}
    rows = []
    for hid, g in f.groupby("mezaheprat", sort=False):
        w, y, b, c = (g[x].to_numpy(float) for x in ("weight_hh", "target", "base", "candidate"))
        rows.append((hid, w.sum(), np.sum(w * np.abs(b-y)), np.sum(w * np.abs(c-y)),
                     np.sum(w * (b-y)**2), np.sum(w * (c-y)**2),
                     np.sum(w * (np.log(b)-np.log(y))**2),
                     np.sum(w * (np.log(c)-np.log(y))**2)))
    a = np.asarray([r[1:] for r in rows], float)
    rng, vals = np.random.default_rng(seed), {k: [] for k in ("mae", "rmse", "log_rmse")}
    n = len(a)
    for _ in range(reps):
        counts = rng.multinomial(n, np.full(n, 1/n))
        z = counts @ a
        vals["mae"].append(1 - z[2] / z[1])
        vals["rmse"].append(1 - np.sqrt(z[4] / z[0]) / np.sqrt(z[3] / z[0]))
        vals["log_rmse"].append(1 - np.sqrt(z[6] / z[0]) / np.sqrt(z[5] / z[0]))
    return {"available": True, "cluster": "mezaheprat", "clusters": n, "replicates": reps,
            "improvement_95pct_ci": {k: [float(x) for x in np.quantile(v, [.025, .975])]
                                      for k, v in vals.items()}}


def weighted_mean(f, col):
    m = f[col].notna() & f.weight_hh.gt(0)
    return float(np.average(f.loc[m, col], weights=f.loc[m, "weight_hh"])) if m.any() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default="home_prior_experiment")
    ap.add_argument("--blend-weight", type=float, default=.25)
    ap.add_argument("--min-transactions", type=int, default=30)
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--plumbing-only", action="store_true")
    args = ap.parse_args()
    if args.bootstrap_reps < 1000:
        raise ValueError("acceptance runner requires at least 1000 bootstrap replicates")

    cells_path, outdir = Path(args.cells).resolve(), Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cells = load_transaction_cells(cells_path)
    config = HomeTransactionPriorConfig(True, str(cells_path), args.blend_weight,
                                        args.min_transactions)
    raw = pd.read_csv(pipeline.CSV_IN, low_memory=False)
    for c in pipeline.NUMERIC:
        if c in raw: raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw[raw.weight_hh > 0].reset_index(drop=True)
    raw["survey_year"] = raw.wave.map(pipeline.WAVE_YEAR)
    raw["group"] = pipeline._grp_series(raw)
    raw["mezaheprat"] = raw["mezaheprat"].astype("string")

    held = []
    sensitivity_specs = [(0.20, 30), (0.30, 30), (0.25, 40)]
    sensitivity_rows = {f"blend_{b:.2f}_min_{n}": [] for b, n in sensitivity_specs}
    for train_years, test_year in FOLDS:
        d = raw[raw.survey_year.isin(train_years + [test_year])].copy().reset_index(drop=True)
        collected = pipeline.collected_by_wave(d, "apt_value_subjective")
        owner = d.tenure_type.eq(1) | d.owns_primary_apt.eq(1)
        primary = d.survey_year.eq(test_year) & owner & d.apt_value_subjective.gt(0)
        secondary = (d.survey_year.eq(test_year) & owner & d.apt_value_subjective.isna()
                     & d.apt_value_estimated.gt(0))
        target = d.apt_value_subjective.where(primary, d.apt_value_estimated.where(secondary))
        kind = pd.Series(np.where(primary, "subjective", np.where(secondary, "cbs", None)), index=d.index)
        masked = d.copy()
        masked.loc[primary | secondary, ["apt_value_subjective", "apt_value_estimated"]] = np.nan
        base, _ = pipeline.hedonic_home(masked, transaction_prior_config=HomeTransactionPriorConfig(),
                                        subjective_collection_override=collected)
        candidate, _ = pipeline.hedonic_home(masked, transaction_prior_config=config,
                                             subjective_collection_override=collected)
        m = primary | secondary
        fold_frame = pd.DataFrame({"train_years": ",".join(map(str, train_years)),
            "test_year": test_year, "kind": kind[m], "target": target[m], "base": base[m],
            "candidate": candidate[m], "weight_hh": d.loc[m, "weight_hh"],
            "group": d.loc[m, "group"], "mezaheprat": d.loc[m, "mezaheprat"]})
        fold_frame["covered"] = transaction_cell_match_mask(masked, cells, config).loc[m].to_numpy()
        held.append(fold_frame)
        for blend, min_n in sensitivity_specs:
            key = f"blend_{blend:.2f}_min_{min_n}"
            scfg = HomeTransactionPriorConfig(True, str(cells_path), blend, min_n)
            sp, _ = pipeline.hedonic_home(masked, transaction_prior_config=scfg,
                                          subjective_collection_override=collected)
            sensitivity_rows[key].append(pd.DataFrame({"kind": kind[m], "target": target[m],
                "base": base[m], "candidate": sp[m], "weight_hh": d.loc[m, "weight_hh"]}))
    held = pd.concat(held, ignore_index=True)
    primary, secondary = held[held.kind.eq("subjective")], held[held.kind.eq("cbs")]

    primary_result = {"overall": compare(primary),
        "by_time_block": {str(y): compare(g) for y, g in primary.groupby("test_year")},
        "by_social_group": {GROUPS.get(int(k), str(k)): compare(g)
                            for k, g in primary.groupby("group")},
        "paired_cluster_bootstrap": bootstrap_ci(primary, args.bootstrap_reps, args.seed)}
    secondary_result = {"overall": compare(secondary),
        "note": "Non-independent CBS concordance only; cannot rescue primary failure."}
    heldout_coverage_effect = {
        "covered": compare(primary[primary.covered]),
        "uncovered": compare(primary[~primary.covered]),
        "note": "Uncovered held-out rows must and do retain the frozen baseline prediction."
    }
    sensitivity = {k: compare(pd.concat(v, ignore_index=True).query("kind == 'subjective'"))
                   for k, v in sensitivity_rows.items()}

    # Calibration by weighted subjective-value quintile.
    cuts = pipeline.weighted_quantile(primary.target, primary.weight_hh, [.2, .4, .6, .8])
    primary = primary.copy(); primary["quintile"] = np.searchsorted(cuts, primary.target) + 1
    quintiles = {str(q): compare(g)["candidate"] for q, g in primary.groupby("quintile")}

    # Exact-cell coverage on genuinely residual owners and explicit fallback identity.
    eligible = ((raw.tenure_type.eq(1) | raw.owns_primary_apt.eq(1))
                & raw.apt_value_subjective.isna() & raw.apt_value_estimated.isna()
                & raw.survey_year.isin([2020, 2021, 2023]))
    match = transaction_cell_match_mask(raw, cells, config) & eligible
    def coverage(mask):
        denom = raw.loc[mask, "weight_hh"].sum()
        return {"n_eligible": int(mask.sum()), "n_matched": int((mask & match).sum()),
                "weighted_coverage": float(raw.loc[mask & match, "weight_hh"].sum()/denom) if denom else None}
    coverage_result = {"overall": coverage(eligible),
        "by_test_year": {str(y): coverage(eligible & raw.survey_year.eq(y)) for y in [2020,2021,2023]},
        "by_social_group": {GROUPS[k]: coverage(eligible & raw.group.eq(k)) for k in sorted(GROUPS)}}
    base_full, _ = pipeline.hedonic_home(raw, transaction_prior_config=HomeTransactionPriorConfig())
    cand_full, _ = pipeline.hedonic_home(raw, transaction_prior_config=config)
    coverage_result["unmatched_max_abs_prediction_difference"] = float(
        (base_full[eligible & ~match] - cand_full[eligible & ~match]).abs().max() or 0)

    # Wave-10 aggregate movement from the complete balance-sheet frame.
    hh0, _ = pipeline.build_household_frame(home_prior_config=HomeTransactionPriorConfig())
    hh1, _ = pipeline.build_household_frame(home_prior_config=config)
    m10 = hh0.wave.eq(10)
    agg = {}
    for col in ("home", "tot_re", "nw"):
        a, b = weighted_mean(hh0[m10], col), weighted_mean(hh1[m10], col)
        agg[col] = {"baseline": a, "candidate": b, "fractional_change": (b-a)/a}
    agg["home_by_social_group"] = {}
    for k in sorted(GROUPS):
        a, b = weighted_mean(hh0[m10 & hh0.grp.eq(k)], "home"), weighted_mean(hh1[m10 & hh1.grp.eq(k)], "home")
        agg["home_by_social_group"][GROUPS[k]] = {"baseline": a, "candidate": b,
                                                   "fractional_change": (b-a)/a if a else None}
    g0 = pipeline.gini_weighted(hh0.loc[m10, "home"], hh0.loc[m10, "weight_hh"])
    g1 = pipeline.gini_weighted(hh1.loc[m10, "home"], hh1.loc[m10, "weight_hh"])
    agg["home_gini"] = {"baseline": g0, "candidate": g1, "absolute_change": g1-g0}

    baseline_manifest_path = ROOT / "data/external_data_review/stage1_baseline.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    baseline_hash_check = {rel: {"expected": expected.lower(),
        "actual": sha256(ROOT/rel), "matches": sha256(ROOT/rel) == expected.lower()}
        for rel, expected in baseline_manifest["sha256"].items()}
    disabled_baseline_reproduced = all(x["matches"] for x in baseline_hash_check.values())
    overall = primary_result["overall"]; ci = primary_result["paired_cluster_bootstrap"]
    group_testable = {g: (x["candidate"]["n"] >= 100 and x["candidate"]["effective_n"] >= 50)
                      for g, x in primary_result["by_social_group"].items()}
    thresholds = {
      "1_primary_accuracy": bool(overall["fractional_improvement"].get("log_rmse", -9) >= .02
        and overall["fractional_improvement"].get("mae", -9) > 0
        and overall["fractional_improvement"].get("rmse", -9) > 0
        and ci.get("available") and ci["improvement_95pct_ci"]["log_rmse"][0] >= 0),
      "2_time_stability": bool(sum(x["fractional_improvement"].get("log_rmse", -9) > 0
                                    for x in primary_result["by_time_block"].values()) >= 2
        and all(min(x["fractional_improvement"].get(k, -9) for k in ("mae","rmse","log_rmse")) >= -.02
                for x in primary_result["by_time_block"].values())),
      "3_social_groups": bool(all(group_testable.values()) and all(
        min(x["fractional_improvement"].get(k, -9) for k in ("mae","rmse","log_rmse")) >= -.02
        for x in primary_result["by_social_group"].values())),
      "4_coverage_and_fallback": bool(coverage_result["overall"]["weighted_coverage"] >= .30
        and all(x["weighted_coverage"] is not None and x["weighted_coverage"] >= .15
                for x in coverage_result["by_test_year"].values())
        and all(x["weighted_coverage"] is not None and x["weighted_coverage"] >= .15
                for x in coverage_result["by_social_group"].values())
        and coverage_result["unmatched_max_abs_prediction_difference"] == 0),
      "5_calibration": bool(.95 <= overall["candidate"]["geometric_prediction_ratio"] <= 1.05
        and all(.90 <= x["candidate"]["geometric_prediction_ratio"] <= 1.10
                for x in primary_result["by_time_block"].values())
        and all(.90 <= x["candidate"]["geometric_prediction_ratio"] <= 1.10
                for x in primary_result["by_social_group"].values())
        and all(.85 <= x["geometric_prediction_ratio"] <= 1.15 for x in quintiles.values())),
      "6_aggregate_plausibility": bool(all(abs(agg[k]["fractional_change"]) < .02 for k in ("home","tot_re","nw"))
        and all(abs(x["fractional_change"]) < .05 for x in agg["home_by_social_group"].values())
        and abs(agg["home_gini"]["absolute_change"]) < .01),
      "7_reproducibility": bool(not args.plumbing_only and disabled_baseline_reproduced and all(
        x["fractional_improvement"].get("log_rmse", -9) > 0 for x in sensitivity.values())),
    }
    provenance = {"plumbing_only": args.plumbing_only,
        "automatic_rejection_reason": ("Current OVER input is a mostly 100-row-capped, "
          "representative-locality extract and is not acceptance-grade; cell transaction "
          "cutoffs are not proven leakage-safe for respondent valuation dates.") if args.plumbing_only else None}
    verdict = "NON_ACCEPTANCE_PLUMBING_ONLY" if args.plumbing_only else (
        "PASS_PENDING_APPROVAL" if all(thresholds.values()) else "FAIL")
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    manifest_summary = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
    report = {"verdict": verdict, "production_default_enabled": False, "provenance": provenance,
      "configuration": vars(args), "folds": [{"train_years": a, "test_year": b} for a,b in FOLDS],
      "hashes": {"cells": sha256(cells_path), "pipeline": sha256(ROOT/'scripts/pipeline.py'),
                 "prior_module": sha256(ROOT/'scripts/home_transaction_prior.py'),
                 **({"manifest": sha256(manifest_path)} if manifest_path else {})},
      "disabled_baseline_hash_check": baseline_hash_check,
      "disabled_baseline_reproduced": disabled_baseline_reproduced,
      "transaction_build_manifest": manifest_summary,
      "transaction_cutoff_assessment": "Not acceptance-safe: sidecar excludes future survey years but does not prove transactions end before each respondent valuation/interview date or the preceding 31 December fallback.",
      "primary_subjective": primary_result, "secondary_cbs_concordance": secondary_result,
      "heldout_covered_vs_uncovered": heldout_coverage_effect, "sensitivity": sensitivity,
      "calibration_quintiles": quintiles, "coverage": coverage_result,
      "aggregate_movement_wave10": agg, "group_sample_sufficiency": group_testable,
      "thresholds": thresholds,
      "command": "python scripts/run_home_prior_experiment.py " + " ".join(sys.argv[1:])}
    json_path, md_path = outdir/f"{args.label}.json", outdir/f"{args.label}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Home transaction prior experiment: {args.label}", "", f"**Verdict: {verdict}.**",
      "", provenance["automatic_rejection_reason"] or "Acceptance-grade provenance supplied.", "",
      "Production default remains disabled.", "", "## Threshold scorecard", ""]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL / INSUFFICIENT'}**" for k,v in thresholds.items()]
    lines += ["", "## Headline evidence", "",
      f"- Primary held-out subjective n: {overall['candidate']['n']}; effective n: {overall['candidate']['effective_n']:.1f}.",
      f"- Weighted log-RMSE improvement: {overall['fractional_improvement'].get('log_rmse', float('nan')):.2%}.",
      f"- Exact-cell weighted residual coverage: {coverage_result['overall']['weighted_coverage']:.2%}.",
      f"- Unmatched maximum prediction difference: {coverage_result['unmatched_max_abs_prediction_difference']:.6g}.",
      "- Full machine-readable metrics, confidence intervals, subgroup sufficiency, calibration, "
      f"aggregate movement, hashes and command: `{json_path.name}`.", "", "## Reuse", "",
      "Replace the cell file and sidecar with acceptance-grade, cutoff-safe cells and rerun without "
      "`--plumbing-only`; do not change the frozen folds or thresholds."]
    md_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
