"""Estimate K26 six-group parameters against a frozen K21–K25 baseline.

K25 locality demographic composition is explicitly carried forward. This avoids
assuming that a K26 ballot number identifies the same geography as a K25 ballot.
The estimates are ecological coefficients, not observed individual voting rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

GROUPS = ["haredi", "dati", "masorti", "hiloni", "arab", "druze"]
SHARES = ["share_" + group for group in GROUPS]
HISTORICAL = ["K21", "K22", "K23", "K24", "K25"]
SD_FLOOR = 1e-6


def truth(values):
    return values.astype(str).str.lower().isin(["true", "1"])


def locality_composition(frame):
    """Construct K25 eligible-weighted shares within known demographic boxes."""
    historical = frame.loc[frame.election.eq("K25")].copy()
    keep = (~truth(historical.is_envelope) & historical.demog_source.ne("none")
            & historical[SHARES].notna().all(axis=1)
            & historical[SHARES].ge(0).all(axis=1)
            & historical[SHARES].le(1).all(axis=1)
            & historical[SHARES].sum(axis=1).sub(1).abs().le(.01)
            & historical.eligible_voters.gt(0))
    usable = historical.loc[keep]
    numerator = usable[SHARES].multiply(usable.eligible_voters, axis=0)
    numerator["locality_code"] = usable.locality_code
    weights = usable.groupby("locality_code").eligible_voters.sum()
    shares = numerator.groupby("locality_code").sum().div(weights, axis=0)
    audit = {
        "source_election": "K25", "demographic_source": "2022 Census fractional shares",
        "method": "K25 eligible-weighted mean of demographically mapped, non-envelope ballot boxes per locality",
        "k25_boxes_total": len(historical), "k25_boxes_used": len(usable),
        "k25_eligible_used": int(usable.eligible_voters.sum()),
        "k25_localities_with_composition": len(shares),
        "within_locality_missing_composition": "K25 mapped-box shares represent the entire K26 locality; no K26 ballot-number identity is inferred",
    }
    return shares, audit


def solve_pair(x, turnout, coalition, weights):
    sw = np.sqrt(weights / weights.sum())
    a = lsq_linear(x * sw[:, None], turnout * sw, bounds=(.2, .95)).x
    b = lsq_linear(x * sw[:, None], coalition * sw, bounds=(0, a)).x
    return a, b, b / a


def weighted_r2(actual, predicted, weights):
    denominator = np.sum(weights * (actual - np.average(actual, weights=weights)) ** 2)
    return float(1 - np.sum(weights * (actual - predicted) ** 2) / denominator) if denominator > 0 else None


def standardized_change(value, mean, sd):
    if not np.isfinite(sd) or sd < SD_FLOOR:
        return None, "Historical SD below 0.000001; boundary-constrained or nearly constant historical coefficient"
    return float((value - mean) / sd), None


def estimate(ballots, dimension, historical, baseline, readiness, bootstrap=200, seed=42):
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if dimension.ballot_symbol.duplicated().any():
        raise ValueError("Duplicate party symbols in K26 dimension")
    vote_columns = [column for column in ballots if column.startswith("votes_")]
    symbols = set(dimension.ballot_symbol.astype(str))
    missing_symbols = [column[6:] for column in vote_columns if column[6:] not in symbols]
    if missing_symbols:
        raise ValueError(f"Party dimension missing symbols: {missing_symbols}")
    if not ballots[vote_columns].sum(axis=1).eq(ballots.valid_votes).all():
        raise ValueError("K26 party votes do not reconcile to valid votes")
    if not (ballots.valid_votes + ballots.invalid_votes).eq(ballots.total_voted).all():
        raise ValueError("K26 valid and invalid votes do not reconcile to voters")
    if set(ballots.snapshot_id.astype(str)) != {readiness["snapshot_id"]}:
        raise ValueError("K26 ballots and readiness snapshots differ")
    coalition = dimension.bloc_core.eq("Netanyahu bloc") | truth(dimension.amcha_included_in_coalition)
    coalition_symbols = dimension.loc[coalition, "ballot_symbol"].tolist()
    coalition_columns = ["votes_" + symbol for symbol in coalition_symbols]
    if any(column not in ballots for column in coalition_columns):
        raise ValueError("Coalition party dimension contains a missing vote column")
    composition, composition_audit = locality_composition(historical)
    regular = ballots.loc[~truth(ballots.external_envelope)].copy()
    if not regular.eligible_voters.gt(0).all():
        raise ValueError("Non-envelope K26 ballots contain nonpositive eligible counts")
    regular["coalition_votes"] = regular[coalition_columns].sum(axis=1)
    localities = regular.groupby("locality_code")[["eligible_voters", "total_voted", "valid_votes", "coalition_votes"]].sum()
    matched = localities.join(composition, how="inner")
    if len(matched) < len(GROUPS):
        raise ValueError("Too few matched localities for six-group estimation")
    x = matched[SHARES].to_numpy(float)
    if np.linalg.matrix_rank(x) != len(GROUPS):
        raise ValueError("Demographic design does not have full six-group rank")
    weights = matched.eligible_voters.to_numpy(float)
    turnout = (matched.total_voted / matched.eligible_voters).to_numpy(float)
    per_eligible = (matched.coalition_votes / matched.eligible_voters).to_numpy(float)
    if np.any(turnout > 1) or np.any(per_eligible > turnout):
        raise ValueError("Matched locality turnout/coalition counts are incoherent")
    a, b, support = solve_pair(x, turnout, per_eligible, weights)
    rng = np.random.default_rng(seed)
    boot_turnout = np.empty((bootstrap, len(GROUPS)))
    boot_support = np.empty_like(boot_turnout)
    for draw in range(bootstrap):
        indices = rng.integers(0, len(matched), len(matched))
        boot_turnout[draw], _, boot_support[draw] = solve_pair(
            x[indices], turnout[indices], per_eligible[indices], weights[indices])
    baseline = baseline.set_index("group").loc[GROUPS]
    rows = []
    for index, group in enumerate(GROUPS):
        for parameter, values, draws in [("theta_vote", a, boot_turnout), ("theta_bibi", support, boot_support)]:
            historical_values = baseline.loc[group, [parameter + "_" + election for election in HISTORICAL]].to_numpy(float)
            mean, sd = float(np.mean(historical_values)), float(np.std(historical_values, ddof=0))
            if not np.isclose(mean, baseline.loc[group, parameter + "_mean"], atol=1e-12):
                raise ValueError("Stored baseline mean does not match historical coefficients")
            if not np.isclose(sd, baseline.loc[group, parameter + "_sd"], atol=1e-12):
                raise ValueError("Stored baseline SD does not match historical coefficients")
            value = float(values[index])
            z, reason = standardized_change(value, mean, sd)
            rows.append({
                "group": group, "parameter": parameter, "historical_mean": mean,
                "historical_sd": sd, "k25_estimate": float(historical_values[-1]),
                "k26_estimate": value, "change_vs_mean_pp": (value - mean) * 100,
                "change_vs_k25_pp": (value - historical_values[-1]) * 100,
                "change_vs_mean_sd": z, "sd_unavailable_reason": reason,
                "bootstrap_p10": float(np.percentile(draws[:, index], 10)),
                "bootstrap_p90": float(np.percentile(draws[:, index], 90)),
                "at_lower_bound": bool(value <= (.2 if parameter == "theta_vote" else 0) + 1e-6),
                "at_upper_bound": bool(value >= (.95 if parameter == "theta_vote" else 1) - 1e-6),
                "snapshot_id": readiness["snapshot_id"], "count_status": readiness["count_status"],
            })
    national_coalition = int(ballots[coalition_columns].sum().sum())
    audit = {
        **composition_audit, "k26_regular_boxes": len(regular),
        "k26_envelope_rows_excluded": int(truth(ballots.external_envelope).sum()),
        "k26_localities_total": len(localities), "k26_localities_matched": len(matched),
        "k26_unmatched_locality_codes": sorted(int(code) for code in localities.index.difference(matched.index)),
        "k26_regular_eligible": int(localities.eligible_voters.sum()),
        "k26_matched_eligible": int(matched.eligible_voters.sum()),
        "k26_matched_eligible_fraction": float(matched.eligible_voters.sum() / localities.eligible_voters.sum()),
        "k26_matched_valid_votes": int(matched.valid_votes.sum()),
        "k26_matched_valid_vote_fraction_regular": float(matched.valid_votes.sum() / localities.valid_votes.sum()),
        "k26_matched_valid_vote_fraction_national": float(matched.valid_votes.sum() / ballots.valid_votes.sum()),
        "coalition_symbols": coalition_symbols, "all_party_symbols_preserved": True,
    }
    result = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "snapshot_id": readiness["snapshot_id"], "count_status": readiness["count_status"],
        "as_of_utc": readiness.get("as_of_utc"), "rows": rows, "coverage": audit,
        "baseline": {"elections": HISTORICAL, "averaging": "equal-weight arithmetic mean", "sd_ddof": 0,
                     "frozen": True, "sd_floor": SD_FLOOR},
        "fit": {"turnout_r2": weighted_r2(turnout, x @ a, weights),
                "coalition_per_eligible_r2": weighted_r2(per_eligible, x @ b, weights),
                "bootstrap_replicates": bootstrap, "bootstrap_seed": seed,
                "bootstrap_interval": "10th–90th percentiles; locality resampling, jointly refitting turnout then coalition in each draw",
                "design_condition_number": float(np.linalg.cond(x * np.sqrt(weights[:, None]))),
                "turnout_bounds": [.2, .95], "coalition_per_eligible_upper_bound": "same-draw turnout coefficient"},
        "national_observed": {"valid_votes": int(ballots.valid_votes.sum()), "total_voted": int(ballots.total_voted.sum()),
                              "coalition_votes": national_coalition,
                              "coalition_share_valid": national_coalition / int(ballots.valid_votes.sum()),
                              "coalition_share_all_voters": national_coalition / int(ballots.total_voted.sum())},
        "limitations": [
            "Group values are ecological model estimates, not observed individual votes or turnout.",
            "K25 locality demographic shares are carried forward; demographic changes within localities are unmeasured.",
            "K25 demographics cover only mapped boxes within each locality; applying their average to the full K26 locality is an explicit assumption.",
            "External-envelope votes are included in national observed totals and excluded from demographic fitting.",
            "Historical SD measures variation across five historical elections, not estimation uncertainty or statistical significance.",
            "Bootstrap intervals condition on the carried-forward composition and exclude uncertainty in that demographic assumption.",
            "Group coalition support uses all voters, including invalid ballots, following the original simulator convention.",
            "The model does not force fitted demographic coefficients to reproduce national outcomes.",
        ],
    }
    if readiness["count_status"] != "official":
        result["limitations"].append("Source status is " + readiness["count_status"] + "; interpret estimates with the snapshot coverage and provenance.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Local voting-project root containing data/clean and data/processed")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    current = args.root / "data/processed/k26/current"
    baseline_path = args.root / "data/processed/params_groups.csv"
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    result = estimate(pd.read_csv(current / "ballots.csv"), pd.read_csv(current / "party_dimension.csv"),
                      pd.read_csv(args.root / "data/clean/model_input_v2.csv"), pd.read_csv(baseline_path),
                      json.loads((current / "readiness.json").read_text(encoding="utf-8")), args.bootstrap, args.seed)
    result["baseline"]["sha256"] = baseline_hash
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != baseline_hash:
        raise RuntimeError("Historical baseline changed during estimation")
    output = current / "model_comparison"
    output.mkdir(exist_ok=True)
    (output / "six_group_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(result["rows"]).to_csv(output / "six_group_comparison.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"snapshot_id": result["snapshot_id"], "fit": result["fit"], "coverage": result["coverage"], "output": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
