"""Robustness test for life-stage interactions using locality bootstrap samples."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent

import context_features as ctx
import run_multi_election_average as multi
import run_party_family_final as final


OUT = HERE / "interaction_validation"
CONTEXT = ctx.ALL_CONTEXT
BASE = final.INCOME + CONTEXT
LIFE_STAGE = ["age_z", "age_z2", "married", "young_married"]
INTERACTIONS = [
    "young_married_x_income",
    "young_married_x_arab",
    "young_married_x_haredi",
    "young_married_x_dati",
    "young_married_x_masorti",
]
MODELS = {
    "context_baseline": BASE,
    "life_stage_main": BASE + LIFE_STAGE,
    "young_married_interactions": BASE + LIFE_STAGE + INTERACTIONS,
}
ADDED = set(LIFE_STAGE + INTERACTIONS)


def install_expanded_features():
    """Add pre-specified life-stage features to both cell and micro predictions."""
    original = multi.base.feature_frame

    def expanded(df):
        out = original(df)
        age_code = pd.to_numeric(df.GilPUF, errors="coerce").fillna(4)
        marital = pd.to_numeric(df.MatzavMishpachtiDeYurePUF, errors="coerce").fillna(0)
        # Census codes: age 2/3 are roughly 15-34; marital status 20 is married.
        out["married"] = marital.eq(20).astype("float32")
        young = age_code.isin([2, 3]).astype("float32")
        out["young_married"] = young * out.married
        out["young_married_x_income"] = out.young_married * out.income_log_z
        for group in ["arab", "haredi", "dati", "masorti"]:
            out[f"young_married_x_{group}"] = out.young_married * out[group]
        return out.astype("float32")

    multi.base.feature_frame = expanded
    return expanded


def fit_weighted(cells, geos, targets, cols, train_locs, pool, epochs, seed, multiplicity=None, added_l2=0.30):
    """Fit final.FamilyModel with whole-locality multiplicities and extra shrinkage."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, fsu, w, g, obs = final.setup(cells, geos, targets, cols)
    model = final.FamilyModel(len(cols), len(final.FAMILIES), False)
    opt = torch.optim.Adam(model.parameters(), lr=.04)
    added_ix = torch.tensor([i for i, c in enumerate(cols) if c in ADDED], dtype=torch.long)
    best, state, stale = math.inf, None, 0
    mult = {} if multiplicity is None else multiplicity
    for ep in range(epochs):
        opt.zero_grad()
        losses = []
        for ei, o in enumerate(obs):
            keep = o["z"].locality.isin(train_locs).to_numpy()
            ii_np = np.where(keep)[0]
            ii = torch.tensor(ii_np)
            pt, pc = model.probs(x, fsu, ei)
            at, af = final.aggregate(pt, pc, w, g, len(geos))
            ix = o["ix"][ii]
            gw = np.sqrt(o["z"].valid.to_numpy(np.float32)[ii_np])
            if multiplicity is not None:
                gw *= o["z"].locality.iloc[ii_np].map(mult).fillna(0).to_numpy(np.float32)
            gw = torch.tensor(gw)
            gw /= gw.mean().clamp_min(1e-8)
            tl = -(o["turn"][ii] * at[ix].clamp(1e-6, 1-1e-6).log() +
                   (1-o["turn"][ii]) * (1-at[ix]).clamp(1e-6).log())
            fl = -(o["fam"][ii] * af[ix].clamp_min(1e-8).log()).sum(1)
            losses.append(((tl + fl) * gw).mean())
        reg = .08 * (model.turn_shared[1:].square().mean() + model.choice_shared[1:].square().mean())
        pooling = pool * (model.turn_dev[:, 1:].square().mean() + model.choice_dev[:, 1:].square().mean())
        extra = torch.tensor(0.0)
        if len(added_ix):
            extra = added_l2 * (model.turn_shared[added_ix].square().mean() +
                                model.choice_shared[added_ix].square().mean() +
                                model.turn_dev[:, added_ix].square().mean() +
                                model.choice_dev[:, added_ix].square().mean())
        loss = torch.stack(losses).mean() + reg + pooling + extra
        loss.backward()
        opt.step()
        value = float(loss.detach())
        if value < best - 1e-6:
            best = value
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(state)
    return model, {"epochs": ep+1, "objective": best, "added_l2": added_l2}


def evaluate(model, cells, geos, targets, cols, test_locs):
    x, fsu, w, g, obs = final.setup(cells, geos, targets, cols)
    rows = []
    with torch.no_grad():
        for ei, election in enumerate(final.ELECTIONS):
            o = obs[ei]
            pt, pc = model.probs(x, fsu, ei)
            at, af = final.aggregate(pt, pc, w, g, len(geos))
            mask = o["z"].locality.isin(test_locs).to_numpy()
            ii = np.where(mask)[0]
            ix = o["ix"].numpy()[ii]
            valid = o["z"].valid.to_numpy()[ii]
            eligible = o["z"].eligible.to_numpy()[ii]
            delta = af.numpy()[ix] - o["fam"].numpy()[ii]
            rows.append({
                "election": election,
                "date": multi.DATES[election],
                "family_mae_pp": float(np.average(np.abs(delta).mean(1), weights=valid) * 100),
                "family_rmse_pp": float(np.sqrt(np.average(np.square(delta).mean(1), weights=valid)) * 100),
                "turnout_mae_pp": float(np.average(np.abs(at.numpy()[ix]-o["turn"].numpy()[ii]), weights=eligible) * 100),
                "n_geographies": int(len(ii)),
            })
    return rows


def weighted_summary(rows):
    z = pd.DataFrame(rows)
    return {
        "family_mae_pp": float(np.average(z.family_mae_pp, weights=z.n_geographies)),
        "family_rmse_pp": float(np.average(z.family_rmse_pp, weights=z.n_geographies)),
        "turnout_mae_pp": float(np.average(z.turnout_mae_pp, weights=z.n_geographies)),
    }


def priority_profiles(model, full, ff, cols, run, training_locs):
    x = torch.tensor(ff[cols].to_numpy(np.float32))
    fsu = torch.tensor(ff.fsu_proxy.to_numpy(np.float32))
    wt = pd.to_numeric(full.MishkalPratPUF, errors="coerce").fillna(0).to_numpy(float)
    age = pd.to_numeric(full.GilPUF, errors="coerce")
    marital = pd.to_numeric(full.MatzavMishpachtiDeYurePUF, errors="coerce")
    masks = {
        "young_married_all": age.isin([2, 3]) & marital.eq(20),
        "young_married_arab": age.isin([2, 3]) & marital.eq(20) & ff.arab.eq(1),
        "young_married_haredi": age.isin([2, 3]) & marital.eq(20) & ff.haredi.eq(1),
        "young_married_dati": age.isin([2, 3]) & marital.eq(20) & ff.dati.eq(1),
        "young_married_masorti": age.isin([2, 3]) & marital.eq(20) & ff.masorti.eq(1),
    }
    rows = []
    with torch.no_grad():
        for ei, election in enumerate(final.ELECTIONS):
            pt, pc = model.probs(x, fsu, ei)
            pt, pc = pt.numpy(), pc.numpy()
            for name, mask in masks.items():
                q = mask.to_numpy()
                segment_locs = pd.to_numeric(full.loc[q, "SmlYishuvPUF"], errors="coerce").dropna().unique()
                support_locs = int(np.intersect1d(segment_locs, np.asarray(training_locs)).size)
                base = {"run": run, "election": election, "date": multi.DATES[election], "segment": name,
                        "puf_rows": int(q.sum()), "weighted_adults": float(wt[q].sum()),
                        "supporting_geographies": support_locs,
                        "reportable": bool(q.sum() >= 250 and wt[q].sum() >= 20000 and support_locs >= 5),
                        "turnout": float(np.average(pt[q], weights=wt[q]))}
                for j, fam in enumerate(final.FAMILIES):
                    rows.append({**base, "family": fam, "probability": float(np.average(pc[q, j], weights=wt[q]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puf", type=Path, default=multi.base.PUF_DEFAULT)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--attempts", type=int, default=25)
    ap.add_argument("--target-accepted", type=int, default=15)
    ap.add_argument("--quick", action="store_true", help="Screen models and run only three bootstrap attempts.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    expanded = install_expanded_features()
    targets, meta = multi.build_targets()
    cells, geos, targets, _, micro, _ = multi.prepare(targets, args)
    cells, lz = ctx.add_context(cells, micro)
    locs = np.array(sorted(targets.locality.unique()))
    rng = np.random.default_rng(20220901)
    shuffled = locs.copy(); rng.shuffle(shuffled)
    n_test = max(1, int(round(.20 * len(shuffled))))
    test_locs = set(shuffled[:n_test]); train_locs = np.array(shuffled[n_test:])

    screen_rows, screen_summary = [], []
    for name, cols in MODELS.items():
        model, info = fit_weighted(cells, geos, targets, cols, set(train_locs), .05, args.epochs, 42)
        rows = evaluate(model, cells, geos, targets, cols, test_locs)
        for row in rows:
            row.update({"model": name, **info}); screen_rows.append(row)
        screen_summary.append({"model": name, **weighted_summary(rows)})
    screen = pd.DataFrame(screen_summary)
    base = screen.loc[screen.model.eq("context_baseline")].iloc[0]
    eligible = screen[(screen.model.ne("context_baseline")) &
                      (screen.family_mae_pp <= base.family_mae_pp - .10) &
                      (screen.family_rmse_pp <= base.family_rmse_pp + .10)]
    best_added = screen[screen.model.ne("context_baseline")].sort_values(["family_mae_pp", "family_rmse_pp"]).iloc[0].model
    selected = "context_baseline" if eligible.empty else eligible.sort_values(["family_mae_pp", "family_rmse_pp"]).iloc[0].model
    # Bootstrap the strongest interaction specification against a paired baseline;
    # the fixed-holdout selection threshold remains the model-selection rule.
    bootstrap_model = selected if selected != "context_baseline" else best_added
    cols = MODELS[bootstrap_model]
    pd.DataFrame(screen_rows).to_csv(OUT / "fixed_holdout_by_election.csv", index=False)
    screen.assign(selected=lambda z: z.model.eq(selected)).to_csv(OUT / "model_screen.csv", index=False)

    attempts = min(args.attempts, 3) if args.quick else args.attempts
    target_accepted = min(args.target_accepted, attempts)
    logs, profile_rows = [], []
    full = multi.load_all_adults(args.puf)
    ff = ctx.add_micro_context(full, lz)
    # ctx.add_micro_context calls the expanded feature function installed above.
    accepted = 0
    for seed in range(8101, 8101 + attempts):
        draw = np.random.default_rng(seed).choice(train_locs, size=len(train_locs), replace=True)
        values, counts = np.unique(draw, return_counts=True)
        multiplicity = dict(zip(values.tolist(), counts.tolist()))
        baseline_model, _ = fit_weighted(cells, geos, targets, MODELS["context_baseline"], set(values), .05,
                                         args.epochs, seed, multiplicity)
        baseline_rows = evaluate(baseline_model, cells, geos, targets, MODELS["context_baseline"], test_locs)
        paired_base = weighted_summary(baseline_rows)
        model, info = fit_weighted(cells, geos, targets, cols, set(values), .05, args.epochs, seed, multiplicity)
        rows = evaluate(model, cells, geos, targets, cols, test_locs)
        summary = weighted_summary(rows)
        finite = all(np.isfinite(list(summary.values())))
        passed = bool(finite and summary["family_mae_pp"] <= base.family_mae_pp + .75 and
                      summary["family_rmse_pp"] <= base.family_rmse_pp + 1.00)
        logs.append({"seed": seed, "accepted": passed, "unique_training_localities": len(values),
                     "model": bootstrap_model,
                     "paired_baseline_mae_pp": paired_base["family_mae_pp"],
                     "paired_baseline_rmse_pp": paired_base["family_rmse_pp"],
                     "mae_improvement_vs_paired_baseline_pp": paired_base["family_mae_pp"]-summary["family_mae_pp"],
                     "rmse_improvement_vs_paired_baseline_pp": paired_base["family_rmse_pp"]-summary["family_rmse_pp"],
                     **summary, **info})
        if passed:
            accepted += 1
            profile_rows.extend(priority_profiles(model, full, ff, cols, seed, values))
        if accepted >= target_accepted:
            break
    pd.DataFrame(logs).to_csv(OUT / "bootstrap_acceptance_log.csv", index=False)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(OUT / "bootstrap_priority_profiles.csv", index=False)
    if len(profiles):
        uncertainty = (profiles.groupby(["election", "date", "segment", "family", "reportable"], as_index=False)
                       .probability.agg(mean="mean", p10=lambda x: x.quantile(.10), p90=lambda x: x.quantile(.90), sd="std", runs="count"))
        uncertainty.to_csv(OUT / "bootstrap_uncertainty.csv", index=False)
    audit = {
        **meta,
        "model": "conservative life-stage interaction search with whole-locality bootstrap",
        "young_family_proxy": "Census age codes 2/3 (approximately 15-34) and marital status code 20; children are not observed in the current feature input",
        "candidate_models": MODELS,
        "selection_rule": "added model requires >=0.10pp fixed-holdout family-MAE improvement and <=0.10pp RMSE deterioration",
        "selected_model": selected,
        "best_added_candidate": best_added,
        "bootstrap_diagnostic_model": bootstrap_model,
        "bootstrap_diagnostic_does_not_override_selection_gate": True,
        "fixed_holdout_seed": 20220901,
        "fixed_holdout_localities": len(test_locs),
        "interaction_l2": .30,
        "bootstrap_seeds_attempted": [int(x["seed"]) for x in logs],
        "accepted_runs": int(sum(x["accepted"] for x in logs)),
        "minimum_reportable_runs": 10,
        "reportable_ensemble": bool(sum(x["accepted"] for x in logs) >= 10),
        "dashboard_updated": False,
    }
    (OUT / "interaction_bootstrap_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(screen.to_string(index=False))
    print("SELECTED", selected)
    print("BOOTSTRAP", json.dumps({"attempted": len(logs), "accepted": audit["accepted_runs"], "reportable": audit["reportable_ensemble"]}))


if __name__ == "__main__":
    main()
