"""Account for changes in modeled voter shares and broad party-family shares.

Shapley components are exact accounting identities, not causal effects or
individual voter transfers. Inputs must use the same geographical support and
group definitions across elections. Output data remain outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .population_electorate import ELECTION_DATES
except ImportError:
    from population_electorate import ELECTION_DATES


RIGHT_FAMILIES = {"likud", "national_religious_right", "shas", "utj", "secular_national_right"}
CENTER_LEFT_OTHER_FAMILIES = {"liberal_center", "zionist_left", "other"}
CATEGORIES = ["right_religious_families", "center_left_and_other", "arab_parties"]
FACTOR_NAMES = ["eligible_mix", "turnout", "within_group_support"]
IDENTITY = ["variant", "election", "group"]
VOTER_COLUMNS = ["variant", "from_election", "to_election", "comparison", "group",
                 "from_voter_share", "to_voter_share", "change_pp",
                 "eligible_mix_component_pp", "turnout_component_pp"]
BLOC_COLUMNS = ["variant", "from_election", "to_election", "comparison", "category",
                "from_bloc_share", "to_bloc_share", "change_pp",
                "eligible_mix_component_pp", "turnout_component_pp", "within_group_support_component_pp"]
GROUP_BLOC_COLUMNS = ["variant", "from_election", "to_election", "comparison", "group", "category",
                      "from_group_bloc_share", "to_group_bloc_share", "change_pp",
                      "eligible_mix_component_pp", "turnout_component_pp", "within_group_support_component_pp"]


def stable_category(families):
    """A merged ballot observation must fit wholly within one stable category."""
    if isinstance(families, str):
        families = json.loads(families) if families.startswith("[") else [families]
    categories = set()
    for family in families:
        if family in RIGHT_FAMILIES:
            categories.add(CATEGORIES[0])
        elif family in CENTER_LEFT_OTHER_FAMILIES:
            categories.add(CATEGORIES[1])
        elif family == "arab":
            categories.add(CATEGORIES[2])
        else:
            raise ValueError("Unknown political family: " + str(family))
    if len(categories) != 1:
        raise ValueError("Merged party families cross stable political categories")
    return categories.pop()


def _weights(eligible_mix, turnout):
    p, t = np.asarray(eligible_mix, float), np.asarray(turnout, float)
    if p.ndim != 1 or t.shape != p.shape or len(p) == 0:
        raise ValueError("Eligible mix and turnout require matching nonempty vectors")
    if not np.isfinite(p).all() or not np.isfinite(t).all() or (p < 0).any() or (t < 0).any() or (t > 1).any():
        raise ValueError("Invalid eligible mix or turnout")
    if not np.isclose(p.sum(), 1, atol=1e-8, rtol=0) or np.dot(p, t) <= 0:
        raise ValueError("Eligible shares must sum to one and imply positive voters")
    return p, t


def voter_shares(eligible_mix, turnout):
    p, t = _weights(eligible_mix, turnout)
    votes = p * t
    return votes / votes.sum()


def group_bloc_shares(eligible_mix, turnout, support):
    p, t = _weights(eligible_mix, turnout)
    s = np.asarray(support, float)
    if s.ndim != 2 or s.shape[0] != len(p) or not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
        raise ValueError("Invalid group political-support matrix")
    if not np.allclose(s.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Political support must sum to one within each group")
    return voter_shares(p, t)[:, None] * s


def shapley_changes(function, before, after):
    """Average all factor-orderings, including changing normalization terms."""
    if len(before) != len(after) or not before:
        raise ValueError("Matching factors are required")
    baseline, final = function(*before), function(*after)
    components = np.zeros((len(before),) + baseline.shape)
    orders = list(permutations(range(len(before))))
    for order in orders:
        state = list(before)
        previous = baseline
        for factor in order:
            state[factor] = after[factor]
            current = function(*state)
            components[factor] += current - previous
            previous = current
    components /= len(orders)
    if not np.allclose(components.sum(axis=0), final - baseline, atol=1e-12, rtol=0):
        raise ValueError("Change attribution does not conserve the observed modeled change")
    return baseline, final, components


def normalized_preferences(preferences):
    """Correct only sub-1e-6 probability-sum roundoff in retained model exports."""
    result = preferences.copy()
    values = pd.to_numeric(result.value, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid political-support probabilities")
    result["value"] = values
    totals = result.groupby(IDENTITY, dropna=False).value.transform("sum").to_numpy(float)
    deviation = float(np.max(np.abs(totals-1))) if len(totals) else 0.
    if (totals <= 0).any() or deviation > 1e-6:
        raise ValueError("Political support must sum to one within each group")
    result["value"] = values / totals
    delta = float(np.max(np.abs(result.value.to_numpy()-values))) if len(values) else 0.
    return result, dict(max_input_probability_sum_deviation=deviation,
                        max_probability_renormalization_delta=delta,
                        permitted_sum_roundoff=1e-6,
                        interpretation="Float64 renormalization of export roundoff only; original inputs retained")


def validate_inputs(composition, preferences):
    required = set(IDENTITY + ["estimated_eligible", "estimated_eligible_share", "turnout", "estimated_votes", "voter_share"])
    if required - set(composition) or set(IDENTITY + ["category", "value"]) - set(preferences):
        raise ValueError("Historical trend input schema is incomplete")
    if composition.empty or preferences.empty:
        raise ValueError("Historical trend inputs must be nonempty")
    if composition.duplicated(IDENTITY).any() or preferences.duplicated(IDENTITY + ["category"]).any():
        raise ValueError("Duplicate election/group identity")
    if set(preferences.category) != set(CATEGORIES):
        raise ValueError("Expected all three stable political categories")
    if composition[IDENTITY].isna().any().any() or preferences[IDENTITY + ["category"]].isna().any().any():
        raise ValueError("Missing election or group identity")
    if set(composition.election) - set(ELECTION_DATES):
        raise ValueError("Unknown election date; supply a reviewed date mapping")
    numeric = composition[list(required - set(IDENTITY))].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric).all().all() or (numeric < 0).any().any() or (numeric.turnout > 1).any():
        raise ValueError("Invalid population, turnout or voter-share values")
    if not np.allclose(composition.estimated_votes, composition.estimated_eligible * composition.turnout, atol=1e-6, rtol=1e-9):
        raise ValueError("Estimated voters differ from eligible population times turnout")
    target_keys = pd.MultiIndex.from_frame(composition[IDENTITY]).sort_values()
    for category in CATEGORIES:
        keys = pd.MultiIndex.from_frame(preferences.loc[preferences.category.eq(category), IDENTITY]).sort_values()
        if not target_keys.equals(keys):
            raise ValueError("Population and political-preference groups differ")
    preferences, _ = normalized_preferences(preferences)
    for variant, frame in composition.groupby("variant"):
        expected_groups = set(frame.group)
        for election, rows in frame.groupby("election"):
            if set(rows.group) != expected_groups:
                raise ValueError("Historical comparisons require the same group definitions")
            eligible = float(rows.estimated_eligible.sum())
            votes = float(rows.estimated_votes.sum())
            if eligible <= 0 or votes <= 0:
                raise ValueError("Election requires positive eligible population and modeled voters")
            if not np.allclose(rows.estimated_eligible_share, rows.estimated_eligible / eligible, atol=1e-8, rtol=0):
                raise ValueError("Eligible population shares do not reconcile")
            if not np.allclose(rows.voter_share, rows.estimated_votes / votes, atol=1e-8, rtol=0):
                raise ValueError("Voter shares do not reconcile")
            support = preferences.loc[preferences.variant.eq(variant) & preferences.election.eq(election)]
            matrix = support.pivot(index="group", columns="category", values="value").reindex(index=rows.group, columns=CATEGORIES)
            group_bloc_shares(rows.estimated_eligible_share, rows.turnout, matrix.to_numpy(float))


def comparisons(elections):
    ordered = sorted(elections, key=lambda value: ELECTION_DATES[value])
    result = [(a, b, "adjacent_elections") for a, b in zip(ordered[:-1], ordered[1:])]
    for a, b, name in [("K16", "K25", "2003_to_2022"), ("K16", "K26", "2003_to_2026")]:
        if a in ordered and b in ordered:
            result.append((a, b, name))
    return result


def build(composition, preferences):
    validate_inputs(composition, preferences)
    preferences, _ = normalized_preferences(preferences)
    voter_rows, bloc_rows, group_bloc_rows, level_rows = [], [], [], []
    for variant, frame in composition.groupby("variant", sort=True):
        groups = sorted(frame.group.unique())
        states = {}
        for election, rows in frame.groupby("election"):
            rows = rows.set_index("group").reindex(groups)
            choices = preferences.loc[preferences.variant.eq(variant) & preferences.election.eq(election)]
            choices = choices.pivot(index="group", columns="category", values="value").reindex(index=groups, columns=CATEGORIES)
            states[election] = (rows.estimated_eligible_share.to_numpy(float), rows.turnout.to_numpy(float), choices.to_numpy(float))
            shares = group_bloc_shares(*states[election]).sum(axis=0)
            for category, share in zip(CATEGORIES, shares):
                level_rows.append(dict(variant=variant, election=election, reference_date=ELECTION_DATES[election],
                                       category=category, modeled_voter_share=float(share)))
        for first, last, label in comparisons(states):
            meta = dict(variant=variant, from_election=first, to_election=last, comparison=label)
            before, after, components = shapley_changes(voter_shares, states[first][:2], states[last][:2])
            for gi, group in enumerate(groups):
                voter_rows.append(dict(meta, group=group, from_voter_share=float(before[gi]), to_voter_share=float(after[gi]),
                                       change_pp=float(100*(after[gi]-before[gi])), eligible_mix_component_pp=float(100*components[0, gi]),
                                       turnout_component_pp=float(100*components[1, gi])))
            before, after, components = shapley_changes(group_bloc_shares, states[first], states[last])
            for bi, category in enumerate(CATEGORIES):
                bloc_rows.append(dict(meta, category=category, from_bloc_share=float(before[:, bi].sum()), to_bloc_share=float(after[:, bi].sum()),
                                      change_pp=float(100*(after[:, bi].sum()-before[:, bi].sum())),
                                      **{name+"_component_pp": float(100*components[fi, :, bi].sum()) for fi, name in enumerate(FACTOR_NAMES)}))
                for gi, group in enumerate(groups):
                    group_bloc_rows.append(dict(meta, group=group, category=category,
                                                from_group_bloc_share=float(before[gi, bi]), to_group_bloc_share=float(after[gi, bi]),
                                                change_pp=float(100*(after[gi, bi]-before[gi, bi])),
                                                **{name+"_component_pp": float(100*components[fi, gi, bi]) for fi, name in enumerate(FACTOR_NAMES)}))
    return (pd.DataFrame(voter_rows, columns=VOTER_COLUMNS), pd.DataFrame(bloc_rows, columns=BLOC_COLUMNS),
            pd.DataFrame(group_bloc_rows, columns=GROUP_BLOC_COLUMNS), pd.DataFrame(level_rows))


def political_voter_base(composition, preferences):
    """Demographic shares among each category's modeled voters, not supporters in all adults."""
    validate_inputs(composition, preferences)
    preferences, _ = normalized_preferences(preferences)
    result = preferences[IDENTITY + ["category", "value"]].merge(
        composition[IDENTITY + ["estimated_votes"]], on=IDENTITY, validate="many_to_one")
    result["estimated_group_category_votes"] = result.estimated_votes * result.value
    keys = ["variant", "election", "category"]
    result["estimated_category_votes"] = result.groupby(keys).estimated_group_category_votes.transform("sum")
    if result.estimated_category_votes.le(0).any():
        raise ValueError("Political voter-base shares require positive category vote totals")
    result["voter_base_share"] = result.estimated_group_category_votes / result.estimated_category_votes
    total_votes = composition.groupby(["variant", "election"]).estimated_votes.sum()
    result["share_of_all_votes"] = result.estimated_group_category_votes / result.set_index(["variant", "election"]).index.map(total_votes)
    return result[IDENTITY + ["category", "voter_base_share", "share_of_all_votes",
                              "estimated_group_category_votes", "estimated_category_votes"]]


def _persistence(composition, variant, group):
    rows = composition.loc[composition.variant.eq(variant) & composition.group.eq(group)].copy()
    rows["date"] = rows.election.map(ELECTION_DATES)
    rows = rows.sort_values("date")
    values = rows.set_index("election").voter_share
    pre = rows.loc[rows.date.lt("2026-01-01")]
    differences = pre.voter_share.diff().dropna()
    net = float(values["K26"] - values["K16"]) if {"K16", "K26"} <= set(values.index) else None
    prior = float(values["K25"] - values["K16"]) if {"K16", "K25"} <= set(values.index) else None
    latest = float(values["K26"] - values["K25"]) if {"K25", "K26"} <= set(values.index) else None
    return dict(pre2026_change_pp=None if prior is None else 100*prior,
                latest_2022_2026_change_pp=None if latest is None else 100*latest,
                pre2026_change_same_direction=None if prior is None or net is None else bool(prior*net > 0),
                pre2026_absolute_change_over_total=None if prior is None or net is None or abs(net) < 1e-12 else abs(prior/net),
                positive_adjacent_changes_before2026=int((differences > 1e-12).sum()),
                negative_adjacent_changes_before2026=int((differences < -1e-12).sum()),
                adjacent_comparisons_before2026=len(differences))


def summary(composition, voter_changes, bloc_changes):
    candidates = []
    for (variant, comparison), rows in voter_changes.loc[voter_changes.comparison.isin(["2003_to_2022", "2003_to_2026"])].groupby(["variant", "comparison"]):
        ranked = rows.assign(magnitude=rows.change_pp.abs()).sort_values("magnitude", ascending=False).head(3)
        for row in ranked.drop(columns="magnitude").to_dict("records"):
            row["persistence"] = _persistence(composition, variant, row["group"])
            candidates.append(row)
    return dict(method="Exact two-factor and three-factor Shapley accounting, averaging all factor orders",
                interpretation="Modeled share changes, not individual voter flows, switching probabilities or causal demographic effects",
                political_categories=dict(right_religious_families=sorted(RIGHT_FAMILIES),
                                          center_left_and_other=sorted(CENTER_LEFT_OTHER_FAMILIES), arab_parties=["arab"]),
                political_category_limit="Stable analytical families, not current coalition membership; center/left and other includes the residual Other family",
                within_group_support_limit="An aggregate modeled support change can include changes in composition within the broad demographic group; it does not identify individual switching",
                matched_geography_requirement="Producer must use identical geographical support across the compared elections",
                long_run_largest_group_share_changes=candidates,
                long_run_bloc_changes=bloc_changes.loc[bloc_changes.comparison.ne("adjacent_elections")].to_dict("records"),
                direct_individual_voter_flows_identified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"electoral_composition": args.composition, "stable_political_preferences": args.preferences}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    composition = pd.read_csv(args.composition)
    preferences = pd.read_csv(args.preferences)
    voter, blocs, group_blocs, levels = build(composition, preferences)
    voter_base = political_voter_base(composition, preferences)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("voter_share_change_decomposition", voter), ("political_share_change_decomposition", blocs),
                        ("political_group_contributions", group_blocs), ("political_share_levels", levels),
                        ("political_voter_base", voter_base)]:
        frame.to_csv(args.output/(name+".csv"), index=False)
    report = summary(composition, voter, blocs)
    _, report["probability_roundoff_correction"] = normalized_preferences(preferences)
    for name, path in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Trend input changed during analysis: " + name)
    report["input_sha256"] = hashes
    report["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/"historical_trends_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(variants=int(composition.variant.nunique()), elections=int(composition.election.nunique()),
                         voter_change_rows=len(voter), political_change_rows=len(blocs), group_contribution_rows=len(group_blocs))))


if __name__ == "__main__":
    main()
