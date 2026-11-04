"""The reviewed K26 bloc contract, independent of ideological party families."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT_PATH = Path(__file__).with_name("k26_party_blocs.json")


def load_contract() -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("election") != "K26":
        raise ValueError("Unsupported K26 party-bloc contract")
    parties = contract.get("coalition_parties", [])
    symbols = [party["symbol"] for party in parties]
    names = [party["party_id"] for party in parties]
    if not parties or len(set(symbols)) != len(symbols) or len(set(names)) != len(names):
        raise ValueError("K26 bloc contract requires unique party symbols and IDs")
    if not contract.get("definition_id"):
        raise ValueError("K26 bloc definition ID is missing")
    return contract


def coalition_symbols() -> frozenset[str]:
    return frozenset(party["symbol"] for party in load_contract()["coalition_parties"])


def canonical_names() -> dict[str, str]:
    return {party["symbol"]: party["party_id"] for party in load_contract()["coalition_parties"]}


def contract_sha256() -> str:
    payload = json.dumps(load_contract(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_dimension(frame) -> None:
    """Reject stale or partial party dimensions before fitting or publishing."""
    if not {"ballot_symbol", "bloc_core"} <= set(frame.columns):
        raise ValueError("K26 dimension lacks ballot symbols or bloc membership")
    symbols = frame.ballot_symbol.astype(str)
    if symbols.duplicated().any():
        raise ValueError("K26 party dimension contains duplicate symbols")
    expected = coalition_symbols()
    missing = expected - set(symbols)
    if missing:
        raise ValueError("K26 dimension is missing coalition parties: " + ", ".join(sorted(missing)))
    membership = frame.bloc_core.eq("Netanyahu bloc")
    stale = symbols.loc[membership.ne(symbols.isin(expected))].tolist()
    if stale:
        raise ValueError("K26 dimension differs from the bloc contract: " + ", ".join(stale))
    if "party_id" in frame:
        names = canonical_names()
        incorrect = [row.ballot_symbol for row in frame.itertuples()
                     if row.ballot_symbol in names and row.party_id != names[row.ballot_symbol]]
        if incorrect:
            raise ValueError("K26 dimension has stale coalition party IDs: " + ", ".join(incorrect))
    if "amcha_included_in_coalition" in frame:
        legacy = frame.amcha_included_in_coalition.astype(str).str.lower().isin(["true", "1"])
        if (legacy & symbols.ne("ך")).any():
            raise ValueError("The legacy Amcha flag cannot override another party's bloc")
