from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.runtime.paths import load_paths, require_path
from shared.runtime.publication import sync_html, verify_route
from products.voting_propensity.k26_publication import integrate
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the voting-propensity explorer")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--site-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--k26-current", type=Path, help="Prepared K26 current directory containing individual_model outputs")
    args = parser.parse_args()
    route = "voting-simulator/demographics.html"
    destination = args.site_root / route
    if not args.verify:
        source_root = require_path(load_paths(args.config), "voting_propensity_project")
        candidates = (
            source_root / "voting-simulator" / "demographics.html",
            source_root.parent / "voting-simulator" / "demographics.html",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError("Voting-propensity dashboard not found beside the configured project")
        transform = None
        if args.k26_current:
            folder = args.k26_current / "individual_model"
            cube = json.loads((folder / "k26_probability_cube.json").read_text(encoding="utf-8"))
            audit = json.loads((folder / "model_audit.json").read_text(encoding="utf-8"))
            history = json.loads((folder / "historical_election_cubes.json").read_text(encoding="utf-8"))
            readiness = json.loads((args.k26_current / "readiness.json").read_text(encoding="utf-8"))
            if readiness["snapshot_id"] != cube["snapshot_id"]:
                raise ValueError("K26 fit is stale relative to current results")
            transform = lambda page: integrate(page, cube, audit, history)
        sync_html(source, destination, route=route, transform=transform)
    verify_route(destination, route)
    print("voting_propensity: publication checks passed")


if __name__ == "__main__":
    main()
