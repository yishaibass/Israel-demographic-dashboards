from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.runtime.paths import load_paths, require_path
from shared.runtime.publication import sync_html, verify_route
from products.coalition_simulator.publication import integrate_publication_shell, verify_publication_shell


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the coalition simulator")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--site-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    route = "voting-simulator/index.html"
    destination = args.site_root / route
    if not args.verify:
        source_root = require_path(load_paths(args.config), "coalition_simulator_project")
        sync_html(
            source_root / "outputs" / "dashboard" / "index_he.html",
            destination,
            route=route,
            transform=integrate_publication_shell,
        )
    verify_route(destination, route)
    verify_publication_shell(destination.read_text(encoding="utf-8"))
    print("coalition_simulator: publication checks passed")


if __name__ == "__main__":
    main()
