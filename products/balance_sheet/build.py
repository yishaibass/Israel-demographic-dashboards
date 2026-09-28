from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.runtime.paths import load_paths, require_path
from shared.runtime.publication import sync_html, verify_route


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the household balance-sheet dashboard")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--site-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    outputs = {"index_vHe.html": "balance-sheet/index.html", "index.html": "balance-sheet/index-en.html"}
    if args.verify:
        for route in outputs.values():
            verify_route(args.site_root / route, route)
        print("balance_sheet: publication checks passed")
        return
    source = require_path(load_paths(args.config), "balance_sheet_project") / "output"
    for name, route in outputs.items():
        sync_html(source / name, args.site_root / route, route=route)
    print("balance_sheet: published Hebrew and English routes")


if __name__ == "__main__":
    main()
