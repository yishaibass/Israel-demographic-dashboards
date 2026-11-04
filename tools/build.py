from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ("balance_sheet", "fiscal_position", "coalition_simulator", "voting_propensity")


def publish_static_routes(site_root: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from shared.runtime.publication import sync_html

    for route in ("index.html", "feedback/index.html"):
        sync_html(ROOT / route, site_root / route, route=route)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify dashboard publication artifacts")
    parser.add_argument("product", choices=("all", *PRODUCTS), nargs="?", default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--site-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true", help="Run checks without publishing")
    parser.add_argument("--k26-current", type=Path, help="Prepared K26 current folder for the separate voting-propensity view")
    args = parser.parse_args()

    selected = PRODUCTS if args.product == "all" else (args.product,)
    for product in selected:
        command = [sys.executable, str(ROOT / "products" / product / "build.py"), "--site-root", str(args.site_root)]
        if args.config:
            command.extend(["--config", str(args.config)])
        if args.verify:
            command.append("--verify")
        if args.k26_current and product == "voting_propensity":
            command.extend(["--k26-current", str(args.k26_current.resolve())])
        subprocess.run(command, cwd=ROOT, check=True)
    if args.product == "all" and not args.verify:
        publish_static_routes(args.site_root)


if __name__ == "__main__":
    main()
