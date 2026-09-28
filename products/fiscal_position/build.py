from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.runtime.paths import load_paths, require_path
from shared.runtime.publication import sync_html, verify_route


def dashboard_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (\{.*?\});\s*\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Dashboard data block not found: {path}")
    return json.loads(match.group(1))


def public_html(text: str) -> str:
    match = re.search(r"const DATA = (\{.*?\});\s*\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Dashboard data block not found")
    data = json.loads(match.group(1))
    capital = data.get("provenance", {}).get("capital_incidence", {})
    capital["config_path"] = "model_config.json"
    capital["equity_artifact"] = "local household-equity artifact"
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    text = text[: match.start(1)] + encoded + text[match.end(1) :]
    title = "<title>ישראל — החשבון הפיסקלי של שלושת המגזרים, 2018 ← 2023</title>"
    return re.sub(r"<title>.*?</title>", title, text, count=1, flags=re.DOTALL)


def verify(path: Path) -> None:
    data = dashboard_data(path)
    state = data["state"]["2023"]
    expected = {"tax_bn": 546.0066, "transfer_bn": 394.7323945015164, "net_bn": 151.2742054984836}
    for key, value in expected.items():
        if abs(float(state[key]) - value) > 1e-9:
            raise ValueError(f"Fiscal control changed: {key}={state[key]} expected {value}")
    verify_route(path, "fiscal-position/index.html")
    expected_title = "<title>ישראל — החשבון הפיסקלי של שלושת המגזרים, 2018 ← 2023</title>"
    if expected_title not in path.read_text(encoding="utf-8"):
        raise ValueError("Fiscal public Hebrew title changed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the household fiscal-position dashboard")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--site-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    route = "fiscal-position/index.html"
    destination = args.site_root / route
    if not args.verify:
        source_root = require_path(load_paths(args.config), "fiscal_position_project")
        source = source_root / "05_net_position" / "fiscal_dashboard_2018_2023.html"
        sync_html(source, destination, route=route, transform=public_html)
    verify(destination)
    print("fiscal_position: controls and publication checks passed")


if __name__ == "__main__":
    main()
