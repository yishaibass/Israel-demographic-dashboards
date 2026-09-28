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


DATA_PATTERN = re.compile(r"const DATA = (\{.*?\});\s*\n", flags=re.DOTALL)
TEMPLATE = Path(__file__).resolve().parent / "dashboard" / "template_he.html"
STALE_DATA_MARKERS = (
    "389.7260685015164",
    "156.2805314984836",
    "203201.3425071551",
    "125173.0675211704",
    "121738.8327922107",
    "188283.7379440805",
    "131262.6905107279",
    "144803.9091573414",
)


def dashboard_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = DATA_PATTERN.search(text)
    if not match:
        raise ValueError(f"Dashboard data block not found: {path}")
    return json.loads(match.group(1))


def current_data_json(text: str) -> str:
    match = DATA_PATTERN.search(text)
    if not match:
        raise ValueError("Dashboard data block not found")
    data = json.loads(match.group(1))
    capital = data.get("provenance", {}).get("capital_incidence", {})
    capital["config_path"] = "model_config.json"
    capital["equity_artifact"] = "local household-equity artifact"
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def render_hebrew(template: str, source: str) -> str:
    """Keep the prior Hebrew shell verbatim and replace only its fiscal data."""
    match = DATA_PATTERN.search(template)
    if not match:
        raise ValueError("Fiscal Hebrew template data block not found")
    return template[: match.start(1)] + current_data_json(source) + template[match.end(1) :]


def verify_template() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    data = dashboard_data(TEMPLATE)
    if data:
        raise ValueError("Fiscal Hebrew template must not embed a model-data payload")
    if any(marker in template for marker in STALE_DATA_MARKERS):
        raise ValueError("Fiscal Hebrew template contains obsolete model data")


def verify(path: Path) -> None:
    verify_template()
    data = dashboard_data(path)
    state = data["state"]["2023"]
    expected = {"tax_bn": 546.0066, "transfer_bn": 394.7323945015164, "net_bn": 151.2742054984836}
    for key, value in expected.items():
        if abs(float(state[key]) - value) > 1e-9:
            raise ValueError(f"Fiscal control changed: {key}={state[key]} expected {value}")
    verify_route(path, "fiscal-position/index.html")
    html = path.read_text(encoding="utf-8")
    if '<html lang="he" dir="rtl">' not in html:
        raise ValueError("Fiscal public page is not the Hebrew RTL shell")
    if "<h1>החשבון הפיסקלי של שלושת המגזרים בישראל</h1>" not in html:
        raise ValueError("Fiscal public Hebrew shell changed")
    expected_title = "<title>ישראל — החשבון הפיסקלי של שלושת המגזרים, 2018 ← 2023</title>"
    if expected_title not in html:
        raise ValueError("Fiscal public Hebrew title changed")
    required_method_text = (
        "הקצבאות מחולקות לפי ענף קצבה",
        "לפי תמהיל דצמבר 2023",
        "כ-₪5.0 מיליארד לשנה כתשלום כספי",
        "כ-₪10.9 מיליארד לשנה כשירות בעין",
        "מעטפת ההוצאה היא אומדן",
        "תקציבי התוכניות לשנת 2023 ופרופילי מקבלי השירות",
        "אין התאמה לרישום מנהלי ברמת משק הבית",
        "מקדמי הקפיטציה הרשמיים לפי גיל ומין",
        "תוספת פריפריה אינה מיושמת",
        "שתי אפשרויות זמינות: זרם ההון לפי מתודולוגיית קרלינסקי",
        "נבחר מפתח השווי הנקי של משק הבית",
    )
    stale_method_text = (
        "פרופיל העשירוני של STIK מ-2018",
        "אותם משקלות קפיטציה של קופות החולים מ-2018",
        "הון כולל שכר דירה זקוף",
        "קצבאות מזומן נלקחות ישירות מהסקר",
    )
    if any(marker not in html for marker in required_method_text):
        raise ValueError("Fiscal Hebrew methodology disclosure is incomplete")
    if any(marker in html for marker in stale_method_text):
        raise ValueError("Fiscal Hebrew page contains a superseded methodology disclosure")
    if any(marker in html for marker in STALE_DATA_MARKERS):
        raise ValueError("Fiscal Hebrew page contains obsolete model data")
    capital = data.get("provenance", {}).get("capital_incidence", {})
    if capital.get("mode") != "household_equity":
        raise ValueError("Fiscal capital-incidence selection changed")
    expected_sectors = {
        "Jewish non-Haredi": (203883.4722029544, 126964.2616026191, 76919.21060033524),
        "Haredi": (114381.966590899, 188873.066673769, -74491.10008286995),
        "Arab": (131404.6440769136, 146630.1779976143, -15225.53392070062),
    }
    required_transfers = {"NII cash transfers", "Public health", "Welfare in-kind"}
    for sector_name, expected_values in expected_sectors.items():
        sector = data["sectors"][sector_name]
        actual_values = (
            sector["tax_hh"]["2023"],
            sector["transfer_hh"]["2023"],
            sector["net_hh"]["2023"],
        )
        if any(abs(float(actual) - expected) > 1e-9 for actual, expected in zip(actual_values, expected_values)):
            raise ValueError(f"Fiscal sector controls changed: {sector_name}={actual_values}")
        transfer_names = {item["name"] for item in sector["transfer_items"]}
        if not required_transfers.issubset(transfer_names):
            raise ValueError(f"Fiscal transfer allocation detail is incomplete: {sector_name}")


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
        source_text = source.read_text(encoding="utf-8")
        sync_html(TEMPLATE, destination, route=route, transform=lambda template: render_hebrew(template, source_text))
    verify(destination)
    print("fiscal_position: controls and publication checks passed")


if __name__ == "__main__":
    main()
