from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.runtime.publication import sync_html, verify_route


DATA_PATTERN = re.compile(r"const DATA = (\{.*?\});\s*\n", flags=re.DOTALL)
TEMPLATE = Path(__file__).resolve().parent / "dashboard" / "template_he.html"
MODEL_CONFIG = Path(__file__).resolve().parent / "model_config.json"
MODE_DATA = Path(__file__).resolve().parent / "data"
STALE_DATA_MARKERS = (
    "389.7260685015164",
    "156.2805314984836",
    "125173.0675211704",
    "188283.7379440805",
    "144803.9091573414",
)

ARTICLE_2018_MONTHLY = {
    "Jewish non-Haredi": {"he": "יהודים לא-חרדים", "tax": 14000.0, "transfer": 7900.0},
    "Haredi": {"he": "חרדים", "tax": 8800.0, "transfer": 12900.0},
    "Arab": {"he": "ערבים", "tax": 9000.0, "transfer": 10100.0},
}
CAPITAL_MODE_HE = {
    "karlinsky_flow": "זרם הכנסה מהון לפי קרלינסקי",
    "household_equity": "שווי נטו ממודל משק הבית",
}


def dashboard_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = DATA_PATTERN.search(text)
    if not match:
        raise ValueError(f"Dashboard data block not found: {path}")
    return json.loads(match.group(1))


def comparison_2018(data: dict) -> dict:
    """Return the mode-labelled, data-driven 2018 article comparison."""
    mode = data.get("run_metadata", {}).get("capital_incidence_mode")
    if mode not in CAPITAL_MODE_HE:
        raise ValueError(f"Unsupported capital-incidence mode: {mode}")
    rows = []
    for sector_name, benchmark in ARTICLE_2018_MONTHLY.items():
        sector = data["sectors"][sector_name]
        tax = float(sector["tax_hh"]["2018"]) / 12
        transfer = float(sector["transfer_hh"]["2018"]) / 12
        rows.append({
            "sector": sector_name,
            "sector_he": benchmark["he"],
            "tax": {"current": tax, "article": benchmark["tax"],
                    "delta_pct": 100 * (tax / benchmark["tax"] - 1)},
            "transfer": {"current": transfer, "article": benchmark["transfer"],
                         "delta_pct": 100 * (transfer / benchmark["transfer"] - 1)},
        })
    return {"mode": mode, "mode_label_he": CAPITAL_MODE_HE[mode], "rows": rows}


def sanitize_data(data: dict) -> dict:
    capital = data.get("provenance", {}).get("capital_incidence", {})
    mode = capital.get("mode")
    # Analysis runs can use a CLI override while producing a candidate.  Those
    # execution details are deliberately not public provenance: the tracked Git
    # selector is authoritative for publication and chooses a matching, frozen
    # aggregate snapshot.  Keep only mode-relevant public fields.
    for key in ("configured_mode", "override_used", "config_path", "equity_artifact"):
        capital.pop(key, None)
    capital["publication_schema"] = "git_authoritative_aggregate_snapshot_v1"
    capital["publication_selector"] = "model_config.json"
    if mode == "household_equity":
        capital["equity_artifact"] = "local household-equity artifact"
    elif mode == "karlinsky_flow":
        capital["capital_key_field"] = "i12cap"
        for year_meta in data.get("provenance", {}).get("capital_incidence_years", {}).values():
            year_meta.pop("artifact_vintage", None)
            year_meta.pop("equity_artifact", None)
    else:
        raise ValueError(f"Unsupported capital-incidence mode: {mode}")
    data["comparison_2018"] = comparison_2018(data)
    return data


def publication_data(mode: str) -> dict:
    if mode not in CAPITAL_MODE_HE:
        raise ValueError(f"Unsupported capital-incidence mode: {mode}")
    path = MODE_DATA / f"{mode}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return sanitize_data(data)


def render_hebrew(template: str, data: dict) -> str:
    """Keep the prior Hebrew shell verbatim and replace only its fiscal data."""
    match = DATA_PATTERN.search(template)
    if not match:
        raise ValueError("Fiscal Hebrew template data block not found")
    payload = json.dumps(sanitize_data(data), ensure_ascii=False, separators=(",", ":"))
    return template[: match.start(1)] + payload + template[match.end(1) :]


def verify_template() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    data = dashboard_data(TEMPLATE)
    if data:
        raise ValueError("Fiscal Hebrew template must not embed a model-data payload")
    if any(marker in template for marker in STALE_DATA_MARKERS):
        raise ValueError("Fiscal Hebrew template contains obsolete model data")


def verify_mode_contract(data: dict, selected_mode: str) -> None:
    capital = data.get("provenance", {}).get("capital_incidence", {})
    if capital.get("mode") != selected_mode:
        raise ValueError("Fiscal capital-incidence selection changed")
    years = data.get("provenance", {}).get("capital_incidence_years", {})
    if set(years) != {"2018", "2023"} or {v.get("mode") for v in years.values()} != {selected_mode}:
        raise ValueError("Fiscal output metadata does not apply one capital method to both years")
    forbidden = {"configured_mode", "override_used", "config_path"}
    if forbidden.intersection(capital):
        raise ValueError("Fiscal public provenance contains shadow analysis configuration")
    if capital.get("publication_schema") != "git_authoritative_aggregate_snapshot_v1":
        raise ValueError("Fiscal public provenance schema is not Git-authoritative")
    vintages = [meta.get("artifact_vintage") for meta in years.values()]
    if selected_mode == "karlinsky_flow":
        if "equity_artifact" in capital or any(v is not None for v in vintages):
            raise ValueError("Flow-mode provenance incorrectly names an equity artifact")
        if capital.get("capital_key_field") != "i12cap":
            raise ValueError("Flow-mode provenance does not identify i12cap")
    elif "equity_artifact" not in capital or vintages != [2018, 2023]:
        raise ValueError("Equity-mode provenance lacks its year-specific artifact vintages")
    comparison = data.get("comparison_2018")
    expected_comparison = comparison_2018(data)
    if comparison != expected_comparison or comparison.get("mode") != selected_mode:
        raise ValueError("Fiscal 2018 comparison is stale or mismatched to the selected capital mode")


def verify(path: Path, selected_mode: str | None = None) -> None:
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
        "שתי אפשרויות זמינות לכל התקופה: זרם ההון לפי מתודולוגיית קרלינסקי",
        "אותו מתג שיטה חל על שתי השנים",
        "טבלת ההשוואה לשנת 2018 מתעדכנת ישירות מנתוני הדשבורד",
    )
    stale_method_text = (
        "פרופיל העשירוני של STIK מ-2018",
        "אותם משקלות קפיטציה של קופות החולים מ-2018",
        "הון כולל שכר דירה זקוף",
        "קצבאות מזומן נלקחות ישירות מהסקר",
        "שכפול 2018 נשאר לפי זרם ההון",
        "−3.8%",
        "₪8.5K",
    )
    if any(marker not in html for marker in required_method_text):
        raise ValueError("Fiscal Hebrew methodology disclosure is incomplete")
    if any(marker in html for marker in stale_method_text):
        raise ValueError("Fiscal Hebrew page contains a superseded methodology disclosure")
    if any(marker in html for marker in STALE_DATA_MARKERS):
        raise ValueError("Fiscal Hebrew page contains obsolete model data")
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))["capital_incidence"]
    selected_mode = selected_mode or config.get("selected")
    if selected_mode not in config.get("options", {}):
        raise ValueError("Fiscal capital-incidence selection is unsupported")
    if config.get("applies_to_years") != [2018, 2023] or config.get("mixed_year_modes_allowed") is not False:
        raise ValueError("Fiscal capital-incidence config permits a mixed-year methodology")
    verify_mode_contract(data, selected_mode)
    expected_sectors = {
      "household_equity": {
        "Jewish non-Haredi": (168975.5339173671, 203883.4722029544, 126964.2616026191, 76919.21060033524),
        "Haredi": (98494.10535834, 114381.966590899, 188873.066673769, -74491.10008286995),
        "Arab": (107163.3556972415, 131404.6440769136, 146630.1779976143, -15225.53392070062),
      },
      "karlinsky_flow": {
        "Jewish non-Haredi": (168219.2844964973, 203201.3425071551, 126964.2616026191, 76237.080904536),
        "Haredi": (104254.7191393617, 121738.8327922107, 188873.066673769, -67134.23388155834),
        "Arab": (108713.4265860596, 131262.6905107279, 146630.1779976143, -15367.48748688639),
      },
    }
    required_transfers = {"NII cash transfers", "Public health", "Welfare in-kind"}
    for sector_name, expected_values in expected_sectors[selected_mode].items():
        sector = data["sectors"][sector_name]
        actual_values = (
            sector["tax_hh"]["2018"],
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
    parser.add_argument("--capital-mode", choices=sorted(CAPITAL_MODE_HE))
    parser.add_argument("--capture-source", type=Path,
                        help="Capture one aggregate dashboard snapshot from an analysis HTML")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))["capital_incidence"]
    mode = args.capital_mode or config["selected"]
    if args.capture_source:
        data = sanitize_data(dashboard_data(args.capture_source))
        verify_mode_contract(data, mode)
        MODE_DATA.mkdir(parents=True, exist_ok=True)
        (MODE_DATA / f"{mode}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    route = "fiscal-position/index.html"
    destination = args.site_root / route
    if not args.verify:
        data = publication_data(mode)
        sync_html(TEMPLATE, destination, route=route, transform=lambda template: render_hebrew(template, data))
    verify(destination, selected_mode=mode)
    print("fiscal_position: controls and publication checks passed")


if __name__ == "__main__":
    main()
