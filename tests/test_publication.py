from pathlib import Path
import re
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.runtime.publication import analytics_signature, expected_analytics_signature
from products.coalition_simulator.publication import verify_publication_shell
from products.fiscal_position.build import (
    CAPITAL_MODE_HE,
    STALE_DATA_MARKERS,
    TEMPLATE,
    comparison_2018,
    dashboard_data,
    publication_data,
    render_hebrew,
    verify as verify_fiscal,
    verify_mode_contract,
)


class PublicationTests(unittest.TestCase):
    def test_analytics_contract(self) -> None:
        expected = {
            "index.html",
            "balance-sheet/index.html",
            "balance-sheet/index-en.html",
            "fiscal-position/index.html",
            "voting-simulator/index.html",
            "feedback/index.html",
            "voting-simulator/demographics.html",
        }
        for relative in expected:
            actual = analytics_signature((ROOT / relative).read_text(encoding="utf-8"))
            self.assertEqual(actual, expected_analytics_signature(relative), relative)

    def test_fiscal_public_title_is_hebrew(self) -> None:
        path = ROOT / "fiscal-position/index.html"
        html = path.read_text(encoding="utf-8")
        self.assertIn("<title>ישראל — החשבון הפיסקלי של שלושת המגזרים, 2018 ← 2023</title>", html)
        self.assertIn('<html lang="he" dir="rtl">', html)
        self.assertIn("<h1>החשבון הפיסקלי של שלושת המגזרים בישראל</h1>", html)
        self.assertIn("הפער מול יהודים לא-חרדים", html)
        self.assertIn('href="../index.html"', html)
        self.assertNotIn("פרופיל העשירוני של STIK מ-2018", html)
        self.assertNotIn("אותם משקלות קפיטציה של קופות החולים מ-2018", html)
        self.assertNotIn("הון כולל שכר דירה זקוף", html)
        self.assertIn("בדיקת השוואה מול הפרסום", html)
        self.assertIn("לפי תמהיל דצמבר 2023", html)
        self.assertIn("מעטפת ההוצאה היא אומדן", html)
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(dashboard_data(TEMPLATE), {})
        for marker in STALE_DATA_MARKERS:
            self.assertNotIn(marker, template)
            self.assertNotIn(marker, html)
        verify_fiscal(path)

    def test_fiscal_capital_modes_publish_matching_data_and_labels(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        for mode, label in CAPITAL_MODE_HE.items():
            with self.subTest(mode=mode):
                data = publication_data(mode)
                verify_mode_contract(data, mode)
                capital = data["provenance"]["capital_incidence"]
                years = data["provenance"]["capital_incidence_years"]
                self.assertEqual(capital["publication_schema"], "git_authoritative_aggregate_snapshot_v1")
                self.assertFalse({"configured_mode", "override_used", "config_path"} & set(capital))
                if mode == "karlinsky_flow":
                    self.assertEqual(capital["capital_key_field"], "i12cap")
                    self.assertNotIn("equity_artifact", capital)
                    self.assertTrue(all("artifact_vintage" not in meta for meta in years.values()))
                else:
                    self.assertIn("equity_artifact", capital)
                    self.assertEqual([years[str(y)]["artifact_vintage"] for y in (2018, 2023)], [2018, 2023])
                comparison = comparison_2018(data)
                self.assertEqual(comparison["mode_label_he"], label)
                for row in comparison["rows"]:
                    expected = data["sectors"][row["sector"]]["tax_hh"]["2018"] / 12
                    self.assertAlmostEqual(row["tax"]["current"], expected, places=9)
                html = render_hebrew(template, data)
                self.assertIn(label, html)
                self.assertNotIn("−3.8%", html)
                self.assertNotIn("₪8.5K", html)
                self.assertNotIn("בחירת מפתח השווי הנקי משנה שבעה", html)
                self.assertIn("מס חברות, מע״מ פיננסי ומלכ״רים, ארנונה עסקית וסולר מחולקים בשלישים", html)
                self.assertIn("במס רכב ובאגרות המפתח הנבחר נכנס לתמהיל של יתרת הגבייה", html)
                self.assertIn("ויתרת מס הנדל״ן מוקצית במלואה לפי מפתח ההון", html)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "fiscal-position" / "index.html"
                    path.parent.mkdir(parents=True)
                    path.write_text(html, encoding="utf-8")
                    verify_fiscal(path, selected_mode=mode)

    def test_voting_simulator_keeps_demographic_integration(self) -> None:
        simulator = (ROOT / "voting-simulator/index.html").read_text(encoding="utf-8")
        explorer = (ROOT / "voting-simulator/demographics.html").read_text(encoding="utf-8")
        home = (ROOT / "index.html").read_text(encoding="utf-8")
        verify_publication_shell(simulator)
        self.assertIn('<title>ניתוח דפוסי הצבעה</title>', explorer)
        self.assertIn('href="voting-simulator/index.html"', home)
        self.assertEqual(analytics_signature(explorer), ((), 0, 0))

    def test_no_personal_paths_in_public_text(self) -> None:
        absolute_user_path = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]", re.IGNORECASE)
        suffixes = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".html"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.name == "paths.toml":
                continue
            if path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertIsNone(absolute_user_path.search(text), str(path))


if __name__ == "__main__":
    unittest.main()
