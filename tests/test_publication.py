from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.runtime.publication import analytics_signature, expected_analytics_signature
from products.coalition_simulator.publication import verify_publication_shell
from products.fiscal_position.build import STALE_DATA_MARKERS, TEMPLATE, dashboard_data, verify as verify_fiscal


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
        self.assertIn("בשכפול 2018", html)
        self.assertIn("לפי תמהיל דצמבר 2023", html)
        self.assertIn("מעטפת ההוצאה היא אומדן", html)
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(dashboard_data(TEMPLATE), {})
        for marker in STALE_DATA_MARKERS:
            self.assertNotIn(marker, template)
            self.assertNotIn(marker, html)
        verify_fiscal(path)

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
