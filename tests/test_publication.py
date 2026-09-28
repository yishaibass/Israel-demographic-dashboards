from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.runtime.publication import analytics_signature, expected_analytics_signature


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
        html = (ROOT / "fiscal-position/index.html").read_text(encoding="utf-8")
        self.assertIn("<title>ישראל — החשבון הפיסקלי של שלושת המגזרים, 2018 ← 2023</title>", html)

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
