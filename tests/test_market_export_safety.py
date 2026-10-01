from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "products" / "balance_sheet" / "model" / "build_household_pnl.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("build_household_pnl_safety", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MarketExportSafetyTest(unittest.TestCase):
    def test_missing_export_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "provide --market-export"):
            MODULE.resolve_market_export_path(None, {})

    def test_export_inside_repository_is_rejected(self):
        target = ROOT / "products" / "balance_sheet" / "processed" / "households.pkl"
        with self.assertRaisesRegex(ValueError, "outside the Git worktree"):
            MODULE.resolve_market_export_path(target, {})

    def test_external_export_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "hes_market_pnl_households_2023.pkl"
            self.assertEqual(MODULE.resolve_market_export_path(target, {}), target.resolve())

    def test_external_environment_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "hes_market_pnl_households_2023.pkl"
            environment = {"IDD_HES_MARKET_EXPORT": str(target)}
            self.assertEqual(MODULE.resolve_market_export_path(None, environment), target.resolve())

    def test_missing_longitudinal_panel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "provide --longitudinal-panel"):
            MODULE.resolve_longitudinal_panel_path(None, {})

    def test_supplied_longitudinal_panel_is_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory) / "cbs_longitudinal_panel_full.csv"
            panel.write_text("wave,weight_hh,year\n10,1,2023\n", encoding="utf-8")
            resolved = MODULE.resolve_longitudinal_panel_path(panel, {})
            with patch.object(MODULE, "build_household_frame", return_value=("frame", {"ok": True})) as builder:
                self.assertEqual(MODULE.load_longitudinal_household_frame(resolved), ("frame", {"ok": True}))
            builder.assert_called_once_with(panel.resolve())

    def test_fresh_checkout_cli_exposes_all_external_paths(self):
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--hes-root", result.stdout)
        self.assertIn("--longitudinal-panel", result.stdout)
        self.assertIn("--market-export", result.stdout)


if __name__ == "__main__":
    unittest.main()
