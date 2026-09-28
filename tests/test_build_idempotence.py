from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "config" / "paths.toml"


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@unittest.skipUnless(LOCAL_CONFIG.is_file(), "full build requires ignored config/paths.toml")
class BuildIdempotenceTests(unittest.TestCase):
    def test_full_build_is_byte_idempotent_and_does_not_dirty_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site"
            command = [sys.executable, str(ROOT / "tools/build.py"), "all", "--site-root", str(site)]
            subprocess.run(command, cwd=ROOT, check=True)
            first = snapshot(site)
            subprocess.run(command, cwd=ROOT, check=True)
            self.assertEqual(first, snapshot(site))
        subprocess.run(["git", "diff", "--exit-code"], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
