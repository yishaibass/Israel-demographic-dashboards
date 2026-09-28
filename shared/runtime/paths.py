from __future__ import annotations

import os
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths(config: str | Path | None = None) -> dict[str, Path]:
    """Load machine-specific paths without putting them in version control."""
    candidate = Path(config) if config else Path(
        os.environ.get("ISRAEL_DASHBOARDS_PATHS", REPO_ROOT / "config" / "paths.toml")
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Local path configuration not found: {candidate}. "
            "Copy config/paths.example.toml to config/paths.toml and fill it in."
        )
    raw = tomllib.loads(candidate.read_text(encoding="utf-8"))
    values = raw.get("paths", raw)
    return {key: Path(value).expanduser() for key, value in values.items() if value}


def require_path(paths: dict[str, Path], key: str) -> Path:
    value = paths.get(key)
    if value is None:
        raise KeyError(f"Missing [paths].{key} in the local path configuration")
    if not value.exists():
        raise FileNotFoundError(f"Configured path does not exist for {key}: {value}")
    return value
