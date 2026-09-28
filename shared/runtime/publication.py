from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_CONFIG = REPO_ROOT / "config" / "publication.toml"
GA_BLOCK = re.compile(
    r"<!-- Google tag \(gtag\.js\) -->\s*"
    r"<script async[^>]*googletagmanager\.com/gtag/js\?id=G-[A-Z0-9]+[^>]*></script>\s*"
    r"<script>.*?</script>",
    flags=re.DOTALL,
)


def publication_contract() -> tuple[str, frozenset[str]]:
    raw = tomllib.loads(PUBLICATION_CONFIG.read_text(encoding="utf-8"))["analytics"]
    return raw["measurement_id"], frozenset(raw["enabled_routes"])


def analytics_signature(html: str) -> tuple[tuple[str, ...], int, int]:
    ids = tuple(dict.fromkeys(re.findall(r"G-[A-Z0-9]+", html)))
    return ids, html.count("googletagmanager.com/gtag/js"), html.count("gtag('config'")


def expected_analytics_signature(route: str) -> tuple[tuple[str, ...], int, int]:
    measurement_id, enabled_routes = publication_contract()
    return ((measurement_id,), 1, 1) if route in enabled_routes else ((), 0, 0)


def set_analytics(html: str, route: str) -> str:
    """Apply the repository-owned Analytics contract without reading destination state."""
    measurement_id, enabled_routes = publication_contract()
    clean = GA_BLOCK.sub("", html)
    clean = re.sub(r"(<head>)\s*", r"\1\n", clean, count=1)
    if route not in enabled_routes:
        return clean
    block = (
        "<!-- Google tag (gtag.js) -->\n"
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>\n'
        "<script>\n"
        "  window.dataLayer = window.dataLayer || [];\n"
        "  function gtag(){dataLayer.push(arguments);}\n"
        "  gtag('js', new Date());\n\n"
        f"  gtag('config', '{measurement_id}');\n"
        "</script>"
    )
    return clean.replace("<head>\n", f"<head>\n{block}\n", 1)


def sync_html(
    source: Path,
    destination: Path,
    *,
    route: str,
    transform: Callable[[str], str] | None = None,
) -> str:
    """Build one deterministic HTML artifact from source and publication contract."""
    incoming = source.read_text(encoding="utf-8")
    if transform is not None:
        incoming = transform(incoming)
    output = set_analytics(incoming, route)
    actual = analytics_signature(output)
    expected = expected_analytics_signature(route)
    if actual != expected:
        raise ValueError(f"Analytics contract failed for {route}: {actual} != {expected}")
    payload = output.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_bytes() != payload:
        destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_route(path: Path, route: str) -> None:
    actual = analytics_signature(path.read_text(encoding="utf-8"))
    expected = expected_analytics_signature(route)
    if actual != expected:
        raise ValueError(f"Analytics contract failed for {route}: {actual} != {expected}")
