"""Deterministic, non-identifying keys for canonical microdata tables."""

from __future__ import annotations

from hashlib import sha256


SPINE_IDS = frozenset({"hes", "longitudinal", "census"})


def _opaque_key(namespace: str, *parts: object) -> str:
    values = [namespace, *(str(part).strip() for part in parts)]
    if any(not value for value in values):
        raise ValueError("key parts must be non-empty")
    digest = sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"{namespace}_{digest}"


def household_key(spine_id: str, reference_year: int, source_household_id: object) -> str:
    """Return a stable key within one spine and year, not a cross-survey linkage key."""
    if spine_id not in SPINE_IDS:
        raise ValueError(f"unsupported spine_id: {spine_id}")
    return _opaque_key("hh", spine_id, int(reference_year), source_household_id)


def person_key(
    spine_id: str,
    reference_year: int,
    source_household_id: object,
    source_person_id: object,
) -> str:
    """Return a stable person key within one spine and year."""
    if spine_id not in SPINE_IDS:
        raise ValueError(f"unsupported spine_id: {spine_id}")
    return _opaque_key(
        "person", spine_id, int(reference_year), source_household_id, source_person_id
    )


def geography_key(
    geography_vintage: int,
    locality_code: object,
    statistical_area_code: object | None = None,
) -> str:
    """Return a vintage-aware locality or statistical-area key."""
    level = "statistical_area" if statistical_area_code not in (None, "") else "locality"
    area = statistical_area_code if level == "statistical_area" else "all"
    return _opaque_key("geo", int(geography_vintage), locality_code, level, area)
