"""Reviewed locality identities for the 2019–2026 reconstruction.

This is a bounded event list, not a complete historical locality register.
Code creation is distinct from physical settlement founding. Missing registry
rows never establish nonexistence. Annual evidence retains an uncertain event
year instead of inventing a January effective date.
"""
from __future__ import annotations

from datetime import date, datetime
import numpy as np


CBS_2022 = "https://www.cbs.gov.il/he/publications/doclib/2024/local_authorities22_1957/intro.pdf"
MERGER_ORDER = "https://www.idf.il/media/dl3mrznl/קמצם-262-סופי.pdf"
CBS_2023 = "https://data.gov.il/api/3/action/datastore_search?resource_id=d47a54ff-87f0-44b3-b33a-f284c0c38e5a&limit=10000"
SPLIT_EVIDENCE = "https://library.mevaker.gov.il/sites/DigitalLibrary/Documents/2023/2023.5/2023.5-205-Diyur-Haredim.pdf"
OUTSIDE_LOCALITIES = -1  # Internal bucket; not CBS or PIBA code0, nor PUF9999.
MERGER_DATE = date(2022, 6, 7)
REVIEWED_IDENTITIES = {566: "Givat Shemesh", 1362: "Gvaot Eden", 3720: "Shaarei Tikva",
                       3778: "Etz Efraim", 3779: "Kochav Yaakov", 3815: "Tel Tsiyon",
                       3826: "Shaar Shomron", 4029: "President Trump Heights"}

# Public schema-only evidence. Population levels and donor shares are NOT stored.
EVENTS = (
    dict(event_id="shaar_shomron_merger", kind="merge", predecessors=(3720, 3778),
         successors=(3826,), earliest="2022-06-07", latest="2022-06-07", precision="day",
         physical_founding=False, source_urls=(CBS_2022, MERGER_ORDER),
         source_location="CBS PDF p38; legal bulletin262 PDF p29 section8",
         note="Old Shaarei Tikva and Etz Efraim codes replaced by Shaar Shomron."),
    dict(event_id="tel_tsiyon_split", kind="split", predecessors=(3779,),
         successors=(3779, 3815), earliest="2023-01-01", latest="2023-12-31", precision="year",
         physical_founding=False, source_urls=(CBS_2023, SPLIT_EVIDENCE),
         source_location="CBS2023 catalog3815; State Comptroller2023 housing report footnote154",
         note="Tel Tsiyon was an inhabited Kochav Yaakov neighborhood before separate code3815; exact code-effective day unverified."),
    dict(event_id="gvaot_eden_recognition", kind="new_locality", predecessors=(),
         successors=(1362,), earliest="2022-01-01", latest="2022-12-31", precision="year",
         physical_founding=True, source_urls=(CBS_2022, CBS_2023),
         source_location="CBS2022 PDF p38; CBS2023 catalog founding year2022",
         note="Official new-locality year2022; day and first resident occupancy not established."),
    dict(event_id="ramat_trump_recognition", kind="recognition", predecessors=(),
         successors=(4029,), earliest="2022-01-01", latest="2022-12-31", precision="year",
         physical_founding=False, source_urls=(CBS_2022, CBS_2023),
         source_location="CBS2022 PDF p38; CBS2022/2023 catalogs founding year2022",
         note="Official locality recognition2022; separate physical occupancy timing has not been established."),
    dict(event_id="givat_shemesh_nonlocality", kind="outside_localities", predecessors=(566,),
         successors=(OUTSIDE_LOCALITIES,), earliest="2022-01-01", latest="2022-12-31", precision="year",
         physical_founding=False, source_urls=(CBS_2022,), source_location="CBS2022 PDF p38",
         note="Institutional settlement became a point; population belongs to outside-locality residents. PIBA may still retain566."),
)

# Same-code name/form changes on the same CBS2022 table do not move population.
SAME_CODE_CHANGES = {1376: "form", 849: "name", 190: "form", 1094: "name",
                     1154: "name", 1767: "name", 1701: "name", 3655: "form"}


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def active_status(code, reference_date):
    """Return active/inactive/uncertain/unreviewed for the CBS locality CODE.

    Inactive code3815 before2023 says nothing about its neighborhood's residents:
    those belong to parent3779. Only reviewed events are authoritative here.
    """
    code, when = int(code), _date(reference_date)
    if code in (3720, 3778):
        return "active" if when < MERGER_DATE else "inactive"
    if code == 3826:
        return "inactive" if when < MERGER_DATE else "active"
    if code == 3779:
        return "active"
    event = next((e for e in EVENTS if code in e["successors"] and e["kind"] != "merge"), None)
    if code == 566:
        if when < date(2022, 1, 1):
            return "active"
        return "uncertain" if when <= date(2022, 12, 31) else "inactive"
    if event:
        if when < _date(event["earliest"]):
            return "inactive"
        return "uncertain" if when <= _date(event["latest"]) else "active"
    return "unreviewed"


def predecessor_group(target_code, reference_date):
    """Prior backbone for growth denominators; NOT donor allocation weights.

    For split3815 both current3779 and3815 share old3779 denominator. Their
    numerators must partition the contemporary family, never each reuse the
    entire parent population. Caller supplies observed same-series counts.
    """
    code, when = int(target_code), _date(reference_date)
    if code == 3826 and when >= MERGER_DATE:
        return (3720, 3778)
    if code in (3779, 3815) and when >= date(2023, 1, 1):
        return (3779,)
    return (code,)


def comparison_family(code, reference_date):
    """Current mutually exclusive codes sharing a historical population mass."""
    code, when = int(code), _date(reference_date)
    if code in (3779, 3815):
        return (3779, 3815) if when >= date(2023, 1, 1) else (3779,)
    if code in (3720, 3778, 3826):
        return (3826,) if when >= MERGER_DATE else (3720, 3778)
    return (code,)


def lifecycle_plan(codes, reference_date):
    statuses = {int(c): active_status(c, reference_date) for c in codes}
    return dict(reference_date=_date(reference_date).isoformat(), status_by_code=statuses,
                inactive_codes=[c for c, s in statuses.items() if s == "inactive"],
                uncertain_codes=[c for c, s in statuses.items() if s == "uncertain"],
                unreviewed_codes=[c for c, s in statuses.items() if s == "unreviewed"],
                scope="reviewed2019–2026 events; older history and unlisted events not certified",
                events=[dict(e) for e in EVENTS])


def allocate_split(parent_value, weights):
    """Partition ONE parent mass with explicit nonnegative contemporary weights.

    Works on scalar or joint-cell arrays. No demographic equality is inferred:
    applying common total-population shares to every cell is a caller assumption.
    """
    parent = np.asarray(parent_value, dtype=float)
    w = {int(c): float(v) for c, v in weights.items()}
    if set(w) != {3779, 3815} or not all(np.isfinite(v) and v >= 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("Split requires finite nonnegative weights for3779 and3815 with positive total")
    if not np.isfinite(parent).all() or (parent < 0).any():
        raise ValueError("Invalid parent population")
    return {c: parent * (v / sum(w.values())) for c, v in w.items()}


def reconcile_values(values_by_code, reference_date, *, overlap="raise", split_weights=None):
    """Reconcile one known source backbone, returning copies and a mass audit.

    Old+new positive merger estimates overlap: reject by default. Explicit
    predecessor/successor policy SELECTS the corresponding source estimate,
    recording discarded overlapping mass; it never sums both backbones.
    Before split recognition,3815 consolidates into3779. After recognition,
    a pooled parent may be split ONLY by explicit split_weights and no existing
    child mass. This helper cannot decide whether an unlabeled3779 source is
    pooled or already excludes3815; the source vintage must decide that.
    """
    if overlap not in ("raise", "predecessors", "successor"):
        raise ValueError("Unknown merger overlap policy")
    values = {int(c): np.asarray(v, dtype=float).copy() for c, v in values_by_code.items()}
    if any(not np.isfinite(v).all() or (v < 0).any() for v in values.values()):
        raise ValueError("Counts must be finite and nonnegative")
    shapes = {v.shape for v in values.values()}
    if len(shapes) > 1:
        raise ValueError("All locality values must have the same cell shape")
    before = sum(float(v.sum()) for v in values.values())
    audit = dict(reference_date=_date(reference_date).isoformat(), overlap_policy=overlap,
                 transferred=[], discarded_overlap=0.0, input_mass=before)
    positive = lambda c: c in values and bool((values[c] > 0).any())
    when = _date(reference_date)
    if when >= MERGER_DATE:
        old = [c for c in (3720, 3778) if c in values]
        if any(positive(c) for c in old) and positive(3826):
            if overlap == "raise":
                raise ValueError("Overlapping3720/3778 and3826 backbones require explicit source-vintage selection")
            removed = [3826] if overlap == "predecessors" else old
            for c in removed:
                audit["discarded_overlap"] += float(values.pop(c).sum())
            old = [c for c in old if c in values]
        for c in old:
            values[3826] = values.get(3826, np.zeros_like(values[c])) + values.pop(c)
            audit["transferred"].append((c, 3826))
    elif positive(3826):
        raise ValueError("Reverse merger allocation requires separately evidenced predecessor shares")
    if when < date(2023, 1, 1) and 3815 in values:
        values[3779] = values.get(3779, np.zeros_like(values[3815])) + values.pop(3815)
        audit["transferred"].append((3815, 3779))
    if split_weights is not None:
        if when <= date(2023, 12, 31):
            raise ValueError("Exact2023 split day is unresolved; explicit partition is supported after2023")
        if positive(3815) or 3779 not in values:
            raise ValueError("Split requires one pooled parent and no existing positive child")
        values.update(allocate_split(values.pop(3779), split_weights))
        audit["transferred"].append((3779, (3779, 3815)))
    if when > date(2022, 12, 31) and 566 in values:
        values[OUTSIDE_LOCALITIES] = values.get(OUTSIDE_LOCALITIES, np.zeros_like(values[566])) + values.pop(566)
        audit["transferred"].append((566, OUTSIDE_LOCALITIES))
    audit["output_mass"] = sum(float(v.sum()) for v in values.values())
    if not np.isclose(before - audit["discarded_overlap"], audit["output_mass"], rtol=1e-12, atol=1e-8):
        raise AssertionError("Lifecycle transform lost population mass")
    audit["uncertain_codes"] = lifecycle_plan(values, when)["uncertain_codes"]
    return values, audit
