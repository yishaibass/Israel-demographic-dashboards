"""Population-source interpretation and temporal controls, using synthetic cells."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from products.voting_propensity.model import national_population_controls as controls


def workbook(path, year=2022, title="POPULATION, BY POPULATION GROUP, RELIGION, SEX AND AGE"):
    with pd.ExcelWriter(path) as writer:
        for sheet in range(4):
            rows = [[title] + [None] * 10, ["Thousands", None, None, None, f"Average {year}"] + [None] * 6]
            if sheet in (0, 2):
                rows.append(["TOTAL", 86, 86, 172, 172, 172, 344, 86, 86, 172, "סך הכל"])
            ages = list(range(45)) if sheet in (0, 2) else list(range(45, 85)) + ["85-89", "90+"]
            for age in ages:
                # Open85+ total is one person; split between two disjoint tails.
                value = .5 if isinstance(age, str) else 1
                right_age = "89-85" if age == "85-89" else age
                rows.append([age, value, value, 2 * value, 2 * value, 2 * value,
                             4 * value, value, value, 2 * value, right_age])
            rows.append([1] + [None] * 10)  # Print footer, not an age cell.
            pd.DataFrame(rows).to_excel(writer, sheet_name=f"sheet{sheet}", index=False, header=False)


def test_parse_single_ages_and_open_tail_without_grouped_or_footer_duplication(tmp_path):
    path = tmp_path / "source.xlsx"
    workbook(path)
    result = controls.parse_workbook(path, 2022)
    assert len(result) == 86 * 2 * 3
    assert not result.duplicated(["year"] + controls.KEYS).any()
    assert set(result.population) == {1000}
    assert result.population.sum() == 516000
    assert set(result.status) == {"observed"}


def test_wrong_publication_table_and_reference_year_are_rejected(tmp_path):
    path = tmp_path / "source.xlsx"
    workbook(path, year=2021)
    with pytest.raises(ValueError, match="reference year"):
        controls.parse_workbook(path, 2022)
    workbook(path, title="INTERNAL MIGRATION BETWEEN LOCALITIES")
    with pytest.raises(ValueError, match="different CBS table"):
        controls.parse_workbook(path, 2022)


def test_source_content_pin_rejects_modified_workbook(tmp_path):
    path = tmp_path / "source.xlsx"
    workbook(path)
    source = dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    controls.verify_source(path, source)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source content changed"):
        controls.verify_source(path, source)


def observed_cells():
    return pd.DataFrame([dict(year=year, age=18, sex="female", broad_group=group,
                              population=amount * growth, status="observed",
                              reference_date=f"{year}-07-01", reference_basis="annual_average", source_id="synthetic")
                         for year, growth in [(2021, 1), (2023, 4), (2025, 16)]
                         for group, amount in [("jewish", 100), ("arab", 50), ("other", 0)]])


def test_interpolation_projection_zero_cells_and_horizon_are_explicit():
    result = controls.annualize(observed_cells(), start=2021, end=2026)
    assert result.query("year == 2022 and broad_group == 'jewish'").population.item() == pytest.approx(200)
    assert result.query("year == 2026 and broad_group == 'jewish'").population.item() == pytest.approx(3200)
    assert (result.query("broad_group == 'other'").population == 0).all()
    assert set(result.query("year == 2026").status) == {"projected"}
    assert (result.population >= 0).all() and np.isfinite(result.population).all()
    with pytest.raises(ValueError, match="horizon"):
        controls.annualize(observed_cells(), start=2020, end=2026)
    with pytest.raises(ValueError, match="Duplicate"):
        controls.annualize(pd.concat([observed_cells(), observed_cells()]), start=2021, end=2026)


def test_reversed_hebrew_age_ranges_are_recognized_only_for_open_tail():
    assert controls._age("89-85") == controls._age("85-89") == (85, 89)
    assert controls._age("80-84") is None
    assert controls._age("84-80") is None
    assert controls._age("TOTAL") is None
