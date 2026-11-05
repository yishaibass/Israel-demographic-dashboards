"""CBS resident population controls; never use votes to estimate population.

Single ages 0--84 and an open 85+ cell, by sex and Jewish/Arab/Other group.
Published annual averages are kept distinct from the Census 2022 date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd


KEYS = ["age", "sex", "broad_group"]
# These are source identities, not filenames inferred from a publication year.
SOURCES = [
    {
        "year": 2003,
        "filename": "national_2003.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/doclib/2004/2.%20shnaton%20population/st02_18.xls",
        "sha256": "68d0bc87ed3082aa270cd4a3ac81c5584810a158d60b0d2237188e162dce5614",
        "bytes": 351232
    },
    {
        "year": 2009,
        "filename": "national_2009.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/doclib/2010/2.%20shnatonpopulation/st02_19.xls",
        "sha256": "96e0c8b7ee515ab67b4771aa47afdf0c4a0160725cacb1f2d219ea46c794644f",
        "bytes": 123392
    },
    {
        "year": 2015,
        "filename": "national_2015.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2016/2.shnatonpopulation/st02_03.xls",
        "sha256": "37e25f549a3b755e5b7df0fd55b39f4d2ab93ce8a1015e5e9c07af58b3db8b21",
        "bytes": 131072
    },
    {
        "year": 2016,
        "filename": "national_2016.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2017/2.%20shnatonpopulation/st02_03.xls",
        "sha256": "1bd88511601ec79ef887754ef64fd8377187f3b4906a11ce966345c3e54f030a",
        "bytes": 130560
    },
    {
        "year": 2017,
        "filename": "national_2017.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2018/2.%20shnatonpopulation/st02_03.xls",
        "sha256": "9d0a11260b89e35a9133c4535a783d271d1c8d8c3792fbdaea361ee68e1265ef",
        "bytes": 130560
    },
    {
        "year": 2018,
        "filename": "national_2018.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2019/2.shnatonpopulation/st02_03.xls",
        "sha256": "d8211ebf6c36c7543f157bd3b1cb782ac77ab87daeefd5ea09030a668e066e0d",
        "bytes": 131072
    },
    {
        "year": 2019,
        "filename": "national_2019.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2020/2.shnatonpopulation/st02_03.xls",
        "sha256": "a7f0eb56073a6420afa160acc14e961fa6e1d939b416dc7c56815687c8614d03",
        "bytes": 131584
    },
    {
        "year": 2020,
        "filename": "national_2020.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2021/2.shnatonpopulation/st02_03.xls",
        "sha256": "b546d15b51cde87164b7df68800eecc463285e5912fe873602a1dc37e70e42e2",
        "bytes": 131584
    },
    {
        "year": 2021,
        "filename": "national_2021.xls",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2022/2.shnatonpopulation/st02_03.xls",
        "sha256": "2d0fb922f3b1dd61c7c4219b0507bb2065b3d10d0aff8cd033430d97d32ba785",
        "bytes": 130560
    },
    {
        "year": 2022,
        "filename": "national_2022.xlsx",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2023/2.shnatonpopulation/st02_03.xlsx",
        "sha256": "2288e7615f88191c2f8d1641a7303c79164f77832ba4e0b2b5212475f9707e5b",
        "bytes": 71602
    },
    {
        "year": 2023,
        "filename": "national_2023.xlsx",
        "source_url": "https://www.cbs.gov.il/he/publications/doclib/2024/2.shnatonpopulation/st02_03.xlsx",
        "sha256": "9f6b759644ed8f7dd271973018cdb659166ad1de6fdecc90317f38f559f6e6f3",
        "bytes": 71972
    },
    {
        "year": 2024,
        "filename": "national_2024.xlsx",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2025/2.ShnatonPopulation/st02_03.xlsx",
        "sha256": "543caa013f490488d01ec4db8f3da347f29ce846af44f6ac03555d51391279d3",
        "bytes": 67395
    },
    {
        "year": 2025,
        "filename": "national_2025.xlsx",
        "source_url": "https://www.cbs.gov.il/he/publications/DocLib/2026/2.shnatonpopulation/st02_03.xlsx",
        "sha256": "530f7b54f662351c2c19c3e7217780b6c94c3a889198f9cbeeb04664ea0f7adf",
        "bytes": 64890
    }
]


def verify_source(path, source):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError(f"Source content changed: {source['filename']}")
    if data[:2] != b"PK" and data[:4] != bytes.fromhex("d0cf11e0"):
        raise ValueError("Expected an Excel workbook")


def _age(value):
    label = str(value).strip()
    if re.fullmatch(r"\d{1,2}(?:\.0)?", label):
        number = int(float(label))
        return (number, number) if number <= 84 else None
    match = re.fullmatch(r"(\d{2})[-–](\d{2})", label)
    if match and min(int(match[1]), int(match[2])) >= 85:
        return min(int(match[1]), int(match[2])), max(int(match[1]), int(match[2]))
    match = re.fullmatch(r"(\d{2})\+", label)
    if match and int(match[1]) >= 85:
        return int(match[1]), 150
    return None


def parse_workbook(path, year):
    book = pd.ExcelFile(path)
    if len(book.sheet_names) < 4:
        raise ValueError("Population table requires the broad-group and Jewish age sheets")
    first = pd.read_excel(book, sheet_name=0, header=None)
    heading = " ".join(str(v) for v in first.iloc[:15].values.flatten())
    if "POPULATION GROUP" not in heading or "SEX AND AGE" not in heading:
        raise ValueError("Workbook is a different CBS table")
    if "Thousands" not in heading:
        raise ValueError("CBS population units are not the reviewed thousands unit")
    basis = re.search(r"Average\s*(?:\(\d+\))?\s*(\d{4})", heading)
    if not basis or int(basis[1]) != year:
        raise ValueError("Workbook reference year does not match its source identity")
    rows, totals = [], {}
    # CBS arranges triplets as females/males/total. Sheets a,b contain Arab,
    # Jewish+Other and total; c,d contain Christian, Muslim and Jewish.
    for sheet_index in range(4):
        frame = pd.read_excel(book, sheet_name=sheet_index, header=None)
        column_groups = [(1, "arab"), (4, "jewish_other")] if sheet_index < 2 else [(7, "jewish")]
        for _, row in frame.iterrows():
            interval = _age(row.iloc[0])
            # The repeated Hebrew age label guards against numeric print footers.
            if interval is not None and (len(row) <= 10 or _age(row.iloc[10]) != interval):
                continue
            is_total = str(row.iloc[0]).strip().upper() == "TOTAL"
            for column, group in column_groups:
                for offset, sex in [(0, "female"), (1, "male")]:
                    value = pd.to_numeric(row.iloc[column + offset], errors="coerce")
                    if pd.isna(value):
                        if interval is not None:
                            raise ValueError(f"Missing population count: {year}/{group}/{interval}/{sex}")
                        continue
                    if is_total:
                        totals[group, sex] = float(value) * 1000
                    elif interval is not None:
                        rows.append(dict(age=min(interval[0], 85), sex=sex,
                                         broad_group=group, population=float(value) * 1000))
    frame = pd.DataFrame(rows)
    # Open-tail components (85--89,90+) are disjoint and summed once.
    frame = frame.groupby(KEYS, as_index=False).population.sum()
    expected = pd.MultiIndex.from_product([range(86), ["female", "male"],
                                          ["arab", "jewish_other", "jewish"]], names=KEYS)
    indexed = frame.set_index(KEYS).reindex(expected)
    if indexed.population.isna().any() or (indexed.population < 0).any():
        raise ValueError("Incomplete or negative age-by-sex population cells")
    if set(totals) != {(group, sex) for group in ["arab", "jewish_other", "jewish"] for sex in ["female", "male"]}:
        raise ValueError("Missing published population total controls")
    # Input tables round to hundreds: at most 86*50 persons per sex/group.
    for (group, sex), total in totals.items():
        actual = frame[(frame.broad_group == group) & (frame.sex == sex)].population.sum()
        if abs(actual - total) > 5000:
            raise ValueError(f"Population age totals fail: {year}/{group}/{sex}")
    pivot = frame.pivot(index=["age", "sex"], columns="broad_group", values="population")
    pivot["other"] = pivot.jewish_other - pivot.jewish
    if (pivot.other < 0).any():
        raise ValueError("Jewish+Other population is below the Jewish population")
    result = pivot[["jewish", "arab", "other"]].reset_index().melt(
        id_vars=["age", "sex"], var_name="broad_group", value_name="population")
    result["year"] = year
    result["reference_date"] = f"{year}-07-01"
    result["reference_date_status"] = "midyear_alignment_only_not_observed_day"
    result["reference_basis"] = "annual_average"
    result["source_id"] = f"cbs_statistical_abstract_average_{year}"
    result["status"] = "observed"
    return result


def annualize(observed, start=2003, end=2026):
    """Interpolate missing years; project at most one year past the last source.

    Projection uses each cell's two-year growth rate when that anchor exists,
    otherwise the latest available interval. It estimates national resident age
    shape only; locality register counts remain separate.
    """
    years = sorted(observed.year.unique())
    if len(years) < 2:
        raise ValueError("At least two observed national years required")
    if start < years[0] or end > years[-1] + 1:
        raise ValueError("Unsupported national population extrapolation horizon")
    if observed.duplicated(["year"] + KEYS).any():
        raise ValueError("Duplicate population controls")
    pieces = []
    for year in range(start, end + 1):
        if year in years:
            pieces.append(observed[observed.year == year].copy())
            continue
        if year < years[-1]:
            low = max(v for v in years if v < year)
            high = min(v for v in years if v > year)
            fraction = (year - low) / (high - low)
            status = "interpolated"
        else:
            high = years[-1]
            low = high - 2 if high - 2 in years else years[-2]
            fraction = (year - low) / (high - low)
            status = "projected"
        left = observed[observed.year == low].set_index(KEYS).sort_index()
        right = observed[observed.year == high].set_index(KEYS).sort_index()
        if not left.index.equals(right.index):
            raise ValueError("Population source cells changed")
        # Linear counts handle genuinely zero rounded Other cells without a
        # pseudocount; log interpolation is used only for strictly positive cells.
        a, b = left.population.to_numpy(), right.population.to_numpy()
        values = np.maximum(0, a + fraction * (b - a))
        positive = (a > 0) & (b > 0)
        values[positive] = np.exp(np.log(a[positive]) + fraction * np.log(b[positive] / a[positive]))
        part = right.reset_index()
        part["population"] = values
        part["year"] = year
        part["reference_date"] = f"{year}-07-01"
        part["status"] = status
        part["source_id"] = f"cbs_average_{low}_{high}_{status}"
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True).sort_values(["year"] + KEYS)


def holdout_validation(observed, years=(2019, 2022, 2024)):
    """Predict excluded annual margins from adjacent observed years.

    Validation reports errors in within-group adult age-by-sex shares. National
    totals are not supplied to either prediction, so mass closure is no test.
    """
    rows = []
    frozen = observed[(observed.year == 2022) & (observed.age >= 18)].set_index(KEYS).population
    frozen = frozen / frozen.groupby(level="broad_group").transform("sum")
    for year in years:
        train = observed[observed.year != year]
        if year not in observed.year.unique() or year in (train.year.min(), train.year.max()):
            continue
        predicted = annualize(train, start=year, end=year)
        actual = observed[(observed.year == year) & (observed.age >= 18)].set_index(KEYS).population
        estimate = predicted[predicted.age >= 18].set_index(KEYS).population
        actual = actual / actual.groupby(level="broad_group").transform("sum")
        estimate = estimate / estimate.groupby(level="broad_group").transform("sum")
        for group in ["jewish", "arab", "other"]:
            a, b, c = [s.xs(group, level="broad_group") for s in [actual, estimate, frozen]]
            rows.append(dict(year=year, broad_group=group, cells=len(a),
                             reconstructed_mae_pp=float((a - b).abs().mean() * 100),
                             fixed2022_mae_pp=float((a - c).abs().mean() * 100) if year != 2022 else None,
                             reconstructed_total_variation=float((a - b).abs().sum() / 2),
                             fixed2022_total_variation=float((a - c).abs().sum() / 2) if year != 2022 else None))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in SOURCES:
        path = args.raw / source["filename"]
        if args.download and not path.exists():
            with urllib.request.urlopen(source["source_url"], timeout=60) as stream:
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError("Downloaded CBS source differs from reviewed content")
            path.write_bytes(data)
        verify_source(path, source)
        frames.append(parse_workbook(path, source["year"]))
    observed = pd.concat(frames, ignore_index=True)
    annual = annualize(observed)
    annual.to_csv(args.output / "national_age_sex_group_controls.csv", index=False)
    validation = holdout_validation(observed)
    validation.to_csv(args.output / "national_population_holdout.csv", index=False)
    audit = dict(publisher="Israel Central Bureau of Statistics", sources=SOURCES,
                 observed_years=sorted(int(v) for v in observed.year.unique()),
                 annual_rows=len(annual), statuses=annual.groupby("status").size().to_dict(),
                 population_universe="resident Israelis, including permanent noncitizen residents; foreign population excluded",
                 date_basis="annual averages; not election-day counts or eligible voters",
                 definition_break="2023 census-based population revision; direct levels retained, no invented reconciliation",
                 holdout=json.loads(validation.to_json(orient="records")))
    (args.output / "national_population_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["observed_years", "annual_rows", "statuses"]}, allow_nan=False))


if __name__ == "__main__":
    main()
