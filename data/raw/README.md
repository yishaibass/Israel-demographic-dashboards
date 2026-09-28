# Raw inputs (not distributed)

No raw or person-level data are included in this repository. Place locally obtained inputs under this directory using the structure below, or set `ISRAEL_DASHBOARD_DATA_DIR` to an equivalent directory.

```text
data/raw/
├── census/
│   └── census_2022_puf.csv
└── elections/
    ├── K21_kalpi_clean.csv
    ├── K22_kalpi_clean.csv
    ├── K23_kalpi_clean.csv
    ├── K24_kalpi_clean.csv
    ├── K25_kalpi_clean.csv
    ├── model_input_v2.csv
    └── party_mapping.xlsx
```

## Census input

`census_2022_puf.csv` is the person-level 2022 Census PUF obtained from the Israel Central Bureau of Statistics (CBS) under the applicable PUF terms. CBS describes PUF files as non-identifiable individual-level files available through its public-use service. The file is not redistributed here; users must obtain it directly from CBS.

- CBS 2022 Census portal: https://census.cbs.gov.il/en
- CBS service page, including Public Use Files: https://www.cbs.gov.il/he/cbsNewBrand/Pages/Service-to-the-Public.aspx

The required PUF columns are enumerated in `modeling/data_pipeline.py` as `RAW_FEATURES`.

## Election inputs

`K21_kalpi_clean.csv` through `K25_kalpi_clean.csv` contain ballot-box results for the five elections used by the model. The source is the Central Elections Committee for the Knesset, which publishes official final election results. The repository does not redistribute the downloaded files.

- 21st Knesset, 9 April 2019: https://votes21.bechirot.gov.il/
- 22nd Knesset, 17 September 2019: https://votes22.bechirot.gov.il/
- 23rd Knesset, 2 March 2020: https://votes23.bechirot.gov.il/
- 24th Knesset, 23 March 2021: https://votes24.bechirot.gov.il/
- 25th Knesset, 1 November 2022: https://votes25.bechirot.gov.il/
- Central Elections Committee: https://www.bechirot.gov.il/

The cleaned files retain the committee's Hebrew fields for locality code, ballot identifier, eligible voters, voters, invalid votes, valid votes and party-list vote columns. Cleaning is limited to encoding, column normalization and numeric conversion.

`model_input_v2.csv` is the locally prepared ballot-to-Census-geography crosswalk used to aggregate ballot boxes to locality × 2011 statistical-area group. It must contain `election`, `locality_code`, `ballot_id`, `sa_code` and `is_envelope`. `party_mapping.xlsx` contains a `Mapping` sheet with `Election`, `Letter` and `Name_English`, documenting the mapping from election-specific lists to stable party families. Neither file contains Census person records.

Users are responsible for complying with the source agencies' current access and licensing terms.
