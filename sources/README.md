# Data sources

Raw and licensed microdata are not stored in Git. The model reads them from a
user-configured data root; copy `config/paths.example.toml` to
`config/paths.toml` and set the local paths.

`catalog.yml` is the source of truth for required datasets. Each entry records
the accepted filenames, local subpath, publisher, access status, source URL,
minimum schema and consuming products. Paths in the catalog are relative to a
configured root and contain no workstation-specific information.

The local canonical store is organized as follows:

```text
<data_root>/
├── סקרי למס/
│   ├── סקר הוצאות משקי בית 2016 - 2023/
│   ├── סקר ארוך טווח 2012 - 2023/
│   └── Census Israel 2022/
└── Israel demographic dashboards/
    ├── administrative/
    ├── balance_sheet_reference/
    └── elections/
```

CBS public-use microdata are licensed and must not be redistributed. Public
administrative and election files remain outside Git so a clone contains code,
documentation and disclosure-safe aggregates only.

