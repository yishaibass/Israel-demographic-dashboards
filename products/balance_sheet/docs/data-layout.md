# Data layout

Download the public-use microdata from the Israel CBS and keep it outside Git. Pass the two parent directories through the command-line arguments shown in the README.

## Household Expenditure Survey

Expected structure:

```text
HES_ROOT/
  H20211021/
    H20211021datamb.csv
    H20211021dataprat.csv
    H20211021dataincmissim.csv
    H20211021datahouse.csv
    H20211021dataprod.csv
  H20221021/
    H20221021datamb.csv
    H20221021dataprat.csv
    H20221021dataincmissim.csv
    H20221021datahouse.csv
    H20221021dataprod.csv
  H20231021/
    H20231021datamb.csv
    H20231021dataprat.csv
    H20231021dataincmissim.csv
    H20231021datahouse.csv
    H20231021dataprod.csv
```

The files are read using their original CBS encoding. Keep the original filenames.

## Longitudinal Survey

Expected release folders under `LONGITUDINAL_ROOT`:

```text
H20121284
H20131282
H201420151283
H20161281
H20171281
H20181281
H20191281
H20201281
H202120221281
H20231281
```

The panel builder contains the exact expected `datamb`, `dataprat` and, where applicable, `datapkida` filenames for each release.

## Data policy

Do not commit downloaded microdata, household identifiers, processed panels or household-level model outputs. The project `.gitignore` excludes the standard data and build directories.
