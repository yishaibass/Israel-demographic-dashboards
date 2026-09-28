# Household fiscal position

This Increment-1 package validates and publishes the current household fiscal-position output. Set `fiscal_position_project` in ignored `config/paths.toml`, then run:

```bash
python products/fiscal_position/build.py
```

The publication step uses `dashboard/template_he.html`, whose Hebrew shell, layout, modules and navigation were recovered from the prior public release. Its four changed methodology disclosures cover the current NII, welfare, health and capital-incidence methods. The template contains no model payload; the builder injects only the current authoritative output. It is deterministic and validates the Hebrew RTL shell, current method text, national and sector controls, capital-incidence selection, and Analytics contract. The underlying analytical pipeline still runs in the configured legacy project and is not yet independently runnable from a fresh clone.
