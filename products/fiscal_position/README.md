# Household fiscal position

This Increment-1 package validates and publishes the current household fiscal-position output. Set `fiscal_position_project` in ignored `config/paths.toml`, then run:

```bash
python products/fiscal_position/build.py
```

The publication step is deterministic and validates the national controls, Analytics contract, and Hebrew public title. The underlying analytical pipeline still runs in the configured legacy project and is not yet independently runnable from a fresh clone.
