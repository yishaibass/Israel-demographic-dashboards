# Coalition simulator

This Increment-1 package publishes the current coalition simulator. Set `coalition_simulator_project` in ignored `config/paths.toml`, then run:

```bash
python products/coalition_simulator/build.py
```

The publication step is deterministic and applies the repository-owned Analytics contract. The underlying analytical pipeline still runs in the configured legacy project and is not yet independently runnable from a fresh clone.
