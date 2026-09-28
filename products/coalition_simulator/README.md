# Coalition simulator

This Increment-1 package publishes the current coalition simulator. Set `coalition_simulator_project` in ignored `config/paths.toml`, then run:

```bash
python products/coalition_simulator/build.py
```

The publication step preserves the coalition model payload, restores the demographic-analysis tab from `voting-simulator/demographics.html`, adds repository navigation, and applies the repository-owned Analytics contract. These publication elements were recovered from remote commit `f0423cf`. The underlying analytical pipeline still runs in the configured legacy project and is not yet independently runnable from a fresh clone.
