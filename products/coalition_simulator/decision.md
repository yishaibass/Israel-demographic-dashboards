# Coalition-simulator decisions

- Keep the simulator separate from the voting-propensity model.
- Publish a self-contained Hebrew dashboard at `voting-simulator/index.html`.
- Preserve current model logic during the first migration increment.
- Keep demographic analysis as an embedded sibling product; publication integration must not alter the coalition model payload.
- Own the demographic tab, iframe resizing, home navigation and Analytics application in this repository's publication build.
- Increment 1 publishes from the configured `coalition_simulator_project`; the analytical pipeline is not yet independently runnable from a fresh clone.
