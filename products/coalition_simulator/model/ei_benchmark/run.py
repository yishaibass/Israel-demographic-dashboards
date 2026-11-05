"""Run the benchmark after activating its isolated research dependencies."""
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from products.coalition_simulator.model.ei_benchmark.runtime import activate

    activate()
    from products.coalition_simulator.model.ei_benchmark.benchmark import main

    main()
