"""Portable entry point for the research population, history and fitting stages."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


STAGES = {
    "population": "demographic_foundation",
    "history": "historical_elections",
    "fit": "fit_foundation_history",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="Arguments for the selected stage; use STAGE --help for details")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(folder))
    module = importlib.import_module(STAGES[args.stage])
    sys.argv = [str(folder / (STAGES[args.stage]+".py"))] + args.arguments
    module.main()


if __name__ == "__main__":
    main()
