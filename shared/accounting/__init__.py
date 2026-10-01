"""Canonical household profit-and-loss accounting."""

from .pnl import (
    PNL_VIEWS,
    build_household_pnl,
    summarize_functional_pnl,
    validate_component_facts,
    validate_national_controls,
)
from .market_adapter import build_hes_market_component_facts

__all__ = [
    "PNL_VIEWS", "build_household_pnl", "build_hes_market_component_facts",
    "summarize_functional_pnl",
    "validate_component_facts", "validate_national_controls",
]
