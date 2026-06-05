"""Small demonstration models used by the command-line interface."""

from __future__ import annotations

import math
from typing import Mapping


def demo_risk_model(features: Mapping[str, int | float]) -> float:
    """Return a synthetic risk probability for demo feature dictionaries."""

    age = float(features.get("age", 35))
    income = float(features.get("income", 50_000))
    debt = float(features.get("debt", 10_000))
    missed_payments = float(features.get("missed_payments", 0))

    score = (
        -1.2
        + (age - 35) * 0.015
        - (income - 50_000) / 100_000
        + debt / 40_000
        + missed_payments * 0.35
    )
    return 1 / (1 + math.exp(-score))
