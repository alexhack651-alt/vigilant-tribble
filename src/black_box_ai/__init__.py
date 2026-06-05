"""Utilities for inspecting black-box AI predictions."""

from .explainer import Explanation, FeatureContribution, explain_prediction
from .models import demo_risk_model

__all__ = [
    "Explanation",
    "FeatureContribution",
    "demo_risk_model",
    "explain_prediction",
]
