"""Command-line interface for the black-box AI explainer."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .explainer import explain_prediction
from .models import demo_risk_model

DEFAULT_INSTANCE = {
    "age": 47,
    "income": 64_000,
    "debt": 28_000,
    "missed_payments": 2,
}
DEFAULT_BASELINE = {
    "age": 35,
    "income": 50_000,
    "debt": 10_000,
    "missed_payments": 0,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explain a demo black-box AI prediction with feature perturbations."
    )
    parser.add_argument(
        "--instance",
        default=json.dumps(DEFAULT_INSTANCE),
        help="JSON object of feature values to score.",
    )
    parser.add_argument(
        "--baseline",
        default=json.dumps(DEFAULT_BASELINE),
        help="JSON object of baseline feature values.",
    )
    args = parser.parse_args(argv)

    instance = _load_json_object(args.instance, "instance")
    baseline = _load_json_object(args.baseline, "baseline")
    explanation = explain_prediction(demo_risk_model, instance, baseline)

    print(f"prediction: {explanation.prediction:.4f}")
    print(f"baseline_prediction: {explanation.baseline_prediction:.4f}")
    print(f"delta: {explanation.delta:+.4f}")
    print("contributions:")
    for contribution in explanation.ranked_contributions():
        print(
            f"  - {contribution.feature}: {contribution.contribution:+.4f} "
            f"(value={contribution.value}, baseline={contribution.baseline})"
        )
    return 0


def _load_json_object(raw_json: str, label: str) -> dict[str, int | float]:
    value = json.loads(raw_json)
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    invalid = [key for key, feature_value in value.items() if not isinstance(feature_value, int | float)]
    if invalid:
        raise SystemExit(f"{label} contains non-numeric values: {', '.join(invalid)}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
