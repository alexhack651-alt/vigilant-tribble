"""Perturbation-based explanations for opaque prediction functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

NumericValue = int | float
FeatureVector = Mapping[str, NumericValue] | Sequence[NumericValue]
PredictionFunction = Callable[[FeatureVector], float]


@dataclass(frozen=True)
class FeatureContribution:
    """Estimated contribution of a single feature to one prediction."""

    feature: str
    value: NumericValue
    baseline: NumericValue
    contribution: float


@dataclass(frozen=True)
class Explanation:
    """A local explanation for one prediction from a black-box model."""

    prediction: float
    baseline_prediction: float
    contributions: tuple[FeatureContribution, ...]

    @property
    def delta(self) -> float:
        """Return the prediction change from baseline to instance."""

        return self.prediction - self.baseline_prediction

    def ranked_contributions(self) -> tuple[FeatureContribution, ...]:
        """Return contributions sorted by absolute magnitude descending."""

        return tuple(
            sorted(
                self.contributions,
                key=lambda contribution: abs(contribution.contribution),
                reverse=True,
            )
        )


def explain_prediction(
    predict: PredictionFunction,
    instance: FeatureVector,
    baseline: FeatureVector,
    feature_names: Iterable[str] | None = None,
) -> Explanation:
    """Explain one prediction by replacing each feature with its baseline value.

    The function treats ``predict`` as a true black box: it only calls the
    function with complete feature vectors and measures how the output changes.
    Contributions are local, approximate, and intended for inspection rather
    than proof of model causality.
    """

    instance_values, names = _normalise_vector(instance, feature_names)
    baseline_values, baseline_names = _normalise_vector(baseline, names)

    if tuple(names) != tuple(baseline_names):
        raise ValueError("instance and baseline must expose the same feature names")

    prediction = float(predict(_restore_vector(instance, names, instance_values)))
    baseline_prediction = float(predict(_restore_vector(baseline, names, baseline_values)))
    contributions: list[FeatureContribution] = []

    for index, name in enumerate(names):
        perturbed = list(instance_values)
        perturbed[index] = baseline_values[index]
        perturbed_prediction = float(predict(_restore_vector(instance, names, perturbed)))
        contributions.append(
            FeatureContribution(
                feature=name,
                value=instance_values[index],
                baseline=baseline_values[index],
                contribution=prediction - perturbed_prediction,
            )
        )

    return Explanation(
        prediction=prediction,
        baseline_prediction=baseline_prediction,
        contributions=tuple(contributions),
    )


def _normalise_vector(
    vector: FeatureVector,
    feature_names: Iterable[str] | None,
) -> tuple[list[NumericValue], list[str]]:
    if isinstance(vector, Mapping):
        names = list(feature_names) if feature_names is not None else list(vector.keys())
        missing = [name for name in names if name not in vector]
        if missing:
            raise ValueError(f"missing feature values: {', '.join(missing)}")
        return [vector[name] for name in names], names

    values = list(vector)
    names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(len(values))]
    if len(values) != len(names):
        raise ValueError("feature_names must match the vector length")
    return values, names


def _restore_vector(
    template: FeatureVector,
    names: Sequence[str],
    values: Sequence[NumericValue],
) -> FeatureVector:
    if isinstance(template, Mapping):
        return dict(zip(names, values, strict=True))
    return list(values)
