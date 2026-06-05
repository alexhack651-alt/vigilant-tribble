# Black Box AI

A small, dependency-free Python toolkit for inspecting **black-box AI** predictions.
It treats a model as opaque: you provide a prediction function, one instance, and a
baseline instance, and the toolkit estimates local feature contributions by
perturbing one feature at a time.

## Why this exists

Black-box AI systems can be useful, but their decisions are hard to audit when
users only see inputs and outputs. This project demonstrates a lightweight
inspection workflow that can be embedded in demos, tutorials, or early prototypes:

1. Score the original instance.
2. Score a neutral baseline instance.
3. Replace each feature with its baseline value.
4. Rank features by the output change caused by that replacement.

The result is an approximate local explanation, not a causal proof.

## Install

```bash
python -m pip install -e .
```

## Use the CLI

Run the built-in synthetic risk model:

```bash
black-box-ai
```

Pass your own feature values as JSON:

```bash
black-box-ai \
  --instance '{"age": 52, "income": 72000, "debt": 40000, "missed_payments": 2}' \
  --baseline '{"age": 35, "income": 50000, "debt": 10000, "missed_payments": 0}'
```

Example output:

```text
prediction: 0.5597
baseline_prediction: 0.2789
delta: +0.2808
contributions:
  - missed_payments: +0.1727 (value=2, baseline=0)
  - debt: +0.1120 (value=28000, baseline=10000)
  - age: +0.0447 (value=47, baseline=35)
  - income: -0.0342 (value=64000, baseline=50000)
```

## Use as a library

```python
from black_box_ai import explain_prediction


def model(features):
    return features["signal"] * 2 - features["drag"]

explanation = explain_prediction(
    model,
    instance={"signal": 4, "drag": 3},
    baseline={"signal": 1, "drag": 1},
)

for contribution in explanation.ranked_contributions():
    print(contribution.feature, contribution.contribution)
```

## Development

Run tests with:

```bash
PYTHONPATH=src pytest
```

Run a quick CLI smoke test with:

```bash
PYTHONPATH=src python -m black_box_ai
```
