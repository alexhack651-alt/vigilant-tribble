from black_box_ai import demo_risk_model, explain_prediction


def test_explain_prediction_with_mapping_features():
    instance = {"age": 45, "income": 75_000, "debt": 30_000, "missed_payments": 1}
    baseline = {"age": 35, "income": 50_000, "debt": 10_000, "missed_payments": 0}

    explanation = explain_prediction(demo_risk_model, instance, baseline)

    assert 0 <= explanation.prediction <= 1
    assert explanation.delta == explanation.prediction - explanation.baseline_prediction
    assert [item.feature for item in explanation.ranked_contributions()][0] in {
        "debt",
        "income",
        "missed_payments",
        "age",
    }


def test_explain_prediction_with_sequence_features():
    def weighted_sum(values):
        return values[0] * 2 + values[1] * -1

    explanation = explain_prediction(
        weighted_sum,
        instance=[4, 3],
        baseline=[1, 1],
        feature_names=["signal", "drag"],
    )

    contributions = {item.feature: item.contribution for item in explanation.contributions}
    assert explanation.prediction == 5
    assert explanation.baseline_prediction == 1
    assert contributions == {"signal": 6, "drag": -2}
