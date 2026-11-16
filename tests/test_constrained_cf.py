"""
tests/test_constrained_cf.py

Unit tests for:
- Constrained counterfactuals with immutable features and intervention cost.
- Insertion-AUC and insertion curves in explanation validation.
"""

import numpy as np

from src.explainability.counterfactuals import nearest_normal_counterfactual
from src.explainability.validation import insertion_auc, insertion_curve


def test_constrained_counterfactual_immutable_feature():
    # 4 features: [amount, distance, user_id, age]
    # user_id (index 2) is immutable!
    obs = np.array([500.0, 100.0, 42.0, 30.0])
    ref = np.array([
        [10.0, 5.0, 99.0, 28.0],
        [20.0, 8.0, 42.0, 31.0],
    ])
    feature_names = ["amount", "distance", "user_id", "age"]

    # When user_id is marked immutable
    res = nearest_normal_counterfactual(
        observation=obs,
        reference=ref,
        feature_names=feature_names,
        immutable_features=["user_id"],
    )

    assert res["found"] is True
    assert "intervention_cost" in res
    assert res["intervention_cost"] >= 0.0

    # Ensure user_id was NOT changed in changes list
    changed_names = [c["feature"] for c in res["changes"]]
    assert "user_id" not in changed_names


def test_insertion_curve_and_auc():
    obs = np.array([5.0, 5.0, 5.0])
    base = np.array([0.0, 0.0, 0.0])

    def score_fn(x):
        return float(np.sum(x))

    order = [0, 1, 2]
    curve = insertion_curve(obs, score_fn, order, baseline=base)
    assert len(curve) == 4
    assert curve[0] == 0.0
    assert curve[-1] == 15.0

    auc = insertion_auc(curve)
    assert 0.0 < auc <= 15.0
