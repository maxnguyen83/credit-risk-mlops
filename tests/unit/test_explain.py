"""Unit tests for the explainability layer's pure parts.

Nothing here needs a trained model, MLflow or the compose stack. The pieces
that turn numbers into sentences and compare two rankings are exactly the ones
a reviewer will question, so they are tested in isolation where the failure
message points at the line that is wrong.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.explain import lime_explainer, shap_explainer
from credit_risk.explain.shap_explainer import ExplainerUnavailable
from credit_risk.features.build import FEATURE_NAMES
from credit_risk.serving import model_loader

CONTRIBUTIONS: list[dict[str, Any]] = [
    {"feature": "PAY_1", "value": 2, "shap": 0.90},
    {"feature": "utilization_mean", "value": 0.94, "shap": 0.70},
    {"feature": "available_credit", "value": 6500.0, "shap": -0.50},
    {"feature": "months_delinquent", "value": 2, "shap": 0.40},
    {"feature": "mean_pay_amt", "value": 3583.33, "shap": 0.10},
]


# ------------------------------------------------------------ top_reasons


def test_top_reasons_returns_k_sentences_ordered_by_contribution() -> None:
    reasons = shap_explainer.top_reasons(CONTRIBUTIONS, k=3)

    assert len(reasons) == 3
    assert "repayment status last month" in reasons[0]
    assert "credit utilisation" in reasons[1]
    assert "months in arrears" in reasons[2]
    assert all(sentence.endswith(".") for sentence in reasons)
    assert all(sentence[0].isupper() for sentence in reasons)


def test_top_reasons_ignores_contributions_that_lowered_the_risk() -> None:
    # available_credit has the third-largest magnitude but pushed the score
    # down, so it is not a reason the answer was adverse.
    reasons = shap_explainer.top_reasons(CONTRIBUTIONS, k=5)
    assert len(reasons) == 4
    assert not any("still available" in sentence for sentence in reasons)


def test_top_reasons_says_so_when_nothing_increased_the_risk() -> None:
    reasons = shap_explainer.top_reasons([{"feature": "PAY_1", "value": -1, "shap": -0.2}])
    assert len(reasons) == 1
    assert "portfolio average" in reasons[0]


def test_top_reasons_never_cites_a_protected_attribute() -> None:
    # SHAP may well rank SEX first. "Your gender increased your risk" is not a
    # notice anyone may send, so it is dropped from the letter -- while staying
    # in the contribution list the compliance officer reads.
    reasons = shap_explainer.top_reasons(
        [
            {"feature": schema.SEX, "value": 2, "shap": 0.95},
            {"feature": schema.AGE, "value": 39, "shap": 0.80},
            {"feature": schema.EDUCATION, "value": 2, "shap": 0.60},
            {"feature": "PAY_1", "value": 2, "shap": 0.30},
        ],
        k=3,
    )
    assert reasons == [
        "The repayment status last month (2) increased the estimated risk of default."
    ]


def test_top_reasons_survives_a_contribution_without_an_observed_value() -> None:
    reasons = shap_explainer.top_reasons([{"feature": "PAY_3", "value": None, "shap": 0.4}])
    assert reasons == [
        "The repayment status three months ago increased the estimated risk of default."
    ]


def test_phrase_for_falls_back_to_the_raw_name() -> None:
    assert shap_explainer.phrase_for("utilization_m3") == "credit utilisation three months ago"
    assert shap_explainer.phrase_for("PAY_AMT6") == "the amount repaid six months ago"
    assert shap_explainer.phrase_for("worst_pay_status").startswith("the worst repayment status")
    assert shap_explainer.phrase_for("some_new_feature") == "some new feature"


# -------------------------------------------------------------- agreement


def _shap_result(features: list[str]) -> dict[str, Any]:
    return {
        "base_value": -1.2,
        "contributions": [
            {"feature": name, "value": 1.0, "shap": 0.9 - index * 0.1}
            for index, name in enumerate(features)
        ],
    }


def _lime_result(conditions: list[str]) -> dict[str, Any]:
    return {
        "contributions": [
            {"feature": condition, "weight": 0.4 - index * 0.1}
            for index, condition in enumerate(conditions)
        ]
    }


def test_agreement_is_total_for_identical_rankings() -> None:
    result = lime_explainer.agreement(
        _shap_result(["PAY_1", "AGE", "LIMIT_BAL"]),
        _lime_result(["PAY_1", "AGE", "LIMIT_BAL"]),
    )
    assert result["top3_overlap"] == 3
    assert "all 3" in result["note"]


def test_agreement_is_zero_for_disjoint_rankings() -> None:
    result = lime_explainer.agreement(
        _shap_result(["PAY_1", "PAY_2", "PAY_3"]),
        _lime_result(["LIMIT_BAL", "AGE", "BILL_AMT1"]),
    )
    assert result["top3_overlap"] == 0
    assert "none" in result["note"]


def test_agreement_counts_a_partial_overlap() -> None:
    result = lime_explainer.agreement(
        _shap_result(["PAY_1", "PAY_2", "PAY_3"]),
        _lime_result(["PAY_1", "AGE", "LIMIT_BAL"]),
    )
    assert result["top3_overlap"] == 1
    assert "1 of the top 3" in result["note"]


def test_agreement_sees_through_limes_discretised_conditions() -> None:
    name = max(FEATURE_NAMES, key=len)
    result = lime_explainer.agreement(
        _shap_result([name, "AGE", "LIMIT_BAL"]),
        _lime_result([f"1.00 < {name} <= 2.00", "AGE", "LIMIT_BAL"]),
    )
    assert result["top3_overlap"] == 3


def test_base_feature_recovers_the_column_from_a_condition() -> None:
    name = max(FEATURE_NAMES, key=len)
    assert lime_explainer.base_feature(f"{name} > 1.00") == name
    assert lime_explainer.base_feature("nothing recognisable") == "nothing recognisable"


# -------------------------------------------------- shap value normalisation


class _FakeExplainer:
    """Stands in for TreeExplainer's several return shapes."""

    def __init__(self, values: Any, expected: Any) -> None:
        self._values = values
        self.expected_value = expected

    def shap_values(self, frame: pd.DataFrame) -> Any:
        return self._values


def test_shap_matrix_collapses_a_per_class_list() -> None:
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    explainer = _FakeExplainer([np.zeros((1, 2)), np.ones((1, 2))], [0.1, 0.2])
    assert shap_explainer.shap_matrix(explainer, frame).tolist() == [[1.0, 1.0]]
    assert shap_explainer.base_value(explainer) == pytest.approx(0.2)


def test_shap_matrix_collapses_a_three_dimensional_array() -> None:
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    values = np.stack([np.zeros((1, 2)), np.ones((1, 2))], axis=-1)
    explainer = _FakeExplainer(values, 0.3)
    assert shap_explainer.shap_matrix(explainer, frame).tolist() == [[1.0, 1.0]]
    assert shap_explainer.base_value(explainer) == pytest.approx(0.3)


def test_shap_matrix_reshapes_a_single_row() -> None:
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    explainer = _FakeExplainer(np.array([0.5, -0.5]), 0.0)
    assert shap_explainer.shap_matrix(explainer, frame).shape == (1, 2)


# ---------------------------------------------------- JSON serialisability


def test_explain_payloads_are_json_serialisable() -> None:
    frame = pd.DataFrame([{"PAY_1": np.float32(2.0), "AGE": np.int64(39)}])
    values = np.array([[0.61, -0.22]], dtype=np.float32)

    contributions = shap_explainer.build_contributions(frame, values)
    shap_block = {"base_value": shap_explainer.base_value(_FakeExplainer(None, np.float32(-1.2)))}
    shap_block["contributions"] = contributions
    lime_block = _lime_result(["PAY_1 > 1.00"])
    payload = {
        "shap": shap_block,
        "lime": lime_block,
        "top_reasons": shap_explainer.top_reasons(contributions),
        "agreement": lime_explainer.agreement(shap_block, lime_block),
    }

    # numpy scalars are not JSON-serialisable; the failure would otherwise
    # surface as a 500 from the response encoder, far from this code.
    encoded = json.loads(json.dumps(payload))
    assert encoded["shap"]["contributions"][0]["feature"] == "PAY_1"
    assert isinstance(encoded["shap"]["contributions"][0]["shap"], float)


def test_build_contributions_orders_by_magnitude() -> None:
    frame = pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 3.0}])
    values = np.array([[0.1, -0.9, 0.4]])
    contributions = shap_explainer.build_contributions(frame, values)
    assert [item["feature"] for item in contributions] == ["b", "c", "a"]
    assert contributions[0]["value"] == pytest.approx(2.0)


# ------------------------------------------------------- degraded explainers


def test_shap_refuses_to_explain_when_no_model_is_loaded() -> None:
    model_loader.MODEL.unload()
    shap_explainer.reset()
    with pytest.raises(ExplainerUnavailable):
        shap_explainer.get_explainer()


def test_lime_refuses_to_explain_when_no_model_is_loaded() -> None:
    model_loader.MODEL.unload()
    lime_explainer.reset()
    with pytest.raises(ExplainerUnavailable):
        lime_explainer.explain_local({})


def test_missing_background_is_reported_not_guessed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    lime_explainer.reset()
    with pytest.raises(ExplainerUnavailable):
        lime_explainer.load_background()


def test_load_background_samples_a_parquet_of_features(tmp_path: Any) -> None:
    columns = list(FEATURE_NAMES)
    frame = pd.DataFrame(np.zeros((10, len(columns))), columns=columns)
    path = tmp_path / "features_train.parquet"
    frame.to_parquet(path)

    background = lime_explainer.load_background(path, n=5)
    assert background.shape == (5, len(columns))


def test_set_background_reorders_to_the_fitted_columns() -> None:
    columns = list(FEATURE_NAMES)
    frame = pd.DataFrame(np.arange(len(columns), dtype=float).reshape(1, -1), columns=columns)
    lime_explainer.set_background(frame[columns[::-1]])
    assert lime_explainer.background().tolist() == [list(range(len(columns)))]
    lime_explainer.reset()


def test_top_reasons_prints_a_value_it_cannot_read_as_a_number() -> None:
    reasons = shap_explainer.top_reasons(
        [{"feature": "worst_pay_status", "value": "n/a", "shap": 0.4}]
    )
    assert "(n/a)" in reasons[0]


def test_shap_refuses_a_model_tree_explainer_cannot_read() -> None:
    shap_explainer.reset()
    with pytest.raises(ExplainerUnavailable, match="TreeExplainer"):
        shap_explainer.get_explainer(model=object(), version="not-a-tree")


def test_explain_global_writes_a_beeswarm(tmp_path: Any) -> None:
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(schema.RANDOM_SEED)
    columns = list(FEATURE_NAMES)
    frame = pd.DataFrame(rng.normal(size=(120, len(columns))), columns=columns)
    target = (frame.iloc[:, 0].to_numpy() > 0).astype(int)
    model = LGBMClassifier(
        n_estimators=8,
        num_leaves=4,
        min_child_samples=5,
        random_state=schema.RANDOM_SEED,
        verbose=-1,
    )
    model.fit(frame, target)

    shap_explainer.reset()
    out = shap_explainer.explain_global(frame, tmp_path / "report" / "beeswarm.png", model=model)
    assert out.is_file()
    assert out.stat().st_size > 0
    shap_explainer.reset()


def test_predict_fn_squares_up_a_one_dimensional_probability() -> None:
    class OneDimensional:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), 0.25)

    proba = lime_explainer.predict_fn(OneDimensional())(np.zeros((3, len(FEATURE_NAMES))))
    assert proba.shape == (3, 2)
    assert proba[0].tolist() == [0.75, 0.25]


def test_set_background_accepts_a_bare_matrix() -> None:
    lime_explainer.set_background(np.zeros((4, len(FEATURE_NAMES))))
    assert lime_explainer.background().shape == (4, len(FEATURE_NAMES))
    lime_explainer.reset()


def test_background_is_loaded_from_disk_on_first_use(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    columns = list(FEATURE_NAMES)
    raw = pd.DataFrame(np.zeros((6, len(columns))), columns=columns)
    raw.to_parquet(tmp_path / "batch_05.parquet")
    monkeypatch.setattr(settings, "processed_dir", tmp_path)

    lime_explainer.reset()
    assert lime_explainer.background().shape == (6, len(columns))
    lime_explainer.reset()


def test_background_builds_features_when_the_parquet_holds_raw_columns(tmp_path: Any) -> None:
    from credit_risk.serving.models import EXAMPLE_APPLICATION

    record = {key: value for key, value in EXAMPLE_APPLICATION.items() if key != "account_id"}
    pd.DataFrame([record, record]).to_parquet(tmp_path / "raw_train.parquet")

    background = lime_explainer.load_background(tmp_path / "raw_train.parquet", n=2)
    assert background.shape == (2, len(FEATURE_NAMES))
