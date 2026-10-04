"""Failure paths. Every one of them is asserted as a metric, not just a status.

A 422 that nobody counts is invisible to Prometheus, which means the
HighErrorRate alert never fires and a broken upstream looks like reduced
traffic. These tests exist so that cannot regress quietly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from credit_risk import schema
from credit_risk.features.build import FEATURE_NAMES
from credit_risk.serving import metrics as m
from credit_risk.serving import model_loader, routes
from credit_risk.serving.main import app
from credit_risk.serving.models import EXAMPLE_APPLICATION

ERROR_FIELDS = {"code", "message", "detail", "request_id"}


def counter_value(counter: Any, **labels: str) -> float:
    """Read a counter through its public collect(), never through _value."""
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total") and sample.labels == labels:
                return float(sample.value)
    return 0.0


def build_stub_model() -> LGBMClassifier:
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
    return model


def payload(**overrides: Any) -> dict[str, Any]:
    return {**EXAMPLE_APPLICATION, **overrides}


@pytest.fixture
def loaded_client(monkeypatch: pytest.MonkeyPatch):
    model = build_stub_model()

    def fake_load(self: model_loader.ModelHolder) -> bool:
        self.install(model, version="test-7", algo="lightgbm", trained_at="2026-09-30T00:00:00Z")
        return True

    monkeypatch.setattr(model_loader.ModelHolder, "load", fake_load)
    # raise_server_exceptions=False so the global handler's response is
    # returned instead of the exception being re-raised into the test.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    model_loader.MODEL.unload()


@pytest.fixture
def degraded_client(monkeypatch: pytest.MonkeyPatch):
    def fake_load(self: model_loader.ModelHolder) -> bool:
        self.unload()
        self.last_error = "ConnectionError: registry unreachable"
        return False

    monkeypatch.setattr(model_loader.ModelHolder, "load", fake_load)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    model_loader.MODEL.unload()


def test_missing_field_is_422_and_counted(loaded_client: TestClient) -> None:
    before = counter_value(m.ERRORS, reason="validation")

    body = payload()
    del body["PAY_1"]
    response = loaded_client.post("/api/v1/predict", json=body)

    assert response.status_code == 422
    assert counter_value(m.ERRORS, reason="validation") == before + 1
    error = response.json()
    assert set(error) == ERROR_FIELDS
    assert error["code"] == "validation_error"
    assert "PAY_1" in error["detail"]


def test_validation_detail_never_echoes_the_submitted_values(loaded_client: TestClient) -> None:
    response = loaded_client.post("/api/v1/predict", json=payload(AGE=None))
    assert response.status_code == 422
    # FastAPI's default 422 body includes the offending input. On credit data
    # that would copy a customer into every client-side log.
    assert str(EXAMPLE_APPLICATION["LIMIT_BAL"]) not in response.text
    assert "AGE" in response.json()["detail"]


def test_wrong_type_is_422_and_counted(loaded_client: TestClient) -> None:
    before = counter_value(m.ERRORS, reason="validation")
    response = loaded_client.post("/api/v1/predict", json=payload(AGE="thirty-nine"))

    assert response.status_code == 422
    assert counter_value(m.ERRORS, reason="validation") == before + 1
    assert response.json()["code"] == "validation_error"


def test_age_outside_the_observed_range_is_422(loaded_client: TestClient) -> None:
    before = counter_value(m.ERRORS, reason="validation")
    too_young = loaded_client.post("/api/v1/predict", json=payload(AGE=schema.AGE_MIN - 1))
    too_old = loaded_client.post("/api/v1/predict", json=payload(AGE=schema.AGE_MAX + 1))

    assert too_young.status_code == 422
    assert too_old.status_code == 422
    assert counter_value(m.ERRORS, reason="validation") == before + 2


def test_unknown_field_is_rejected_rather_than_ignored(loaded_client: TestClient) -> None:
    # The raw extract calls the first repayment column PAY_0. Silently dropping
    # it would score the account with a missing feature.
    response = loaded_client.post("/api/v1/predict", json=payload(PAY_0=2))
    assert response.status_code == 422
    assert "PAY_0" in response.json()["detail"]


def test_predict_is_503_and_counted_when_no_model_is_loaded(degraded_client: TestClient) -> None:
    before = counter_value(m.ERRORS, reason="model_not_loaded")
    response = degraded_client.post("/api/v1/predict", json=payload())

    assert response.status_code == 503
    assert counter_value(m.ERRORS, reason="model_not_loaded") == before + 1
    error = response.json()
    assert set(error) == ERROR_FIELDS
    assert error["code"] == "model_not_loaded"
    assert "registry unreachable" in error["detail"]
    assert error["request_id"]


def test_every_inference_route_is_503_while_degraded(degraded_client: TestClient) -> None:
    assert degraded_client.post("/api/v1/explain", json=payload()).status_code == 503
    batch = degraded_client.post("/api/v1/predict/batch", json={"applications": [payload()]})
    assert batch.status_code == 503
    assert degraded_client.get("/api/v1/fairness/report").status_code == 503


def test_scoring_failure_is_500_with_its_own_reason(
    loaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(model: Any, frame: pd.DataFrame) -> np.ndarray:
        raise ValueError("estimator refused the frame")

    monkeypatch.setattr(model_loader, "predict_probability", boom)
    before = counter_value(m.ERRORS, reason="prediction_failed")
    response = loaded_client.post("/api/v1/predict", json=payload())

    assert response.status_code == 500
    assert counter_value(m.ERRORS, reason="prediction_failed") == before + 1
    assert response.json()["code"] == "prediction_failed"


def test_unhandled_exception_becomes_a_500_error_response(
    loaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(probability: float) -> str:
        raise RuntimeError("something nobody anticipated")

    monkeypatch.setattr(routes, "risk_band", boom)
    before = counter_value(m.ERRORS, reason="internal")
    response = loaded_client.post("/api/v1/predict", json=payload())

    assert response.status_code == 500
    assert counter_value(m.ERRORS, reason="internal") == before + 1
    error = response.json()
    assert set(error) == ERROR_FIELDS
    assert error["code"] == "internal_error"
    assert error["request_id"]
    # The stack trace stays in the logs; the client gets a class name at most.
    assert "something nobody anticipated" not in response.text


def test_explainer_failure_is_503_and_counted(
    loaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from credit_risk.explain import shap_explainer

    def unavailable(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise shap_explainer.ExplainerUnavailable("no background sample on disk")

    monkeypatch.setattr(shap_explainer, "explain_local", unavailable)
    before = counter_value(m.ERRORS, reason="explainer_unavailable")
    response = loaded_client.post("/api/v1/explain", json=payload())

    assert response.status_code == 503
    assert counter_value(m.ERRORS, reason="explainer_unavailable") == before + 1
    assert response.json()["code"] == "explainer_unavailable"


def test_empty_batch_is_rejected(loaded_client: TestClient) -> None:
    response = loaded_client.post("/api/v1/predict/batch", json={"applications": []})
    assert response.status_code == 422


def test_batch_scoring_failure_is_500_with_its_own_reason(
    loaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(records: Any) -> pd.DataFrame:
        raise ValueError("feature builder rejected the batch")

    monkeypatch.setattr(model_loader, "to_feature_frame_many", boom)
    before = counter_value(m.ERRORS, reason="prediction_failed")
    response = loaded_client.post("/api/v1/predict/batch", json={"applications": [payload()]})

    assert response.status_code == 500
    assert counter_value(m.ERRORS, reason="prediction_failed") == before + 1
    assert response.json()["code"] == "prediction_failed"


def test_an_explainer_blowing_up_is_500_not_503(
    loaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from credit_risk.explain import lime_explainer

    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("lime could not fit a surrogate")

    monkeypatch.setattr(lime_explainer, "explain_local", boom)
    before = counter_value(m.ERRORS, reason="explain_failed")
    response = loaded_client.post("/api/v1/explain", json=payload())

    # 503 says "come back later"; this is a bug, and the label has to say so.
    assert response.status_code == 500
    assert counter_value(m.ERRORS, reason="explain_failed") == before + 1
    assert response.json()["code"] == "explain_failed"


def test_explain_reports_a_scoring_failure_before_it_reaches_an_explainer(
    loaded_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(model: Any, frame: pd.DataFrame) -> np.ndarray:
        raise ValueError("estimator refused the frame")

    monkeypatch.setattr(model_loader, "predict_probability", boom)
    before = counter_value(m.ERRORS, reason="prediction_failed")
    response = loaded_client.post("/api/v1/explain", json=payload())

    assert response.status_code == 500
    assert counter_value(m.ERRORS, reason="prediction_failed") == before + 1
    assert response.json()["code"] == "prediction_failed"


def test_an_unknown_path_is_404_in_the_error_shape(loaded_client: TestClient) -> None:
    before = {
        reason: counter_value(m.ERRORS, reason=reason) for reason in ("validation", "internal")
    }
    response = loaded_client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    error = response.json()
    assert set(error) == ERROR_FIELDS
    assert error["code"] == "not_found"
    assert "/api/v1" in error["message"]
    assert error["request_id"] == response.headers["X-Request-ID"]
    # A probe for a path that does not exist is not the service failing.
    assert {
        reason: counter_value(m.ERRORS, reason=reason) for reason in ("validation", "internal")
    } == before


def test_a_wrong_method_is_405_in_the_error_shape_and_keeps_allow(
    loaded_client: TestClient,
) -> None:
    response = loaded_client.get("/api/v1/predict")

    assert response.status_code == 405
    error = response.json()
    assert set(error) == ERROR_FIELDS
    assert error["code"] == "method_not_allowed"
    # RFC 9110 requires Allow on a 405; rebuilding the body must not drop it.
    assert "POST" in response.headers["Allow"]
