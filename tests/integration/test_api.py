"""End-to-end tests over the ASGI app with a stand-in model.

The model holder is monkeypatched to a tiny LightGBM fitted here, so the whole
file runs without MLflow, without MinIO and without a trained artefact. It is a
real tree rather than a mock because SHAP's TreeExplainer is part of what is
under test -- a mock would have proved only that the mock was called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.explain import lime_explainer, shap_explainer
from credit_risk.fairness import mitigation
from credit_risk.features.build import FEATURE_NAMES, FeatureBuildError
from credit_risk.serving import main as serving_main
from credit_risk.serving import metrics as m
from credit_risk.serving import model_loader, routes
from credit_risk.serving.main import app
from credit_risk.serving.models import (
    EXAMPLE_APPLICATION,
    ExplainResponse,
    FairnessReportResponse,
    HealthResponse,
    PredictResponse,
)

OPENAPI_DOCUMENT = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"

# Every metric name the alert rules in monitoring/ query. If one of these stops
# being exported, the alerts silently never fire -- which is worse than a
# broken alert, because a green dashboard is mistaken for a healthy system.
ALERT_METRIC_NAMES = (
    "credit_predictions_total",
    "credit_errors_total",
    "credit_prediction_latency_seconds",
    "credit_model_loaded",
    "credit_model_info",
    "credit_high_risk_share",
    "credit_selection_rate",
    "credit_feature_psi",
    "credit_explain_duration_seconds",
)


def build_stub_model() -> tuple[LGBMClassifier, pd.DataFrame]:
    """A small real tree over the production feature space, plus its background."""
    rng = np.random.default_rng(schema.RANDOM_SEED)
    columns = list(FEATURE_NAMES)
    frame = pd.DataFrame(rng.normal(size=(240, len(columns))), columns=columns)
    signal = frame.iloc[:, 0].to_numpy() + rng.normal(scale=0.5, size=len(frame))
    target = (signal > 0).astype(int)
    model = LGBMClassifier(
        n_estimators=12,
        num_leaves=4,
        min_child_samples=5,
        random_state=schema.RANDOM_SEED,
        verbose=-1,
    )
    model.fit(frame, target)
    return model, frame


def payload(**overrides: Any) -> dict[str, Any]:
    """The documented example, optionally tweaked."""
    return {**EXAMPLE_APPLICATION, **overrides}


def record(**overrides: Any) -> dict[str, Any]:
    """The same example as a scoring record: no account_id, which is not a feature."""
    return {key: value for key, value in payload(**overrides).items() if key != "account_id"}


def predictions_total() -> float:
    """credit_predictions_total summed over every decision label.

    Read through the public collect(), never through the private _value, so the
    test breaks when the exposition changes rather than when an internal does.
    """
    return sum(
        float(sample.value)
        for metric in m.PREDICTIONS.collect()
        for sample in metric.samples
        if sample.name == "credit_predictions_total"
    )


def exported_series(text: str, name: str) -> list[str]:
    """Sample lines for a metric, excluding its HELP and TYPE comments.

    `name in text` is satisfied by `# HELP credit_feature_psi ...`, which a
    metric that exports nothing at all still emits. That is how a permanently
    empty PSI gauge passed this file for as long as it did.
    """
    return [
        line for line in text.splitlines() if not line.startswith("#") and line.startswith(name)
    ]


@pytest.fixture
def loaded_client(monkeypatch: pytest.MonkeyPatch):
    """A client whose lifespan installs the stub model instead of calling MLflow."""
    model, background = build_stub_model()

    def fake_load(self: model_loader.ModelHolder) -> bool:
        self.install(model, version="test-7", algo="lightgbm", trained_at="2026-09-30T00:00:00Z")
        return True

    monkeypatch.setattr(model_loader.ModelHolder, "load", fake_load)
    shap_explainer.reset()
    lime_explainer.reset()
    lime_explainer.set_background(background)

    with TestClient(app) as client:
        yield client

    model_loader.MODEL.unload()
    shap_explainer.reset()
    lime_explainer.reset()


@pytest.fixture
def degraded_client(monkeypatch: pytest.MonkeyPatch):
    """A client that booted with an unreachable registry."""

    def fake_load(self: model_loader.ModelHolder) -> bool:
        self.unload()
        self.last_error = "MlflowException: registry unreachable"
        return False

    monkeypatch.setattr(model_loader.ModelHolder, "load", fake_load)
    with TestClient(app) as client:
        yield client
    model_loader.MODEL.unload()


# ------------------------------------------------------------------ health


def test_health_reports_which_model_is_serving(loaded_client: TestClient) -> None:
    body = loaded_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-7"
    assert body["algo"] == "lightgbm"
    assert body["model_name"] == settings.model_name
    assert body["uptime_seconds"] >= 0.0


def test_health_is_200_but_degraded_without_a_model(degraded_client: TestClient) -> None:
    response = degraded_client.get("/health")
    # 200 on purpose: a 503 here restarts the container and destroys the logs
    # that explain the outage.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "registry unreachable" in body["detail"]


# ----------------------------------------------------------------- predict


def test_predict_returns_the_documented_schema(loaded_client: TestClient) -> None:
    response = loaded_client.post("/api/v1/predict", json=payload())
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "account_id",
        "default_probability",
        "decision",
        "risk_band",
        "threshold_used",
        "threshold_policy",
        "model_name",
        "model_version",
        "served_at",
        "request_id",
    }
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"intervene", "monitor"}
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["account_id"] == EXAMPLE_APPLICATION["account_id"]
    assert body["model_version"] == "test-7"
    assert body["served_at"].endswith("Z")
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_predict_honours_an_upstream_request_id(loaded_client: TestClient) -> None:
    response = loaded_client.post(
        "/api/v1/predict", json=payload(), headers={"X-Request-ID": "abc123"}
    )
    assert response.json()["request_id"] == "abc123"


def test_predict_decision_follows_the_threshold(loaded_client: TestClient) -> None:
    body = loaded_client.post("/api/v1/predict", json=payload()).json()
    expected = "intervene" if body["default_probability"] >= body["threshold_used"] else "monitor"
    assert body["decision"] == expected


# ------------------------------------------------------------------- batch


def test_batch_accepts_the_documented_maximum(loaded_client: TestClient) -> None:
    applications = [payload(account_id=f"A-{index}") for index in range(settings.max_batch_size)]
    response = loaded_client.post("/api/v1/predict/batch", json={"applications": applications})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == settings.max_batch_size
    assert len(body["predictions"]) == settings.max_batch_size
    assert body["intervene_count"] == sum(
        1 for item in body["predictions"] if item["decision"] == "intervene"
    )
    assert body["high_risk_count"] == sum(
        1 for item in body["predictions"] if item["risk_band"] == "high"
    )
    assert body["predictions"][0]["account_id"] == "A-0"
    assert all(0.0 <= item["default_probability"] <= 1.0 for item in body["predictions"])


def test_batch_rejects_one_over_the_maximum(loaded_client: TestClient) -> None:
    applications = [payload()] * (settings.max_batch_size + 1)
    response = loaded_client.post("/api/v1/predict/batch", json={"applications": applications})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


# ----------------------------------------------------------------- explain


def test_explain_returns_shap_lime_and_agreement(loaded_client: TestClient) -> None:
    response = loaded_client.post("/api/v1/explain", json=payload())
    assert response.status_code == 200
    body = response.json()

    assert body["shap"]["contributions"]
    assert {"feature", "value", "shap"} == set(body["shap"]["contributions"][0])
    assert body["lime"]["contributions"]
    assert {"feature", "weight"} == set(body["lime"]["contributions"][0])
    assert 1 <= len(body["top_reasons"]) <= 3
    assert all(reason.endswith(".") for reason in body["top_reasons"])
    assert body["agreement"]["note"]
    assert isinstance(body["shap"]["base_value"], float)

    # Recomputed from what was actually returned. Asserting a bound the response
    # model already enforces (`ge=0`, and TOP_K pins the ceiling) proved nothing
    # about whether the two rankings had been compared at all.
    shap_top = [item["feature"] for item in body["shap"]["contributions"][:3]]
    lime_ranked = sorted(
        body["lime"]["contributions"], key=lambda item: abs(item["weight"]), reverse=True
    )
    lime_top = [lime_explainer.base_feature(item["feature"]) for item in lime_ranked[:3]]
    assert body["agreement"]["top3_overlap"] == len(set(shap_top) & set(lime_top))


def test_explain_is_reproducible_for_the_same_record(loaded_client: TestClient) -> None:
    first = loaded_client.post("/api/v1/explain", json=payload()).json()
    second = loaded_client.post("/api/v1/explain", json=payload()).json()
    # LIME is stochastic; the per-request re-seed is what makes this hold.
    assert first["lime"]["contributions"] == second["lime"]["contributions"]


# ---------------------------------------------------------------- fairness


def test_fairness_report_covers_every_protected_group(loaded_client: TestClient) -> None:
    loaded_client.post("/api/v1/predict", json=payload(SEX=1))
    loaded_client.post("/api/v1/predict", json=payload(SEX=2))

    body = loaded_client.get("/api/v1/fairness/report").json()
    assert body["protected_attribute"] == schema.PRIMARY_PROTECTED
    groups = {item["group"] for item in body["groups"]}
    assert {"sex_male", "sex_female"} <= groups
    assert body["max_demographic_parity_difference"] == schema.MAX_DEMOGRAPHIC_PARITY_DIFF
    assert body["max_equalized_odds_difference"] == schema.MAX_EQUALIZED_ODDS_DIFF
    assert body["note"]


# --------------------------------------------------------------------- ops


def test_metrics_is_plain_text_and_exports_every_alerted_series(
    loaded_client: TestClient,
) -> None:
    m.reset_windows()

    # Enough traffic for every gauge that is computed over a window rather than
    # set directly: PSI refreshes once per PSI_REFRESH_EVERY observations and
    # needs PSI_MIN_SAMPLES in the window first, and a protected group is only
    # published once it holds SELECTION_RATE_MIN_SAMPLES slots.
    volume = max(m.PSI_REFRESH_EVERY, m.PSI_MIN_SAMPLES, m.SELECTION_RATE_MIN_SAMPLES)
    loaded_client.post(
        "/api/v1/predict/batch",
        json={"applications": [payload(account_id=f"A-{index}") for index in range(volume)]},
    )
    loaded_client.post("/api/v1/explain", json=payload())
    # credit_errors_total has no children until something fails, so the series
    # has to be earned here rather than inherited from whichever test ran first.
    loaded_client.post("/api/v1/predict", json=payload(AGE="not a number"))

    response = loaded_client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for name in ALERT_METRIC_NAMES:
        assert exported_series(
            response.text, name
        ), f"{name} is queried by an alert rule but exports no series"


def test_a_prediction_increments_the_counter_exactly_once(loaded_client: TestClient) -> None:
    # credit_predictions_total is the denominator of HighErrorRate. Counted
    # twice, a 10% error rate reads as 5% and never crosses the 5% threshold --
    # and nothing about the dashboard looks wrong while that is true.
    before = predictions_total()
    loaded_client.post("/api/v1/predict", json=payload())
    assert predictions_total() == before + 1

    before = predictions_total()
    loaded_client.post(
        "/api/v1/predict/batch", json={"applications": [payload(), payload(), payload()]}
    )
    assert predictions_total() == before + 3


def test_the_drift_gauge_reports_once_there_is_enough_traffic(
    loaded_client: TestClient,
) -> None:
    # The lifespan installs the PSI baseline; without it observe_features
    # returns immediately, credit_feature_psi never gets a child, and
    # FeatureDriftHigh cannot fire no matter how far the population moves.
    m.reset_windows()
    applications = [payload(account_id=f"A-{index}") for index in range(m.PSI_REFRESH_EVERY)]
    loaded_client.post("/api/v1/predict/batch", json={"applications": applications})

    series = exported_series(loaded_client.get("/metrics").text, "credit_feature_psi")
    assert series, "no credit_feature_psi sample after enough traffic to refresh it"
    reported = {line.split('feature="')[1].split('"')[0] for line in series}
    assert reported == set(serving_main.DRIFT_WATCH_FEATURES)


def test_every_watched_feature_is_one_the_builder_produces() -> None:
    # A typo here does not fail loudly: set_baseline would raise inside the
    # lifespan's own try/except and the service would boot with no drift gauge
    # and one WARNING line nobody reads.
    assert set(serving_main.DRIFT_WATCH_FEATURES) <= set(FEATURE_NAMES)
    assert len(serving_main.DRIFT_WATCH_FEATURES) <= m.MAX_PSI_FEATURES


def test_the_lifespan_publishes_exactly_one_model_info_series(
    loaded_client: TestClient,
) -> None:
    series = exported_series(loaded_client.get("/metrics").text, "credit_model_info")
    assert len(series) == 1
    assert 'version="test-7"' in series[0]


def test_version_reports_api_and_model_build(loaded_client: TestClient) -> None:
    body = loaded_client.get("/version").json()
    assert body["api_version"] == settings.api_version
    assert body["model_name"] == settings.model_name
    assert body["model_version"] == "test-7"
    assert body["git_sha"] == settings.git_sha
    assert body["threshold_policy"] == settings.threshold_policy


def test_openapi_documents_predict_with_a_usable_example(loaded_client: TestClient) -> None:
    spec = loaded_client.get("/openapi.json").json()
    assert "/api/v1/predict" in spec["paths"]
    assert "/api/v1/predict/batch" in spec["paths"]
    assert "/api/v1/explain" in spec["paths"]
    assert "/api/v1/fairness/report" in spec["paths"]
    # The example has to survive into the published schema, otherwise /docs
    # shows a form of 23 zeros that nobody can paste anywhere useful.
    assert str(EXAMPLE_APPLICATION["account_id"]) in json.dumps(spec)


def test_docs_are_served(loaded_client: TestClient) -> None:
    assert loaded_client.get("/docs").status_code == 200


@pytest.mark.parametrize(
    "response_model", [PredictResponse, ExplainResponse, FairnessReportResponse, HealthResponse]
)
def test_every_response_example_is_a_valid_response(response_model: Any) -> None:
    # An example that no longer matches its model is worse than none: /docs
    # would show a client a body the API can never send.
    examples = response_model.model_json_schema()["examples"]
    assert examples
    for example in examples:
        assert response_model.model_validate(example).model_dump(mode="json") == example


def test_openapi_publishes_a_response_example_for_each_documented_answer(
    loaded_client: TestClient,
) -> None:
    schemas = loaded_client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("PredictResponse", "ExplainResponse", "FairnessReportResponse", "HealthResponse"):
        assert schemas[name]["examples"], name


def test_the_committed_openapi_document_matches_the_app() -> None:
    # docs/openapi.json lets a reviewer read the contract without booting the
    # stack, which is only worth anything while it is the contract.
    committed = json.loads(OPENAPI_DOCUMENT.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(app.openapi()))
    assert committed == current, "docs/openapi.json is stale; regenerate it with `make openapi`"


# ------------------------------------------------- pure helpers under /serving


def test_risk_band_is_anchored_on_the_portfolio_base_rate() -> None:
    assert routes.risk_band(0.90) == "high"
    assert routes.risk_band(routes.HIGH_RISK_PROBABILITY) == "high"
    assert routes.risk_band(schema.BASE_POSITIVE_RATE) == "medium"
    assert routes.risk_band(0.01) == "low"


def test_decide_is_inclusive_at_the_threshold() -> None:
    assert routes.decide(0.42, 0.42) == "intervene"
    assert routes.decide(0.41, 0.42) == "monitor"


def test_group_label_never_leaks_an_account() -> None:
    assert routes.group_label({schema.SEX: 1}) == "sex_male"
    assert routes.group_label({schema.SEX: 2}) == "sex_female"
    assert routes.group_label({}) == "sex_unknown"
    assert routes.group_label({schema.SEX: "x"}) == "sex_unknown"


def test_base_policy_applies_one_threshold_to_everyone() -> None:
    male, policy = routes.applied_threshold({schema.SEX: 1})
    female, _ = routes.applied_threshold({schema.SEX: 2})
    assert male == female == settings.decision_threshold
    assert policy == settings.threshold_policy


def test_group_aware_policy_applies_the_fitted_cutoffs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "threshold_policy", routes.GROUP_AWARE_POLICY)
    routes.set_group_thresholds({"1": 0.31, "2": 0.55})
    try:
        male, policy = routes.applied_threshold({schema.SEX: 1})
        female, _ = routes.applied_threshold({schema.SEX: 2})
        assert (male, female) == (0.31, 0.55)
        assert policy == routes.GROUP_AWARE_POLICY

        # A group nobody fitted a cutoff for is judged by the base threshold,
        # not silently approved or silently rejected.
        unseen, fallback = routes.applied_threshold({schema.SEX: 9})
        assert unseen == settings.decision_threshold
        assert fallback == "base"
    finally:
        routes.set_group_thresholds(None)


def test_group_thresholds_come_from_the_pipeline_handoff(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Written through mitigation's own path helper: serving and training have to
    # agree on the filename, and a test that spells it out itself would keep
    # passing after one side renamed it.
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    mitigation.save_group_thresholds({"1": 0.4, "2": 0.6})
    routes.set_group_thresholds(None)
    try:
        assert dict(routes.group_threshold_map()) == {"1": 0.4, "2": 0.6}
    finally:
        routes.set_group_thresholds(None)


def test_missing_group_thresholds_fall_back_rather_than_fail(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    routes.set_group_thresholds(None)
    try:
        assert dict(routes.group_threshold_map()) == {}
    finally:
        routes.set_group_thresholds(None)


def test_model_holder_names_the_estimator_when_the_registry_is_terse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from credit_risk.models import registry

    estimator = object()
    monkeypatch.setattr(registry, "load_production_model", lambda **kwargs: estimator)
    monkeypatch.setattr(
        registry,
        "production_version_metadata",
        lambda **kwargs: registry.VersionMetadata(model_name="credit-risk", version="7"),
    )

    holder = model_loader.ModelHolder()
    assert holder.load() is True
    assert holder.version == "7"
    assert holder.algo == "object"


def test_model_holder_degrades_when_the_registry_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from credit_risk.models import registry

    monkeypatch.setattr(registry, "load_production_model", lambda **kwargs: None)
    monkeypatch.setattr(registry, "production_version_metadata", lambda **kwargs: None)

    holder = model_loader.ModelHolder()
    assert holder.load() is False
    assert holder.is_loaded is False
    assert "no model" in (holder.last_error or "")


def test_describe_version_accepts_a_string_a_mapping_and_an_object() -> None:
    assert model_loader.describe_version("7") == ("7", "unknown", "unknown")
    assert model_loader.describe_version(None) == ("unknown", "unknown", "unknown")

    mapping = {"version": "9", "algo": "lightgbm", "trained_at": "2026-09-30"}
    assert model_loader.describe_version(mapping) == ("9", "lightgbm", "2026-09-30")

    class Info:
        version = "11"
        algorithm = "logreg"
        created_at = "2026-01-01"

    assert model_loader.describe_version(Info()) == ("11", "logreg", "2026-01-01")


def test_positive_class_index_reads_classes_rather_than_guessing() -> None:
    class Flipped:
        classes_ = np.array([1, 0])

    class Normal:
        classes_ = np.array([0, 1])

    class Unlabelled:
        classes_ = np.array(["a", "b", "c"])

    assert model_loader.positive_class_index(Flipped()) == 0
    assert model_loader.positive_class_index(Normal()) == 1
    assert model_loader.positive_class_index(object()) == 1
    # Nothing recognisable as the positive class: take the last column, which
    # is what every binary sklearn estimator means by it.
    assert model_loader.positive_class_index(Unlabelled()) == 2


def test_predict_probability_picks_the_positive_column() -> None:
    class Flipped:
        classes_ = np.array([1, 0])

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([[0.8, 0.2]])

    frame = pd.DataFrame([{"a": 1.0}])
    assert model_loader.predict_probability(Flipped(), frame)[0] == pytest.approx(0.8)


def test_frame_from_features_rejects_something_it_cannot_use() -> None:
    with pytest.raises(TypeError):
        model_loader.frame_from_features("not features")


def test_unreadable_group_thresholds_fall_back_to_the_base_cutoff(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    mitigation.group_thresholds_path().write_text("{not json", encoding="utf-8")
    routes.set_group_thresholds(None)
    try:
        assert dict(routes.group_threshold_map()) == {}
    finally:
        routes.set_group_thresholds(None)


def test_feature_row_skips_columns_that_are_not_numbers() -> None:
    frame = pd.DataFrame([{"utilization_mean": 0.9, "note": "not a number"}])
    assert routes.feature_row(frame) == {"utilization_mean": 0.9}


def test_fairness_report_includes_a_group_the_codebook_does_not_know(
    loaded_client: TestClient,
) -> None:
    # A group that only shows up in traffic still has to appear in the report;
    # an unmonitored group is exactly where a fairness problem would hide.
    m.SELECTION_RATE.labels(group="sex_unknown").set(0.31)
    body = loaded_client.get("/api/v1/fairness/report").json()
    groups = {item["group"]: item for item in body["groups"]}
    assert "sex_unknown" in groups
    assert groups["sex_unknown"]["selection_rate"] == pytest.approx(0.31)
    assert body["demographic_parity_difference"] is not None


def test_order_columns_leaves_an_unrecognised_frame_alone() -> None:
    frame = pd.DataFrame([{"b": 1.0, "a": 2.0}])
    assert list(model_loader.order_columns(frame).columns) == ["b", "a"]


def test_frame_from_features_accepts_the_shapes_a_builder_might_return() -> None:
    columns = list(FEATURE_NAMES)
    values = list(range(len(columns)))

    from_mapping = model_loader.frame_from_features(dict(zip(columns, values, strict=True)))
    from_series = model_loader.frame_from_features(pd.Series(values, index=columns))
    from_array = model_loader.frame_from_features(np.array(values, dtype=float))

    for frame in (from_mapping, from_series, from_array):
        assert list(frame.columns) == columns
        assert frame.shape == (1, len(columns))


def test_batch_feature_frame_builds_every_record_in_one_pass() -> None:
    frame = model_loader.to_feature_frame_many([record(), record(SEX=2)])
    assert frame.shape == (2, len(FEATURE_NAMES))
    assert list(frame.columns) == list(FEATURE_NAMES)
    # One frame built once, so the two rows have to agree with the single-record
    # path they would otherwise silently diverge from.
    single = model_loader.to_feature_frame(record())
    assert frame.iloc[0].tolist() == single.iloc[0].tolist()


def test_batch_feature_frame_rejects_an_empty_batch() -> None:
    # BatchPredictRequest forbids it, so this only happens when something
    # reaches past the schema -- where pandas would otherwise raise "No objects
    # to concatenate" from four frames down the stack.
    with pytest.raises(FeatureBuildError):
        model_loader.to_feature_frame_many([])


def test_batch_feature_frame_names_a_column_the_records_are_missing() -> None:
    incomplete = record()
    del incomplete[schema.LIMIT_BAL]
    with pytest.raises(FeatureBuildError, match=schema.LIMIT_BAL):
        model_loader.to_feature_frame_many([incomplete])


def test_predict_probability_handles_one_dimensional_and_single_column_output() -> None:
    class OneDimensional:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([0.25])

    class SingleColumn:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([[0.25]])

    frame = pd.DataFrame([{"a": 1.0}])
    assert model_loader.predict_probability(OneDimensional(), frame)[0] == pytest.approx(0.25)
    assert model_loader.predict_probability(SingleColumn(), frame)[0] == pytest.approx(0.25)


def test_model_holder_unpacks_a_registry_that_returns_model_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from credit_risk.models import registry

    estimator = object()
    metadata = {"version": "12", "algo": "lightgbm", "trained_at": "2026-09-01"}
    monkeypatch.setattr(registry, "load_production_model", lambda **kwargs: (estimator, metadata))
    monkeypatch.setattr(registry, "production_version_metadata", lambda **kwargs: None)

    holder = model_loader.ModelHolder()
    assert holder.load() is True
    assert (holder.version, holder.algo, holder.trained_at) == ("12", "lightgbm", "2026-09-01")
