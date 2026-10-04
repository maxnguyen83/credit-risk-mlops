"""The cutoff the API decides at: where it comes from, and that it holds capacity.

Three layers, cheapest first. `resolve_threshold` is pure and every branch is
pinned here. The loader is driven with a stubbed registry to show it reads the
version's tags, loads that exact version, and reports a fallback instead of
hiding it. Then the whole path runs for real -- a model trained, logged and
registered into a throwaway SQLite MLflow, promoted, loaded by the API's own
lifespan, and a held-out pool scored through `/predict/batch` -- to show the
served selection rate lands on the 10% capacity rather than wherever 0.5 falls.

The real-data version of that last test, on the `serving_pool` split, lives in
`tests/model/test_served_capacity.py`, because CI runs the `needs_data` model
tests after the dataset is downloaded and runs nothing else that way.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.explain import lime_explainer, shap_explainer
from credit_risk.features.build import FEATURE_NAMES, REQUIRED_INPUT_COLUMNS
from credit_risk.models import registry
from credit_risk.models.evaluate import evaluate
from credit_risk.models.train import feature_frame, make_lightgbm
from credit_risk.serving import main as serving_main
from credit_risk.serving import metrics as m
from credit_risk.serving import model_loader, routes
from credit_risk.serving.main import app
from credit_risk.serving.model_loader import DecisionThreshold, resolve_threshold
from tests.integration.test_api import build_stub_model, payload
from tests.model.test_performance import FAIR_ENOUGH, synthetic_clean_frame

CAPACITY = 0.10
# The tolerance the capacity has to be held to on accounts the threshold was not
# computed on. Sampling noise on 4,000-5,000 accounts is about half of it.
CAPACITY_TOLERANCE = 0.01

REGISTRY_CUTOFF = 0.6123


# ------------------------------------------------------- resolve_threshold


def test_a_tagged_version_decides_at_its_own_cutoff() -> None:
    chosen = resolve_threshold("0.5166605772575003", source="registry", configured=0.5)
    assert chosen == DecisionThreshold(0.5166605772575003, "registry")


def test_env_mode_ignores_the_tag() -> None:
    chosen = resolve_threshold("0.5166605772575003", source="env", configured=0.42)
    assert chosen == DecisionThreshold(0.42, "env")


@pytest.mark.parametrize(
    ("tag", "why"),
    [
        (None, "carries no threshold_at_k tag"),
        ("not-a-number", "is not a number"),
        ("1.5", "is not a probability"),
        ("-0.1", "is not a probability"),
        ("nan", "is not a probability"),
        ("inf", "is not a probability"),
    ],
)
def test_a_missing_or_broken_tag_falls_back_and_says_why(tag: str | None, why: str) -> None:
    chosen = resolve_threshold(tag, source="registry", configured=0.5)
    assert chosen.value == 0.5
    assert chosen.source == "fallback"
    assert chosen.reason is not None and why in chosen.reason


def test_the_settings_choose_when_the_caller_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "decision_threshold", 0.37)
    monkeypatch.setattr(settings, "threshold_source", "env")
    assert resolve_threshold("0.6") == DecisionThreshold(0.37, "env")
    monkeypatch.setattr(settings, "threshold_source", "registry")
    assert resolve_threshold("0.6") == DecisionThreshold(0.6, "registry")
    assert resolve_threshold(None).value == 0.37


def test_an_empty_holder_reports_the_constant_and_why() -> None:
    holder = model_loader.ModelHolder()
    assert holder.threshold == settings.decision_threshold
    assert holder.threshold_source == "fallback"
    assert holder.decision.reason == "no model is loaded"


# ------------------------------------------------------------------ loader


def stub_registry(
    monkeypatch: pytest.MonkeyPatch, tags: dict[str, str], version: str = "2"
) -> dict[str, Any]:
    """Point the loader at a registry that holds one tagged version."""
    calls: dict[str, Any] = {}
    estimator = object()

    def load(**kwargs: Any) -> object:
        calls.update(kwargs)
        return estimator

    metadata = registry.VersionMetadata(
        model_name=settings.model_name, version=version, run_id="run-2", tags=tags
    )
    monkeypatch.setattr(registry, "production_version_metadata", lambda **kwargs: metadata)
    monkeypatch.setattr(registry, "load_production_model", load)
    return calls


def test_the_loader_reads_the_threshold_tag_of_the_version_it_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_registry(
        monkeypatch,
        {
            registry.THRESHOLD_TAG: "0.5166605772575003",
            registry.TRAINED_AT_TAG: "2026-10-03T02:40:12Z",
        },
    )

    holder = model_loader.ModelHolder()
    assert holder.load() is True

    assert holder.threshold == 0.5166605772575003
    assert holder.threshold_source == "registry"
    assert holder.run_id == "run-2"
    assert holder.trained_at == "2026-10-03T02:40:12Z"
    # Loaded by the version's own URI, so the tags just read describe this
    # estimator even if another version is promoted a moment later.
    assert calls["model_uri"] == f"models:/{settings.model_name}/2"


def test_an_untagged_version_falls_back_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stub_registry(monkeypatch, {})

    holder = model_loader.ModelHolder()
    with caplog.at_level(logging.WARNING, logger=model_loader.log.name):
        assert holder.load() is True

    assert holder.threshold == settings.decision_threshold
    assert holder.threshold_source == "fallback"
    # The log line says what to run, not just that something is off.
    assert "tag-threshold --version 2" in caplog.text


def test_env_mode_decides_at_the_constant_even_when_the_tag_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "threshold_source", "env")
    stub_registry(monkeypatch, {registry.THRESHOLD_TAG: "0.5166605772575003"})

    holder = model_loader.ModelHolder()
    assert holder.load() is True

    assert holder.threshold == settings.decision_threshold
    assert holder.threshold_source == "env"


def test_a_failed_load_resets_the_threshold_with_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = model_loader.ModelHolder()
    holder.install(object(), version="9", threshold=DecisionThreshold(0.61, "registry"))
    monkeypatch.setattr(registry, "production_version_metadata", lambda **kwargs: None)
    monkeypatch.setattr(registry, "load_production_model", lambda **kwargs: None)

    assert holder.load() is False

    assert holder.threshold == settings.decision_threshold
    assert holder.threshold_source == "fallback"
    assert holder.run_id == model_loader.UNKNOWN


# --------------------------------------------------------------------- API


@pytest.fixture
def tagged_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The API serving a version whose registry tag says REGISTRY_CUTOFF."""
    model, background = build_stub_model()

    def fake_load(self: model_loader.ModelHolder) -> bool:
        self.install(
            model,
            version="9",
            algo="lightgbm",
            trained_at="2026-10-03T02:40:12Z",
            run_id="run-9",
            threshold=DecisionThreshold(REGISTRY_CUTOFF, "registry"),
        )
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


def test_health_reports_the_run_the_threshold_and_where_it_came_from(
    tagged_client: TestClient,
) -> None:
    body = tagged_client.get("/health").json()
    assert body["model_version"] == "9"
    assert body["run_id"] == "run-9"
    assert body["threshold"] == REGISTRY_CUTOFF
    assert body["threshold_source"] == "registry"
    # Additive: every field /health carried before is still there.
    for field in ("status", "model_loaded", "algo", "trained_at", "threshold_policy", "detail"):
        assert field in body


def test_every_decision_is_made_at_the_loaded_threshold(tagged_client: TestClient) -> None:
    single = tagged_client.post("/api/v1/predict", json=payload()).json()
    batch = tagged_client.post(
        "/api/v1/predict/batch",
        json={"applications": [payload(SEX=1), payload(SEX=2, PAY_1=0, PAY_2=0)]},
    ).json()
    explained = tagged_client.post("/api/v1/explain", json=payload()).json()

    for body in (single, explained, *batch["predictions"]):
        assert body["threshold_used"] == REGISTRY_CUTOFF
        expected = "intervene" if body["default_probability"] >= REGISTRY_CUTOFF else "monitor"
        assert body["decision"] == expected

    report = tagged_client.get("/api/v1/fairness/report").json()
    assert {group["threshold"] for group in report["groups"]} == {REGISTRY_CUTOFF}


def test_the_high_risk_share_counts_against_the_loaded_threshold(
    tagged_client: TestClient,
) -> None:
    # The live share and its baseline must describe the cutoff decisions are
    # made at; otherwise HighRiskShareShift compares two different questions.
    assert m.decision_threshold() == REGISTRY_CUTOFF

    background = lime_explainer.background()
    frame = pd.DataFrame(background, columns=list(FEATURE_NAMES))
    proba = model_loader.predict_probability(model_loader.MODEL.model, frame)
    expected = float((proba >= REGISTRY_CUTOFF).mean())
    assert serving_main.prime_high_risk_baseline(frame) == pytest.approx(expected)


def test_group_aware_policy_falls_back_to_the_loaded_threshold(
    tagged_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "threshold_policy", routes.GROUP_AWARE_POLICY)
    routes.set_group_thresholds({"1": 0.31})
    try:
        assert routes.applied_threshold({schema.SEX: 1}) == (0.31, routes.GROUP_AWARE_POLICY)
        # No fitted cutoff for SEX=2: judged by this model's base cutoff, not by
        # a constant from configuration.
        assert routes.applied_threshold({schema.SEX: 2}) == (REGISTRY_CUTOFF, "base")
    finally:
        routes.set_group_thresholds(None)


def boot_with(monkeypatch: pytest.MonkeyPatch, **install: Any) -> None:
    """Make the lifespan install the stub model with `install`, not call MLflow."""
    model, _ = build_stub_model()

    def fake_load(self: model_loader.ModelHolder) -> bool:
        self.install(model, **install)
        return True

    monkeypatch.setattr(model_loader.ModelHolder, "load", fake_load)


def test_the_metrics_threshold_returns_to_the_constant_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boot_with(monkeypatch, version="9", threshold=DecisionThreshold(REGISTRY_CUTOFF, "registry"))
    with TestClient(app):
        assert m.decision_threshold() == REGISTRY_CUTOFF
    assert m.decision_threshold() == settings.decision_threshold


def test_health_says_fallback_when_the_version_has_no_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    boot_with(monkeypatch, version="2", run_id="3914189dcbc645808759fd94c2900e5f")
    with TestClient(app) as client:
        body = client.get("/health").json()
        decided = client.post("/api/v1/predict", json=payload()).json()
    model_loader.MODEL.unload()

    assert body["threshold_source"] == "fallback"
    assert body["threshold"] == settings.decision_threshold
    assert decided["threshold_used"] == settings.decision_threshold


def test_health_still_answers_with_a_nonsense_configured_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # /health reports; it does not validate. A misconfigured constant has to be
    # visible there, not turned into a 500 by the response model.
    monkeypatch.setattr(settings, "decision_threshold", 1.5)
    boot_with(monkeypatch, version="2")
    with TestClient(app) as client:
        response = client.get("/health")
    model_loader.MODEL.unload()

    assert response.status_code == 200
    assert response.json()["threshold"] == 1.5


# ------------------------------------------------------------- end to end


@dataclass(frozen=True)
class Served:
    """What the API did with a pool of accounts, next to what was registered."""

    health: dict[str, Any]
    logged_threshold: float
    run_id: str
    decisions: list[str]
    thresholds_used: set[float]
    offline_selected: int

    @property
    def selection_rate(self) -> float:
        return sum(1 for decision in self.decisions if decision == "intervene") / len(
            self.decisions
        )


def as_requests(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as the API's request body: exactly the input columns, plain JSON types."""
    projected = frame.loc[:, list(REQUIRED_INPUT_COLUMNS)]
    records: list[dict[str, Any]] = json.loads(projected.to_json(orient="records"))
    return records


def serve_and_score(
    estimator: Any,
    evaluation: pd.DataFrame,
    pool: pd.DataFrame,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Served:
    """Register `estimator` the way the pipeline does, serve it, score `pool`.

    The threshold is computed on `evaluation` by the same `evaluate()` training
    calls and logged under the same metric name, so the registry reads it off
    the run exactly as it does for a DAG run. Nothing on the serving side is
    stubbed: the lifespan loads the promoted version from MLflow.
    """
    monkeypatch.setattr(settings, "threshold_source", "registry")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        experiment = mlflow.create_experiment("served", artifact_location=str(tmp_path / "art"))
        mlflow.set_experiment(experiment_id=experiment)

        X_eval = feature_frame(evaluation)
        metrics = evaluate(
            evaluation[schema.TARGET],
            estimator.predict_proba(X_eval)[:, 1],
            capacity_fraction=CAPACITY,
        )
        with mlflow.start_run() as run:
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
            mlflow.sklearn.log_model(sk_model=estimator, artifact_path="model")
        run_id = str(run.info.run_id)

        # Gate inputs supplied, as in the registry happy-path test: whether a
        # fixture model clears PR-AUC is not what this test is about.
        decision = registry.register_if_passes(run_id, {"pr_auc": 0.61}, FAIR_ENOUGH)
        assert decision.registered and decision.version is not None, decision.reasons
        assert registry.promote(decision.version, settings.model_stage).registered

        lime_explainer.reset()
        lime_explainer.set_background(X_eval.head(200).to_numpy())
        requests = as_requests(pool)
        decisions: list[str] = []
        thresholds_used: set[float] = set()
        with TestClient(app) as client:
            health = client.get("/health").json()
            for start in range(0, len(requests), settings.max_batch_size):
                chunk = requests[start : start + settings.max_batch_size]
                body = client.post("/api/v1/predict/batch", json={"applications": chunk}).json()
                decisions.extend(item["decision"] for item in body["predictions"])
                thresholds_used.update(item["threshold_used"] for item in body["predictions"])
            served_threshold = model_loader.MODEL.threshold
        offline = estimator.predict_proba(feature_frame(pool))[:, 1]
        return Served(
            health=health,
            logged_threshold=metrics["threshold_at_k"],
            run_id=run_id,
            decisions=decisions,
            thresholds_used=thresholds_used,
            offline_selected=int((offline >= served_threshold).sum()),
        )
    finally:
        model_loader.MODEL.unload()
        lime_explainer.reset()
        shap_explainer.reset()
        mlflow.set_tracking_uri(previous_uri)


def assert_capacity_is_held(served: Served, n_pool: int) -> None:
    """The checks both end-to-end tests make, synthetic and real."""
    assert served.health["threshold_source"] == "registry"
    assert served.health["run_id"] == served.run_id
    # The exact number the evaluation computed, through tag and loader intact.
    assert served.health["threshold"] == served.logged_threshold
    assert served.thresholds_used == {served.logged_threshold}
    assert len(served.decisions) == n_pool
    # The API applies the cutoff exactly as offline scoring would -- the same
    # `>=`, the same features -- so the counts agree to the account.
    assert served.decisions.count("intervene") == served.offline_selected
    assert math.isclose(served.selection_rate, CAPACITY, abs_tol=CAPACITY_TOLERANCE), (
        f"served selection rate {served.selection_rate:.4f} is outside "
        f"{CAPACITY} +/- {CAPACITY_TOLERANCE}"
    )


def test_the_served_selection_rate_holds_capacity_on_accounts_never_scored(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic stand-in for the serving pool: same population, unseen rows.

    Train, evaluation and pool are disjoint slices of one synthetic portfolio,
    so the pool plays the part batch 6 plays in production.
    """
    frame = synthetic_clean_frame(n=16_000, seed=11)
    train, evaluation, pool = frame.iloc[:8_000], frame.iloc[8_000:12_000], frame.iloc[12_000:]
    model: LGBMClassifier = make_lightgbm(n_estimators=120, learning_rate=0.1, num_leaves=15)
    model.fit(feature_frame(train), train[schema.TARGET])

    served = serve_and_score(model, evaluation, pool, tmp_path, monkeypatch)

    assert_capacity_is_held(served, len(pool))
    # And 0.5 would not have: the point of the change. On this model the fixed
    # cutoff selects a visibly different share of the same pool.
    at_half = float((model.predict_proba(feature_frame(pool))[:, 1] >= 0.5).mean())
    assert not math.isclose(at_half, CAPACITY, abs_tol=CAPACITY_TOLERANCE)


def test_scoring_helpers_project_requests_to_the_input_schema() -> None:
    frame = synthetic_clean_frame(n=50, seed=3)
    requests = as_requests(frame)
    assert set(requests[0]) == set(REQUIRED_INPUT_COLUMNS)
    assert all(isinstance(value, int | float) for value in requests[0].values())
    assert np.isfinite(list(requests[0].values())).all()
