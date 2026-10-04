"""Which version the API serves: the champion alias, else the Production stage.

The registry promotes by moving `@champion`; a registry populated before aliases
were used has only the stage. The API has to serve the alias when there is one,
fall back to the stage when there is not, and say in `/health` which of the two
it did -- "version 2" alone does not tell an operator whether the alias they
just set has been picked up. The end-to-end test logs real models into a
throwaway SQLite MLflow and lets the API's own lifespan load them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pytest
from fastapi.testclient import TestClient
from mlflow.client import MlflowClient

from credit_risk.config import settings
from credit_risk.explain import lime_explainer, shap_explainer
from credit_risk.models import registry
from credit_risk.serving import model_loader
from credit_risk.serving.main import app
from tests.integration.test_api import build_stub_model, payload

FAIR_ENOUGH = {"demographic_parity_difference": 0.01, "equalized_odds_difference": 0.02}


def stub_resolution(monkeypatch: pytest.MonkeyPatch, resolved_by: str, uri: str) -> None:
    metadata = registry.VersionMetadata(
        model_name=settings.model_name,
        version="3",
        run_id="run-3",
        resolved_by=resolved_by,
        resolved_uri=uri,
    )
    monkeypatch.setattr(registry, "production_version_metadata", lambda **kwargs: metadata)
    monkeypatch.setattr(registry, "load_production_model", lambda **kwargs: object())


@pytest.mark.parametrize(
    ("resolved_by", "uri"),
    [
        ("alias", f"models:/{settings.model_name}@{settings.model_alias}"),
        ("stage", f"models:/{settings.model_name}/{settings.model_stage}"),
    ],
)
def test_the_loader_records_how_the_version_was_found(
    monkeypatch: pytest.MonkeyPatch, resolved_by: str, uri: str
) -> None:
    stub_resolution(monkeypatch, resolved_by, uri)

    holder = model_loader.ModelHolder()
    assert holder.load() is True

    assert holder.model_ref == resolved_by
    assert holder.model_ref_uri == uri


def test_a_stage_fallback_is_logged_with_the_command_that_ends_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stub_resolution(monkeypatch, "stage", f"models:/{settings.model_name}/Production")

    with caplog.at_level(logging.WARNING, logger=model_loader.log.name):
        model_loader.ModelHolder().load()

    assert "set-champion --version 3" in caplog.text


def test_nothing_loaded_means_no_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    holder = model_loader.ModelHolder()
    holder.install(object(), version="3", model_ref="alias", model_ref_uri="models:/x@champion")
    monkeypatch.setattr(registry, "production_version_metadata", lambda **kwargs: None)
    monkeypatch.setattr(registry, "load_production_model", lambda **kwargs: None)

    assert holder.load() is False

    assert holder.model_ref == model_loader.UNKNOWN
    assert holder.model_ref_uri == model_loader.UNKNOWN


# ------------------------------------------------------------- end to end


@pytest.fixture
def sqlite_registry(tmp_path: Path) -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.create_experiment("alias", artifact_location=str(tmp_path / "art"))
    mlflow.set_experiment(experiment_id=experiment)
    model, background = build_stub_model()
    lime_explainer.reset()
    lime_explainer.set_background(background.head(50).to_numpy())
    try:
        yield
    finally:
        model_loader.MODEL.unload()
        lime_explainer.reset()
        shap_explainer.reset()
        mlflow.set_tracking_uri(previous)


def register_logged_model() -> str:
    model, _ = build_stub_model()
    with mlflow.start_run() as run:
        mlflow.log_metric("test_pr_auc", 0.56)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    decision = registry.register_if_passes(str(run.info.run_id), {"pr_auc": 0.56}, FAIR_ENOUGH)
    assert decision.registered and decision.version is not None, decision.reasons
    return decision.version


def served() -> dict[str, Any]:
    with TestClient(app) as client:
        health: dict[str, Any] = client.get("/health").json()
        scored = client.post("/api/v1/predict", json=payload()).json()
    assert scored["model_version"] == health["model_version"]
    return health


def test_the_api_serves_the_alias_and_falls_back_to_the_stage(sqlite_registry: None) -> None:
    first, second = register_logged_model(), register_logged_model()

    # The live registry's shape: a version in Production, no alias anywhere.
    registry.promote(first, settings.model_stage)
    health = served()
    assert (health["model_version"], health["model_ref"]) == (first, "stage")
    assert health["model_ref_uri"] == f"models:/{settings.model_name}/{settings.model_stage}"

    # Promotion as registration now does it: alias plus stage. A restart --
    # here, a new lifespan -- picks it up and says it came from the alias.
    assert registry.set_champion(second).registered
    health = served()
    assert (health["model_version"], health["model_ref"]) == (second, "alias")
    assert health["model_ref_uri"] == f"models:/{settings.model_name}@{settings.model_alias}"

    # The alias decides even when the stage points elsewhere.
    registry.promote(first, settings.model_stage)
    assert MlflowClient().get_model_version(settings.model_name, first).current_stage == (
        settings.model_stage
    )
    assert served()["model_version"] == second
