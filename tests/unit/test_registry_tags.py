"""The tags a registered version carries for serving, and the backfill that adds them.

Run against a throwaway SQLite-backed MLflow, not a mock: the production
registry is SQL-backed (Postgres), and "a version tag written here is read back
there" is the property under test. No model artefact is logged -- registering a
version and tagging it never opens one -- so every test here is fast enough for
the main CI job.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import mlflow
import pytest
from mlflow.client import MlflowClient

from credit_risk.config import Settings, settings
from credit_risk.models import registry
from credit_risk.models.registry import (
    CAPACITY_TAG,
    GIT_SHA_TAG,
    RUN_GIT_COMMIT_TAG,
    RUN_ID_TAG,
    TAGGED_BY_TAG,
    THRESHOLD_TAG,
    TRAINED_AT_TAG,
    backfill_version_tags,
    production_version_metadata,
    register_if_passes,
    tag_version,
    version_metadata,
    version_tags,
)

MODEL = "credit-risk-tags-test"

# The capacity threshold version 2 logged as test_threshold_at_k. A value with
# all seventeen significant digits, so a tag that rounded it would fail here.
LOGGED_THRESHOLD = 0.5166605772575003

FAIR_ENOUGH = {"demographic_parity_difference": 0.01, "equalized_odds_difference": 0.02}


@pytest.fixture
def sqlite_mlflow(tmp_path: Any) -> Iterator[str]:
    """A fresh SQL-backed tracking store and registry for one test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        "registry-tags", artifact_location=str(tmp_path / "artifacts")
    )
    # Set as well as created: the fluent API caches the active experiment id,
    # and a stale one from another test's store would not exist in this one.
    mlflow.set_experiment(experiment_id=experiment_id)
    yield uri
    mlflow.set_tracking_uri(previous)


def logged_run(
    *,
    threshold: float | None = LOGGED_THRESHOLD,
    capacity: float | None = 0.1,
    commit: str | None = None,
) -> str:
    """A run shaped like the one training logs: prefixed metrics, capacity param."""
    with mlflow.start_run() as run:
        mlflow.log_param("capacity_fraction", 0.1)
        if threshold is not None:
            mlflow.log_metric("test_threshold_at_k", threshold)
        if capacity is not None:
            mlflow.log_metric("test_capacity_fraction", capacity)
        mlflow.log_metric("test_pr_auc", 0.5668)
        if commit is not None:
            mlflow.set_tag(RUN_GIT_COMMIT_TAG, commit)
    return str(run.info.run_id)


def started_at(run_id: str) -> str:
    start = MlflowClient().get_run(run_id).info.start_time
    return datetime.fromtimestamp(start / 1000.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def old_style_version(run_id: str) -> str:
    """A version registered the way it was before versions carried tags."""
    MlflowClient().create_registered_model(MODEL)
    created = MlflowClient().create_model_version(MODEL, f"runs:/{run_id}/model", run_id)
    return str(created.version)


# ----------------------------------------------------------- registration


def test_registration_tags_the_version_with_everything_serving_needs(
    sqlite_mlflow: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "git_sha", "abc1234")
    run_id = logged_run()
    handoff = {"pr_auc": 0.5668, "threshold_at_k": LOGGED_THRESHOLD, "capacity_fraction": 0.1}

    decision = register_if_passes(run_id, handoff, FAIR_ENOUGH, model_name=MODEL)

    assert decision.registered is True
    tags = MlflowClient().get_model_version(MODEL, decision.version).tags
    assert tags == {
        THRESHOLD_TAG: repr(LOGGED_THRESHOLD),
        CAPACITY_TAG: "0.1",
        TRAINED_AT_TAG: started_at(run_id),
        GIT_SHA_TAG: "abc1234",
        RUN_ID_TAG: run_id,
        TAGGED_BY_TAG: "registration",
    }
    # Serving parses the tag back with float(); it must be the same number the
    # evaluation computed, not a rounded neighbour of it.
    assert float(tags[THRESHOLD_TAG]) == LOGGED_THRESHOLD
    # The decision reports what it wrote, so registration.json records it too.
    assert decision.tags == tags


def test_registration_reads_the_threshold_off_the_run_when_the_caller_lacks_it(
    sqlite_mlflow: str,
) -> None:
    run_id = logged_run()

    decision = register_if_passes(run_id, {"pr_auc": 0.61}, FAIR_ENOUGH, model_name=MODEL)

    assert decision.tags[THRESHOLD_TAG] == repr(LOGGED_THRESHOLD)
    assert decision.tags[CAPACITY_TAG] == "0.1"


def test_git_sha_comes_from_the_environment_and_defaults_to_unknown(
    sqlite_mlflow: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    assert Settings(_env_file=None).git_sha == "unknown"
    monkeypatch.setenv("GIT_SHA", "f00dfee")
    assert Settings(_env_file=None).git_sha == "f00dfee"

    monkeypatch.setattr(settings, "git_sha", "unknown")
    decision = register_if_passes(logged_run(), {"pr_auc": 0.61}, FAIR_ENOUGH, model_name=MODEL)
    assert decision.tags[GIT_SHA_TAG] == "unknown"


def test_a_run_that_logged_no_threshold_gets_no_threshold_tag(
    sqlite_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    run_id = logged_run(threshold=None, capacity=None)

    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        decision = register_if_passes(run_id, {"pr_auc": 0.61}, FAIR_ENOUGH, model_name=MODEL)

    # Left out, not guessed: the missing tag is what makes serving fall back
    # and say so. The capacity still comes through, from the logged param.
    assert decision.registered is True
    assert THRESHOLD_TAG not in decision.tags
    assert decision.tags[CAPACITY_TAG] == "0.1"
    assert f"tag-threshold --version {decision.version}" in caplog.text


def test_version_tags_are_deterministic_and_tagging_twice_changes_nothing(
    sqlite_mlflow: str,
) -> None:
    run_id = logged_run()
    first = version_tags(run_id, tagged_by="registration", git_sha="abc")
    second = version_tags(run_id, tagged_by="registration", git_sha="abc")
    # trained_at is the run's start, not "now", so the two calls agree.
    assert first == second

    version = old_style_version(run_id)
    assert tag_version(MODEL, version, first) == []
    after_once = MlflowClient().get_model_version(MODEL, version).tags
    assert tag_version(MODEL, version, first) == []
    assert MlflowClient().get_model_version(MODEL, version).tags == after_once == first


def test_a_tag_the_store_refuses_is_reported_and_registration_still_stands(
    sqlite_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    class RefusesTheThreshold(MlflowClient):
        def set_model_version_tag(self, name: str, version: str, key: str, value: Any) -> None:
            if key == THRESHOLD_TAG:
                raise RuntimeError("tracking server gone")
            super().set_model_version_tag(name, version, key, value)

    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        decision = register_if_passes(
            logged_run(),
            {"pr_auc": 0.61},
            FAIR_ENOUGH,
            model_name=MODEL,
            client=RefusesTheThreshold(),
        )

    # The model was registered; losing one tag does not undo that, and the
    # caller is told exactly which tag is missing and how to put it back.
    assert decision.registered is True
    assert THRESHOLD_TAG not in decision.tags
    assert RUN_ID_TAG in decision.tags
    assert "carries no threshold_at_k tag" in caplog.text


def test_version_tags_survive_a_run_the_store_cannot_read() -> None:
    class Unreachable(MlflowClient):
        def get_run(self, run_id: str) -> Any:
            raise RuntimeError("connection refused")

    tags = version_tags(
        "abc", {"threshold_at_k": 0.42}, tagged_by="registration", client=Unreachable()
    )

    assert tags[THRESHOLD_TAG] == "0.42"
    assert tags[TRAINED_AT_TAG] == "unknown"
    assert tags[GIT_SHA_TAG] == "unknown"


# --------------------------------------------------------------- reading


def test_production_version_metadata_returns_the_tags_serving_reads(sqlite_mlflow: str) -> None:
    assert production_version_metadata(model_name=MODEL) is None

    decision = register_if_passes(logged_run(), {"pr_auc": 0.61}, FAIR_ENOUGH, model_name=MODEL)
    assert decision.version is not None
    # Registered is not serving: nothing is in Production until promoted.
    assert production_version_metadata(model_name=MODEL) is None
    registry.promote(decision.version, "Production", model_name=MODEL)

    metadata = production_version_metadata(model_name=MODEL)
    assert metadata is not None
    assert metadata.version == decision.version
    assert metadata.model_uri == f"models:/{MODEL}/{decision.version}"
    assert metadata.tags[THRESHOLD_TAG] == repr(LOGGED_THRESHOLD)
    assert metadata.run_id == decision.tags[RUN_ID_TAG]
    assert metadata.trained_at == decision.tags[TRAINED_AT_TAG]


def test_version_metadata_is_none_for_a_version_that_does_not_exist(sqlite_mlflow: str) -> None:
    assert version_metadata("99", model_name=MODEL) is None


# --------------------------------------------------------------- backfill


def run_backfill(capsys: pytest.CaptureFixture[str], uri: str, *extra: str) -> tuple[int, Any, str]:
    """`python -m credit_risk.models.registry tag-threshold --version 1`, in process."""
    code = registry.main(
        ["tag-threshold", "--version", "1", "--model-name", MODEL, "--tracking-uri", uri, *extra]
    )
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_backfill_tags_a_version_registered_before_tags_existed(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = logged_run(commit="0123abc")
    assert old_style_version(run_id) == "1"

    code, payload, err = run_backfill(capsys, sqlite_mlflow)

    assert code == 0
    assert "restart" in err
    tags = MlflowClient().get_model_version(MODEL, "1").tags
    assert float(tags[THRESHOLD_TAG]) == LOGGED_THRESHOLD
    assert tags[CAPACITY_TAG] == "0.1"
    assert tags[RUN_ID_TAG] == run_id
    assert tags[TRAINED_AT_TAG] == started_at(run_id)
    # The commit MLflow recorded on the run, never the backfiller's own
    # checkout -- that would attribute an old model to new code.
    assert tags[GIT_SHA_TAG] == "0123abc"
    assert tags[TAGGED_BY_TAG] == "backfill"
    assert payload["written"] == tags
    assert payload["reasons"] == []


def test_backfill_is_idempotent(sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]) -> None:
    old_style_version(logged_run())
    assert run_backfill(capsys, sqlite_mlflow)[0] == 0
    first = MlflowClient().get_model_version(MODEL, "1").tags

    code, payload, _ = run_backfill(capsys, sqlite_mlflow)

    assert code == 0
    assert payload["written"] == {}
    assert payload["unchanged"] == first
    assert MlflowClient().get_model_version(MODEL, "1").tags == first


def test_backfill_never_overwrites_what_registration_wrote(sqlite_mlflow: str) -> None:
    run_id = logged_run()
    version = old_style_version(run_id)
    tag_version(MODEL, version, {GIT_SHA_TAG: "abc1234", TAGGED_BY_TAG: "registration"})

    result = backfill_version_tags(version, model_name=MODEL)

    tags = MlflowClient().get_model_version(MODEL, version).tags
    assert result.ok
    assert tags[GIT_SHA_TAG] == "abc1234"
    assert tags[TAGGED_BY_TAG] == "registration"
    assert float(tags[THRESHOLD_TAG]) == LOGGED_THRESHOLD
    assert set(result.unchanged) == {GIT_SHA_TAG, TAGGED_BY_TAG}


def test_backfill_dry_run_writes_nothing(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    old_style_version(logged_run())

    code, payload, err = run_backfill(capsys, sqlite_mlflow, "--dry-run")

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["written"][THRESHOLD_TAG] == repr(LOGGED_THRESHOLD)
    assert MlflowClient().get_model_version(MODEL, "1").tags == {}
    assert "restart" not in err


def test_backfill_refuses_a_run_that_logged_no_threshold(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    old_style_version(logged_run(threshold=None))

    code, payload, err = run_backfill(capsys, sqlite_mlflow)

    # Writing everything except the threshold would make the version look
    # finished while it still decides at the fallback.
    assert code == 2
    assert "logged no" in payload["reasons"][0]
    assert "TAG-THRESHOLD FAILED" in err
    assert MlflowClient().get_model_version(MODEL, "1").tags == {}


def test_backfill_reports_a_version_that_does_not_exist(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, _ = run_backfill(capsys, sqlite_mlflow)

    assert code == 2
    assert "not in the registry" in payload["reasons"][0]


def test_backfill_reports_a_version_with_no_run_behind_it(sqlite_mlflow: str) -> None:
    MlflowClient().create_registered_model(MODEL)
    MlflowClient().create_model_version(MODEL, "s3://bucket/model")

    result = backfill_version_tags("1", model_name=MODEL)

    assert not result.ok
    assert "no run id" in result.reasons[0]


def test_backfill_reports_a_tag_the_store_will_not_take(sqlite_mlflow: str) -> None:
    class ReadOnly(MlflowClient):
        def set_model_version_tag(self, name: str, version: str, key: str, value: Any) -> None:
            raise RuntimeError("read-only replica")

    version = old_style_version(logged_run())

    result = backfill_version_tags(version, model_name=MODEL, client=ReadOnly())

    assert not result.ok
    assert result.written == {}
    assert any(THRESHOLD_TAG in reason for reason in result.reasons)


def test_backfill_leaves_a_disagreeing_threshold_tag_alone_and_says_so(
    sqlite_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    version = old_style_version(logged_run())
    tag_version(MODEL, version, {THRESHOLD_TAG: "0.6"})

    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        result = backfill_version_tags(version, model_name=MODEL)

    assert result.ok
    assert MlflowClient().get_model_version(MODEL, version).tags[THRESHOLD_TAG] == "0.6"
    assert "leaving the tag as it is" in caplog.text
