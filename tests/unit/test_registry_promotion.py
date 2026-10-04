"""Champion and challenger: what registration does with a model that passed the gate.

Passing the gate makes a model registrable, not better than the one serving.
These tests pin the promotion rule -- an empty registry promotes, a candidate no
worse than the champion beyond the configured tolerance promotes, anything else
waits as the challenger for a person to decide -- and the alias that rule
moves, against a throwaway SQLite-backed MLflow rather than a mock, for the
same reason as `test_registry_tags.py`: the production registry is SQL-backed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow.client import MlflowClient

from credit_risk.config import settings
from credit_risk.models import registry
from credit_risk.models import train as train_module
from credit_risk.models.registry import (
    CHALLENGER_ALIAS,
    CHALLENGER_STAGE,
    COMPARED_WITH_TAG,
    PROMOTION_DECISION_TAG,
    PROMOTION_REASON_TAG,
    TEST_SPLIT_PARAM,
    compare_with_champion,
    production_version_metadata,
    register_if_passes,
    set_champion,
)

MODEL = settings.model_name
CHAMPION = settings.model_alias
FAIR_ENOUGH = {"demographic_parity_difference": 0.01, "equalized_odds_difference": 0.02}
SPLIT = "964ef99fee80e8d71354dc8298762508731e20922cf2ef2ca37cef1d8bd78106"


@pytest.fixture
def sqlite_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A fresh SQL-backed tracking store and registry, with a known tolerance."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        "registry-promotion", artifact_location=str(tmp_path / "artifacts")
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    monkeypatch.setattr(settings, "promotion_pr_auc_tolerance", 0.005)
    yield uri
    mlflow.set_tracking_uri(previous)


def logged_run(pr_auc: float, *, split: str | None = SPLIT) -> str:
    """A run shaped like the one training logs: prefixed metric, split hash param."""
    with mlflow.start_run() as run:
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_metric("test_threshold_at_k", 0.52)
        if split is not None:
            mlflow.log_param(TEST_SPLIT_PARAM, split)
    return str(run.info.run_id)


def handoff(tmp_path: Path, uri: str, run_id: str, pr_auc: float) -> Path:
    """The training_result.json the DAG's register_model task reads."""
    path = tmp_path / f"training_result_{run_id}.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": "lightgbm",
                "tracking_uri": uri,
                "metrics": {"pr_auc": pr_auc, "threshold_at_k": 0.52, "capacity_fraction": 0.1},
                "fairness": FAIR_ENOUGH,
            }
        )
    )
    return path


def register(tmp_path: Path, uri: str, pr_auc: float, *extra: str, **run: Any) -> dict[str, Any]:
    """`python -m credit_risk.models.registry`, in process; returns registration.json."""
    run_id = logged_run(pr_auc, **run)
    path = handoff(tmp_path, uri, run_id, pr_auc)
    assert registry.main(["--result", str(path), *extra]) == 0
    written: dict[str, Any] = json.loads((tmp_path / "registration.json").read_text())
    return written


def version(number: str) -> Any:
    return MlflowClient().get_model_version(MODEL, number)


def alias_holder(alias: str) -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(MODEL, alias).version)
    except Exception:  # noqa: BLE001 - "no such alias" is the answer being tested
        return None


# ----------------------------------------------------------- first model


def test_the_first_model_into_an_empty_registry_becomes_champion(
    sqlite_mlflow: str, tmp_path: Path
) -> None:
    # deploy.yml and the DAG bootstrap an empty stack this way: nothing to
    # compare with must mean promote, or a fresh stack never serves anything.
    written = register(tmp_path, sqlite_mlflow, 0.56)

    assert written["promoted"] is True
    assert written["alias"] == CHAMPION
    assert written["stage"] == settings.model_stage
    assert alias_holder(CHAMPION) == "1"
    assert version("1").current_stage == settings.model_stage
    assert version("1").tags[PROMOTION_DECISION_TAG] == "champion"
    assert "no champion" in version("1").tags[PROMOTION_REASON_TAG]


# --------------------------------------------------------- the comparison


def test_a_worse_candidate_is_registered_as_challenger_and_not_promoted(
    sqlite_mlflow: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register(tmp_path, sqlite_mlflow, 0.60)

    written = register(tmp_path, sqlite_mlflow, 0.55)

    # Registered -- it passed the gate and stays available for review -- but
    # the champion and the stage it holds are untouched.
    assert written["registered"] is True and written["version"] == "2"
    assert written["promoted"] is False
    assert written["alias"] == CHALLENGER_ALIAS
    assert written["promotion"]["decision"] == "challenger"
    assert alias_holder(CHAMPION) == "1"
    assert alias_holder(CHALLENGER_ALIAS) == "2"
    assert version("1").current_stage == settings.model_stage
    assert version("2").current_stage == CHALLENGER_STAGE
    tags = version("2").tags
    assert tags[PROMOTION_DECISION_TAG] == "challenger"
    assert "0.0500 below" in tags[PROMOTION_REASON_TAG]
    assert tags[COMPARED_WITH_TAG] == "1"
    # The log says what to run once somebody has looked at it.
    assert "set-champion --version 2" in capsys.readouterr().err


def test_a_candidate_within_the_tolerance_is_promoted(sqlite_mlflow: str, tmp_path: Path) -> None:
    register(tmp_path, sqlite_mlflow, 0.600)

    written = register(tmp_path, sqlite_mlflow, 0.597)

    assert written["promoted"] is True
    assert alias_holder(CHAMPION) == "2"
    assert version("2").current_stage == settings.model_stage
    # One version per stage: the previous champion is archived, not deleted.
    assert version("1").current_stage == "Archived"
    assert version("2").tags[COMPARED_WITH_TAG] == "1"


def test_a_better_candidate_replaces_the_champion(sqlite_mlflow: str, tmp_path: Path) -> None:
    register(tmp_path, sqlite_mlflow, 0.55)
    assert register(tmp_path, sqlite_mlflow, 0.58)["promoted"] is True
    assert alias_holder(CHAMPION) == "2"


def test_the_tolerance_comes_from_configuration(
    sqlite_mlflow: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register(tmp_path, sqlite_mlflow, 0.600)
    monkeypatch.setattr(settings, "promotion_pr_auc_tolerance", 0.0)

    assert register(tmp_path, sqlite_mlflow, 0.597)["promoted"] is False
    assert alias_holder(CHAMPION) == "1"


def test_a_candidate_scored_on_another_held_out_split_waits_for_review(
    sqlite_mlflow: str, tmp_path: Path
) -> None:
    register(tmp_path, sqlite_mlflow, 0.55)

    # Higher, but measured on different rows: the two numbers do not compare.
    written = register(tmp_path, sqlite_mlflow, 0.70, split="0" * 64)

    assert written["promoted"] is False
    assert alias_holder(CHAMPION) == "1"
    assert "held-out split" in version("2").tags[PROMOTION_REASON_TAG]


def test_a_champion_with_no_logged_pr_auc_is_not_replaced_blindly(sqlite_mlflow: str) -> None:
    # A champion whose run logged no PR-AUC -- registered by hand, or by
    # something other than this pipeline. There is nothing to compare with,
    # and "cannot tell" must not read as "the candidate is better".
    with mlflow.start_run() as run:
        mlflow.log_param("model", "lightgbm")
    champion = register_if_passes(str(run.info.run_id), {"pr_auc": 0.6}, FAIR_ENOUGH)
    assert champion.version is not None
    assert set_champion(champion.version).registered

    verdict = compare_with_champion(logged_run(0.58), {"pr_auc": 0.58})

    assert verdict.promote is False
    assert verdict.champion_version == champion.version
    assert "cannot compare" in verdict.reasons[0]


def test_no_promote_registers_without_touching_alias_or_stage(
    sqlite_mlflow: str, tmp_path: Path
) -> None:
    written = register(tmp_path, sqlite_mlflow, 0.56, "--no-promote")

    assert written["registered"] is True
    assert "promoted" not in written
    assert alias_holder(CHAMPION) is None
    assert version("1").current_stage == "None"


# ------------------------------------------------- resolution for serving


def test_serving_resolution_prefers_the_alias_and_falls_back_to_the_stage(
    sqlite_mlflow: str,
) -> None:
    assert production_version_metadata(model_name=MODEL) is None

    first = register_if_passes(logged_run(0.56), {"pr_auc": 0.56}, FAIR_ENOUGH)
    second = register_if_passes(logged_run(0.57), {"pr_auc": 0.57}, FAIR_ENOUGH)
    assert first.version == "1" and second.version == "2"

    # A registry from before aliases: the stage is all there is.
    registry.promote("1", settings.model_stage)
    legacy = production_version_metadata()
    assert legacy is not None
    assert (legacy.version, legacy.resolved_by) == ("1", "stage")
    assert legacy.resolved_uri == f"models:/{MODEL}/{settings.model_stage}"

    # Once an alias exists it decides, even if the stage says otherwise.
    MlflowClient().set_registered_model_alias(MODEL, CHAMPION, "2")
    current = production_version_metadata()
    assert current is not None
    assert (current.version, current.resolved_by) == ("2", "alias")
    assert current.resolved_uri == f"models:/{MODEL}@{CHAMPION}"
    # Still loaded by its own number, so the tags read describe the model loaded.
    assert current.model_uri == f"models:/{MODEL}/2"


# ------------------------------------------------------- set-champion cli


def run_set_champion(
    capsys: pytest.CaptureFixture[str], uri: str, number: str, *extra: str
) -> tuple[int, Any, str]:
    code = registry.main(
        ["set-champion", "--version", number, "--model-name", MODEL, "--tracking-uri", uri, *extra]
    )
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_set_champion_aliases_a_version_registered_before_aliases_and_is_idempotent(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The live registry's shape: version 2 in Production, no alias at all.
    for pr_auc in (0.56, 0.5668):
        register_if_passes(logged_run(pr_auc), {"pr_auc": pr_auc}, FAIR_ENOUGH)
    registry.promote("1", settings.model_stage)
    registry.promote("2", settings.model_stage)

    code, payload, err = run_set_champion(capsys, sqlite_mlflow, "2")

    assert code == 0
    assert alias_holder(CHAMPION) == "2"
    assert version("2").current_stage == settings.model_stage
    assert payload["changed"] == [f"alias {CHAMPION}"]
    assert "restart" in err
    assert version("2").tags[PROMOTION_DECISION_TAG] == "champion"

    code, payload, _ = run_set_champion(capsys, sqlite_mlflow, "2")
    assert code == 0
    assert payload["changed"] == []


def test_set_champion_promotes_a_reviewed_challenger(
    sqlite_mlflow: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    register(tmp_path, sqlite_mlflow, 0.60)
    register(tmp_path, sqlite_mlflow, 0.55)
    capsys.readouterr()

    code, payload, _ = run_set_champion(capsys, sqlite_mlflow, "2")

    assert code == 0
    assert alias_holder(CHAMPION) == "2"
    assert alias_holder(CHALLENGER_ALIAS) is None
    assert version("2").current_stage == settings.model_stage
    assert version("1").current_stage == "Archived"
    assert set(payload["changed"]) == {
        f"alias {CHAMPION}",
        f"stage {settings.model_stage}",
        f"dropped alias {CHALLENGER_ALIAS}",
    }


def test_set_champion_dry_run_changes_nothing(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    register_if_passes(logged_run(0.56), {"pr_auc": 0.56}, FAIR_ENOUGH)

    code, payload, err = run_set_champion(capsys, sqlite_mlflow, "1", "--dry-run")

    assert code == 0
    assert payload["dry_run"] is True
    assert f"alias {CHAMPION}" in payload["changed"]
    assert alias_holder(CHAMPION) is None
    assert version("1").current_stage == "None"
    assert "restart" not in err


def test_set_champion_reports_a_version_that_does_not_exist(
    sqlite_mlflow: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, err = run_set_champion(capsys, sqlite_mlflow, "9")

    assert code == 2
    assert "not in the registry" in payload["reasons"][0]
    assert "SET-CHAMPION FAILED" in err


# --------------------------------------------------------- store failures


class RefusesAliases(MlflowClient):
    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        raise RuntimeError("read-only replica")


def test_a_champion_alias_the_store_refuses_leaves_the_stage_alone(sqlite_mlflow: str) -> None:
    register_if_passes(logged_run(0.56), {"pr_auc": 0.56}, FAIR_ENOUGH)

    outcome = set_champion("1", client=RefusesAliases())

    # Alias first, and nothing else when it fails: a half-promoted version
    # whose stage moved while serving still resolves the old alias would be
    # two answers to "what is serving".
    assert outcome.registered is False
    assert "read-only replica" in outcome.reasons[0]
    assert version("1").current_stage == "None"


def test_a_challenger_keeps_its_explanation_when_the_alias_cannot_be_written(
    sqlite_mlflow: str,
) -> None:
    register_if_passes(logged_run(0.56), {"pr_auc": 0.56}, FAIR_ENOUGH)

    outcome = registry.set_challenger(
        "1", reason="pr_auc 0.5600 is 0.0400 below", client=RefusesAliases()
    )

    assert outcome.registered is False
    assert version("1").tags[PROMOTION_REASON_TAG] == "pr_auc 0.5600 is 0.0400 below"
    assert version("1").current_stage == "None"


# ------------------------------------------------------------- contracts


def test_the_split_param_registry_reads_is_the_one_training_writes() -> None:
    assert TEST_SPLIT_PARAM == train_module.TEST_SHA_PARAM
