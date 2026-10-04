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
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

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


def register_with_code(
    tmp_path: Path, uri: str, pr_auc: float, *extra: str, **run: Any
) -> tuple[int, dict[str, Any]]:
    """`python -m credit_risk.models.registry`, in process: (exit code, registration.json)."""
    run_id = logged_run(pr_auc, **run)
    path = handoff(tmp_path, uri, run_id, pr_auc)
    code = registry.main(["--result", str(path), *extra])
    written: dict[str, Any] = json.loads((tmp_path / "registration.json").read_text())
    return code, written


def register(tmp_path: Path, uri: str, pr_auc: float, *extra: str, **run: Any) -> dict[str, Any]:
    """The same, for a registration that is expected to succeed."""
    code, written = register_with_code(tmp_path, uri, pr_auc, *extra, **run)
    assert code == 0, written
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


# ------------------------------------------- unreadable is not the same as empty


def unavailable(*args: Any, **kwargs: Any) -> Any:
    # What the REST store raises on a 5xx, a timeout or an exhausted DB pool:
    # an MlflowException whose code is not one of the not-found codes.
    raise MlflowException("503 Service Unavailable: connection pool exhausted")


class CannotRead(MlflowClient):
    get_model_version_by_alias = unavailable
    get_latest_versions = unavailable


def test_an_unreadable_empty_registry_is_not_taken_for_an_empty_one(sqlite_mlflow: str) -> None:
    # Before: every lookup error was None, None was "no champion", and the
    # candidate was promoted over whatever the unreadable registry held.
    verdict = compare_with_champion(logged_run(0.56), {"pr_auc": 0.56}, client=CannotRead())

    assert verdict.promote is False
    assert "could not be read" in verdict.reasons[0]
    assert "connection pool exhausted" in verdict.reasons[0]


def test_an_unreadable_registry_parks_the_candidate_and_leaves_the_champion_alone(
    sqlite_mlflow: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register(tmp_path, sqlite_mlflow, 0.55)
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", unavailable)
    monkeypatch.setattr(MlflowClient, "get_latest_versions", unavailable)

    code, written = register_with_code(tmp_path, sqlite_mlflow, 0.70)
    monkeypatch.undo()

    # Better on paper, but nobody could check against what is serving.
    assert code == 0
    assert written["promoted"] is False
    assert written["alias"] == CHALLENGER_ALIAS
    assert alias_holder(CHAMPION) == "1"
    assert version("1").current_stage == settings.model_stage
    assert version("2").current_stage == CHALLENGER_STAGE
    assert "could not be read" in version("2").tags[PROMOTION_REASON_TAG]


def test_a_missing_alias_and_an_empty_stage_still_mean_no_champion(sqlite_mlflow: str) -> None:
    # The not-found answers -- INVALID_PARAMETER_VALUE for the alias,
    # RESOURCE_DOES_NOT_EXIST for the model -- are the empty registry.
    verdict = compare_with_champion(logged_run(0.56), {"pr_auc": 0.56})
    assert verdict.promote is True
    assert "no champion" in verdict.reasons[0]


def test_serving_falls_back_to_the_stage_when_the_alias_cannot_be_read_and_says_why(
    sqlite_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    register_if_passes(logged_run(0.56), {"pr_auc": 0.56}, FAIR_ENOUGH)
    registry.promote("1", settings.model_stage)

    class AliasDown(MlflowClient):
        get_model_version_by_alias = unavailable

    with caplog.at_level(logging.WARNING, logger=registry.logger.name):
        found = production_version_metadata(client=AliasDown())

    # Serving stays up on the stage; the reason is not swallowed.
    assert found is not None
    assert (found.version, found.resolved_by) == ("1", "stage")
    assert any("connection pool exhausted" in error for error in found.resolution_errors)
    assert "connection pool exhausted" in caplog.text


# ------------------------------------------ a promotion that fails is a failure


def test_a_stage_transition_that_fails_after_the_alias_moved_fails_the_command(
    sqlite_mlflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register(tmp_path, sqlite_mlflow, 0.55)
    capsys.readouterr()
    monkeypatch.setattr(MlflowClient, "transition_model_version_stage", unavailable)

    code, written = register_with_code(tmp_path, sqlite_mlflow, 0.60)
    monkeypatch.undo()

    # Non-zero, so the DAG's register_model task fails and the failure alert
    # fires, instead of a green task over a half-promoted registry.
    assert code == registry.EXIT_PROMOTION_FAILED
    assert written["promoted"] is False
    assert written["promotion_failed"] is True
    err = capsys.readouterr().err
    assert "PROMOTION FAILED" in err
    # The split state, in words: the alias moved, the stage did not.
    assert f"@{CHAMPION} now names v2" in err
    assert alias_holder(CHAMPION) == "2"
    assert version("1").current_stage == settings.model_stage
    assert version("2").current_stage == "None"


def test_set_champion_reports_a_stage_it_could_not_move(sqlite_mlflow: str) -> None:
    register_if_passes(logged_run(0.56), {"pr_auc": 0.56}, FAIR_ENOUGH)

    class StageDown(MlflowClient):
        transition_model_version_stage = unavailable

    outcome = set_champion("1", client=StageDown())

    assert outcome.registered is False
    assert outcome.changed == [f"alias {CHAMPION}"]
    assert any("set-champion --version 1" in reason for reason in outcome.reasons)


def test_an_alias_write_the_store_refuses_fails_the_command(
    sqlite_mlflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise MlflowException("read-only replica")

    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", refuse)

    code, written = register_with_code(tmp_path, sqlite_mlflow, 0.56)
    monkeypatch.undo()

    assert code == registry.EXIT_PROMOTION_FAILED
    err = capsys.readouterr().err
    assert "PROMOTION FAILED" in err
    assert "nothing was changed" in err
    assert alias_holder(CHAMPION) is None
    assert version("1").current_stage == "None"


def test_a_challenger_that_cannot_be_recorded_fails_the_command(
    sqlite_mlflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register(tmp_path, sqlite_mlflow, 0.60)
    capsys.readouterr()
    monkeypatch.setattr(MlflowClient, "transition_model_version_stage", unavailable)

    code, written = register_with_code(tmp_path, sqlite_mlflow, 0.55)
    monkeypatch.undo()

    assert code == registry.EXIT_PROMOTION_FAILED
    assert written["promotion_failed"] is True
    assert "CHALLENGER NOT RECORDED" in capsys.readouterr().err
    assert alias_holder(CHAMPION) == "1"


# --------------------------------------------------- comparison edge cases


def test_the_tolerance_boundary_is_inclusive(sqlite_mlflow: str, tmp_path: Path) -> None:
    register(tmp_path, sqlite_mlflow, 0.600)

    # 0.600 - 0.595 is 0.0050000000000000044 in floating point: exactly the
    # tolerance, so it must promote, not lose to the last bit.
    assert register(tmp_path, sqlite_mlflow, 0.595)["promoted"] is True
    assert alias_holder(CHAMPION) == "2"


def test_a_candidate_from_the_champions_own_run_is_promoted_whatever_its_number(
    sqlite_mlflow: str,
) -> None:
    run_id = logged_run(0.60)
    champion = register_if_passes(run_id, {"pr_auc": 0.60}, FAIR_ENOUGH)
    assert champion.version is not None and set_champion(champion.version).registered

    # The same run registered again is the same model; a stale or partial
    # number in the hand-off must not demote it.
    verdict = compare_with_champion(run_id, {"pr_auc": 0.10})

    assert verdict.promote is True
    assert "same run" in verdict.reasons[0]


def test_the_comparison_is_with_the_alias_when_alias_and_stage_disagree(
    sqlite_mlflow: str,
) -> None:
    stage_only = register_if_passes(logged_run(0.70), {"pr_auc": 0.70}, FAIR_ENOUGH)
    aliased = register_if_passes(logged_run(0.55), {"pr_auc": 0.55}, FAIR_ENOUGH)
    assert stage_only.version == "1" and aliased.version == "2"
    registry.promote("1", settings.model_stage)
    MlflowClient().set_registered_model_alias(MODEL, CHAMPION, "2")

    # 0.58 beats the alias's 0.55 and loses to the stage's 0.70. The alias is
    # what serves, so it is what the candidate has to beat.
    verdict = compare_with_champion(logged_run(0.58), {"pr_auc": 0.58})

    assert verdict.champion_version == "2"
    assert verdict.promote is True


def test_a_champion_without_split_hashes_is_compared_as_logged_and_says_so(
    sqlite_mlflow: str, tmp_path: Path
) -> None:
    # The live registry's version 2: its run predates the split hashes, so the
    # held-out split cannot be checked. The comparison still runs, and the
    # version records that it was unverified.
    register(tmp_path, sqlite_mlflow, 0.5668, split=None)

    written = register(tmp_path, sqlite_mlflow, 0.5701)

    assert written["promoted"] is True
    reason = version("2").tags[PROMOTION_REASON_TAG]
    assert "held-out split hash not recorded on champion v1" in reason


# ------------------------------------------------------------- contracts


def test_the_split_param_registry_reads_is_the_one_training_writes() -> None:
    assert TEST_SPLIT_PARAM == train_module.TEST_SHA_PARAM
