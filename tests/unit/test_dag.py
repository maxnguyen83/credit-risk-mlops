"""The Airflow DAG parses, and its task graph is the one the docs describe.

A DAG that fails to import does not fail anything visible: Airflow drops it
from the UI and writes the traceback to a scheduler log. These tests load
``dags/`` through a real ``DagBag`` so an import error, a renamed task or a lost
edge is a red test instead.

Airflow is installed only in the Airflow image, not in the project virtualenv
or in CI, so the module skips itself when ``airflow`` is not importable. The
Airflow interpreter in that image cannot import ``credit_risk`` (ADR 0006), so
nothing here imports it and the shared conftest, which does, is skipped. One
way to run it against the built image, without touching the running stack:

    docker run --rm --entrypoint bash -e PIP_USER=false -v "$PWD":/repo:ro -w /repo \\
      credit-risk-airflow:2.8.4 -c "pip install -q --target /tmp/pt pytest==8.3.4 &&
      PYTHONPATH=/tmp/pt python -m pytest --noconftest -p no:cacheprovider tests/unit/test_dag.py"

``make dag-test`` runs the whole DAG once instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow", reason="Airflow is only installed in the Airflow image")

from airflow.models import DAG, DagBag  # noqa: E402 - must follow the skip

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"
DAG_ID = "credit_risk_pipeline"

# Task id -> the task ids it waits on. Mirrors the diagram in ARCHITECTURE.md
# §3.1. clean_and_split waits on validate_raw although it reads nothing from it:
# that edge is what stops a frame that failed validation from being split.
EXPECTED_UPSTREAM: dict[str, set[str]] = {
    "download_raw": set(),
    "validate_raw": {"download_raw"},
    "clean_and_split": {"download_raw", "validate_raw"},
    "build_features": {"clean_and_split"},
    "train_candidates": {"build_features"},
    "evaluate_and_gate": {"train_candidates"},
    "register_model": {"evaluate_and_gate"},
    "publish_report": {"register_model"},
}


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


@pytest.fixture(scope="module")
def pipeline(dagbag: DagBag) -> DAG:
    # dagbag.dags, not dagbag.get_dag(): get_dag() also queries the metadata DB,
    # which a test environment has no reason to have initialised.
    assert DAG_ID in dagbag.dags, f"{DAG_ID} not found in {DAGS_DIR}"
    dag: DAG = dagbag.dags[DAG_ID]
    return dag


def test_the_dag_folder_imports_without_errors(dagbag: DagBag) -> None:
    assert dagbag.import_errors == {}


def test_the_pipeline_is_the_only_dag(dagbag: DagBag) -> None:
    assert set(dagbag.dag_ids) == {DAG_ID}


def test_the_pipeline_has_exactly_the_eight_documented_tasks(pipeline: DAG) -> None:
    assert sorted(pipeline.task_ids) == sorted(EXPECTED_UPSTREAM)


def test_every_task_waits_on_exactly_the_documented_upstream(pipeline: DAG) -> None:
    actual = {task.task_id: set(task.upstream_task_ids) for task in pipeline.tasks}
    assert actual == EXPECTED_UPSTREAM


def test_catchup_is_off_so_unpausing_does_not_queue_a_backlog_of_runs(pipeline: DAG) -> None:
    # @daily from 2026-01-01 with catchup on would queue ~270 sequential
    # training runs on the SequentialExecutor the moment the DAG is unpaused.
    assert pipeline.catchup is False
    assert pipeline.max_active_runs == 1
