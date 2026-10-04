"""The batch-scoring DAG parses, runs its three steps in order, and finds its input.

Like ``test_dag.py`` this needs Airflow, which only the Airflow image has, so
the module skips itself elsewhere. Nothing here imports ``credit_risk``: the
Airflow interpreter cannot (ADR 0006). Run it against the built image the way
``test_dag.py`` describes, with this file's path in place of that one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("airflow", reason="Airflow is only installed in the Airflow image")

from airflow.exceptions import AirflowException, AirflowFailException  # noqa: E402
from airflow.models import DAG, DagBag  # noqa: E402 - must follow the skip

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"
DAG_ID = "credit_risk_scoring"

EXPECTED_UPSTREAM: dict[str, set[str]] = {
    "validate_batch": set(),
    "score_batch": {"validate_batch"},
    "publish_call_list": {"score_batch"},
}


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


@pytest.fixture(scope="module")
def scoring(dagbag: DagBag) -> DAG:
    assert DAG_ID in dagbag.dags, f"{DAG_ID} not found in {DAGS_DIR}"
    dag: DAG = dagbag.dags[DAG_ID]
    return dag


@pytest.fixture(scope="module")
def dag_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "credit_risk_scoring_under_test", DAGS_DIR / "credit_risk_scoring.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dag_imports_without_errors(dagbag: DagBag, scoring: DAG) -> None:
    assert dagbag.import_errors == {}


def test_validate_then_score_then_publish(scoring: DAG) -> None:
    actual = {task.task_id: set(task.upstream_task_ids) for task in scoring.tasks}
    assert actual == EXPECTED_UPSTREAM


def test_it_runs_when_triggered_and_only_one_run_at_a_time(scoring: DAG) -> None:
    assert scoring.schedule_interval is None
    assert scoring.is_paused_upon_creation is False
    assert scoring.max_active_runs == 1
    assert {"credit-risk", "scoring"} <= set(scoring.tags)


# ----------------------------------------------------------- input choice


def test_the_conf_names_the_input_relative_to_the_data_dir(
    dag_module: ModuleType, tmp_path: Path
) -> None:
    chosen = dag_module.resolve_input({"input": "incoming/march.csv"}, tmp_path)

    assert chosen == (tmp_path / "incoming" / "march.csv").resolve()


@pytest.mark.parametrize("escape", ["../secrets.csv", "/etc/passwd"])
def test_the_conf_cannot_reach_outside_the_data_dir(
    dag_module: ModuleType, tmp_path: Path, escape: str
) -> None:
    with pytest.raises(AirflowFailException, match="outside"):
        dag_module.resolve_input({"input": escape}, tmp_path / "data")


def test_without_conf_the_newest_incoming_file_is_scored(
    dag_module: ModuleType, tmp_path: Path
) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for age, name in enumerate(["new.parquet", "old.csv", "notes.txt", ".partial.csv"]):
        path = incoming / name
        path.write_text("x")
        os.utime(path, (1_000_000 - age * 100, 1_000_000 - age * 100))
    os.utime(incoming / ".partial.csv", (2_000_000, 2_000_000))  # newest, but hidden
    os.utime(incoming / "notes.txt", (2_000_000, 2_000_000))  # newest, but not data

    assert dag_module.resolve_input({}, tmp_path) == incoming / "new.parquet"


def test_with_nothing_incoming_the_serving_pool_is_scored(
    dag_module: ModuleType, tmp_path: Path
) -> None:
    expected = tmp_path / "processed" / "serving_pool.parquet"

    assert dag_module.resolve_input(None, tmp_path) == expected
    (tmp_path / "incoming").mkdir()
    assert dag_module.resolve_input({}, tmp_path) == expected


# ------------------------------------------------------------- the tasks


class _Recorder:
    """Stands in for subprocess.run: records argv and env, answers as scripted."""

    def __init__(self, code: int = 0, stdout: str = "") -> None:
        self.code = code
        self.stdout = stdout
        self.argv: list[str] = []
        self.env: dict[str, str] = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.argv = list(argv)
        self.env = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(argv, self.code, stdout=self.stdout, stderr="log")


class _DagRun:
    def __init__(self, conf: dict[str, Any] | None = None) -> None:
        self.dag_id = DAG_ID
        self.run_id = "manual__2026-10-04T09:30:00+00:00"
        self.conf = conf or {}

    def get_task_instances(self, state: Any = None, **kwargs: Any) -> list[Any]:
        return []


@pytest.fixture
def recorder(dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Recorder:
    capture = _Recorder()
    monkeypatch.setattr(dag_module.subprocess, "run", capture)
    monkeypatch.setenv(dag_module.DATA_DIR_VARIABLE, str(tmp_path))
    monkeypatch.setenv(dag_module.PYTHON_VARIABLE, "/venv/bin/python")
    monkeypatch.delenv(dag_module.API_URL_VARIABLE, raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    return capture


def test_validate_batch_scores_the_resolved_input_and_returns_the_run_dir(
    scoring: DAG, recorder: _Recorder, tmp_path: Path
) -> None:
    recorder.stdout = f"{tmp_path}/scored/20261004T093000Z\n"

    run_dir = scoring.get_task("validate_batch").python_callable(
        dag_run=_DagRun({"input": "incoming/march.csv"})
    )

    assert run_dir == f"{tmp_path}/scored/20261004T093000Z"
    assert recorder.argv == [
        "/venv/bin/python",
        "-m",
        "credit_risk.scoring.batch",
        "validate",
        "--input",
        str((tmp_path / "incoming" / "march.csv").resolve()),
        "--data-dir",
        str(tmp_path),
    ]
    # Inside the Airflow container localhost is Airflow itself, never the API.
    assert recorder.env["CREDIT_API_URL"] == "http://credit-api:8000"
    assert recorder.env["DATA_DIR"] == str(tmp_path)


def test_an_api_url_set_on_the_container_wins(
    scoring: DAG, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CREDIT_API_URL", "http://elsewhere:9000")
    recorder.stdout = "/data/scored/x\n"

    scoring.get_task("score_batch").python_callable("/data/scored/x")

    assert recorder.env["CREDIT_API_URL"] == "http://elsewhere:9000"
    assert recorder.argv[3:6] == ["score", "--run-dir", "/data/scored/x"]


def test_publish_records_the_run_id_and_returns_the_summary(
    scoring: DAG, recorder: _Recorder, tmp_path: Path
) -> None:
    summary = {"n_scored": 5000, "n_call_list": 500}
    recorder.stdout = json.dumps(summary, indent=2)

    returned = scoring.get_task("publish_call_list").python_callable(
        "/data/scored/x", dag_run=_DagRun()
    )

    assert returned == summary
    assert recorder.argv[3:] == [
        "publish",
        "--run-dir",
        "/data/scored/x",
        "--data-dir",
        str(tmp_path),
        "--dag-run-id",
        "manual__2026-10-04T09:30:00+00:00",
    ]


def test_an_unavailable_api_is_retried_and_bad_input_is_not(
    scoring: DAG, dag_module: ModuleType, recorder: _Recorder
) -> None:
    score = scoring.get_task("score_batch")
    assert score.retries >= 1

    recorder.code = dag_module.TRANSIENT_EXIT_CODE
    with pytest.raises(AirflowException) as excinfo:
        score.python_callable("/data/scored/x")
    assert not isinstance(excinfo.value, AirflowFailException)

    recorder.code = 2
    with pytest.raises(AirflowFailException, match="exited with 2"):
        scoring.get_task("validate_batch").python_callable(dag_run=_DagRun())


# ---------------------------------------------------------- failure alerts


class _Posted:
    def __init__(self) -> None:
        self.bodies: list[list[dict[str, Any]]] = []

    def __call__(self, request: Any, timeout: float | None = None) -> Any:
        self.bodies.append(json.loads(request.data))

        class _Response:
            status = 200

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        return _Response()


@pytest.fixture
def posted(dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Posted:
    capture = _Posted()
    monkeypatch.setattr(dag_module.urllib.request, "urlopen", capture)
    monkeypatch.delenv(dag_module.ALERTS_URL_VARIABLE, raising=False)
    return capture


def test_a_failed_scoring_run_alerts_like_the_training_pipeline(
    scoring: DAG, dag_module: ModuleType, posted: _Posted
) -> None:
    assert scoring.on_failure_callback.__name__ == "notify_failure"

    class FailedRun(_DagRun):
        def get_task_instances(self, state: Any = None, **kwargs: Any) -> list[Any]:
            task = type("TI", (), {"task_id": "score_batch", "log_url": "http://af/log"})
            return [task()]

    dag_module.notify_failure({"dag_run": FailedRun(), "dag": scoring, "reason": "task_failure"})

    [[alert]] = posted.bodies
    assert alert["labels"] == {
        "alertname": "PipelineTaskFailed",
        "severity": "critical",
        "dag_id": DAG_ID,
        "task_id": "score_batch",
    }
    assert "latest.json" in alert["annotations"]["description"]


def test_a_successful_run_resolves_its_own_alerts(
    scoring: DAG, dag_module: ModuleType, posted: _Posted
) -> None:
    assert scoring.on_success_callback.__name__ == "resolve_failure_alerts"

    dag_module.resolve_failure_alerts({"dag": scoring, "dag_run": _DagRun()})

    [alerts] = posted.bodies
    assert {alert["labels"]["task_id"] for alert in alerts} == {*EXPECTED_UPSTREAM, "dagrun"}
    assert {alert["labels"]["dag_id"] for alert in alerts} == {DAG_ID}
    assert all(alert["startsAt"] == alert["endsAt"] for alert in alerts)
