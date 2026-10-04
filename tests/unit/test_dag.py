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

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("airflow", reason="Airflow is only installed in the Airflow image")

from airflow.exceptions import AirflowException, AirflowFailException  # noqa: E402
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


# ------------------------------------------------------------ run_module


@pytest.fixture(scope="module")
def dag_module() -> ModuleType:
    """The DAG file as a plain module, so its helpers can be called directly."""
    spec = importlib.util.spec_from_file_location(
        "credit_risk_pipeline_under_test", DAGS_DIR / "credit_risk_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exits(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, code: int, stdout: str = ""
) -> None:
    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["python"], code, stdout=stdout, stderr="boom")

    monkeypatch.setattr(module.subprocess, "run", run)


def test_a_transient_exit_raises_an_exception_airflow_retries(
    dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before this every non-zero exit was AirflowFailException, so the two
    retries in DEFAULT_ARGS never applied to a network blip."""
    _exits(monkeypatch, dag_module, dag_module.TRANSIENT_EXIT_CODE)

    with pytest.raises(AirflowException) as excinfo:
        dag_module.run_module("credit_risk.data.download")

    assert not isinstance(excinfo.value, AirflowFailException)


def test_any_other_non_zero_exit_fails_without_a_retry(
    dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exits(monkeypatch, dag_module, 1)

    with pytest.raises(AirflowFailException, match="exited with 1"):
        dag_module.run_module("credit_risk.data.validate")


def test_a_clean_exit_returns_what_the_module_printed(
    dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exits(monkeypatch, dag_module, 0, stdout="/opt/airflow/data/raw/x.parquet\n")

    assert dag_module.run_module("credit_risk.data.download").strip().endswith("x.parquet")


def test_the_download_task_has_retries_to_use(pipeline: DAG) -> None:
    assert pipeline.get_task("download_raw").retries >= 1


# ------------------------------------------------------- validation evidence


class _RecordingTaskInstance:
    """Stands in for the TaskInstance Airflow injects as ``ti``."""

    def __init__(self) -> None:
        self.pushed: dict[str, Any] = {}

    def xcom_push(self, key: str, value: Any, **kwargs: Any) -> None:
        self.pushed[key] = value


def test_a_failed_validation_still_leaves_its_report_in_xcom(
    pipeline: DAG, dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The docstring promised this; a failing task used to push nothing."""
    report = {"ok": False, "bad_row_fraction": 1.0, "checks": [{"name": "age_range"}]}
    _exits(monkeypatch, dag_module, 1, stdout=json.dumps(report))
    ti = _RecordingTaskInstance()

    with pytest.raises(AirflowFailException):
        pipeline.get_task("validate_raw").python_callable("/data/raw.parquet", ti=ti)

    assert ti.pushed == {dag_module.VALIDATION_REPORT_XCOM_KEY: report}


def test_a_passing_validation_returns_and_pushes_the_same_report(
    pipeline: DAG, dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"ok": True, "bad_row_fraction": 0.0, "checks": []}
    _exits(monkeypatch, dag_module, 0, stdout=json.dumps(report))
    ti = _RecordingTaskInstance()

    returned = pipeline.get_task("validate_raw").python_callable("/data/raw.parquet", ti=ti)

    assert returned == report
    assert ti.pushed == {dag_module.VALIDATION_REPORT_XCOM_KEY: report}


# ---------------------------------------------------------- failure alerts


class _FailedTask:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.log_url = f"http://localhost:8080/log?task_id={task_id}"
        self.try_number = 1


class _DagRun:
    def __init__(self, failed: list[str]) -> None:
        self.dag_id = DAG_ID
        self.run_id = "scheduled__2026-10-04T00:00:00+00:00"
        self._failed = [_FailedTask(task_id) for task_id in failed]

    def get_task_instances(self, state: Any = None, **kwargs: Any) -> list[_FailedTask]:
        return self._failed


class _Posted:
    """Captures what the callback sends instead of letting it reach a network."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, request: Any, timeout: float | None = None) -> Any:
        self.requests.append(request)

        class _Response:
            status = 200

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        return _Response()

    def alerts(self) -> list[dict[str, Any]]:
        assert len(self.requests) == 1, f"expected one POST, saw {len(self.requests)}"
        return list(json.loads(self.requests[0].data))


@pytest.fixture
def posted(dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Posted:
    capture = _Posted()
    monkeypatch.setattr(dag_module.urllib.request, "urlopen", capture)
    monkeypatch.delenv(dag_module.ALERTS_URL_VARIABLE, raising=False)
    return capture


def test_the_dag_alerts_when_a_run_fails(pipeline: DAG) -> None:
    """Before this a refused gate or a failed task notified nobody."""
    callback = pipeline.on_failure_callback
    assert callback is not None and callback.__name__ == "notify_failure"


def test_a_failed_run_posts_one_alert_per_failed_task(
    dag_module: ModuleType, posted: _Posted
) -> None:
    context = {"dag_run": _DagRun(["evaluate_and_gate"]), "reason": "task_failure"}

    dag_module.notify_failure(context)

    request = posted.requests[0]
    assert request.full_url == "http://alertmanager:9093/api/v2/alerts"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    [alert] = posted.alerts()
    assert alert["labels"] == {
        "alertname": "PipelineTaskFailed",
        "severity": "critical",
        "dag_id": DAG_ID,
        "task_id": "evaluate_and_gate",
    }
    assert "evaluate_and_gate" in alert["annotations"]["summary"]
    assert "scheduled__2026-10-04" in alert["annotations"]["description"]
    assert alert["generatorURL"].startswith("http://localhost:8080/log")
    assert alert["startsAt"] < alert["endsAt"]


def test_the_alertmanager_url_comes_from_the_environment(
    dag_module: ModuleType, posted: _Posted, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(dag_module.ALERTS_URL_VARIABLE, "http://am.example.test/api/v2/alerts")

    dag_module.notify_failure({"dag_run": _DagRun(["download_raw"])})

    assert posted.requests[0].full_url == "http://am.example.test/api/v2/alerts"


def test_an_empty_url_turns_alerting_off(
    dag_module: ModuleType, posted: _Posted, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(dag_module.ALERTS_URL_VARIABLE, "")

    dag_module.notify_failure({"dag_run": _DagRun(["download_raw"])})

    assert posted.requests == []


def test_an_unreachable_alertmanager_never_breaks_the_callback(
    dag_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(request: Any, timeout: float | None = None) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(dag_module.urllib.request, "urlopen", refused)

    dag_module.notify_failure({"dag_run": _DagRun(["train_candidates"])})  # must not raise


def test_a_context_with_nothing_useful_in_it_never_breaks_the_callback(
    dag_module: ModuleType, posted: _Posted
) -> None:
    class Broken:
        run_id = "manual__x"

        def get_task_instances(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("metadata DB unavailable")

    dag_module.notify_failure({})
    dag_module.notify_failure({"dag_run": Broken()})

    labels = [json.loads(r.data)[0]["labels"] for r in posted.requests]
    assert [label["task_id"] for label in labels] == ["unknown", "unknown"]


def test_a_successful_run_resolves_the_failure_alerts(
    pipeline: DAG, dag_module: ModuleType, posted: _Posted
) -> None:
    """Without this a failure alert would stay firing for its whole lifetime
    after a re-run had already fixed the pipeline."""
    assert pipeline.on_success_callback.__name__ == "resolve_failure_alerts"

    dag_module.resolve_failure_alerts({"dag": pipeline})

    alerts = posted.alerts()
    assert {alert["labels"]["task_id"] for alert in alerts} == set(pipeline.task_ids)
    assert all(alert["labels"]["alertname"] == "PipelineTaskFailed" for alert in alerts)
    assert all(alert["endsAt"] == alert["startsAt"] for alert in alerts)
