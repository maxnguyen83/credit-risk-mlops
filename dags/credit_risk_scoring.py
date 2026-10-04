"""Airflow DAG: score a file of accounts and publish the monthly call list.

Triggered, never scheduled: a portfolio extract arrives when it arrives, and
someone -- the console, or a person in the UI -- says "score this one". The
conf names the file relative to the data directory::

    {"input": "incoming/2026-10.csv"}

With no conf it scores the newest ``.csv`` or ``.parquet`` in
``<data_dir>/incoming/``, and with nothing there the demo pool
``processed/serving_pool.parquet``.

Three tasks, one per stage of ``python -m credit_risk.scoring.batch``, sharing
one run directory that ``validate_batch`` creates and hands on through XCom:

- ``validate_batch`` checks every row with the API's own request model. A file
  with no valid row exits 2 and fails the run without a retry.
- ``score_batch`` scores through ``POST /api/v1/predict/batch``. An API that
  stays unreachable exits 75 (TRANSIENT_EXIT_CODE), which Airflow retries; a
  model that changes mid-batch exits 1, which it does not.
- ``publish_call_list`` ranks, explains the call list, writes the files and
  then ``scored/latest.json``. Missing explanations never fail it.

The runner and the Alertmanager callbacks are copies of the ones in
``credit_risk_pipeline.py``, for the same reason that file gives: the
project is not importable from this interpreter (ADR 0006), and importing the
training DAG's module to share them would register that DAG twice.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException, AirflowFailException

log = logging.getLogger(__name__)

PYTHON_VARIABLE: Final = "CREDIT_RISK_PYTHON"
DEFAULT_PYTHON: Final = "python"

DATA_DIR_VARIABLE: Final = "CREDIT_RISK_DATA_DIR"
DEFAULT_AIRFLOW_HOME: Final = "/opt/airflow"

# The child reads this through Settings.credit_api_url. Defaulted to the compose
# service name here because inside the Airflow container the module's own
# default, localhost, is Airflow itself.
API_URL_VARIABLE: Final = "CREDIT_API_URL"
DEFAULT_API_URL: Final = "http://credit-api:8000"

SCORING_MODULE: Final = "credit_risk.scoring.batch"

INPUT_CONF_KEY: Final = "input"
INCOMING_DIR: Final = "incoming"
DEFAULT_INPUT: Final = "processed/serving_pool.parquet"
INPUT_SUFFIXES: Final[frozenset[str]] = frozenset({".csv", ".parquet"})

# Explaining a 500-account call list is the long pole: SHAP and LIME per
# account. An hour means something is wedged.
SUBPROCESS_TIMEOUT: Final = 3600

# EX_TEMPFAIL: the module's "the API is down, try later". Copied from
# credit_risk.scoring.batch.EXIT_TRANSIENT, which this interpreter cannot
# import; tests/unit/test_batch_scoring.py holds the two equal.
TRANSIENT_EXIT_CODE: Final = 75

DAG_ID: Final = "credit_risk_scoring"

ALERTS_URL_VARIABLE: Final = "ALERTMANAGER_ALERTS_URL"
DEFAULT_ALERTS_URL: Final = "http://alertmanager:9093/api/v2/alerts"
FAILURE_ALERT_NAME: Final = "PipelineTaskFailed"
FAILURE_ALERT_SEVERITY: Final = "critical"
# Posted once with an explicit end, so Alertmanager does not resolve it after
# resolve_timeout. A day outlasts any plausible gap before somebody re-runs it.
FAILURE_ALERT_TTL: Final = timedelta(hours=24)
ALERT_POST_TIMEOUT_SECONDS: Final = 5.0
RUN_ALERT_TASK_ID: Final = "dagrun"

DEFAULT_ARGS: Final[dict[str, Any]] = {
    "owner": "p1-data",
    # Only exit 75 is retried (see _raise_for_exit): an API restarting under
    # a deploy is back within a minute or two.
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}


def _project_python() -> str:
    """Interpreter that has the project installed: Variable, then env, then ``python``."""
    fallback = os.environ.get(PYTHON_VARIABLE, DEFAULT_PYTHON)
    try:
        from airflow.models import Variable

        return str(Variable.get(PYTHON_VARIABLE, default_var=fallback))
    except Exception:  # noqa: BLE001 - no metadata DB in unit tests or `airflow dags test`
        return fallback


def _data_dir() -> Path:
    """Directory the scoring reads and writes, from the container's point of view."""
    override = os.environ.get(DATA_DIR_VARIABLE)
    if override:
        return Path(override)
    return Path(os.environ.get("AIRFLOW_HOME", DEFAULT_AIRFLOW_HOME)) / "data"


def _subprocess_env() -> dict[str, str]:
    """The child's environment: data paths on the mounted volume, the API by service name."""
    env = os.environ.copy()
    data_dir = _data_dir()
    # setdefault: a value already set on the container is a deliberate choice.
    env.setdefault("DATA_DIR", str(data_dir))
    env.setdefault("RAW_DIR", str(data_dir / "raw"))
    env.setdefault("PROCESSED_DIR", str(data_dir / "processed"))
    env.setdefault(API_URL_VARIABLE, DEFAULT_API_URL)
    return env


def _execute(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``<project python> -m <module>`` and hand back what happened."""
    command = [_project_python(), "-m", module, *args]
    log.info("running: %s", " ".join(command))
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT,
            env=_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        # Not retryable: three attempts at an hour each outlive dagrun_timeout.
        raise AirflowFailException(
            f"{module} exceeded the {SUBPROCESS_TIMEOUT}s budget and was killed"
        ) from exc

    if completed.stderr:
        log.info("stderr from %s:\n%s", module, completed.stderr.strip())
    return completed


def _raise_for_exit(module: str, completed: subprocess.CompletedProcess[str]) -> None:
    """Exit 75 raises AirflowException (retried); any other non-zero, AirflowFailException."""
    if completed.returncode == 0:
        return
    detail = (
        f"stdout: {completed.stdout.strip()[-2000:]} stderr: {completed.stderr.strip()[-2000:]}"
    )
    if completed.returncode == TRANSIENT_EXIT_CODE:
        raise AirflowException(
            f"{module} failed transiently (exit {TRANSIENT_EXIT_CODE}); Airflow will retry. "
            + detail
        )
    raise AirflowFailException(f"{module} exited with {completed.returncode}. {detail}")


def run_module(module: str, *args: str) -> str:
    """Run ``<project python> -m <module>`` and return its stdout, which must not be empty.

    Every subcommand prints its result -- the run directory or the summary --
    so silence means the module ran no entry point at all, and that must not
    report green.
    """
    completed = _execute(module, *args)
    _raise_for_exit(module, completed)
    if not completed.stdout.strip():
        raise AirflowFailException(f"{module} exited 0 without printing anything, so nothing ran")
    log.info("stdout from %s:\n%s", module, completed.stdout.strip())
    return completed.stdout


def resolve_input(conf: Mapping[str, Any] | None, data_dir: Path) -> Path:
    """The file to score: the conf's, else the newest incoming file, else the demo pool.

    A conf path is resolved against the data directory and refused if it
    leaves it -- the conf is typed into a form, and ``../`` would otherwise
    point the scorer at any file the Airflow user can read.
    """
    requested = (conf or {}).get(INPUT_CONF_KEY)
    if requested:
        root = data_dir.resolve()
        candidate = (root / str(requested)).resolve()
        if root not in candidate.parents:
            raise AirflowFailException(
                f"conf input {requested!r} is outside the data directory {data_dir}"
            )
        return candidate

    incoming = data_dir / INCOMING_DIR
    if incoming.is_dir():
        # Hidden files are uploads still being written (".name.csv.part" style).
        candidates = [
            path
            for path in incoming.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in INPUT_SUFFIXES
        ]
        if candidates:
            return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    return data_dir / DEFAULT_INPUT


def _last_line(stdout: str) -> str:
    return stdout.strip().splitlines()[-1].strip()


def _alerts_url() -> str:
    return os.environ.get(ALERTS_URL_VARIABLE, DEFAULT_ALERTS_URL).strip()


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def _alert_labels(dag_id: str, task_id: str) -> dict[str, str]:
    # Nothing run-specific, so a later success resolves exactly this alert.
    return {
        "alertname": FAILURE_ALERT_NAME,
        "severity": FAILURE_ALERT_SEVERITY,
        "dag_id": dag_id,
        "task_id": task_id,
    }


def _post_alerts(alerts: list[dict[str, Any]]) -> bool:
    """POST alerts to Alertmanager. Returns whether it accepted them; never raises."""
    url = _alerts_url()
    if not url:
        log.info("%s is empty; not sending %d alert(s)", ALERTS_URL_VARIABLE, len(alerts))
        return False
    request = urllib.request.Request(
        url,
        data=json.dumps(alerts).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=ALERT_POST_TIMEOUT_SECONDS) as response:  # noqa: S310
            status = int(response.status)
    except Exception as exc:  # noqa: BLE001 - an alert must never fail the thing it reports
        log.warning("could not post %d alert(s) to %s: %s", len(alerts), url, exc)
        return False
    log.info("posted %d alert(s) to %s: HTTP %d", len(alerts), url, status)
    return 200 <= status < 300


def _failed_task_instances(context: dict[str, Any]) -> list[Any]:
    """The failed task instances of this run, best effort; [] when there are none."""
    dag_run = context.get("dag_run")
    if dag_run is None:
        return []
    try:
        return list(dag_run.get_task_instances(state=["failed"]))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not list the failed tasks of the run: %s", exc)
        return []


# latest.json is written last and atomically, so any failure leaves it on the
# previous run -- which is what the risk desk needs to know first.
_CONSEQUENCE: Final = (
    "No new call list was published: scored/latest.json still names the last successful run. "
)


def notify_failure(context: dict[str, Any]) -> None:
    """DAG on_failure_callback: one Alertmanager alert per failed task; never raises."""
    try:
        dag_run = context.get("dag_run")
        dag_id = str(getattr(dag_run, "dag_id", None) or DAG_ID)
        run_id = str(getattr(dag_run, "run_id", None) or "unknown")
        reason = str(context.get("reason") or "task_failure")
        now = datetime.now(timezone.utc)  # noqa: UP017
        timing = {"startsAt": _rfc3339(now), "endsAt": _rfc3339(now + FAILURE_ALERT_TTL)}
        alerts = []
        for ti in _failed_task_instances(context):
            task_id = str(getattr(ti, "task_id", None) or "unknown")
            log_url = str(getattr(ti, "log_url", None) or "")
            alerts.append(
                {
                    "labels": _alert_labels(dag_id, task_id),
                    "annotations": {
                        "summary": f"{dag_id}: task {task_id} failed",
                        "description": (
                            f"Task {task_id} failed in run {run_id} of {dag_id} ({reason}). "
                            + _CONSEQUENCE
                            + f"Log: {log_url or 'see the Airflow UI'}"
                        ),
                        "run_id": run_id,
                        "log_url": log_url,
                    },
                    **timing,
                    "generatorURL": log_url,
                }
            )
        if not alerts:
            alerts.append(
                {
                    "labels": _alert_labels(dag_id, RUN_ALERT_TASK_ID),
                    "annotations": {
                        "summary": f"{dag_id}: run {run_id} failed ({reason})",
                        "description": (
                            f"Run {run_id} of {dag_id} failed ({reason}) with no failed task. "
                            + _CONSEQUENCE
                        ),
                        "run_id": run_id,
                    },
                    **timing,
                }
            )
        _post_alerts(alerts)
    except Exception:  # noqa: BLE001
        log.exception("could not build the failure alert; the run's own state is unaffected")


def resolve_failure_alerts(context: dict[str, Any]) -> None:
    """DAG on_success_callback: resolve any failure alert a previous run raised."""
    try:
        dag = context.get("dag")
        dag_run = context.get("dag_run")
        dag_id = str(getattr(dag, "dag_id", None) or DAG_ID)
        run_id = str(getattr(dag_run, "run_id", None) or "unknown")
        task_ids = [str(task_id) for task_id in getattr(dag, "task_ids", [])]
        now = _rfc3339(datetime.now(timezone.utc))  # noqa: UP017
        alerts = [
            {
                "labels": _alert_labels(dag_id, task_id),
                "annotations": {
                    "summary": f"{dag_id}: {task_id} is no longer failing",
                    "description": f"Run {run_id} of {dag_id} succeeded.",
                    "run_id": run_id,
                },
                "startsAt": now,
                "endsAt": now,
            }
            for task_id in [*task_ids, RUN_ALERT_TASK_ID]
        ]
        _post_alerts(alerts)
    except Exception:  # noqa: BLE001
        log.exception("could not resolve the failure alerts")


@dag(
    dag_id=DAG_ID,
    description="Score a file of accounts through the credit API and publish the call list",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    on_failure_callback=notify_failure,
    on_success_callback=resolve_failure_alerts,
    dagrun_timeout=timedelta(hours=2),
    tags=["ddm501", "credit-risk", "scoring"],
    doc_md=__doc__,
)
def credit_risk_scoring() -> None:
    """Validate, score and publish one batch of accounts."""

    @task(task_id="validate_batch")
    def validate_batch_task(dag_run: Any = None) -> str:
        """Choose the input, validate it into a new run directory, return that directory."""
        data_dir = _data_dir()
        source = resolve_input(getattr(dag_run, "conf", None), data_dir)
        log.info("scoring %s", source)
        stdout = run_module(
            SCORING_MODULE, "validate", "--input", str(source), "--data-dir", str(data_dir)
        )
        return _last_line(stdout)

    @task(task_id="score_batch")
    def score_batch_task(run_dir: str) -> str:
        """Score the valid rows through the API, in chunks of its batch cap."""
        return _last_line(
            run_module(
                SCORING_MODULE, "score", "--run-dir", run_dir, "--data-dir", str(_data_dir())
            )
        )

    @task(task_id="publish_call_list")
    def publish_call_list_task(run_dir: str, dag_run: Any = None) -> dict[str, Any]:
        """Rank, explain the call list, publish; the summary becomes the task's XCom."""
        args = ["publish", "--run-dir", run_dir, "--data-dir", str(_data_dir())]
        run_id = getattr(dag_run, "run_id", None)
        if run_id:
            args += ["--dag-run-id", str(run_id)]
        return dict(json.loads(run_module(SCORING_MODULE, *args)))

    # The run directory travels as each task's return value, so the XCom
    # hand-off is also the dependency: no task can start without the one before.
    publish_call_list_task(score_batch_task(validate_batch_task()))


credit_risk_scoring()
