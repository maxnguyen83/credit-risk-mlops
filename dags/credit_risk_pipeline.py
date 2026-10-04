"""Airflow DAG: ingest, validate, split, train, gate, register, report.

Three decisions are visible in this file and all three were made on purpose.

**Fail fast on bad data.** ``validate_raw`` runs as its own task and exits
non-zero when more than ``schema.MAX_BAD_ROW_FRACTION`` of rows are broken.
Training on a frame that failed validation produces a model that looks fine in
MLflow and is wrong in production, which costs far more to discover than a red
DAG costs to fix.

**One subprocess per step, never an in-process import.** Two reasons, and they
are independent. Airflow 2.8.4 pins ``pandas==2.1.4``; the project runs on
``pandas==2.2.3`` with LightGBM, SHAP and pydantic-settings on top, and
installing both sets of pins into one interpreter resolves to something neither
side was tested against. And in the shipped image the project is installed only
into ``/opt/credit-risk/venv`` -- the Airflow interpreter has no ``credit_risk``
on its path at all, so an in-process ``from credit_risk...`` import inside a
task is a ``ModuleNotFoundError`` waiting for the first run. Everything goes
through :func:`run_module`, which shells out to the interpreter named by
``CREDIT_RISK_PYTHON``. The module level of this file imports only stdlib and
Airflow, so DAG parsing cannot fail because a project dependency is missing.

**Explicit data directory.** ``config.PROJECT_ROOT`` is derived from the
package's own location, which for a pip-installed package is inside the venv --
a directory no volume mounts. :func:`run_module` therefore exports
``DATA_DIR``/``RAW_DIR``/``PROCESSED_DIR`` pointing at ``$AIRFLOW_HOME/data``,
where compose mounts the host's ``./data``. Without it the parquet is written
into the container filesystem and disappears with the container.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException, AirflowFailException
from airflow.models.baseoperator import chain

log = logging.getLogger(__name__)

PYTHON_VARIABLE: Final = "CREDIT_RISK_PYTHON"
DEFAULT_PYTHON: Final = "python"

DATA_DIR_VARIABLE: Final = "CREDIT_RISK_DATA_DIR"
DEFAULT_AIRFLOW_HOME: Final = "/opt/airflow"

# Entry points, named once. They run in the other interpreter, so a typo here
# surfaces as "No module named ..." rather than an import error at parse time
# -- hence keeping them together where they are easy to check against the
# source tree.
DOWNLOAD_MODULE: Final = "credit_risk.data.download"
VALIDATE_MODULE: Final = "credit_risk.data.validate"
SPLIT_MODULE: Final = "credit_risk.data.split"
FEATURES_MODULE: Final = "credit_risk.features.build"
TRAIN_MODULE: Final = "credit_risk.models.train"
EVALUATE_MODULE: Final = "credit_risk.models.evaluate"
REGISTRY_MODULE: Final = "credit_risk.models.registry"
REPORT_MODULE: Final = "credit_risk.models.report"

# Subprocess budget. HPO with 5-fold CV on 20,000 rows is minutes, not hours;
# an hour means something is wedged and the task should say so.
SUBPROCESS_TIMEOUT: Final = 3600

# Exit status a module uses for "failed for a reason that may clear on its own"
# (EX_TEMPFAIL). credit_risk.data.download.EXIT_TRANSIENT is the same number; it
# is copied rather than imported because this interpreter cannot import the
# project (ADR 0006), and tests/data_quality/test_raw.py holds the two equal.
TRANSIENT_EXIT_CODE: Final = 75

# XCom key validate_raw pushes its report under, on success and on failure.
VALIDATION_REPORT_XCOM_KEY: Final = "validation_report"

DAG_ID: Final = "credit_risk_pipeline"

# Failure alerts go straight to Alertmanager's v2 API, so a refused gate or a
# dead task reaches the same receivers as every Prometheus alert. The URL is the
# compose service name; set the variable to another URL to point elsewhere, or
# to an empty string to switch the alerts off.
ALERTS_URL_VARIABLE: Final = "ALERTMANAGER_ALERTS_URL"
DEFAULT_ALERTS_URL: Final = "http://alertmanager:9093/api/v2/alerts"
FAILURE_ALERT_NAME: Final = "PipelineTaskFailed"
FAILURE_ALERT_SEVERITY: Final = "critical"
# Posted once, so it needs an explicit end: without one Alertmanager resolves it
# after resolve_timeout (5m) and reports a broken pipeline as fixed. A day spans
# the @daily schedule; a successful run resolves it sooner.
FAILURE_ALERT_TTL: Final = timedelta(hours=24)
# A callback that waits on a dead Alertmanager holds up the scheduler loop.
ALERT_POST_TIMEOUT_SECONDS: Final = 5.0

DEFAULT_ARGS: Final[dict[str, Any]] = {
    "owner": "p1-data",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=20),
    "depends_on_past": False,
}


def _project_python() -> str:
    """Interpreter that has the project installed.

    Variable first so it can be changed from the UI without a redeploy, env
    var second for local runs, and a plain ``python`` last so a fresh checkout
    does something sensible.
    """
    fallback = os.environ.get(PYTHON_VARIABLE, DEFAULT_PYTHON)
    try:
        from airflow.models import Variable

        return str(Variable.get(PYTHON_VARIABLE, default_var=fallback))
    except Exception:  # noqa: BLE001 - no metadata DB in unit tests or `airflow dags test`
        return fallback


def _data_dir() -> Path:
    """Directory the pipeline reads and writes, from the container's point of view."""
    override = os.environ.get(DATA_DIR_VARIABLE)
    if override:
        return Path(override)
    return Path(os.environ.get("AIRFLOW_HOME", DEFAULT_AIRFLOW_HOME)) / "data"


def _subprocess_env() -> dict[str, str]:
    """The child's environment, with the data paths pinned to the mounted volume."""
    env = os.environ.copy()
    data_dir = _data_dir()
    # pydantic-settings maps these onto Settings.data_dir / raw_dir /
    # processed_dir case-insensitively. setdefault, not assignment: a value
    # already set on the container (compose, .env) is somebody's deliberate
    # choice and outranks this fallback.
    env.setdefault("DATA_DIR", str(data_dir))
    env.setdefault("RAW_DIR", str(data_dir / "raw"))
    env.setdefault("PROCESSED_DIR", str(data_dir / "processed"))
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
        # AirflowFailException, not the bare TimeoutExpired: a plain exception
        # is retryable, and three attempts at the one-hour budget is three
        # hours against a two-hour dagrun_timeout. The run would be killed
        # mid-retry and leave no failure record worth reading.
        raise AirflowFailException(
            f"{module} exceeded the {SUBPROCESS_TIMEOUT}s budget and was killed"
        ) from exc

    if completed.stderr:
        log.info("stderr from %s:\n%s", module, completed.stderr.strip())
    return completed


def _raise_for_exit(module: str, completed: subprocess.CompletedProcess[str]) -> None:
    """Turn a non-zero exit into the right kind of Airflow failure.

    TRANSIENT_EXIT_CODE raises a plain AirflowException, which Airflow retries
    with DEFAULT_ARGS' backoff: the module has already retried what it could
    and is saying "later". Every other code is a real defect -- bad data, a
    refused gate, a crash -- and raises AirflowFailException, because retrying
    those only delays the alert.
    """
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


def run_module(module: str, *args: str, expect_output: bool = True) -> str:
    """Run ``<project python> -m <module>`` and return its stdout.

    A non-zero exit fails the task without a retry, except TRANSIENT_EXIT_CODE,
    which is retried (see :func:`_raise_for_exit`).
    """
    completed = _execute(module, *args)
    _raise_for_exit(module, completed)
    if expect_output and not completed.stdout.strip():
        # Every module invoked here prints its result to stdout: the parquet
        # path, the validation report, the splits manifest, the run summary.
        # A module with no `if __name__ == "__main__"` block also exits 0 --
        # `-m` imports it, defines some functions and stops -- so the exit
        # status alone cannot tell "ran and succeeded" from "was never wired
        # up as an entry point". Silence means the second, and a fairness gate
        # that never executed must not report green.
        raise AirflowFailException(
            f"{module} exited 0 without printing anything, so nothing ran. "
            f'It has no `if __name__ == "__main__"` entry point.'
        )
    log.info("stdout from %s:\n%s", module, completed.stdout.strip())
    return completed.stdout


def _alerts_url() -> str:
    return os.environ.get(ALERTS_URL_VARIABLE, DEFAULT_ALERTS_URL).strip()


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def _alert_labels(dag_id: str, task_id: str) -> dict[str, str]:
    # The identity of an alert in Alertmanager. Nothing run-specific goes here,
    # so a later success can resolve exactly the alert a failure raised.
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
    """The failed task instances of this run, best effort."""
    dag_run = context.get("dag_run")
    if dag_run is not None:
        try:
            failed = list(dag_run.get_task_instances(state=["failed"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not list the failed tasks of the run: %s", exc)
        else:
            if failed:
                return failed
    ti = context.get("task_instance") or context.get("ti")
    return [ti] if ti is not None else []


def notify_failure(context: dict[str, Any]) -> None:
    """DAG on_failure_callback: one Alertmanager alert per failed task.

    Runs where Airflow runs DAG callbacks -- the DAG processor under the
    scheduler, in process under ``airflow dags test``. It must never raise:
    a callback that throws loses the alert and buries the run's own failure
    under a second traceback.
    """
    try:
        dag_run = context.get("dag_run")
        dag_id = str(getattr(dag_run, "dag_id", None) or DAG_ID)
        run_id = str(getattr(dag_run, "run_id", None) or "unknown")
        reason = str(context.get("reason") or "task_failure")
        now = datetime.now(timezone.utc)  # noqa: UP017
        alerts = []
        for ti in _failed_task_instances(context) or [None]:
            task_id = str(getattr(ti, "task_id", None) or "unknown")
            log_url = str(getattr(ti, "log_url", None) or "")
            alerts.append(
                {
                    "labels": _alert_labels(dag_id, task_id),
                    "annotations": {
                        "summary": f"{dag_id}: task {task_id} failed",
                        "description": (
                            f"Task {task_id} failed in run {run_id} of {dag_id} ({reason}). "
                            "Nothing downstream of it ran, so no new model was registered. "
                            f"Log: {log_url or 'see the Airflow UI'}"
                        ),
                        "run_id": run_id,
                        "log_url": log_url,
                    },
                    "startsAt": _rfc3339(now),
                    "endsAt": _rfc3339(now + FAILURE_ALERT_TTL),
                    "generatorURL": log_url,
                }
            )
        _post_alerts(alerts)
    except Exception:  # noqa: BLE001
        log.exception("could not build the failure alert; the run's own state is unaffected")


def resolve_failure_alerts(context: dict[str, Any]) -> None:
    """DAG on_success_callback: resolve any failure alert a previous run raised.

    Posts every task's alert with endsAt = now. Alertmanager resolves the ones
    that were firing and sends nothing for the ones that never were.
    """
    try:
        dag = context.get("dag")
        dag_id = str(getattr(dag, "dag_id", None) or DAG_ID)
        task_ids = [str(task_id) for task_id in getattr(dag, "task_ids", [])]
        now = _rfc3339(datetime.now(timezone.utc))  # noqa: UP017
        alerts = [
            {"labels": _alert_labels(dag_id, task_id), "startsAt": now, "endsAt": now}
            for task_id in task_ids
        ]
        if alerts:
            _post_alerts(alerts)
    except Exception:  # noqa: BLE001
        log.exception("could not resolve the failure alerts")


@dag(
    dag_id=DAG_ID,
    description="UCI credit-default ingestion, training, fairness gate and registration",
    schedule="@daily",
    # timezone.utc rather than datetime.UTC: the project floor is Python
    # 3.11 but the Airflow image tag decides which interpreter parses this
    # file, and a DAG that fails to import is invisible in the UI.
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    on_failure_callback=notify_failure,
    on_success_callback=resolve_failure_alerts,
    dagrun_timeout=timedelta(hours=2),
    tags=["ddm501", "credit-risk"],
    doc_md=__doc__,
)
def credit_risk_pipeline() -> None:
    """Daily end-to-end pipeline for the credit default early-warning model."""

    @task(task_id="download_raw")
    def download_raw_task() -> str:
        """Fetch the UCI archive and land it as parquet. Idempotent by design.

        The module retries a dropped connection or a 5xx itself, and exits
        TRANSIENT_EXIT_CODE when the archive stays unreachable; that one exit
        is retried by Airflow (DEFAULT_ARGS), every other failure is not.
        """
        return run_module(DOWNLOAD_MODULE).strip()

    @task(task_id="validate_raw")
    def validate_raw_task(raw_path: str, ti: Any = None) -> dict[str, Any]:
        """Schema, ranges and null checks on the raw frame.

        The module exits 1 when more than schema.MAX_BAD_ROW_FRACTION of rows
        are broken, which fails this task and stops the DAG here. That is the
        intent: no downstream task should ever train on a frame that failed
        validation, and a red task is the cheapest possible way to say so.

        The report is pushed to XCom under VALIDATION_REPORT_XCOM_KEY *before*
        the exit status is checked, so a failed run still leaves the evidence
        behind. A passing run also returns it, as the task's return_value.
        Fewer broken rows than the tolerance pass this task; clean_and_split
        then quarantines them rather than training on them.
        """
        completed = _execute(VALIDATE_MODULE, "--raw", raw_path)
        try:
            report: dict[str, Any] | None = json.loads(completed.stdout)
        except ValueError:
            report = None
        if report is not None and ti is not None:
            ti.xcom_push(key=VALIDATION_REPORT_XCOM_KEY, value=report)
        _raise_for_exit(VALIDATE_MODULE, completed)
        if report is None:
            raise AirflowFailException(
                f"{VALIDATE_MODULE} exited 0 without a JSON report on stdout; nothing was checked"
            )
        return report

    @task(task_id="clean_and_split")
    def clean_and_split_task(raw_path: str) -> dict[str, Any]:
        """Normalise codes, cut six deterministic batches, write the manifest."""
        return json.loads(run_module(SPLIT_MODULE, "--raw", raw_path))

    @task(task_id="build_features")
    def build_features_task(manifest: dict[str, Any]) -> dict[str, Any]:
        """Preflight the model input matrix for the train and test splits.

        Deliberately not a materialisation step. models/train.py builds its own
        matrix from the split it loads -- that is what makes features/build.py
        the single source and blocks train/serve skew -- so a feature parquet
        written here would be a file nothing reads and a second copy of the
        matrix to keep in step. What this buys is a red task in seconds when
        the feature contract is broken, instead of one after a five-minute
        hyperparameter search that was doomed at its first row.
        """
        paths = [manifest["splits"][name]["path"] for name in ("train", "test")]
        return json.loads(run_module(FEATURES_MODULE, *paths))

    @task(task_id="train_candidates")
    def train_candidates_task() -> None:
        """LogisticRegression baseline plus LightGBM with HPO, logged to MLflow."""
        run_module(TRAIN_MODULE)

    @task(task_id="evaluate_and_gate")
    def evaluate_and_gate_task() -> None:
        """Score on the held-out batch and apply the performance and fairness gates."""
        run_module(EVALUATE_MODULE)

    @task(task_id="register_model")
    def register_model_task() -> None:
        """Register the gated candidate in the MLflow Model Registry."""
        run_module(REGISTRY_MODULE)

    @task(task_id="publish_report")
    def publish_report_task() -> None:
        """Publish the HTML run report: experiment comparison, SHAP, fairness."""
        run_module(REPORT_MODULE)

    raw_path = download_raw_task()
    validation = validate_raw_task(raw_path)
    manifest = clean_and_split_task(raw_path)
    features = build_features_task(manifest)
    trained = train_candidates_task()
    gated = evaluate_and_gate_task()
    registered = register_model_task()
    published = publish_report_task()

    # Cleaning depends on validation passing, not on anything it returns --
    # without this edge Airflow would happily clean and split in parallel with
    # the check that is supposed to stop it. chain() rather than `>>` so the
    # dependency is a call, not a bare expression a linter may read as dead.
    chain(validation, manifest)
    chain(features, trained, gated, registered, published)


credit_risk_pipeline()
