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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
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


def run_module(module: str, *args: str, expect_output: bool = True) -> str:
    """Run ``<project python> -m <module>`` and return its stdout.

    Failures raise AirflowFailException rather than a generic error: a module
    that exits non-zero is a real defect, and retrying it two more times only
    delays the alert.
    """
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
    if completed.returncode != 0:
        raise AirflowFailException(
            f"{module} exited with {completed.returncode}. "
            f"stdout: {completed.stdout.strip()[-2000:]} "
            f"stderr: {completed.stderr.strip()[-2000:]}"
        )
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


@dag(
    dag_id="credit_risk_pipeline",
    description="UCI credit-default ingestion, training, fairness gate and registration",
    schedule="@daily",
    # timezone.utc rather than datetime.UTC: the project floor is Python
    # 3.11 but the Airflow image tag decides which interpreter parses this
    # file, and a DAG that fails to import is invisible in the UI.
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    dagrun_timeout=timedelta(hours=2),
    tags=["ddm501", "credit-risk"],
    doc_md=__doc__,
)
def credit_risk_pipeline() -> None:
    """Daily end-to-end pipeline for the credit default early-warning model."""

    @task(task_id="download_raw")
    def download_raw_task() -> str:
        """Fetch the UCI archive and land it as parquet. Idempotent by design."""
        return run_module(DOWNLOAD_MODULE).strip()

    @task(task_id="validate_raw")
    def validate_raw_task(raw_path: str) -> dict[str, Any]:
        """Schema, ranges and null checks on the raw frame.

        The module exits 1 when more than schema.MAX_BAD_ROW_FRACTION of rows
        are broken, which fails this task and stops the DAG here. That is the
        intent: no downstream task should ever train on a frame that failed
        validation, and a red task is the cheapest possible way to say so.
        The report lands in XCom either way, so a failed run still leaves the
        evidence behind.
        """
        return json.loads(run_module(VALIDATE_MODULE, "--raw", raw_path))

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
