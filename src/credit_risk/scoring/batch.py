"""Batch scoring: a file of accounts in, a ranked call list out.

The risk team can call about one customer in ten each month
(``settings.intervention_capacity_fraction``). This module turns a portfolio
extract into that list: it checks every row against the API's own request
model, scores the valid ones through ``POST /api/v1/predict/batch``, ranks them
by probability of default, asks ``POST /api/v1/explain`` why for the accounts
that made the list, and publishes the result under ``<data_dir>/scored/``.

It scores through the running API rather than loading the model itself, for
the reason ``serving/routes.py`` imports ``features/build.py``: one path from a
record to a decision. A second loader here could serve another version, apply
another threshold policy, or skip the metrics the fairness alert reads -- and
nothing would say so.

Each stage reads what the previous one wrote into one run directory, so the
DAG can run them as separate tasks:

    python -m credit_risk.scoring.batch validate --input PATH   # prints the run dir
    python -m credit_risk.scoring.batch score    --run-dir DIR
    python -m credit_risk.scoring.batch publish  --run-dir DIR [--dag-run-id ID]
    python -m credit_risk.scoring.batch run      --input PATH   # all three, locally

Exit 0 on success; 2 for bad input (nothing in the file can be scored, a column
the API does not take, a run directory missing a stage); 75 when the API stayed
unreachable after retries, which the DAG retries; 1 for anything else,
including a served model that changed while the batch was being scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests
from pydantic import ValidationError

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.data.validate import BATCH_COL
from credit_risk.serving.models import (
    BatchPredictResponse,
    CreditApplication,
    PredictResponse,
    utc_now_iso,
)

log = logging.getLogger(__name__)

ACCOUNT_ID: Final = "account_id"
INPUT_ROW: Final = "input_row"

# The API's own field list, read off its request model. A field added to or
# removed from CreditApplication changes what this module expects with it.
INPUT_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in CreditApplication.model_fields if name != ACCOUNT_ID
)

# Columns that may ride along and are dropped before validation. The label is
# what the batch is meant to predict; AGE_GROUP and the batch number are
# bookkeeping that data/split.py adds, so the demo serving_pool.parquet carries
# them. Anything else unknown fails the file: the API refuses unknown fields
# (extra="forbid") so a stale extract is told so rather than scored.
DROPPED_COLUMNS: Final[tuple[str, ...]] = (schema.TARGET, schema.AGE_GROUP, BATCH_COL)

SCORED_DIR: Final = "scored"
LATEST_JSON: Final = "latest.json"
VALID_PARQUET: Final = "valid.parquet"
REJECTED_CSV: Final = "rejected.csv"
VALIDATION_JSON: Final = "validation.json"
SCORED_PARQUET: Final = "scored.parquet"
SCORING_JSON: Final = "scoring.json"
SCORES_CSV: Final = "scores.csv"
CALL_LIST_CSV: Final = "call_list.csv"
SUMMARY_JSON: Final = "summary.json"

SCORES_COLUMNS: Final[tuple[str, ...]] = (
    "rank",
    ACCOUNT_ID,
    "default_probability",
    "risk_band",
    "decision",
    "in_call_list",
)
CALL_LIST_COLUMNS: Final[tuple[str, ...]] = (*SCORES_COLUMNS[:-1], "top_reasons")
PROBABILITY_DECIMALS: Final = 4
REASON_SEPARATOR: Final = " | "

BATCH_PATH: Final = "/api/v1/predict/batch"
EXPLAIN_PATH: Final = "/api/v1/explain"

# (connect, read). A full 1,000-row batch is well under a second of model time;
# the read budget is for a container that is busy or still loading its model.
# One explanation is SHAP plus a LIME fit, the most expensive call the API has.
BATCH_TIMEOUT: Final[tuple[float, float]] = (5.0, 120.0)
EXPLAIN_TIMEOUT: Final[tuple[float, float]] = (5.0, 30.0)

# One try and two retries, 2 s then 4 s apart: enough to ride out an API
# restart between chunks. Longer outages are the DAG's to wait out (exit 75).
ATTEMPTS: Final = 3
BACKOFF_SECONDS: Final = 2.0

# For /explain these mean "the service is down, not this record": every later
# call would fail the same way, so the rest of the list goes out without
# reasons instead of each account waiting out its own timeout.
SERVICE_DOWN_STATUS: Final[frozenset[int]] = frozenset({502, 503, 504})

# Under this policy each group is judged at its own cutoff, so a run has more
# than one threshold by design (config.Settings.threshold_policy).
GROUP_AWARE_POLICY: Final = "group_aware_equalized_odds"

EXIT_FAILED: Final = 1
EXIT_BAD_INPUT: Final = 2
# EX_TEMPFAIL, the exit both DAGs retry; credit_risk.data.download uses the same.
EXIT_TRANSIENT: Final = 75


class BatchScoringError(RuntimeError):
    """A batch that cannot be scored as asked."""


class BadInputError(BatchScoringError):
    """The input file, or the run directory a stage reads, cannot be used."""


class ApiUnavailableError(BatchScoringError):
    """The API did not answer, or answered 5xx. ``status`` is None for no answer."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ApiResponseError(BatchScoringError):
    """The API refused the request (4xx) or answered in a shape we do not know."""


class ServedModelChangedError(BatchScoringError):
    """Chunks of one batch were scored by different models or thresholds."""


# --------------------------------------------------------------- validation


@dataclass(frozen=True)
class Validation:
    """Rows the API will accept, rows it would refuse, and why."""

    valid: pd.DataFrame  # input_row, account_id, *INPUT_COLUMNS
    rejected: pd.DataFrame  # row, account_id, reason
    n_input: int
    rejected_by_reason: dict[str, int]


def load_input(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet extract."""
    if not path.is_file():
        raise BadInputError(f"{path} does not exist")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            # Identifiers as text: read as numbers, "00123" would become 123.
            return pd.read_csv(path, dtype={ACCOUNT_ID: str, schema.ID_COL: str})
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except (ValueError, OSError) as exc:
        raise BadInputError(f"{path} could not be read: {exc}") from exc
    raise BadInputError(f"{path.name}: expected a .csv or .parquet file")


def _is_missing(value: Any) -> bool:
    # An empty cell, NaN or infinity is a missing value. JSON -- the API's wire
    # format -- cannot carry NaN or infinity, so the nearest thing the API ever
    # sees is null, and a null is what it refuses.
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and not math.isfinite(value)


def _account_id(value: Any, row: int) -> str:
    """The account reference: the file's own, or the input row it came from."""
    if _is_missing(value):
        # A call list is useless without someone to call; the row number at
        # least leads back to the line in the file.
        return f"row-{row}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _check_columns(frame: pd.DataFrame) -> None:
    present = {str(column) for column in frame.columns}
    missing = [column for column in INPUT_COLUMNS if column not in present]
    known = {*INPUT_COLUMNS, *DROPPED_COLUMNS, ACCOUNT_ID, schema.ID_COL}
    unknown = sorted(present - known)
    problems = []
    if missing:
        problems.append(f"missing column(s) the API requires: {', '.join(missing)}")
    if unknown:
        problems.append(f"column(s) the API does not accept: {', '.join(unknown)}")
    if problems:
        raise BadInputError("; ".join(problems))


def validate_frame(frame: pd.DataFrame) -> Validation:
    """Check every row with the API's own request model, so the rules cannot drift."""
    _check_columns(frame)
    if ACCOUNT_ID in frame.columns:
        ids = frame[ACCOUNT_ID].tolist()
    elif schema.ID_COL in frame.columns:
        ids = frame[schema.ID_COL].tolist()
    else:
        ids = [None] * len(frame)

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    records = frame[list(INPUT_COLUMNS)].to_dict(orient="records")
    for row, (record, raw_id) in enumerate(zip(records, ids, strict=True), start=1):
        account_id = _account_id(raw_id, row)
        candidate = {key: None if _is_missing(value) else value for key, value in record.items()}
        try:
            application = CreditApplication.model_validate({ACCOUNT_ID: account_id, **candidate})
        except ValidationError as exc:
            # Field and rule only, never the submitted value -- the same
            # promise the API's ErrorResponse makes.
            problems = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
            reasons.update(problems)
            rejected.append({"row": row, ACCOUNT_ID: account_id, "reason": "; ".join(problems)})
            continue
        valid.append({INPUT_ROW: row, **application.model_dump()})

    return Validation(
        valid=pd.DataFrame(valid, columns=[INPUT_ROW, ACCOUNT_ID, *INPUT_COLUMNS]),
        rejected=pd.DataFrame(rejected, columns=["row", ACCOUNT_ID, "reason"]),
        n_input=len(frame),
        rejected_by_reason=dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
    )


def new_run_dir(data_dir: Path, now: datetime | None = None) -> Path:
    """Create ``<data_dir>/scored/<UTC stamp>``; a second run in the same second gets ``-2``."""
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    parent = data_dir / SCORED_DIR
    parent.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = parent / (stamp if suffix == 1 else f"{stamp}-{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def validate_input(input_path: Path, run_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Validate a file into ``run_dir``; raise BadInputError if no row can be scored.

    More than ``schema.MAX_BAD_ROW_FRACTION`` rejected does not stop the run --
    the valid accounts still deserve a score -- but it is recorded in the
    report and the summary, and logged, rather than dropped quietly.
    """
    result = validate_frame(load_input(input_path))
    n_rejected = len(result.rejected)
    fraction = n_rejected / result.n_input if result.n_input else 0.0
    report: dict[str, Any] = {
        "input_path": _relative(input_path, data_dir),
        "input_sha256": _sha256(input_path),
        "n_input": result.n_input,
        "n_rejected": n_rejected,
        "rejected_fraction": round(fraction, 4),
        "rejected_over_tolerance": fraction > schema.MAX_BAD_ROW_FRACTION,
        "rejected_by_reason": result.rejected_by_reason,
    }

    # The evidence goes to disk before any verdict, so a file that fails
    # outright still leaves rejected.csv saying why.
    run_dir.mkdir(parents=True, exist_ok=True)
    if n_rejected:
        result.rejected.to_csv(run_dir / REJECTED_CSV, index=False)
    _write_json(run_dir / VALIDATION_JSON, report)
    if result.valid.empty:
        raise BadInputError(
            f"no valid row in {input_path} ({result.n_input} read, {n_rejected} rejected); "
            f"see {run_dir / REJECTED_CSV}"
        )
    result.valid.to_parquet(run_dir / VALID_PARQUET, index=False)

    log.info(
        "validated %d rows: %d valid, %d rejected", result.n_input, len(result.valid), n_rejected
    )
    if report["rejected_over_tolerance"]:
        log.warning(
            "%.1f%% of rows were rejected (tolerance %.0f%%); scoring the rest -- see %s",
            100 * fraction,
            100 * schema.MAX_BAD_ROW_FRACTION,
            run_dir / REJECTED_CSV,
        )
    return report


# --------------------------------------------------------------- API client


class ApiClient:
    """The two endpoints batch scoring uses, with retries where they are safe."""

    def __init__(
        self,
        api_url: str,
        *,
        session: Any = None,
        attempts: int = ATTEMPTS,
        backoff: float = BACKOFF_SECONDS,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._attempts = attempts
        self._backoff = backoff
        self._sleep = sleep if sleep is not None else time.sleep

    def _post_once(self, path: str, payload: Any, timeout: tuple[float, float]) -> Any:
        url = self.api_url + path
        try:
            response = self._session.post(url, json=payload, timeout=timeout)
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            raise ApiUnavailableError(f"{type(exc).__name__} calling {url}: {exc}") from exc
        status = int(response.status_code)
        if status >= 500:
            raise ApiUnavailableError(f"HTTP {status} from {url}", status=status)
        if status >= 400:
            raise ApiResponseError(f"HTTP {status} from {url}: {_error_summary(response)}")
        try:
            return response.json()
        except ValueError as exc:
            raise ApiResponseError(f"{url} answered HTTP {status} with a non-JSON body") from exc

    def _post(self, path: str, payload: Any, timeout: tuple[float, float]) -> Any:
        """POST, retrying a dropped connection, a timeout and any 5xx with backoff."""
        for attempt in range(1, self._attempts + 1):
            try:
                return self._post_once(path, payload, timeout)
            except ApiUnavailableError as exc:
                if attempt >= self._attempts:
                    raise ApiUnavailableError(
                        f"{exc} (gave up after {attempt} attempts)", status=exc.status
                    ) from exc
                delay = self._backoff * 2 ** (attempt - 1)
                log.warning(
                    "attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt,
                    self._attempts,
                    exc,
                    delay,
                )
                self._sleep(delay)
        raise ApiUnavailableError(f"no attempt made to call {self.api_url + path}")

    def predict_batch(self, applications: list[dict[str, Any]]) -> BatchPredictResponse:
        """Score one chunk; the answer is parsed with the API's own response model."""
        body = self._post(BATCH_PATH, {"applications": applications}, BATCH_TIMEOUT)
        try:
            response = BatchPredictResponse.model_validate(body)
        except ValidationError as exc:
            raise ApiResponseError(
                f"{BATCH_PATH} answered in a shape this client does not know "
                f"({exc.error_count()} problem(s))"
            ) from exc
        sent = [application.get(ACCOUNT_ID) for application in applications]
        if [prediction.account_id for prediction in response.predictions] != sent:
            raise ApiResponseError(
                f"{BATCH_PATH} returned {len(response.predictions)} predictions for "
                f"{len(sent)} accounts, or not in the order they were sent"
            )
        return response

    def explain(self, application: dict[str, Any]) -> dict[str, Any]:
        """Explain one account. One attempt: reasons are best effort, and expensive."""
        body = self._post_once(EXPLAIN_PATH, application, EXPLAIN_TIMEOUT)
        if not isinstance(body, dict):
            raise ApiResponseError(f"{EXPLAIN_PATH} answered with a non-object body")
        return body


def _error_summary(response: Any) -> str:
    """The API's error code and message; never the payload we sent."""
    try:
        body = response.json()
    except ValueError:
        return "no JSON body"
    if not isinstance(body, dict):
        return "no error object"
    return " -- ".join(str(body[key]) for key in ("code", "message", "detail") if body.get(key))


# ------------------------------------------------------------------ scoring


def _single_threshold(predictions: Sequence[PredictResponse]) -> float | None:
    """The one cutoff every decision in the run was made at; None under per-group cutoffs."""
    thresholds = sorted({prediction.threshold_used for prediction in predictions})
    if len(thresholds) == 1:
        return thresholds[0]
    if any(prediction.threshold_policy == GROUP_AWARE_POLICY for prediction in predictions):
        return None
    raise ServedModelChangedError(
        f"decisions in this run were made at {len(thresholds)} thresholds "
        f"({', '.join(str(t) for t in thresholds)}) under one policy: the served threshold "
        "changed mid-run; nothing was written -- run the batch again"
    )


def score_run(run_dir: Path, client: ApiClient, chunk_size: int | None = None) -> dict[str, Any]:
    """Score ``valid.parquet`` in chunks of ``settings.max_batch_size``.

    Every chunk must come back from the same model. A deploy that lands
    mid-batch would otherwise leave half the list ranked by one model and half
    by another, which no threshold or capacity figure describes -- so the run
    fails and writes nothing rather than publish a mixture.
    """
    size = settings.max_batch_size if chunk_size is None else chunk_size
    if size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {size}")
    valid = _read_parquet(run_dir / VALID_PARQUET, step="validate")
    applications = valid.drop(columns=[INPUT_ROW]).to_dict(orient="records")
    if not applications:
        raise BadInputError(f"{run_dir / VALID_PARQUET} holds no rows")

    predictions: list[PredictResponse] = []
    served: tuple[str, str] | None = None
    for start in range(0, len(applications), size):
        chunk = applications[start : start + size]
        response = client.predict_batch(chunk)
        identity = (response.model_name, response.model_version)
        if served is None:
            served = identity
        elif identity != served:
            raise ServedModelChangedError(
                f"rows {start + 1}-{start + len(chunk)} were scored by {identity[0]} version "
                f"{identity[1]}, earlier rows by {served[0]} version {served[1]}; "
                "nothing was written -- run the batch again"
            )
        predictions.extend(response.predictions)
        log.info("scored %d/%d accounts", len(predictions), len(applications))
    assert served is not None  # at least one chunk was sent

    threshold = _single_threshold(predictions)
    scored = pd.DataFrame(
        {
            INPUT_ROW: valid[INPUT_ROW].tolist(),
            ACCOUNT_ID: [prediction.account_id for prediction in predictions],
            "default_probability": [prediction.default_probability for prediction in predictions],
            "risk_band": [prediction.risk_band for prediction in predictions],
            "decision": [prediction.decision for prediction in predictions],
        }
    )
    scored.to_parquet(run_dir / SCORED_PARQUET, index=False)
    report: dict[str, Any] = {
        "n_scored": len(scored),
        "model_name": served[0],
        "model_version": served[1],
        "threshold_used": threshold,
    }
    _write_json(run_dir / SCORING_JSON, report)
    return report


# ------------------------------------------------------- ranking and reasons


def call_list_size(n: int, capacity_fraction: float) -> int:
    """How many accounts the team calls: ``ceil(capacity_fraction * n)``."""
    if not 0.0 < capacity_fraction <= 1.0:
        raise BadInputError(f"capacity fraction must be in (0, 1], got {capacity_fraction}")
    # Through the decimal the setting was written as: in binary floating point
    # 0.07 * 100 is 7.000000000000001, and its ceiling would call one account
    # more than the team has room for.
    return math.ceil(Decimal(repr(capacity_fraction)) * n)


def rank_scores(scored: pd.DataFrame, capacity_fraction: float) -> pd.DataFrame:
    """Rank by probability, highest first, and mark the call list.

    A stable sort, so tied accounts keep their order in the file and a re-run
    on the same file produces the same list.
    """
    size = call_list_size(len(scored), capacity_fraction)
    ranked = scored.sort_values("default_probability", ascending=False, kind="mergesort")
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["in_call_list"] = ranked["rank"] <= size
    return ranked


def fetch_reasons(
    applications: Sequence[dict[str, Any]],
    client: ApiClient,
    model_version: str,
    limit: int | None = None,
) -> list[str]:
    """``top_reasons`` per account, joined with " | "; "" where none could be had.

    Never raises for an API failure: a call list without reasons is still a
    call list. A record the explainer fails on is skipped; an explainer that is
    down (no answer, 502/503/504) is not asked again for the rest. Reasons
    from a different model version than the one that ranked the list would
    explain a score nobody gave, so they are dropped too.
    """
    reasons = [""] * len(applications)
    budget = len(applications) if limit is None else min(limit, len(applications))
    for position in range(budget):
        try:
            body = client.explain(applications[position])
        except ApiUnavailableError as exc:
            if exc.status is None or exc.status in SERVICE_DOWN_STATUS:
                log.warning(
                    "explain is unavailable (%s); the remaining %d account(s) go without reasons",
                    exc,
                    budget - position,
                )
                break
            log.warning("explain failed for rank %d: %s", position + 1, exc)
            continue
        except ApiResponseError as exc:
            log.warning("explain failed for rank %d: %s", position + 1, exc)
            continue
        if str(body.get("model_version")) != model_version:
            log.warning(
                "explain for rank %d came from model version %s, not %s; reasons dropped",
                position + 1,
                body.get("model_version"),
                model_version,
            )
            continue
        top = body.get("top_reasons")
        if isinstance(top, list) and all(isinstance(reason, str) for reason in top):
            reasons[position] = REASON_SEPARATOR.join(top)
    return reasons


# ------------------------------------------------------------------ publish


def publish_run(
    run_dir: Path,
    client: ApiClient,
    *,
    data_dir: Path,
    capacity_fraction: float | None = None,
    max_reasons: int | None = None,
    dag_run_id: str | None = None,
) -> dict[str, Any]:
    """Rank, explain the call list, write the outputs, then point latest.json at them.

    latest.json is replaced last and atomically, so a reader that follows it
    always lands on a run whose files are complete -- the previous one, if this
    publish dies halfway.
    """
    fraction = (
        settings.intervention_capacity_fraction if capacity_fraction is None else capacity_fraction
    )
    call_list_size(1, fraction)  # refuse a bad capacity before anything is written
    if max_reasons is not None and max_reasons < 0:
        raise BadInputError(f"max_reasons must be 0 or more, got {max_reasons}")

    valid = _read_parquet(run_dir / VALID_PARQUET, step="validate")
    scored = _read_parquet(run_dir / SCORED_PARQUET, step="score")
    validation = _read_json(run_dir / VALIDATION_JSON, step="validate")
    scoring = _read_json(run_dir / SCORING_JSON, step="score")

    ranked = rank_scores(scored, fraction)
    ranked["default_probability"] = ranked["default_probability"].round(PROBABILITY_DECIMALS)
    call_list = ranked[ranked["in_call_list"]].copy()

    by_row = dict(
        zip(
            valid[INPUT_ROW].tolist(),
            valid.drop(columns=[INPUT_ROW]).to_dict(orient="records"),
            strict=True,
        )
    )
    reasons = fetch_reasons(
        [by_row[row] for row in call_list[INPUT_ROW].tolist()],
        client,
        str(scoring["model_version"]),
        limit=max_reasons,
    )
    call_list["top_reasons"] = reasons

    ranked[list(SCORES_COLUMNS)].to_csv(run_dir / SCORES_CSV, index=False)
    call_list[list(CALL_LIST_COLUMNS)].to_csv(run_dir / CALL_LIST_CSV, index=False)

    summary: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "run_dir": _relative(run_dir, data_dir),
        "input_path": validation["input_path"],
        "input_sha256": validation["input_sha256"],
        "n_input": validation["n_input"],
        "n_rejected": validation["n_rejected"],
        "rejected_fraction": validation["rejected_fraction"],
        "rejected_over_tolerance": validation["rejected_over_tolerance"],
        "rejected_by_reason": validation["rejected_by_reason"],
        "n_scored": len(ranked),
        "n_call_list": len(call_list),
        "capacity_fraction": fraction,
        "n_above_threshold": int((ranked["decision"] == "intervene").sum()),
        "model_name": scoring["model_name"],
        "model_version": scoring["model_version"],
        "threshold_used": scoring["threshold_used"],
        "max_reasons": max_reasons,
        "n_reasons_missing": sum(1 for reason in reasons if not reason),
        "dag_run_id": dag_run_id,
    }
    _write_json(run_dir / SUMMARY_JSON, summary)
    _write_json_atomic(data_dir / SCORED_DIR / LATEST_JSON, summary)
    log.info(
        "published %d scores, %d on the call list (%d without reasons) to %s",
        summary["n_scored"],
        summary["n_call_list"],
        summary["n_reasons_missing"],
        run_dir,
    )
    return summary


# ------------------------------------------------------------------ helpers


def _relative(path: Path, base: Path) -> str:
    """``path`` relative to the data dir when it is inside it, else absolute.

    Relative because the Airflow container, the API and a laptop mount the data
    directory at different places; ``scored/<stamp>`` means the same in all three.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write beside ``path`` and rename over it: a reader sees the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
        # mkstemp creates the file 0600; the console reading latest.json may
        # run as another user than the Airflow container that wrote it.
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(temporary, 0o666 & ~mask)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_parquet(path: Path, *, step: str) -> pd.DataFrame:
    if not path.is_file():
        raise BadInputError(f"{path} is missing; run the `{step}` step first")
    return pd.read_parquet(path)


def _read_json(path: Path, *, step: str) -> dict[str, Any]:
    if not path.is_file():
        raise BadInputError(f"{path} is missing; run the `{step}` step first")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise BadInputError(f"{path} does not hold a JSON object")
    return loaded


# ---------------------------------------------------------------------- CLI


def _validate_command(args: argparse.Namespace) -> str:
    run_dir = args.run_dir or new_run_dir(args.data_dir)
    validate_input(args.input, run_dir, args.data_dir)
    return str(run_dir)


def _score_command(args: argparse.Namespace) -> str:
    score_run(args.run_dir, ApiClient(args.api_url))
    return str(args.run_dir)


def _publish(args: argparse.Namespace, run_dir: Path, client: ApiClient) -> str:
    summary = publish_run(
        run_dir,
        client,
        data_dir=args.data_dir,
        capacity_fraction=args.capacity_fraction,
        max_reasons=args.max_reasons,
        dag_run_id=args.dag_run_id,
    )
    return json.dumps(summary, indent=2)


def _publish_command(args: argparse.Namespace) -> str:
    return _publish(args, args.run_dir, ApiClient(args.api_url))


def _run_command(args: argparse.Namespace) -> str:
    run_dir = Path(_validate_command(args))
    client = ApiClient(args.api_url)
    score_run(run_dir, client)
    return _publish(args, run_dir, client)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help="root of scored/ and of relative paths in the summary (DATA_DIR)",
    )
    common.add_argument(
        "--api-url",
        default=settings.credit_api_url,
        help="base URL of the credit API (CREDIT_API_URL)",
    )

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--input", type=Path, required=True, help="CSV or parquet to score")

    existing = argparse.ArgumentParser(add_help=False)
    existing.add_argument("--run-dir", type=Path, required=True, help="directory `validate` made")

    publishing = argparse.ArgumentParser(add_help=False)
    publishing.add_argument(
        "--capacity-fraction",
        type=float,
        default=settings.intervention_capacity_fraction,
        help="share of scored accounts on the call list",
    )
    publishing.add_argument(
        "--max-reasons",
        type=int,
        default=None,
        help="explain at most this many call-list accounts (default: all of them)",
    )
    publishing.add_argument("--dag-run-id", default=None, help="recorded in the summary")

    parser = argparse.ArgumentParser(
        prog="python -m credit_risk.scoring.batch",
        description="Score a file of accounts through the credit API and publish a call list",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", parents=[common, source], help="check rows; print the run dir"
    )
    validate.add_argument("--run-dir", type=Path, default=None, help="default: a new one")
    validate.set_defaults(handler=_validate_command)
    commands.add_parser(
        "score", parents=[common, existing], help="score the valid rows"
    ).set_defaults(handler=_score_command)
    commands.add_parser(
        "publish", parents=[common, existing, publishing], help="rank, explain, publish"
    ).set_defaults(handler=_publish_command)
    commands.add_parser(
        "run", parents=[common, source, publishing], help="validate, score and publish"
    ).set_defaults(handler=_run_command, run_dir=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m credit_risk.scoring.batch <command>``."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        output = args.handler(args)
    except BadInputError as exc:
        log.error("bad input: %s", exc)
        return EXIT_BAD_INPUT
    except ApiUnavailableError as exc:
        # Not a traceback and exit 1: the DAG fails those without a retry, and
        # an API restarting under a deploy is the textbook case for one.
        log.error("%s -- exiting %d so the scheduler retries later", exc, EXIT_TRANSIENT)
        return EXIT_TRANSIENT
    except BatchScoringError as exc:
        log.error("%s", exc)
        return EXIT_FAILED
    # stdout is the contract for the DAG, which reads it; logs go to stderr.
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
