"""Reading what the scoring DAG published: ``scored/latest.json`` and the files it points at.

These files are written by another container, possibly while being read, and
by code that may change shape. So every read here is defensive: a missing key
is a null, a missing file is an absent download, a half-written summary is an
empty state -- and none of it is an exception the page has to survive.

``run_dir`` is the one value read from a file that becomes part of a path, so
it is resolved and confined to the results folder before anything is opened.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path, PurePosixPath
from typing import Any, Final

log = logging.getLogger(__name__)

SCORED_DIR: Final = "scored"
LATEST_FILE: Final = "latest.json"

SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "generated_at",
    "run_dir",
    "input_path",
    "input_sha256",
    "n_input",
    "n_rejected",
    "rejected_by_reason",
    "n_scored",
    "n_call_list",
    "capacity_fraction",
    "n_above_threshold",
    "model_name",
    "model_version",
    "threshold_used",
    "n_reasons_missing",
    "dag_run_id",
)

# The only files the console will hand out. A name not listed here is a 404,
# whatever it resolves to.
DOWNLOADABLE: Final[tuple[str, ...]] = ("call_list.csv", "scores.csv", "rejected.csv")

REASON_SEPARATOR: Final = " | "

NOTHING_YET: Final = "Chưa có lần chấm điểm nào. Hãy chạy pipeline chấm điểm để có danh sách gọi."
UNREADABLE: Final = (
    "Kết quả lần chấm gần nhất chưa đọc được (có thể đang được ghi). Thử lại sau ít giây."
)


class NoResults(Exception):
    """There is nothing (readable) to show or download."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _scored_root(data_dir: Path) -> Path:
    return data_dir / SCORED_DIR


def load_summary(data_dir: Path) -> dict[str, Any]:
    """The latest summary with every known key present, absent ones as None."""
    path = _scored_root(data_dir) / LATEST_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NoResults(NOTHING_YET) from exc
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", path, type(exc).__name__)
        raise NoResults(UNREADABLE) from exc
    if not isinstance(raw, dict):
        raise NoResults(UNREADABLE)
    return {key: _finite(raw.get(key)) for key in SUMMARY_KEYS}


def _finite(value: Any) -> Any:
    """NaN and infinity as None. Python's json reads them, but a JSON response cannot carry them."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def run_directory(data_dir: Path, run_dir: Any) -> Path | None:
    """The folder of the latest run, or None if run_dir does not name one inside ``scored/``.

    An absolute run_dir is a path inside the Airflow container (its mount point
    differs from this one), so only its last component is kept.
    """
    if not isinstance(run_dir, str) or not run_dir.strip():
        return None
    candidate = PurePosixPath(run_dir.strip())
    if candidate.is_absolute():
        candidate = PurePosixPath(candidate.name)
    root = _scored_root(data_dir).resolve()
    resolved = (root / candidate).resolve()
    if root not in resolved.parents:
        return None
    return resolved


def _contained_file(folder: Path | None, name: str) -> Path | None:
    """``folder/name`` if it is a regular file that resolves inside ``folder``.

    Resolved again because the file itself may be a link out of the results
    folder, even when the folder is not.
    """
    if folder is None:
        return None
    path = (folder / name).resolve()
    if folder not in path.parents or not path.is_file():
        return None
    return path


def result_file(data_dir: Path, name: str) -> Path:
    """A downloadable file of the latest run, or :class:`NoResults`."""
    if name not in DOWNLOADABLE:
        raise NoResults(f"Không có file {name}.")
    folder = run_directory(data_dir, load_summary(data_dir)["run_dir"])
    path = _contained_file(folder, name)
    if path is None:
        raise NoResults(f"Lần chấm gần nhất không có file {name}.")
    return path


def _as_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _as_float(value: str | None) -> float | None:
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _row(raw: dict[str, str | None]) -> dict[str, Any]:
    reasons = raw.get("top_reasons") or ""
    return {
        "rank": _as_int(raw.get("rank")),
        "account_id": raw.get("account_id"),
        "default_probability": _as_float(raw.get("default_probability")),
        "risk_band": raw.get("risk_band"),
        "decision": raw.get("decision"),
        "top_reasons": [part.strip() for part in reasons.split(REASON_SEPARATOR) if part.strip()],
    }


def read_call_list(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    """The first ``limit`` rows of the call list, and how many rows it has in all."""
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            total += 1
            if len(rows) < limit:
                rows.append(_row(raw))
    return rows, total


def latest(data_dir: Path, limit: int) -> dict[str, Any]:
    """Everything section 3 of the page shows; raises :class:`NoResults` for the empty state."""
    summary = load_summary(data_dir)
    folder = run_directory(data_dir, summary["run_dir"])

    downloads = {
        name: f"/api/results/latest/{name}" if _contained_file(folder, name) else None
        for name in DOWNLOADABLE
    }

    rows: list[dict[str, Any]] = []
    total: int | None = None
    call_list = _contained_file(folder, "call_list.csv")
    if call_list is not None:
        try:
            rows, total = read_call_list(call_list, limit)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            log.warning("cannot read the call list: %s", type(exc).__name__)

    return {
        "available": True,
        "summary": summary,
        "rows": rows,
        "n_rows_total": total,
        "downloads": downloads,
    }
