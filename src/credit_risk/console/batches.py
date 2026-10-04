"""Checking and storing a customer file uploaded for scoring.

The checks here are structural only -- is it text, is it CSV, does the header
carry every model input, is there at least one row. Whether the values are
valid is the scoring DAG's ``validate_batch`` step, which already owns those
rules; repeating them here would be a second copy to drift.

The saved file's name is always chosen by the server. Nothing from the request
-- no filename, no header, no query string -- reaches the filesystem path.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from credit_risk import schema

log = logging.getLogger(__name__)

# The model's raw inputs, in the order the feature builder expects them. ID and
# the target are not required: ID is carried through when present, and a file
# of customers to score has no label yet.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    schema.LIMIT_BAL,
    schema.SEX,
    schema.EDUCATION,
    schema.MARRIAGE,
    schema.AGE,
    *schema.PAY_COLS,
    *schema.BILL_COLS,
    *schema.PAY_AMT_COLS,
)

INCOMING_DIR: Final = "incoming"
SAMPLE_INPUT: Final = "processed/serving_pool.parquet"

# `application/csv` is not registered but some clients send it; the page itself
# always sends `text/csv`.
CSV_MEDIA_TYPES: Final = frozenset({"text/csv", "application/csv"})

# More same-second uploads than this is not a person at a browser.
_MAX_NAME_ATTEMPTS: Final = 100


class UploadRejected(Exception):
    """An upload the console will not pass on, with a message for the person who sent it."""

    def __init__(self, status_code: int, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class ParsedUpload:
    """An upload that passed the structural checks, normalised to UTF-8 without a BOM."""

    text: str
    columns: list[str]
    n_rows: int


def check_content_type(content_type: str | None) -> None:
    """Accept a CSV body only. A multipart form would be saved as its own envelope."""
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type not in CSV_MEDIA_TYPES:
        raise UploadRejected(
            415,
            "unsupported_media_type",
            "Chỉ nhận file CSV (Content-Type: text/csv).",
            media_type or None,
        )


def too_large(max_bytes: int) -> UploadRejected:
    """The rejection for a body over the limit, shared by both places that detect it."""
    megabytes = max_bytes / (1024 * 1024)
    return UploadRejected(
        413,
        "too_large",
        f"File vượt quá giới hạn {megabytes:.1f} MB. Hãy chia nhỏ danh sách khách hàng.",
        max_bytes,
    )


def parse_upload(payload: bytes) -> ParsedUpload:
    """Decode and check an uploaded CSV, or raise :class:`UploadRejected` saying why."""
    if not payload.strip():
        raise UploadRejected(400, "empty_file", "File rỗng, không có dữ liệu để chấm điểm.")
    try:
        # utf-8-sig drops the byte-order mark Excel writes at the start of a CSV.
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadRejected(
            400,
            "not_utf8",
            "File không phải CSV dạng văn bản UTF-8. Nếu là file Excel, hãy lưu lại dưới dạng "
            "CSV UTF-8.",
        ) from exc

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = [cell.strip() for cell in next(reader, [])]
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise UploadRejected(
                422,
                "missing_columns",
                f"File thiếu {len(missing)} cột bắt buộc: {', '.join(missing)}.",
                missing,
            )
        n_rows = sum(1 for row in reader if any(cell.strip() for cell in row))
    except csv.Error as exc:
        raise UploadRejected(
            400, "malformed_csv", "File CSV bị lỗi định dạng và không đọc được.", str(exc)
        ) from exc

    if n_rows == 0:
        raise UploadRejected(422, "no_rows", "File chỉ có dòng tiêu đề, chưa có khách hàng nào.")
    return ParsedUpload(text=text, columns=header, n_rows=n_rows)


def save_upload(data_dir: Path, text: str, *, now: datetime) -> str:
    """Write the upload to ``incoming/<UTC stamp>.csv`` and return that path relative to data_dir.

    Exclusive create, so a second upload in the same second gets a suffix
    instead of silently replacing a file a DAG run may be reading.
    """
    incoming = data_dir / INCOMING_DIR
    incoming.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    for attempt in range(_MAX_NAME_ATTEMPTS):
        name = f"{stamp}.csv" if attempt == 0 else f"{stamp}-{attempt}.csv"
        try:
            with (incoming / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
        except FileExistsError:
            continue
        log.info("saved upload %s/%s (%d bytes)", INCOMING_DIR, name, len(text))
        return f"{INCOMING_DIR}/{name}"
    raise UploadRejected(
        429, "too_many_uploads", "Quá nhiều file được tải lên cùng lúc, hãy thử lại sau."
    )


def discard(data_dir: Path, relative: str) -> None:
    """Remove an upload whose run could not be started, so nothing is left half-done."""
    try:
        (data_dir / relative).unlink()
    except OSError:
        log.warning("could not remove %s after a failed start", relative)
