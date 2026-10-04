"""Fetch the UCI credit-default archive and land it as parquet.

The source is a 5.5 MB zip holding one Excel 97-2003 sheet. Excel is read
exactly once -- here. Every later step reads the parquet instead, which keeps
`xlrd` (read-only, legacy-format, easy to break on a version bump) out of the
training and serving paths entirely.

Run it directly:

    python -m credit_risk.data.download [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from credit_risk import schema
from credit_risk.config import settings

log = logging.getLogger(__name__)

RAW_PARQUET_NAME: Final = "credit_default_raw.parquet"

# Provenance sidecar. Written next to the parquet rather than inside it so the
# digest can be read without loading 30,000 rows, and so `split.py` can copy it
# into the splits manifest -- which is what makes "did the data change or did
# the model change?" answerable from two JSON files.
RAW_META_SUFFIX: Final = ".meta.json"

# archive.ics.uci.edu answers 403 to some default library user agents. A plain
# browser string is the difference between a green CI job and a mystery outage
# that only reproduces off a laptop.
USER_AGENT: Final = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# (connect, read). The read budget is generous because UCI is occasionally slow
# and a retry costs a full 5.5 MB of bandwidth.
REQUEST_TIMEOUT: Final[tuple[int, int]] = (10, 180)

# Measured 2026-09-30. UCI publishes no checksum, so this is the only integrity
# signal we have; a mismatch is reported, not enforced, because a legitimate
# re-export would otherwise wedge the pipeline.
EXPECTED_ZIP_BYTES: Final = 5_539_494


class DownloadError(RuntimeError):
    """The archive could not be fetched, or was not the archive we expected."""


def raw_parquet_path() -> Path:
    """Canonical location of the raw dataset after conversion."""
    return settings.raw_dir / RAW_PARQUET_NAME


def raw_meta_path(parquet_path: Path | None = None) -> Path:
    """Sidecar holding the digest and row count of the file we actually got."""
    target = raw_parquet_path() if parquet_path is None else Path(parquet_path)
    return target.with_suffix(RAW_META_SUFFIX)


def load_raw_metadata(parquet_path: Path | None = None) -> dict[str, Any]:
    """Read the sidecar, or return ``{}`` when there is nothing to read.

    Empty rather than raising: a parquet placed by hand -- a fixture, a sample
    carved out for a notebook -- has no sidecar, and that is a gap in the
    provenance record, not a reason to stop the pipeline. Callers record what
    they got; a manifest with ``source_sha256: null`` says "unknown origin"
    out loud, which is the honest answer.
    """
    path = raw_meta_path(parquet_path)
    if not path.exists():
        log.info("no provenance sidecar at %s; source digest will be unknown", path)
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        log.warning("%s is not readable JSON (%s); continuing without provenance", path, exc)
        return {}
    if not isinstance(loaded, dict):
        log.warning("%s does not hold a JSON object; ignoring it", path)
        return {}
    return loaded


def _fetch(url: str) -> bytes:
    """GET the archive, returning its bytes."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def _read_archive(payload: bytes) -> pd.DataFrame:
    """Extract the single .xls member and parse it into a DataFrame."""
    buffer = io.BytesIO(payload)
    if not zipfile.is_zipfile(buffer):
        # A captive portal or an HTML error page served with status 200 lands
        # here rather than inside pandas with an unreadable traceback.
        raise DownloadError(
            f"payload of {len(payload)} bytes is not a zip archive; "
            f"first bytes were {payload[:32]!r}"
        )

    with zipfile.ZipFile(buffer) as archive:
        names = archive.namelist()
        member = schema.DATASET_MEMBER
        if member not in names:
            member = next((n for n in names if n.lower().endswith(".xls")), "")
        if not member:
            raise DownloadError(f"no .xls member in archive; found {names}")

        with tempfile.TemporaryDirectory() as tmp:
            extracted = Path(archive.extract(member, path=tmp))
            # header=1: row 0 of the sheet is a merged banner cell, not the
            # column names. Reading with the default header yields 24 columns
            # called "Unnamed: n" and a first data row full of strings.
            return pd.read_excel(extracted, header=schema.EXCEL_HEADER_ROW)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_download(target: Path) -> tuple[bool, str]:
    """Whether ``target`` is a finished download, and if not, why not.

    Finished means the parquet exists *and* a sidecar describes it. A parquet
    on its own is what a crash between the two writes used to leave behind,
    and trusting it meant every later manifest said ``source_sha256: null``.
    """
    if not target.exists():
        return False, "no parquet"
    metadata = load_raw_metadata(target)
    if not metadata.get("sha256"):
        return False, "no provenance sidecar describes it"
    recorded = metadata.get("parquet_sha256")
    if recorded is None:
        # A sidecar written before the parquet digest was recorded. It still
        # names the archive, which is all the manifest needs from it.
        return True, "sidecar predates the parquet digest"
    if recorded != _file_sha256(target):
        return False, "it is not the parquet its sidecar describes"
    return True, "parquet matches its sidecar"


def _write_download(frame: pd.DataFrame, metadata: dict[str, Any], target: Path) -> None:
    """Write the parquet and its sidecar so that no crash leaves an orphan parquet.

    Both go to temporary files first and are renamed into place, the sidecar
    before the parquet. A crash before the first rename changes nothing; one
    between the renames leaves a sidecar whose ``parquet_sha256`` the parquet on
    disk (old, or absent) does not match, which the next run re-downloads.
    """
    meta_path = raw_meta_path(target)
    tmp_parquet = target.with_name(target.name + ".tmp")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    try:
        frame.to_parquet(tmp_parquet, index=False)
        sidecar = {**metadata, "parquet_sha256": _file_sha256(tmp_parquet)}
        tmp_meta.write_text(json.dumps(sidecar, indent=2) + "\n")
        os.replace(tmp_meta, meta_path)
        os.replace(tmp_parquet, target)
    finally:
        tmp_parquet.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)


def download_raw(dest: Path | None = None, force: bool = False) -> Path:
    """Download the UCI archive and write it to parquet, returning that path.

    Idempotent: a finished download -- the parquet plus the sidecar that
    describes it -- short-circuits the network unless ``force`` is set, so
    re-running the DAG does not re-download 5.5 MB. A parquet with no sidecar,
    or one its sidecar does not describe, is downloaded again.
    """
    target = Path(dest) if dest is not None else raw_parquet_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        complete, why = _completed_download(target)
        if complete:
            log.info("raw parquet already at %s (%s); skipping download", target, why)
            return target
        if target.exists():
            log.warning("raw parquet at %s exists but %s; downloading again", target, why)

    log.info("downloading %s", schema.DATASET_URL)
    payload = _fetch(schema.DATASET_URL)
    digest = hashlib.sha256(payload).hexdigest()
    log.info("fetched %d bytes, sha256=%s", len(payload), digest)

    if len(payload) != EXPECTED_ZIP_BYTES:
        log.warning(
            "archive is %d bytes, expected %d -- the row counts recorded in "
            "schema.py describe a file we may no longer be holding",
            len(payload),
            EXPECTED_ZIP_BYTES,
        )

    frame = _read_archive(payload)
    metadata: dict[str, Any] = {
        "url": schema.DATASET_URL,
        "sha256": digest,
        "n_bytes": len(payload),
        "member": schema.DATASET_MEMBER,
        "n_rows": int(frame.shape[0]),
        "n_cols": int(frame.shape[1]),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_download(frame, metadata, target)

    log.info("wrote %s (%d rows x %d cols)", target, frame.shape[0], frame.shape[1])
    return target


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw parquet produced by :func:`download_raw`."""
    source = Path(path) if path is not None else raw_parquet_path()
    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing; run `python -m credit_risk.data.download` first"
        )
    return pd.read_parquet(source)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m credit_risk.data.download``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if parquet exists")
    parser.add_argument("--dest", type=Path, default=None, help="override the output parquet path")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    path = download_raw(dest=args.dest, force=args.force)
    # stdout is the contract for shell and subprocess callers; logs go to stderr.
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
