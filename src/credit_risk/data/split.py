"""Normalise the raw frame and cut it into reproducible batches.

The dataset is a static file, so "ingestion" would otherwise be a copy. We
split the 30,000 accounts into six ID-ordered batches of 5,000 and treat each
as one arrival: batches 1-4 train, batch 5 is the held-out test set, batch 6 is
the pool the traffic generator draws from. This is simulation and DATASHEET.md
says so out loud.

Determinism is the point. Every split carries a content hash in
``splits_manifest.json``, so "did the data change or did the model change?" is
a question with an answer rather than an argument.

Rows that fail an error-level check -- of the raw frame, or of the cleaned
frame (a category code the folds cannot fix) -- are not trained on. Up to
``schema.MAX_BAD_ROW_FRACTION`` of the file may fail and the run still goes
ahead -- without those rows: they are written, as received, to
``quarantine.parquet`` with the checks they failed, and the manifest counts
them. Batches are assigned before anything is set aside, so a quarantined row
leaves a hole in its own batch instead of shifting every later account into an
earlier one.

Batches are cut by ``ID``, and ``ID`` is not time: the default rate differs
between splits (22.8% train, 20.4% test, 21.2% serving pool on the published
file). DATASHEET.md records this as a known property of the simulation.

Run it directly:

    python -m credit_risk.data.split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.data.download import load_raw, load_raw_metadata, raw_parquet_path
from credit_risk.data.validate import (
    BATCH_COL,
    DataValidationError,
    ValidationReport,
    assert_ok,
    validate_clean,
    validate_raw,
)
from credit_risk.features.build import normalize_codes

log = logging.getLogger(__name__)

# Re-exported: BATCH_COL is defined in validate.py (see the note there), but
# this is the module that puts the column on the frame, so this is where every
# consumer looks for its name.
__all__ = [
    "BATCH_COL",
    "CLEAN_CHECK_PREFIX",
    "INTEGER_COLUMNS",
    "MANIFEST_NAME",
    "QUARANTINE_COLUMNS",
    "QUARANTINE_FILE",
    "QUARANTINE_REASON_COL",
    "SPLIT_BATCHES",
    "SPLIT_FILES",
    "SplitError",
    "assign_batches",
    "build_splits",
    "clean",
    "frame_sha256",
    "load_split",
    "main",
    "quarantine_rows",
    "split_frames",
    "write_splits",
]

MANIFEST_NAME: Final = "splits_manifest.json"

# Rows set aside by build_splits, written next to the splits on every run --
# empty when nothing failed, so a file left over from an earlier run never
# describes rows this run did not set aside.
QUARANTINE_FILE: Final = "quarantine.parquet"
# The error checks a quarantined row failed, ";"-joined. Checks of the cleaned
# frame carry CLEAN_CHECK_PREFIX, so "education_codes" (a raw check) and
# "clean:education_codes" (the same code, still invalid after folding) differ.
QUARANTINE_REASON_COL: Final = "quarantine_reason"
CLEAN_CHECK_PREFIX: Final = "clean:"
# The one column layout of quarantine.parquet, whether or not it holds rows.
QUARANTINE_COLUMNS: Final[tuple[str, ...]] = (
    *schema.RAW_COLUMNS,
    BATCH_COL,
    QUARANTINE_REASON_COL,
)

# Every column of the published file is an integer (measured: 30,000 x 25,
# all int64). A single null makes pandas hold its whole column as float64, and
# quarantining that row does not undo it -- so the type is restored afterwards.
INTEGER_COLUMNS: Final[tuple[str, ...]] = schema.RAW_COLUMNS

SPLIT_FILES: Final[dict[str, str]] = {
    "train": "train.parquet",
    "test": "test.parquet",
    "serving_pool": "serving_pool.parquet",
}

SPLIT_BATCHES: Final[dict[str, tuple[int, ...]]] = {
    "train": schema.TRAIN_BATCHES,
    "test": (schema.TEST_BATCH,),
    "serving_pool": (schema.SERVING_BATCH,),
}


class SplitError(ValueError):
    """The frame cannot be cut into the splits the pipeline promises."""


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, fold undocumented codes, add AGE_GROUP, fix the column order.

    Folding rather than dropping: the 345 EDUCATION and 54 MARRIAGE rows with
    codes outside the published dictionary are real accounts, and deleting
    them would quietly change the base rates the fairness analysis reports.
    """
    out = df.rename(columns=schema.RENAME_MAP).copy()

    missing = [c for c in schema.CLEAN_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"cannot clean: columns absent after rename: {missing}")

    # features.build owns the fold, and is called rather than copied, because
    # the serving path builds features without ever cleaning: a second copy of
    # the mapping here is a second chance for the two paths to disagree about
    # what EDUCATION=5 means.
    out = normalize_codes(out)

    # AGE_GROUP is a fairness slice, not a model feature: the mitigation and
    # monitoring code groups by it, the feature builder never sees it.
    out[schema.AGE_GROUP] = np.where(out[schema.AGE] <= schema.AGE_GROUP_CUTOFF, "young", "older")

    return out[[*schema.CLEAN_COLUMNS, schema.AGE_GROUP]]


def assign_batches(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 1..6 ``batch`` column by sorted ID, and return the sorted frame.

    No randomness anywhere: the same input produces the same batches on any
    machine, in any pandas version, which is what makes the manifest hashes
    worth writing down.
    """
    if schema.ID_COL not in df.columns:
        raise KeyError(f"cannot assign batches: {schema.ID_COL} is not a column")

    # mergesort is the stable one; with duplicate IDs an unstable sort would
    # let batch membership depend on input row order.
    ordered = df.sort_values(schema.ID_COL, kind="mergesort").reset_index(drop=True)

    position = np.arange(len(ordered))
    # Clipping at N_BATCHES keeps a short or oversized frame from inventing a
    # seventh batch that nothing downstream knows how to read.
    batch = np.minimum(position // schema.BATCH_SIZE + 1, schema.N_BATCHES)
    ordered[BATCH_COL] = batch.astype("int64")
    return ordered


def frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of a frame, independent of parquet encoding details.

    Hashing the written file would make the digest depend on the pyarrow
    version stamped into the footer, which turns a library upgrade into a
    false "the data changed" alarm.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode())
    hasher.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return hasher.hexdigest()


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slice a batched frame into train / test / serving_pool."""
    if BATCH_COL not in df.columns:
        raise KeyError(f"{BATCH_COL} column is missing; call assign_batches() first")
    return {
        name: df[df[BATCH_COL].isin(batches)].reset_index(drop=True)
        for name, batches in SPLIT_BATCHES.items()
    }


def quarantine_rows(
    raw: pd.DataFrame, report: ValidationReport
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch the raw frame, then split it into (kept, quarantined).

    ``report`` must be ``validate_raw`` of this very frame: its row masks are
    positional. Batches are assigned first, over every row, so the accounts
    that are kept stay in the batch they arrived in. The quarantined rows keep
    their raw values plus the batch and the checks they failed.
    """
    if report.n_rows != len(raw):
        raise ValueError(f"report covers {report.n_rows} rows, frame has {len(raw)}")
    flagged = raw.assign(**{QUARANTINE_REASON_COL: report.row_failures()})
    batched = assign_batches(flagged)
    bad = (batched[QUARANTINE_REASON_COL] != "").to_numpy()
    quarantined = batched.loc[bad].reset_index(drop=True)
    kept = batched.loc[~bad].drop(columns=QUARANTINE_REASON_COL).reset_index(drop=True)
    return _restore_integers(kept), quarantined


def _restore_integers(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the integer columns that a removed null left as float64 back to int64.

    Only when every remaining value is a whole number: a column that really
    holds fractions is left alone and logged, never truncated.
    """
    out = frame.copy()
    for column in INTEGER_COLUMNS:
        if column not in out.columns or not is_float_dtype(out[column]):
            continue
        values = out[column].to_numpy(dtype="float64")
        if not (np.isfinite(values).all() and (values == np.round(values)).all()):
            log.warning("%s holds non-integer values; leaving it as float64", column)
            continue
        out[column] = values.astype("int64")
    return out


def _reject_cleaned(
    kept: pd.DataFrame, cleaned: pd.DataFrame, report: ValidationReport
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Set aside the rows the cleaned-frame contract rejects.

    Some defects only show after cleaning: EDUCATION=7 is a raw *warning*,
    because the fold might fix it, and an *error* once the fold has left it
    alone. ``cleaned`` is ``kept`` cleaned row for row, so the report's
    positional mask selects the same accounts in both; the quarantined copy is
    taken from ``kept``, as received. Returns (kept, cleaned, quarantined).
    """
    rejected = report.bad_row_mask()
    reasons = [
        ";".join(f"{CLEAN_CHECK_PREFIX}{name}" for name in failed.split(";"))
        for failed, bad in zip(report.row_failures(), rejected, strict=True)
        if bad
    ]
    quarantined = kept.loc[rejected].assign(**{QUARANTINE_REASON_COL: reasons})
    return (
        kept.loc[~rejected].reset_index(drop=True),
        cleaned.loc[~rejected].reset_index(drop=True),
        quarantined.reset_index(drop=True),
    )


def _quarantine_frame(quarantined: pd.DataFrame | None) -> pd.DataFrame:
    """The rows to write to quarantine.parquet, in its one fixed schema.

    Raw values as float64 -- a quarantined row may hold a null, and the
    published integers convert exactly -- the batch as int64 and the reasons as
    strings, in :data:`QUARANTINE_COLUMNS` order. An empty quarantine gets the
    same schema as a full one, so a reader never sees it change shape.
    """
    source = pd.DataFrame(columns=list(QUARANTINE_COLUMNS)) if quarantined is None else quarantined
    out = pd.DataFrame({column: source[column].astype("float64") for column in schema.RAW_COLUMNS})
    out[BATCH_COL] = source[BATCH_COL].astype("int64")
    out[QUARANTINE_REASON_COL] = source[QUARANTINE_REASON_COL].astype("string")
    return out.reset_index(drop=True)


def _quarantine_counts(quarantined: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reasons in quarantined.get(QUARANTINE_REASON_COL, pd.Series(dtype="object")):
        for name in str(reasons).split(";"):
            if name:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def write_splits(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    source: Mapping[str, Any] | None = None,
    allow_empty_splits: bool = False,
    quarantined: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write the three splits plus a manifest, and return the manifest.

    Returning it rather than only writing it lets the DAG push the row counts
    and hashes straight into XCom without re-reading the file it just wrote.

    ``source`` is the download's provenance sidecar; its digest is copied into
    the manifest so the two questions "did the data change?" and "did the model
    change?" can be told apart from the files alone.

    ``quarantined`` holds the rows :func:`build_splits` set aside. They are
    written to :data:`QUARANTINE_FILE` in one fixed schema and counted in the
    manifest; ``None`` writes an empty quarantine, so the file always
    describes this run.

    An empty split raises unless ``allow_empty_splits`` is set. It is reachable
    without one: ``assign_batches`` clips at :data:`schema.N_BATCHES`, so a
    frame shorter than 25,001 rows never reaches batch 5 and the held-out test
    set comes out with zero rows -- a valid manifest, two empty parquet files,
    and a PR-AUC gate downstream scoring nothing at all.
    """
    target_dir = Path(out_dir) if out_dir is not None else settings.processed_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    frames = split_frames(df)
    provenance = dict(source or {})
    held = _quarantine_frame(quarantined)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # Rows received: the ones split below plus the ones set aside.
        "source_rows": int(len(df) + len(held)),
        # None rather than absent when the sidecar is missing: a null reads as
        # "origin unknown", a missing key reads as "nobody thought about it".
        "source_sha256": provenance.get("sha256"),
        "source_n_rows": provenance.get("n_rows"),
        "n_batches": schema.N_BATCHES,
        "batch_size": schema.BATCH_SIZE,
        "degraded": False,
        "empty_splits": [],
        "quarantined_rows": int(len(held)),
        "splits": {},
    }

    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        detail = (
            f"splits {empty} are empty: {len(df)} rows do not reach batch "
            f"{schema.N_BATCHES} at {schema.BATCH_SIZE} rows per batch"
        )
        if not allow_empty_splits:
            raise SplitError(f"{detail}; refusing to write a held-out set nothing can score")
        # Sample mode is a legitimate way to work, but it must be recorded:
        # every metric computed from these files is computed from no rows.
        log.error("%s -- manifest marked degraded", detail)
        manifest["degraded"] = True
        manifest["empty_splits"] = empty

    for name, frame in frames.items():
        path = target_dir / SPLIT_FILES[name]
        frame.to_parquet(path, index=False)
        manifest["splits"][name] = {
            "path": str(path),
            "file": SPLIT_FILES[name],
            "batches": list(SPLIT_BATCHES[name]),
            "n_rows": int(len(frame)),
            "n_cols": int(frame.shape[1]),
            "positive_rate": round(float(frame[schema.TARGET].mean()), 6)
            if schema.TARGET in frame.columns and len(frame)
            else None,
            "sha256": frame_sha256(frame),
        }
        log.info("wrote %s (%d rows)", path, len(frame))

    quarantine_path = target_dir / QUARANTINE_FILE
    held.to_parquet(quarantine_path, index=False)
    manifest["quarantine"] = {
        "path": str(quarantine_path),
        "file": QUARANTINE_FILE,
        "n_rows": int(len(held)),
        "by_check": _quarantine_counts(held),
        "sha256": frame_sha256(held),
    }
    if len(held):
        log.warning(
            "quarantined %d rows that failed validation (%s); see %s",
            len(held),
            manifest["quarantine"]["by_check"],
            quarantine_path,
        )

    (target_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(name: str, out_dir: Path | None = None) -> pd.DataFrame:
    """Read one written split back. Names are the keys of :data:`SPLIT_FILES`."""
    if name not in SPLIT_FILES:
        raise KeyError(f"unknown split {name!r}; expected one of {sorted(SPLIT_FILES)}")
    target_dir = Path(out_dir) if out_dir is not None else settings.processed_dir
    path = target_dir / SPLIT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m credit_risk.data.split` first")
    return pd.read_parquet(path)


def build_splits(
    raw: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    expect_full_dataset: bool = True,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    """Validate, quarantine, clean, batch and write in one call. Used by the DAG and the CLI.

    A row that fails an error check -- of the raw frame, or of the cleaned
    frame (a code the folds cannot fix) -- never reaches a split. Batches are
    assigned over the whole file first; the failing rows are then written to
    :data:`QUARANTINE_FILE` as received and counted in the manifest. If more
    than ``schema.MAX_BAD_ROW_FRACTION`` of the file fails, across both stages,
    the run stops instead. The frame that is written is re-validated and must
    pass every error check.

    ``raw_path`` only says where to look for the provenance sidecar; the frame
    itself is the one passed in. Sampling mode (``expect_full_dataset=False``)
    also permits empty splits, because a sample short enough to skip the
    30,000-row assertion is a sample too short to fill six batches.
    """
    raw_report = validate_raw(raw, expect_full_dataset=expect_full_dataset)
    assert_ok(raw_report, context="raw dataset")
    kept, quarantined = quarantine_rows(raw, raw_report)

    # clean() returns the published column set, which has no batch column; the
    # batch kept was assigned over the whole file and is put back by position.
    cleaned = clean(kept)
    cleaned[BATCH_COL] = kept[BATCH_COL].to_numpy()
    clean_report = validate_clean(
        cleaned,
        expect_full_dataset=expect_full_dataset,
        expected_rows=schema.RAW_N_ROWS - len(quarantined),
    )
    # A whole-frame failure condemns every row and stops the run here.
    assert_ok(clean_report, context="cleaned dataset")
    kept, cleaned, late = _reject_cleaned(kept, cleaned, clean_report)
    if len(late):
        quarantined = (
            pd.concat([quarantined, late[quarantined.columns]], ignore_index=True)
            .sort_values(schema.ID_COL, kind="mergesort")
            .reset_index(drop=True)
        )

    # The tolerance is about the file: 4% rejected raw and 4% rejected after
    # cleaning is 8% of the accounts gone, whichever stage caught them.
    set_aside = len(quarantined) / len(raw) if len(raw) else 1.0
    if set_aside > schema.MAX_BAD_ROW_FRACTION:
        raise DataValidationError(
            f"{len(quarantined)} of {len(raw)} rows ({set_aside:.1%}) failed the raw or "
            f"the cleaned checks, above the {schema.MAX_BAD_ROW_FRACTION:.0%} tolerance. "
            f"Failing checks: {_quarantine_counts(quarantined)}"
        )

    final = validate_clean(
        cleaned,
        expect_full_dataset=expect_full_dataset,
        expected_rows=schema.RAW_N_ROWS - len(quarantined),
    )
    if not final.ok:
        named = ", ".join(f"{c.name} ({c.n_bad_rows} rows)" for c in final.failures)
        raise DataValidationError(f"cleaned dataset still fails after quarantine: {named}")
    return write_splits(
        cleaned,
        out_dir,
        source=load_raw_metadata(raw_path),
        allow_empty_splits=not expect_full_dataset,
        quarantined=quarantined,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m credit_risk.data.split``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=None, help="raw parquet to read")
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write the splits")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="skip the 30,000-row assertion (for working on a sample)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    source = args.raw if args.raw is not None else raw_parquet_path()
    manifest = build_splits(
        load_raw(source),
        args.out_dir,
        expect_full_dataset=not args.allow_partial,
        raw_path=source,
    )
    # stdout is the contract for subprocess callers; logs go to stderr.
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
