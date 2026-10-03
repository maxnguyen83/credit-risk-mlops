"""Normalise the raw frame and cut it into reproducible batches.

The dataset is a static file, so "ingestion" would otherwise be a copy. We
split the 30,000 accounts into six ID-ordered batches of 5,000 and treat each
as one arrival: batches 1-4 train, batch 5 is the held-out test set, batch 6 is
the pool the traffic generator draws from. This is simulation and DATASHEET.md
says so out loud.

Determinism is the point. Every split carries a content hash in
``splits_manifest.json``, so "did the data change or did the model change?" is
a question with an answer rather than an argument.

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

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.data.download import load_raw, load_raw_metadata, raw_parquet_path
from credit_risk.data.validate import BATCH_COL, assert_ok, validate_clean, validate_raw
from credit_risk.features.build import normalize_codes

log = logging.getLogger(__name__)

# Re-exported: BATCH_COL is defined in validate.py (see the note there), but
# this is the module that puts the column on the frame, so this is where every
# consumer looks for its name.
__all__ = [
    "BATCH_COL",
    "MANIFEST_NAME",
    "SPLIT_BATCHES",
    "SPLIT_FILES",
    "SplitError",
    "assign_batches",
    "build_splits",
    "clean",
    "frame_sha256",
    "load_split",
    "main",
    "split_frames",
    "write_splits",
]

MANIFEST_NAME: Final = "splits_manifest.json"

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


def write_splits(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    source: Mapping[str, Any] | None = None,
    allow_empty_splits: bool = False,
) -> dict[str, Any]:
    """Write the three splits plus a manifest, and return the manifest.

    Returning it rather than only writing it lets the DAG push the row counts
    and hashes straight into XCom without re-reading the file it just wrote.

    ``source`` is the download's provenance sidecar; its digest is copied into
    the manifest so the two questions "did the data change?" and "did the model
    change?" can be told apart from the files alone.

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
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_rows": int(len(df)),
        # None rather than absent when the sidecar is missing: a null reads as
        # "origin unknown", a missing key reads as "nobody thought about it".
        "source_sha256": provenance.get("sha256"),
        "source_n_rows": provenance.get("n_rows"),
        "n_batches": schema.N_BATCHES,
        "batch_size": schema.BATCH_SIZE,
        "degraded": False,
        "empty_splits": [],
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
    """Validate, clean, batch and write in one call. Used by the DAG and the CLI.

    ``raw_path`` only says where to look for the provenance sidecar; the frame
    itself is the one passed in. Sampling mode (``expect_full_dataset=False``)
    also permits empty splits, because a sample short enough to skip the
    30,000-row assertion is a sample too short to fill six batches.
    """
    assert_ok(validate_raw(raw, expect_full_dataset=expect_full_dataset), context="raw dataset")
    cleaned = clean(raw)
    assert_ok(
        validate_clean(cleaned, expect_full_dataset=expect_full_dataset),
        context="cleaned dataset",
    )
    return write_splits(
        assign_batches(cleaned),
        out_dir,
        source=load_raw_metadata(raw_path),
        allow_empty_splits=not expect_full_dataset,
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
