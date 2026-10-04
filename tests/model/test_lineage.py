"""Which data a training run saw, recorded on the run itself.

The splits manifest names a content hash for every split, but it is one file
that every `clean_and_split` overwrites. Until a run carried those hashes, the
question "was this model trained on the data in data/processed today?" could
only be answered for the most recent run, and then only by trusting that
nothing had been re-split since. These tests pin that a run logs the hashes of
the frames it actually trained and scored on, adopts the manifest's source
digest only when the manifest describes those same frames, and keeps its own
copy of that manifest as an artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from credit_risk.data.split import BATCH_COL, MANIFEST_NAME, frame_sha256, write_splits
from credit_risk.models import train as train_module
from credit_risk.models.train import (
    MANIFEST_ARTIFACT_DIR,
    MANIFEST_PARAM,
    SOURCE_SHA_PARAM,
    TEST_SHA_PARAM,
    TRAIN_SHA_PARAM,
    data_lineage,
    load_training_splits,
    train_all,
)
from tests.model.test_performance import synthetic_clean_frame, synthetic_split

SOURCE_DIGEST = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"


def written_splits(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits written by the data module, with a provenance digest, read back."""
    frame = synthetic_clean_frame(n=600)
    frame[BATCH_COL] = np.repeat([1, 2, 3, 4, 5, 6], 100)
    write_splits(frame, tmp_path, source={"sha256": SOURCE_DIGEST, "n_rows": 600})
    return load_training_splits(tmp_path)


def test_a_run_on_the_written_splits_adopts_the_manifest(tmp_path: Path) -> None:
    train_df, test_df = written_splits(tmp_path)

    lineage = data_lineage(train_df, test_df, tmp_path)

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    # Hashes of what was read back from parquet equal the ones the split step
    # wrote down: the manifest and the run describe the same rows.
    assert lineage.params[TRAIN_SHA_PARAM] == manifest["splits"]["train"]["sha256"]
    assert lineage.params[TEST_SHA_PARAM] == manifest["splits"]["test"]["sha256"]
    assert lineage.params[SOURCE_SHA_PARAM] == SOURCE_DIGEST
    assert lineage.params[MANIFEST_PARAM] == "matched"
    assert lineage.manifest_path == tmp_path / MANIFEST_NAME


def test_a_manifest_that_describes_other_data_is_not_adopted(tmp_path: Path) -> None:
    written_splits(tmp_path)
    train_df, test_df = synthetic_split(n=400)

    lineage = data_lineage(train_df, test_df, tmp_path)

    # The frames' own hashes are still recorded; the manifest's source digest
    # is not, because it is the provenance of somebody else's rows.
    assert lineage.params[TRAIN_SHA_PARAM] == frame_sha256(train_df)
    assert lineage.params[TEST_SHA_PARAM] == frame_sha256(test_df)
    assert lineage.params[SOURCE_SHA_PARAM] == "unknown"
    assert lineage.params[MANIFEST_PARAM] == "mismatch"
    assert lineage.manifest_path is None


@pytest.mark.parametrize(("content", "status"), [(None, "absent"), ("{not json", "unreadable")])
def test_no_usable_manifest_still_records_the_frame_hashes(
    tmp_path: Path, content: str | None, status: str
) -> None:
    if content is not None:
        (tmp_path / MANIFEST_NAME).write_text(content)
    train_df, test_df = synthetic_split(n=400)

    lineage = data_lineage(train_df, test_df, tmp_path)

    assert lineage.params[TEST_SHA_PARAM] == frame_sha256(test_df)
    assert lineage.params[MANIFEST_PARAM] == status
    assert lineage.manifest_path is None


@pytest.mark.slow
def test_every_training_run_logs_its_lineage_and_keeps_the_manifest(tmp_path: Path) -> None:
    uri = (tmp_path / "mlruns").as_uri()
    previous = mlflow.get_tracking_uri()
    train_df, test_df = synthetic_split(n=1500)
    manifest = {
        "generated_at": "2026-10-03T02:36:20+00:00",
        "source_sha256": SOURCE_DIGEST,
        "splits": {
            "train": {"sha256": frame_sha256(train_df)},
            "test": {"sha256": frame_sha256(test_df)},
        },
    }
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / MANIFEST_NAME).write_text(json.dumps(manifest))
    try:
        result = train_all(
            train_df,
            test_df,
            experiment="lineage",
            tracking_uri=uri,
            grid={"n_estimators": [40], "learning_rate": [0.1], "num_leaves": [15]},
            folds=2,
            thresholds_path=tmp_path / "group_thresholds.json",
            processed_dir=processed,
        )

        for candidate in result.candidates:
            assert candidate.run_id is not None
            params = mlflow.get_run(candidate.run_id).data.params
            assert params[TRAIN_SHA_PARAM] == frame_sha256(train_df)
            assert params[TEST_SHA_PARAM] == frame_sha256(test_df)
            assert params[SOURCE_SHA_PARAM] == SOURCE_DIGEST
            assert params[MANIFEST_PARAM] == "matched"
            # The run's own copy: the file in data/processed is rewritten by
            # the next split, this one is not.
            local = mlflow.artifacts.download_artifacts(
                run_id=candidate.run_id,
                artifact_path=f"{MANIFEST_ARTIFACT_DIR}/{MANIFEST_NAME}",
                dst_path=str(tmp_path / candidate.name),
            )
            assert json.loads(Path(local).read_text()) == manifest

        # The hand-off carries it too, so the register step need not re-read it.
        handoff = tmp_path / "training_result.json"
        train_module.save_training_result(result, handoff)
        assert json.loads(handoff.read_text())["data_lineage"][TEST_SHA_PARAM] == frame_sha256(
            test_df
        )
    finally:
        mlflow.set_tracking_uri(previous)
