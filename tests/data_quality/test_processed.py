"""Data-quality tests for everything downstream of cleaning.

These run on a synthetic frame with the same schema and the same defects as
the published file, so they gate every pull request rather than only the runs
where the 5.5 MB download succeeded.

The reproducibility tests are the load-bearing ones. If a rerun can silently
produce different batches, then "the metrics moved" has two possible causes
and no way to tell them apart.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.data.download import raw_meta_path
from credit_risk.data.split import (
    BATCH_COL,
    MANIFEST_NAME,
    SPLIT_FILES,
    SplitError,
    assign_batches,
    build_splits,
    clean,
    frame_sha256,
    load_split,
    main,
    split_frames,
    write_splits,
)
from credit_risk.data.validate import validate_clean


@pytest.fixture
def full_batched(make_raw_frame: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """A full-portfolio frame: 30,000 accounts, cleaned and batched."""
    return assign_batches(clean(make_raw_frame(n_rows=schema.RAW_N_ROWS)))


# --------------------------------------------------------------- cleaning


def test_education_codes_are_a_subset_of_the_dictionary(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    assert set(synthetic_clean_df[schema.EDUCATION]) <= set(schema.EDUCATION_CODES)


def test_marriage_codes_are_a_subset_of_the_dictionary(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    assert set(synthetic_clean_df[schema.MARRIAGE]) <= set(schema.MARRIAGE_CODES)


def test_undocumented_rows_are_folded_not_dropped(
    synthetic_raw_df: pd.DataFrame, synthetic_clean_df: pd.DataFrame
) -> None:
    """Deleting them would quietly move the base rates the fairness work reports."""
    assert len(synthetic_clean_df) == len(synthetic_raw_df)
    assert synthetic_clean_df.loc[0, schema.EDUCATION] == schema.EDUCATION_OTHER
    assert synthetic_clean_df.loc[1, schema.EDUCATION] == schema.EDUCATION_OTHER
    assert synthetic_clean_df.loc[2, schema.EDUCATION] == schema.EDUCATION_OTHER
    assert synthetic_clean_df.loc[3, schema.MARRIAGE] == schema.MARRIAGE_OTHER


def test_pay_0_is_renamed_and_gone(synthetic_clean_df: pd.DataFrame) -> None:
    assert schema.PAY_0_RAW not in synthetic_clean_df.columns
    assert schema.PAY_1 in synthetic_clean_df.columns


def test_target_column_is_renamed(synthetic_clean_df: pd.DataFrame) -> None:
    assert schema.TARGET_RAW not in synthetic_clean_df.columns
    assert schema.TARGET in synthetic_clean_df.columns


def test_age_group_is_added_with_both_values(synthetic_clean_df: pd.DataFrame) -> None:
    assert set(synthetic_clean_df[schema.AGE_GROUP]) == {"young", "older"}
    young = synthetic_clean_df[schema.AGE_GROUP] == "young"
    assert (synthetic_clean_df.loc[young, schema.AGE] <= schema.AGE_GROUP_CUTOFF).all()
    assert (synthetic_clean_df.loc[~young, schema.AGE] > schema.AGE_GROUP_CUTOFF).all()


def test_column_order_is_the_published_contract(synthetic_clean_df: pd.DataFrame) -> None:
    assert tuple(synthetic_clean_df.columns) == (*schema.CLEAN_COLUMNS, schema.AGE_GROUP)


def test_cleaned_frame_satisfies_the_clean_validator(synthetic_clean_df: pd.DataFrame) -> None:
    report = validate_clean(synthetic_clean_df, expect_full_dataset=False)
    assert report.ok, [c.to_dict() for c in report.failures]


def test_clean_refuses_a_frame_missing_a_column(synthetic_raw_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=schema.LIMIT_BAL):
        clean(synthetic_raw_df.drop(columns=[schema.LIMIT_BAL]))


def test_clean_does_not_mutate_its_input(synthetic_raw_df: pd.DataFrame) -> None:
    before = synthetic_raw_df.copy(deep=True)
    clean(synthetic_raw_df)
    pd.testing.assert_frame_equal(synthetic_raw_df, before)


# --------------------------------------------------------------- batching


def test_batch_assignment_is_deterministic(synthetic_clean_df: pd.DataFrame) -> None:
    """Same accounts in any order must yield byte-identical batches."""
    first = assign_batches(synthetic_clean_df)
    shuffled = assign_batches(synthetic_clean_df.sample(frac=1.0, random_state=7))

    assert frame_sha256(first) == frame_sha256(shuffled)
    pd.testing.assert_frame_equal(first, shuffled)


def test_assign_batches_requires_an_id(synthetic_clean_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=schema.ID_COL):
        assign_batches(synthetic_clean_df.drop(columns=[schema.ID_COL]))


def test_the_portfolio_splits_into_six_equal_batches(full_batched: pd.DataFrame) -> None:
    counts = full_batched[BATCH_COL].value_counts().sort_index()
    assert list(counts.index) == list(range(1, schema.N_BATCHES + 1))
    assert set(counts) == {schema.BATCH_SIZE}


def test_batches_follow_sorted_id(full_batched: pd.DataFrame) -> None:
    """Batch 1 is the lowest 5,000 IDs, not a random 5,000."""
    assert full_batched[schema.ID_COL].is_monotonic_increasing
    first_batch = full_batched[full_batched[BATCH_COL] == 1][schema.ID_COL]
    assert int(first_batch.max()) == schema.BATCH_SIZE


def test_splits_are_disjoint_and_account_for_everyone(full_batched: pd.DataFrame) -> None:
    frames = split_frames(full_batched)
    ids = {name: set(frame[schema.ID_COL]) for name, frame in frames.items()}

    assert len(frames["train"]) == len(schema.TRAIN_BATCHES) * schema.BATCH_SIZE
    assert len(frames["test"]) == schema.BATCH_SIZE
    assert len(frames["serving_pool"]) == schema.BATCH_SIZE

    assert not ids["train"] & ids["test"]
    assert not ids["train"] & ids["serving_pool"]
    assert not ids["test"] & ids["serving_pool"]
    assert len(ids["train"] | ids["test"] | ids["serving_pool"]) == schema.RAW_N_ROWS


def test_split_frames_refuses_an_unbatched_frame(synthetic_clean_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=BATCH_COL):
        split_frames(synthetic_clean_df)


# --------------------------------------------------------------- manifest


def test_write_splits_emits_files_and_a_manifest(
    full_batched: pd.DataFrame, tmp_path: Path
) -> None:
    manifest = write_splits(full_batched, tmp_path)

    for name, filename in SPLIT_FILES.items():
        assert (tmp_path / filename).exists()
        entry = manifest["splits"][name]
        assert entry["n_rows"] == len(pd.read_parquet(tmp_path / filename))
        assert len(entry["sha256"]) == 64
        assert entry["batches"]

    on_disk = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert on_disk["source_rows"] == schema.RAW_N_ROWS
    assert on_disk["splits"].keys() == manifest["splits"].keys()


def test_rerunning_write_splits_produces_identical_hashes(
    full_batched: pd.DataFrame, tmp_path: Path
) -> None:
    """Reruns must be verifiably the same, or drift becomes unattributable."""
    first = write_splits(full_batched, tmp_path / "run1")
    second = write_splits(full_batched, tmp_path / "run2")

    digests = {name: entry["sha256"] for name, entry in first["splits"].items()}
    assert digests == {name: entry["sha256"] for name, entry in second["splits"].items()}


def test_load_split_round_trips(full_batched: pd.DataFrame, tmp_path: Path) -> None:
    write_splits(full_batched, tmp_path)
    loaded = load_split("test", tmp_path)
    assert len(loaded) == schema.BATCH_SIZE
    assert set(loaded[BATCH_COL]) == {schema.TEST_BATCH}


def test_load_split_rejects_an_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="validation"):
        load_split("validation", tmp_path)


def test_load_split_says_what_to_run_when_nothing_was_written(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="credit_risk.data.split"):
        load_split("train", tmp_path)


def test_cli_validates_cleans_and_writes(
    synthetic_raw_df: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "processed"
    synthetic_raw_df.to_parquet(raw_path, index=False)

    exit_code = main(["--raw", str(raw_path), "--out-dir", str(out_dir), "--allow-partial"])

    assert exit_code == 0
    assert (out_dir / MANIFEST_NAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["source_rows"] == len(synthetic_raw_df)


# ------------------------------------------------------- short-frame guard


def test_a_frame_too_short_to_reach_batch_five_is_refused(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """The held-out test set silently coming out empty is worse than a red task.

    assign_batches clips at N_BATCHES, so anything under 25,001 rows never
    reaches batch 5. Left alone this writes a valid manifest and two empty
    parquet files, and the PR-AUC gate downstream scores zero rows and reports
    whatever an empty frame reports.
    """
    short = assign_batches(clean(make_raw_frame(n_rows=1000)))

    with pytest.raises(SplitError, match="test"):
        write_splits(short, tmp_path)

    assert not (tmp_path / SPLIT_FILES["train"]).exists(), "nothing should be written on refusal"


def test_sample_mode_writes_the_short_frame_but_marks_it_degraded(
    synthetic_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """`--allow-partial` is a real workflow; it just may not pass for a full run."""
    manifest = build_splits(synthetic_raw_df, tmp_path, expect_full_dataset=False)

    assert manifest["degraded"] is True
    assert set(manifest["empty_splits"]) == {"test", "serving_pool"}
    assert manifest["splits"]["test"]["n_rows"] == 0


def test_a_full_portfolio_is_not_degraded(full_batched: pd.DataFrame, tmp_path: Path) -> None:
    manifest = write_splits(full_batched, tmp_path)
    assert manifest["degraded"] is False
    assert manifest["empty_splits"] == []


# -------------------------------------------------------------- provenance


def test_the_manifest_records_which_file_the_splits_came_from(
    synthetic_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Versioning, end to end: "did the data change or did the model change?".

    download.py hashes the archive it fetched into a sidecar. Without that
    digest in the manifest the question stops halfway -- the splits are
    hashed, but nothing says which source produced them.
    """
    raw_path = tmp_path / "raw.parquet"
    synthetic_raw_df.to_parquet(raw_path, index=False)
    digest = "a" * 64
    raw_meta_path(raw_path).write_text(
        json.dumps({"sha256": digest, "n_rows": len(synthetic_raw_df)}) + "\n"
    )

    manifest = build_splits(
        synthetic_raw_df, tmp_path / "out", expect_full_dataset=False, raw_path=raw_path
    )

    assert manifest["source_sha256"] == digest
    assert manifest["source_n_rows"] == len(synthetic_raw_df)


def test_an_unknown_origin_is_recorded_as_unknown(
    synthetic_raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A null says "nobody knows"; an absent key says "nobody asked"."""
    raw_path = tmp_path / "raw.parquet"
    synthetic_raw_df.to_parquet(raw_path, index=False)

    manifest = build_splits(
        synthetic_raw_df, tmp_path / "out", expect_full_dataset=False, raw_path=raw_path
    )

    assert manifest["source_sha256"] is None
    assert "source_n_rows" in manifest
