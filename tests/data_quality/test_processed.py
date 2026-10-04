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
    QUARANTINE_FILE,
    QUARANTINE_REASON_COL,
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
from credit_risk.data.validate import DataValidationError, validate_clean


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


# -------------------------------------------------------------- quarantine


def _full_raw_with(
    make_raw_frame: Callable[..., pd.DataFrame], column: str, value: int, n_bad: int
) -> tuple[pd.DataFrame, list[int]]:
    """30,000 raw rows with ``n_bad`` of them spread across every batch set to ``value``."""
    raw = make_raw_frame(n_rows=schema.RAW_N_ROWS)
    positions = list(range(7, schema.RAW_N_ROWS, schema.RAW_N_ROWS // n_bad))[:n_bad]
    raw.loc[positions, column] = value
    return raw, [int(raw.loc[p, schema.ID_COL]) for p in positions]


def _written_splits(out_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: load_split(name, out_dir) for name in SPLIT_FILES}


def test_rows_failing_an_error_check_are_quarantined_not_trained_on(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """0.33% of the file is under the 5% tolerance, so the run goes ahead --
    without those rows. Before this, all 100 impossible ages reached
    train.parquet because the gate only looked at the ratio."""
    raw, bad_ids = _full_raw_with(make_raw_frame, schema.AGE, 150, n_bad=100)

    manifest = build_splits(raw, tmp_path)

    splits = _written_splits(tmp_path)
    for name, frame in splits.items():
        assert frame[schema.AGE].max() <= schema.AGE_MAX, f"an impossible age reached {name}"
        assert not set(frame[schema.ID_COL]) & set(bad_ids)
    assert sum(len(frame) for frame in splits.values()) == schema.RAW_N_ROWS - 100

    quarantined = pd.read_parquet(tmp_path / QUARANTINE_FILE)
    assert sorted(quarantined[schema.ID_COL]) == sorted(bad_ids)
    assert set(quarantined[QUARANTINE_REASON_COL]) == {"age_range"}
    # Kept as received, so the rows can be inspected or fixed at the source.
    assert set(quarantined[schema.AGE]) == {150}

    assert manifest["quarantined_rows"] == 100
    assert manifest["quarantine"]["n_rows"] == 100
    assert manifest["quarantine"]["by_check"] == {"age_range": 100}
    assert manifest["source_rows"] == schema.RAW_N_ROWS


def test_an_undocumented_sex_code_does_not_become_a_third_group(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    raw, _ = _full_raw_with(make_raw_frame, schema.SEX, 3, n_bad=60)

    manifest = build_splits(raw, tmp_path)

    for frame in _written_splits(tmp_path).values():
        assert set(frame[schema.SEX]) <= set(schema.SEX_CODES)
    assert manifest["quarantine"]["by_check"] == {"sex_codes": 60}


def test_quarantine_leaves_every_other_account_in_its_batch(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """A quarantined row leaves a hole in its own batch. Re-batching what is
    left would shift the next 5,000-account boundary and quietly change the
    membership of the held-out test set because of a typo in batch 1."""
    raw, bad_ids = _full_raw_with(make_raw_frame, schema.LIMIT_BAL, -1, n_bad=30)
    expected = assign_batches(clean(make_raw_frame(n_rows=schema.RAW_N_ROWS)))
    expected = expected[~expected[schema.ID_COL].isin(bad_ids)]

    build_splits(raw, tmp_path)

    written = pd.concat(_written_splits(tmp_path).values())
    batch_of = dict(zip(written[schema.ID_COL], written[BATCH_COL], strict=True))
    assert batch_of == dict(zip(expected[schema.ID_COL], expected[BATCH_COL], strict=True))


def test_a_clean_file_quarantines_nothing_and_hashes_exactly_as_before(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    raw = make_raw_frame(n_rows=schema.RAW_N_ROWS)

    manifest = build_splits(raw, tmp_path / "built")
    reference = write_splits(assign_batches(clean(raw)), tmp_path / "reference")

    assert manifest["quarantined_rows"] == 0
    assert len(pd.read_parquet(tmp_path / "built" / QUARANTINE_FILE)) == 0
    for name in SPLIT_FILES:
        assert manifest["splits"][name]["sha256"] == reference["splits"][name]["sha256"]


def test_a_rerun_replaces_the_previous_quarantine(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """A quarantine file left over from an earlier run would describe rows that
    were never set aside this time."""
    raw, _ = _full_raw_with(make_raw_frame, schema.AGE, 150, n_bad=10)
    build_splits(raw, tmp_path)

    build_splits(make_raw_frame(n_rows=schema.RAW_N_ROWS), tmp_path)

    assert len(pd.read_parquet(tmp_path / QUARANTINE_FILE)) == 0


def test_more_bad_rows_than_the_tolerance_still_stops_the_run(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    raw, _ = _full_raw_with(make_raw_frame, schema.AGE, 150, n_bad=1600)

    with pytest.raises(DataValidationError, match="age_range"):
        build_splits(raw, tmp_path)

    assert not (tmp_path / QUARANTINE_FILE).exists()


@pytest.mark.parametrize(
    ("column", "code", "check"),
    [(schema.EDUCATION, 7, "education_codes"), (schema.MARRIAGE, 9, "marriage_codes")],
)
def test_a_code_cleaning_cannot_fold_is_quarantined_too(
    make_raw_frame: Callable[..., pd.DataFrame],
    tmp_path: Path,
    column: str,
    code: int,
    check: str,
) -> None:
    """EDUCATION=7 is only a warning on the raw frame (the folds might fix it),
    survives the fold (it is not 0/5/6), and fails the cleaned-frame contract.
    Checking that contract with the 5% ratio alone let all 60 rows into train."""
    raw, bad_ids = _full_raw_with(make_raw_frame, column, code, n_bad=60)

    manifest = build_splits(raw, tmp_path)

    for frame in _written_splits(tmp_path).values():
        assert code not in set(frame[column])
    quarantined = pd.read_parquet(tmp_path / QUARANTINE_FILE)
    assert sorted(quarantined[schema.ID_COL]) == sorted(bad_ids)
    assert set(quarantined[QUARANTINE_REASON_COL]) == {f"clean:{check}"}
    assert set(quarantined[column]) == {code}, "kept as received, not as cleaned"
    assert manifest["quarantined_rows"] == 60
    assert manifest["quarantine"]["by_check"] == {f"clean:{check}": 60}


def test_raw_and_clean_rejections_share_one_tolerance(
    make_raw_frame: Callable[..., pd.DataFrame], tmp_path: Path
) -> None:
    """4% impossible ages and 4% unfoldable codes are each under 5%, and 8%
    of the file together. The tolerance is about the file, not each stage."""
    raw = make_raw_frame(n_rows=schema.RAW_N_ROWS)
    raw.loc[raw.index[0:1200], schema.AGE] = 150
    raw.loc[raw.index[1200:2400], schema.EDUCATION] = 7

    with pytest.raises(DataValidationError, match="8.0%"):
        build_splits(raw, tmp_path)
