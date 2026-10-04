"""Tests for the data-quality validators and the fail-fast gate.

Every corruption below is injected one at a time. A validator that fails
"something is wrong" is not much use at 2am; these tests pin each defect to
the single named check that is supposed to find it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.data import split as split_module
from credit_risk.data.split import assign_batches
from credit_risk.data.validate import (
    BATCH_COL,
    DataValidationError,
    assert_ok,
    main,
    validate_clean,
    validate_raw,
)

SAMPLE_ROWS = 200


def test_a_good_sample_passes(synthetic_raw_df: pd.DataFrame) -> None:
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)
    assert report.ok, [c.to_dict() for c in report.failures]
    assert report.bad_row_fraction == 0.0
    assert report.n_rows == SAMPLE_ROWS
    assert report.n_cols == schema.RAW_N_COLS


def test_undocumented_codes_warn_but_do_not_fail(synthetic_raw_df: pd.DataFrame) -> None:
    """Cleaning handles these; failing the DAG over them would be wrong."""
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)
    assert {c.name for c in report.warnings} == {"education_codes", "marriage_codes"}
    assert report.check("education_codes").n_bad_rows == 3
    assert report.check("marriage_codes").n_bad_rows == 1
    assert report.ok


def test_wrong_row_count_condemns_the_whole_frame(synthetic_raw_df: pd.DataFrame) -> None:
    report = validate_raw(synthetic_raw_df)  # default: the full 30,000 are expected
    assert not report.check("row_count").passed
    assert not report.ok
    assert report.bad_row_fraction == 1.0


@pytest.mark.parametrize(
    ("column", "value", "check_name"),
    [
        (schema.AGE, 200, "age_range"),
        (schema.LIMIT_BAL, -1, "limit_bal_positive"),
        (schema.PAY_AMT_COLS[0], -5, "pay_amt_non_negative"),
        (schema.SEX, 3, "sex_codes"),
        (schema.PAY_0_RAW, 99, "pay_status_range"),
        (schema.TARGET_RAW, 7, "target_binary"),
    ],
)
def test_each_corruption_trips_its_own_check(
    synthetic_raw_df: pd.DataFrame, column: str, value: Any, check_name: str
) -> None:
    synthetic_raw_df.loc[0, column] = value
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)

    assert {c.name for c in report.failures} == {check_name}
    assert report.check(check_name).n_bad_rows == 1
    assert report.bad_row_fraction == pytest.approx(1 / SAMPLE_ROWS)


def test_injected_null_is_caught(synthetic_raw_df: pd.DataFrame) -> None:
    synthetic_raw_df[schema.AGE] = synthetic_raw_df[schema.AGE].astype("float64")
    synthetic_raw_df.loc[0, schema.AGE] = np.nan

    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)
    assert {c.name for c in report.failures} == {"no_nulls"}
    assert report.check("no_nulls").n_bad_rows == 1


def test_non_numeric_column_is_caught(synthetic_raw_df: pd.DataFrame) -> None:
    synthetic_raw_df[schema.AGE] = synthetic_raw_df[schema.AGE].astype(str)
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)
    assert not report.check("numeric_dtypes").passed


def test_missing_column_condemns_every_row(synthetic_raw_df: pd.DataFrame) -> None:
    frame = synthetic_raw_df.drop(columns=[schema.AGE])
    report = validate_raw(frame, expect_full_dataset=False)

    assert not report.check("column_set").passed
    assert schema.AGE in report.check("column_set").detail
    assert report.check("age_range").detail.startswith("required column")
    assert report.bad_row_fraction == 1.0


def test_unexpected_column_is_caught(synthetic_raw_df: pd.DataFrame) -> None:
    synthetic_raw_df["surprise"] = 1
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)
    assert not report.check("column_set").passed
    assert "surprise" in report.check("column_set").detail


def test_assert_ok_tolerates_a_defect_below_the_threshold(
    synthetic_raw_df: pd.DataFrame,
) -> None:
    """One bad row in 200 is a data-entry error, not a reason to stop the bank."""
    synthetic_raw_df.loc[0, schema.AGE] = 200
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)

    assert not report.ok
    assert_ok(report)  # 0.5% is under MAX_BAD_ROW_FRACTION


def test_the_report_names_the_rows_it_counted_and_why(synthetic_raw_df: pd.DataFrame) -> None:
    """The gate's ratio and the rows a caller sets aside must be the same rows."""
    synthetic_raw_df.loc[3, schema.AGE] = 200
    synthetic_raw_df.loc[3, schema.SEX] = 3
    synthetic_raw_df.loc[9, schema.LIMIT_BAL] = 0
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)

    mask = report.bad_row_mask()
    assert list(np.flatnonzero(mask)) == [3, 9]
    assert mask.mean() == pytest.approx(report.bad_row_fraction)
    failures = report.row_failures()
    assert failures[3] == "age_range;sex_codes"
    assert failures[9] == "limit_bal_positive"
    assert sum(1 for reasons in failures if reasons) == 2
    # Warnings are not errors and never put a row aside: rows 0-3 carry the
    # undocumented EDUCATION/MARRIAGE codes and only row 3 is listed.
    assert failures[0] == ""


def test_assert_ok_raises_above_the_threshold(synthetic_raw_df: pd.DataFrame) -> None:
    n_bad = int(SAMPLE_ROWS * (schema.MAX_BAD_ROW_FRACTION + 0.05))
    synthetic_raw_df.loc[: n_bad - 1, schema.AGE] = 200
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)

    with pytest.raises(DataValidationError, match="age_range"):
        assert_ok(report, context="unit test")


def test_assert_ok_rejects_an_empty_frame() -> None:
    """Zero rows is a pipeline failure, not a frame with nothing wrong with it."""
    empty = pd.DataFrame({column: pd.Series(dtype="int64") for column in schema.RAW_COLUMNS})
    report = validate_raw(empty, expect_full_dataset=False)

    assert report.bad_row_fraction == 1.0
    with pytest.raises(DataValidationError):
        assert_ok(report)


def test_report_is_json_serialisable(synthetic_raw_df: pd.DataFrame) -> None:
    """The DAG puts this on an XCom and MLflow stores it as an artifact."""
    payload = validate_raw(synthetic_raw_df, expect_full_dataset=False).to_dict()
    restored = json.loads(json.dumps(payload))

    assert restored["ok"] is True
    assert restored["n_rows"] == SAMPLE_ROWS
    assert {c["name"] for c in restored["checks"]} >= {"no_nulls", "sex_codes", "age_range"}


def test_unknown_check_name_raises(synthetic_raw_df: pd.DataFrame) -> None:
    report = validate_raw(synthetic_raw_df, expect_full_dataset=False)
    with pytest.raises(KeyError):
        report.check("no_such_check")


def test_clean_frame_passes_the_clean_contract(synthetic_clean_df: pd.DataFrame) -> None:
    report = validate_clean(synthetic_clean_df, expect_full_dataset=False)
    assert report.ok, [c.to_dict() for c in report.failures]


def test_clean_contract_rejects_a_surviving_undocumented_code(
    synthetic_clean_df: pd.DataFrame,
) -> None:
    """Warning on the raw frame, hard error once cleaning was supposed to fix it."""
    synthetic_clean_df.loc[0, schema.EDUCATION] = 5
    report = validate_clean(synthetic_clean_df, expect_full_dataset=False)

    assert not report.check("education_codes").passed
    assert report.check("education_codes").severity == "error"
    assert not report.ok


def test_clean_contract_rejects_the_raw_column_names(synthetic_raw_df: pd.DataFrame) -> None:
    report = validate_clean(synthetic_raw_df, expect_full_dataset=False)
    assert not report.check("pay_0_renamed").passed
    assert not report.check("target_renamed").passed


def test_clean_contract_tolerates_the_batch_column(synthetic_clean_df: pd.DataFrame) -> None:
    batched = assign_batches(synthetic_clean_df)
    report = validate_clean(batched, expect_full_dataset=False)
    assert report.check("column_set").passed
    assert report.ok


def test_clean_contract_rejects_a_bad_age_group(synthetic_clean_df: pd.DataFrame) -> None:
    synthetic_clean_df.loc[0, schema.AGE_GROUP] = "middle"
    report = validate_clean(synthetic_clean_df, expect_full_dataset=False)
    assert not report.check("age_group_values").passed


def test_the_batch_column_has_exactly_one_name() -> None:
    """The splitter and the validator must agree, or a rename condemns every row.

    validate_clean tolerates the batch column by name. When that name was a
    literal in one module and a constant in the other, renaming the constant
    left the literal behind: column_set then reported the batch column as
    unexpected, an ERROR-severity failure condemns the whole frame, and the
    DAG died on bad_row_fraction == 1.0 with a message about the wrong thing.
    """
    assert split_module.BATCH_COL is BATCH_COL


# ------------------------------------------------------------------- cli


def _write_raw(frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "raw.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_cli_prints_the_report_and_exits_zero_on_a_good_frame(
    synthetic_raw_df: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_raw(synthetic_raw_df, tmp_path)

    assert main(["--raw", str(path), "--allow-partial"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["n_rows"] == len(synthetic_raw_df)


def test_cli_exits_non_zero_but_still_prints_the_evidence(
    synthetic_raw_df: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed run has to leave the report behind, or nobody can see why."""
    synthetic_raw_df.loc[:, schema.AGE] = 200
    path = _write_raw(synthetic_raw_df, tmp_path)

    assert main(["--raw", str(path), "--allow-partial"]) == 1

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["bad_row_fraction"] == 1.0
    assert "age_range" in {check["name"] for check in printed["checks"] if not check["passed"]}
