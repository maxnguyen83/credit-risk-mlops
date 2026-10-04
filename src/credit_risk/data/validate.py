"""Data-quality checks that run before anything trains on the data.

Two ideas hold this module together.

First, a check reports rather than throws. A validator that raises on the
first problem tells you about one broken column at a time; a validator that
returns a :class:`ValidationReport` tells you about all of them in one run,
and the report is JSON so it can be attached to an Airflow XCom or an MLflow
artifact and read months later.

Second, not every anomaly is fatal. The undocumented ``EDUCATION`` codes
0/5/6 and ``MARRIAGE`` code 0 are genuine defects in the published file, but
cleaning folds them into the documented "other" bucket, so they are warnings.
Nulls, impossible ages and negative payments are errors: they mean the file we
received is not the file the pipeline was written against.

:func:`assert_ok` turns a report into the DAG's fail-fast gate. Passing that
gate does not make the bad rows good: it means there are few enough of them to
set aside. :meth:`ValidationReport.row_failures` names the checks each row
failed, and ``split.build_splits`` quarantines every row it names instead of
training on it.

Run it directly:

    python -m credit_risk.data.validate [--raw PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from credit_risk import schema

log = logging.getLogger(__name__)

Severity = Literal["error", "warning"]

ERROR: Severity = "error"
WARNING: Severity = "warning"

#: Name of the column the splitter stamps on each row. It lives here, and not
#: in ``data.split`` where it is used, because ``split`` imports this module
#: and not the other way round -- a constant defined over there could only
#: reach :func:`validate_clean` through an import cycle. Spelling it as a
#: literal in the tolerated-columns list was the alternative, and it meant a
#: rename in one file would make the validator condemn every row of a
#: perfectly good frame and fail the DAG with a message about the wrong thing.
BATCH_COL: Final = "batch"


class DataValidationError(RuntimeError):
    """Raised when a frame is too damaged to continue with."""


@dataclass(frozen=True)
class Check:
    """One named assertion about a frame, plus why it passed or failed."""

    name: str
    passed: bool
    detail: str
    severity: Severity = ERROR
    n_bad_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
            "n_bad_rows": self.n_bad_rows,
        }


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating one frame."""

    n_rows: int
    n_cols: int
    checks: tuple[Check, ...] = field(default_factory=tuple)
    bad_row_fraction: float = 0.0
    # Positional masks of the rows each failed ERROR check condemns, keyed by
    # check name. Not part of the JSON report or of equality: they are the
    # working data the quarantine step needs, one boolean per row.
    row_masks: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    @property
    def ok(self) -> bool:
        """True when every error-severity check passed. Warnings do not count."""
        return all(check.passed for check in self.checks if check.severity == ERROR)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.severity == ERROR and not c.passed)

    @property
    def warnings(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.severity == WARNING and not c.passed)

    def bad_row_mask(self) -> np.ndarray:
        """One boolean per row: True where any ERROR check failed.

        Its mean is :attr:`bad_row_fraction`, so the rows the gate counted are
        exactly the rows a caller sets aside.
        """
        mask = np.zeros(self.n_rows, dtype=bool)
        for condemned in self.row_masks.values():
            mask |= condemned
        return mask

    def row_failures(self) -> list[str]:
        """Per row, the failed ERROR checks as ``"name;name"``; empty for a good row."""
        failed: list[list[str]] = [[] for _ in range(self.n_rows)]
        for name, condemned in self.row_masks.items():
            for position in np.flatnonzero(condemned):
                failed[position].append(name)
        return [";".join(names) for names in failed]

    def check(self, name: str) -> Check:
        """Look one check up by name. Raises KeyError if it was never run."""
        for candidate in self.checks:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no check named {name!r}; ran {[c.name for c in self.checks]}")

    def to_dict(self) -> dict[str, Any]:
        """A plain dict, safe for json.dumps and for an XCom payload."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "ok": self.ok,
            "bad_row_fraction": round(self.bad_row_fraction, 6),
            "checks": [c.to_dict() for c in self.checks],
        }


# A check result is the check itself plus, when the check is row-level, the
# mask of rows it condemns. None means "this is a whole-frame check".
_Result = tuple[Check, np.ndarray | None]


def _scalar(value: Any) -> Any:
    """Unwrap numpy scalars so details format the same on every numpy version."""
    return value.item() if hasattr(value, "item") else value


def _describe_counts(values: pd.Series) -> str:
    counts = {_scalar(k): int(v) for k, v in values.value_counts().items()}
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts, key=str))


def _missing_columns(df: pd.DataFrame, name: str, missing: Sequence[str]) -> _Result:
    """Every row is suspect when a column the check needs is not there."""
    detail = f"required column(s) absent: {list(missing)}"
    return Check(name, False, detail, ERROR, len(df)), None


def _check_row_count(df: pd.DataFrame, expected: int, enforce: bool) -> _Result:
    if not enforce:
        return Check("row_count", True, f"not enforced (sample mode); {len(df)} rows"), None
    passed = len(df) == expected
    detail = f"{len(df)} rows" + ("" if passed else f", expected {expected}")
    return Check("row_count", passed, detail, ERROR, 0 if passed else len(df)), None


def _check_columns(
    df: pd.DataFrame,
    expected: Sequence[str],
    optional: Sequence[str] = (),
) -> _Result:
    missing = [c for c in expected if c not in df.columns]
    allowed = set(expected) | set(optional)
    unexpected = [c for c in df.columns if c not in allowed]
    passed = not missing and not unexpected
    detail = (
        f"{len(df.columns)} columns as expected"
        if passed
        else f"missing={missing} unexpected={unexpected}"
    )
    return Check("column_set", passed, detail, ERROR, 0 if passed else len(df)), None


def _check_no_nulls(df: pd.DataFrame) -> _Result:
    mask = df.isna().any(axis=1).to_numpy()
    n_bad = int(mask.sum())
    detail = "no nulls" if n_bad == 0 else f"{n_bad} rows contain at least one null"
    return Check("no_nulls", n_bad == 0, detail, ERROR, n_bad), mask


def _check_numeric_dtypes(df: pd.DataFrame, cols: Sequence[str]) -> _Result:
    present = [c for c in cols if c in df.columns]
    offenders = [c for c in present if not is_numeric_dtype(df[c])]
    passed = not offenders
    detail = "all numeric" if passed else f"non-numeric: {offenders}"
    return Check("numeric_dtypes", passed, detail, ERROR, 0 if passed else len(df)), None


def _bounded(
    df: pd.DataFrame,
    name: str,
    cols: Sequence[str],
    *,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
    severity: Severity = ERROR,
) -> _Result:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return _missing_columns(df, name, missing)

    values = df[list(cols)].to_numpy(dtype="float64")
    bad = np.zeros(values.shape, dtype=bool)
    if low is not None:
        bad |= (values < low) if low_inclusive else (values <= low)
    if high is not None:
        bad |= values > high

    mask = bad.any(axis=1)
    n_bad = int(mask.sum())
    bound = f"[{low}, {high}]" if low_inclusive else f"({low}, {high}]"
    detail = f"within {bound}" if n_bad == 0 else f"{n_bad} rows outside {bound}"
    return Check(name, n_bad == 0, detail, severity, n_bad), mask


def _in_set(
    df: pd.DataFrame,
    name: str,
    col: str,
    allowed: Iterable[Any],
    *,
    severity: Severity = ERROR,
) -> _Result:
    if col not in df.columns:
        return _missing_columns(df, name, [col])

    allowed_set = set(allowed)
    mask = ~df[col].isin(allowed_set).to_numpy()
    n_bad = int(mask.sum())
    if n_bad == 0:
        detail = f"{col} within {sorted(allowed_set, key=str)}"
    else:
        seen = _describe_counts(df.loc[mask, col])
        detail = f"{col} has {seen} outside {sorted(allowed_set, key=str)}"
    return Check(name, n_bad == 0, detail, severity, n_bad), mask


def _absent(df: pd.DataFrame, name: str, col: str) -> _Result:
    passed = col not in df.columns
    detail = f"{col} is gone" if passed else f"{col} is still present"
    return Check(name, passed, detail, ERROR, 0 if passed else len(df)), None


def _assemble(df: pd.DataFrame, results: Sequence[_Result]) -> ValidationReport:
    """Fold check results into a report, accumulating the bad-row mask."""
    n_rows = len(df)
    bad = np.zeros(n_rows, dtype=bool)
    checks: list[Check] = []
    row_masks: dict[str, np.ndarray] = {}

    for check, mask in results:
        checks.append(check)
        if check.severity != ERROR or check.passed:
            continue
        # A failed whole-frame check (wrong columns, wrong row count) condemns
        # every row: the frame is not the shape the pipeline was written for.
        condemned = mask if mask is not None else np.ones(n_rows, dtype=bool)
        row_masks[check.name] = condemned
        bad |= condemned

    # An empty frame scores 1.0 rather than dividing by zero -- "no rows" is a
    # pipeline failure, not a clean bill of health.
    fraction = float(bad.sum()) / n_rows if n_rows else 1.0
    return ValidationReport(
        n_rows=n_rows,
        n_cols=int(df.shape[1]),
        checks=tuple(checks),
        bad_row_fraction=fraction,
        row_masks=row_masks,
    )


def validate_raw(df: pd.DataFrame, *, expect_full_dataset: bool = True) -> ValidationReport:
    """Validate the frame exactly as it came out of the UCI spreadsheet.

    ``expect_full_dataset`` exists so unit tests can validate a small sample
    without the 30,000-row assertion drowning out every other finding.
    """
    raw_pay_cols = (schema.PAY_0_RAW, *schema.PAY_COLS[1:])
    results: list[_Result] = [
        _check_row_count(df, schema.RAW_N_ROWS, expect_full_dataset),
        _check_columns(df, schema.RAW_COLUMNS),
        _check_no_nulls(df),
        _check_numeric_dtypes(df, schema.RAW_COLUMNS),
        _bounded(df, "age_range", [schema.AGE], low=schema.AGE_MIN, high=schema.AGE_MAX),
        _bounded(df, "limit_bal_positive", [schema.LIMIT_BAL], low=0, low_inclusive=False),
        _bounded(df, "pay_amt_non_negative", schema.PAY_AMT_COLS, low=0),
        _bounded(df, "pay_status_range", raw_pay_cols, low=schema.PAY_MIN, high=schema.PAY_MAX),
        _in_set(df, "sex_codes", schema.SEX, schema.SEX_CODES),
        _in_set(df, "target_binary", schema.TARGET_RAW, (0, 1)),
        # Warnings: the published dictionary is wrong, not the data. clean()
        # folds these into "other" and validate_clean asserts they are gone.
        _in_set(df, "education_codes", schema.EDUCATION, schema.EDUCATION_CODES, severity=WARNING),
        _in_set(df, "marriage_codes", schema.MARRIAGE, schema.MARRIAGE_CODES, severity=WARNING),
    ]
    return _assemble(df, results)


def validate_clean(
    df: pd.DataFrame,
    *,
    expect_full_dataset: bool = True,
    expected_rows: int | None = None,
) -> ValidationReport:
    """Validate the frame after :func:`credit_risk.data.split.clean`.

    This is the contract every downstream consumer relies on, so the codes
    that were merely warnings on the raw frame are hard errors here.

    ``expected_rows`` defaults to the full file. The splitter passes the full
    file minus what it quarantined, so a run that set rows aside is checked
    against the count it should have kept, not failed for keeping fewer.
    """
    expected = (*schema.CLEAN_COLUMNS, schema.AGE_GROUP)
    want = schema.RAW_N_ROWS if expected_rows is None else expected_rows
    results: list[_Result] = [
        _check_row_count(df, want, expect_full_dataset),
        # BATCH_COL is added after cleaning; tolerate it so the same report
        # can be run on a split frame without a spurious "unexpected column".
        _check_columns(df, expected, optional=(BATCH_COL,)),
        _absent(df, "pay_0_renamed", schema.PAY_0_RAW),
        _absent(df, "target_renamed", schema.TARGET_RAW),
        _check_no_nulls(df),
        _check_numeric_dtypes(df, schema.CLEAN_COLUMNS),
        _in_set(df, "education_codes", schema.EDUCATION, schema.EDUCATION_CODES),
        _in_set(df, "marriage_codes", schema.MARRIAGE, schema.MARRIAGE_CODES),
        _in_set(df, "sex_codes", schema.SEX, schema.SEX_CODES),
        _in_set(df, "age_group_values", schema.AGE_GROUP, ("young", "older")),
        _in_set(df, "target_binary", schema.TARGET, (0, 1)),
        _bounded(df, "age_range", [schema.AGE], low=schema.AGE_MIN, high=schema.AGE_MAX),
        _bounded(df, "limit_bal_positive", [schema.LIMIT_BAL], low=0, low_inclusive=False),
        _bounded(df, "pay_amt_non_negative", schema.PAY_AMT_COLS, low=0),
        _bounded(df, "pay_status_range", schema.PAY_COLS, low=schema.PAY_MIN, high=schema.PAY_MAX),
    ]
    return _assemble(df, results)


def assert_ok(report: ValidationReport, *, context: str = "dataset") -> None:
    """Fail loudly when too much of the frame is broken.

    The DAG calls this instead of inspecting the report itself: the decision
    "how bad is too bad" belongs in one place, next to the threshold, and the
    answer is schema.MAX_BAD_ROW_FRACTION.

    Returning means "few enough to set aside", not "nothing wrong": the rows
    behind a failed check are still bad, and the caller must drop them
    (``split.build_splits`` quarantines them) rather than train on them.
    """
    if report.bad_row_fraction <= schema.MAX_BAD_ROW_FRACTION:
        return

    named = ", ".join(f"{c.name} ({c.n_bad_rows} rows)" for c in report.failures) or "none reported"
    raise DataValidationError(
        f"{context}: {report.bad_row_fraction:.1%} of rows failed validation, "
        f"above the {schema.MAX_BAD_ROW_FRACTION:.0%} tolerance. Failing checks: {named}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m credit_risk.data.validate``.

    The DAG runs this as its own task rather than relying on the validation
    inside ``split.build_splits``, because a red *validate_raw* task names the
    problem where an operator will look for it. Exit code 1 means the frame is
    too damaged to train on; the report is on stdout either way, so a run that
    fails still leaves the evidence behind. Exit 0 with failed checks in the
    report means the damage is under the tolerance: ``clean_and_split`` sets
    those rows aside in ``quarantine.parquet`` rather than training on them.
    """
    # Imported here, not at module scope: download.py pulls in `requests`, and
    # this module is on the import path of every training and serving process.
    from credit_risk.data.download import load_raw, raw_parquet_path

    parser = argparse.ArgumentParser(description="Validate the raw credit-default parquet")
    parser.add_argument("--raw", type=Path, default=None, help="parquet to validate")
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
    report = validate_raw(load_raw(source), expect_full_dataset=not args.allow_partial)
    for warning in report.warnings:
        log.warning("data warning %s: %s", warning.name, warning.detail)
    for failure in report.failures:
        log.warning("data error %s: %s", failure.name, failure.detail)

    # stdout is the contract for subprocess callers; logs go to stderr.
    print(json.dumps(report.to_dict(), indent=2))

    try:
        assert_ok(report, context=str(source))
    except DataValidationError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
