"""Tests for the feature builder.

The parity test is the one that matters: it is the only automated check that
the model is served the same arithmetic it was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.features import build as build_module
from credit_risk.features.build import (
    FEATURE_NAMES,
    NO_DELINQUENCY_SENTINEL,
    PAYMENT_RATIO_CAP,
    REQUIRED_INPUT_COLUMNS,
    FeatureBuildError,
    build_features,
    build_features_from_record,
    main,
    normalize_codes,
)


def _record(**overrides: Any) -> dict[str, Any]:
    """A benign account: revolving credit, mid utilisation, paying steadily."""
    record: dict[str, Any] = {
        schema.LIMIT_BAL: 200_000,
        schema.SEX: 2,
        schema.EDUCATION: 2,
        schema.MARRIAGE: 1,
        schema.AGE: 35,
    }
    record.update(dict.fromkeys(schema.PAY_COLS, 0))
    record.update(dict.fromkeys(schema.BILL_COLS, 50_000))
    record.update(dict.fromkeys(schema.PAY_AMT_COLS, 5_000))
    record.update(overrides)
    return record


def _frame(*records: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def test_feature_names_are_the_output_contract(synthetic_clean_df: pd.DataFrame) -> None:
    features = build_features(synthetic_clean_df)
    assert tuple(features.columns) == FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), "duplicate feature name"
    assert (features.dtypes == "float64").all()


def test_index_is_preserved_for_joining_back(synthetic_clean_df: pd.DataFrame) -> None:
    subset = synthetic_clean_df.iloc[10:20]
    assert list(build_features(subset).index) == list(subset.index)


def test_no_nan_or_inf_on_the_synthetic_frame(synthetic_clean_df: pd.DataFrame) -> None:
    features = build_features(synthetic_clean_df)
    assert np.isfinite(features.to_numpy()).all()


def test_no_nan_or_inf_on_pathological_rows() -> None:
    """Every guarded division gets a row that would otherwise divide by zero."""
    rows = [
        _record(**{schema.LIMIT_BAL: 0}),
        _record(**dict.fromkeys(schema.BILL_COLS, 0)),
        _record(**dict.fromkeys(schema.PAY_AMT_COLS, 0)),
        _record(
            **{schema.LIMIT_BAL: 0},
            **dict.fromkeys(schema.BILL_COLS, 0),
            **dict.fromkeys(schema.PAY_AMT_COLS, 0),
        ),
        _record(**dict.fromkeys(schema.BILL_COLS, -1_000)),
        _record(**dict.fromkeys(schema.PAY_COLS, schema.PAY_MIN)),
        _record(**dict.fromkeys(schema.PAY_COLS, schema.PAY_MAX)),
    ]
    features = build_features(_frame(*rows))
    assert np.isfinite(features.to_numpy()).all()
    assert not features.isna().to_numpy().any()


def test_zero_limit_gives_zero_utilisation_not_infinity() -> None:
    features = build_features(_frame(_record(**{schema.LIMIT_BAL: 0})))
    utilisation = [c for c in FEATURE_NAMES if c.startswith("utilization_m")]
    assert (features.loc[0, utilisation] == 0.0).all()
    assert features.loc[0, "utilization_mean"] == 0.0
    assert features.loc[0, "payment_to_limit_ratio"] == 0.0


def test_payment_ratio_is_capped() -> None:
    """A tiny bill settled with a huge payment must not dominate the matrix."""
    record = _record(**{schema.BILL_COLS[1]: 1, schema.PAY_AMT_COLS[0]: 500_000})
    features = build_features(_frame(record))
    assert features.loc[0, "payment_ratio_m1"] == PAYMENT_RATIO_CAP


def test_nothing_owed_counts_as_fully_paid() -> None:
    record = _record(**{schema.BILL_COLS[1]: 0, schema.PAY_AMT_COLS[0]: 0})
    features = build_features(_frame(record))
    assert features.loc[0, "payment_ratio_m1"] == 1.0


def test_utilisation_trend_is_positive_when_borrowing_grows() -> None:
    rising = _record(
        **{schema.LIMIT_BAL: 100_000},
        **dict(zip(schema.BILL_COLS, range(60_000, 0, -10_000), strict=True)),
    )
    falling = _record(
        **{schema.LIMIT_BAL: 100_000},
        **dict(zip(schema.BILL_COLS, range(10_000, 70_000, 10_000), strict=True)),
    )
    features = build_features(_frame(rising, falling))
    assert features.loc[0, "utilization_trend"] > 0
    assert features.loc[1, "utilization_trend"] < 0


def test_delinquency_features_grow_with_months_late() -> None:
    """Directional sanity: more late months can never mean less delinquency."""
    rows = []
    for late_months in range(len(schema.PAY_COLS) + 1):
        status = {
            column: (2 if index < late_months else 0)
            for index, column in enumerate(schema.PAY_COLS)
        }
        rows.append(_record(**status))

    features = build_features(_frame(*rows)).reset_index(drop=True)

    assert list(features["months_delinquent"]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert features["months_delinquent"].is_monotonic_increasing
    assert features["max_consecutive_delinquent"].is_monotonic_increasing
    assert features.loc[0, "months_since_last_delinquency"] == NO_DELINQUENCY_SENTINEL
    assert (features.loc[1:, "months_since_last_delinquency"] == 1.0).all()
    assert features.loc[0, "worst_pay_status"] == 0.0
    assert (features.loc[1:, "worst_pay_status"] == 2.0).all()


def test_only_the_most_recent_gap_sets_months_since() -> None:
    """Late in April only: five months of clean behaviour since."""
    status = dict.fromkeys(schema.PAY_COLS, 0)
    status[schema.PAY_COLS[5]] = 3
    features = build_features(_frame(_record(**status)))
    assert features.loc[0, "months_since_last_delinquency"] == 6.0
    assert features.loc[0, "max_consecutive_delinquent"] == 1.0


def test_train_and_serve_paths_agree_exactly(synthetic_clean_df: pd.DataFrame) -> None:
    """The single check standing between this project and train/serve skew."""
    for position in (0, 1, 4, 5, 42, 199):
        row = synthetic_clean_df.iloc[[position]]
        record = row.iloc[0].to_dict()

        from_frame = build_features(row).reset_index(drop=True)
        from_record = build_features_from_record(record).reset_index(drop=True)

        pd.testing.assert_frame_equal(from_frame, from_record, check_exact=True)


def test_record_path_ignores_extra_keys() -> None:
    """ID and the label ride along in real payloads; they must change nothing."""
    record = _record()
    noisy = {**record, schema.ID_COL: 123, schema.TARGET: 1, schema.AGE_GROUP: "young"}
    pd.testing.assert_frame_equal(
        build_features_from_record(record),
        build_features_from_record(noisy),
        check_exact=True,
    )


def test_missing_input_column_is_reported_by_name() -> None:
    frame = _frame(_record()).drop(columns=[schema.BILL_COLS[0]])
    with pytest.raises(FeatureBuildError, match=schema.BILL_COLS[0]):
        build_features(frame)


def test_missing_record_field_is_reported_by_name() -> None:
    record = _record()
    del record[schema.PAY_COLS[0]]
    with pytest.raises(FeatureBuildError, match=schema.PAY_COLS[0]):
        build_features_from_record(record)


def test_non_finite_input_is_rejected_rather_than_scored() -> None:
    frame = _frame(_record())
    frame[schema.LIMIT_BAL] = frame[schema.LIMIT_BAL].astype("float64")
    frame.loc[0, schema.LIMIT_BAL] = np.nan
    with pytest.raises(FeatureBuildError, match="non-finite"):
        build_features(frame)


def test_contract_and_construction_cannot_drift(
    monkeypatch: pytest.MonkeyPatch, synthetic_clean_df: pd.DataFrame
) -> None:
    """Adding a name to the contract without building it must fail loudly."""
    monkeypatch.setattr(build_module, "FEATURE_NAMES", (*FEATURE_NAMES, "invented_feature"))
    with pytest.raises(FeatureBuildError, match="invented_feature"):
        build_module.build_features(synthetic_clean_df)


def test_required_columns_exclude_identifier_slice_and_label() -> None:
    """Leaking ID, AGE_GROUP or the target into the matrix is how models cheat."""
    assert schema.ID_COL not in REQUIRED_INPUT_COLUMNS
    assert schema.TARGET not in REQUIRED_INPUT_COLUMNS
    assert schema.AGE_GROUP not in REQUIRED_INPUT_COLUMNS
    assert schema.ID_COL not in FEATURE_NAMES
    assert schema.TARGET not in FEATURE_NAMES


def test_empty_frame_produces_an_empty_matrix() -> None:
    empty = pd.DataFrame({column: pd.Series(dtype="float64") for column in REQUIRED_INPUT_COLUMNS})
    features = build_features(empty)
    assert features.empty
    assert tuple(features.columns) == FEATURE_NAMES


# ----------------------------------------------------- code normalisation


@pytest.mark.parametrize("code", schema.UNDOCUMENTED_EDUCATION)
def test_undocumented_education_reaches_the_model_as_other(code: int) -> None:
    """Train/serve skew in its narrowest form.

    EDUCATION is a plain numeric column, so code 5 is not an unseen label --
    it is a larger number on an axis every tree split already uses. The API
    accepts it (ge=0, le=6) and only the cleaning step used to fold it, and
    the cleaning step is the one thing a live request never runs through.
    """
    features = build_features_from_record(_record(**{schema.EDUCATION: code}))
    assert features[schema.EDUCATION].iloc[0] == float(schema.EDUCATION_OTHER)


@pytest.mark.parametrize("code", schema.UNDOCUMENTED_MARRIAGE)
def test_undocumented_marriage_reaches_the_model_as_other(code: int) -> None:
    features = build_features_from_record(_record(**{schema.MARRIAGE: code}))
    assert features[schema.MARRIAGE].iloc[0] == float(schema.MARRIAGE_OTHER)


def test_documented_codes_are_left_alone() -> None:
    normalised = normalize_codes(_frame(_record(**{schema.EDUCATION: 3, schema.MARRIAGE: 2})))
    assert normalised.loc[0, schema.EDUCATION] == 3
    assert normalised.loc[0, schema.MARRIAGE] == 2


def test_normalize_codes_does_not_mutate_its_input() -> None:
    frame = _frame(_record(**{schema.EDUCATION: 5, schema.MARRIAGE: 0}))
    before = frame.copy(deep=True)
    normalize_codes(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_normalize_codes_ignores_a_frame_without_those_columns() -> None:
    """build_features names every missing column at once; this must not pre-empt it."""
    frame = pd.DataFrame({schema.AGE: [30, 40]})
    pd.testing.assert_frame_equal(normalize_codes(frame), frame)


def test_both_paths_agree_on_an_undocumented_code() -> None:
    """The fold has to sit in the shared function, not on one side of it."""
    record = _record(**{schema.EDUCATION: 6, schema.MARRIAGE: 0})
    pd.testing.assert_frame_equal(
        build_features(_frame(record)).reset_index(drop=True),
        build_features_from_record(record).reset_index(drop=True),
        check_exact=True,
    )


# ------------------------------------------------------------------- cli


def test_cli_preflights_a_split_and_reports_its_shape(
    synthetic_clean_df: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "train.parquet"
    synthetic_clean_df.to_parquet(path, index=False)

    assert main([str(path)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "train": {"n_rows": len(synthetic_clean_df), "n_features": len(FEATURE_NAMES)}
    }


def test_cli_fails_on_a_frame_it_cannot_build(
    synthetic_clean_df: pd.DataFrame, tmp_path: Path
) -> None:
    """A zero exit here would let the DAG spend five minutes training on nothing."""
    path = tmp_path / "train.parquet"
    synthetic_clean_df.drop(columns=[schema.LIMIT_BAL]).to_parquet(path, index=False)
    with pytest.raises(FeatureBuildError, match=schema.LIMIT_BAL):
        main([str(path)])
