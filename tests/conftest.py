"""Shared fixtures: a synthetic stand-in for the UCI file.

Tests that need data should not need the network. The frames built here have
the real schema, the real value ranges and -- deliberately -- the real
defects: the undocumented EDUCATION codes 0/5/6, the undocumented MARRIAGE
code 0, boundary ages, an account late every month and an account that never
used its card. A fixture that only contains well-behaved rows tests the happy
path and nothing else.

The repayment history is a serially correlated process driven by one latent
risk trait, and that is the load-bearing detail. An earlier version drew all
six PAY_* columns as independent uniforms, which left
``max_consecutive_delinquent``, ``months_since_last_delinquency`` and both
trend features as pure noise -- so the fixture was unusable for anyone testing
a model, and every other track wrote its own. Balances and payments hang off
the same trait, so utilisation and payment-ratio trends mean something too.

It is a fixture, not a claim about the world: the class balance is pinned to
the measured 22.12% and SEX=1 is riskier than SEX=2 in the same direction as
the real file, but no number produced from it belongs in the model card.

Everything here depends on pandas, numpy and this project only. Other tracks
import these fixtures, and a fixture that drags in LightGBM would make their
unit tests slow and their CI flaky.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.data.split import clean

# Enough rows to place every forced edge case without them overlapping.
MIN_SYNTHETIC_ROWS = 6

RAW_PAY_STATUS_COLS = (schema.PAY_0_RAW, *schema.PAY_COLS[1:])

# How much of last month's state carries into this one. High enough that a
# delinquent account tends to stay delinquent, which is what makes a run of
# late months -- the thing max_consecutive_delinquent counts -- occur at all.
_PERSISTENCE = 0.75


def make_synthetic_raw(n_rows: int = 200, seed: int = 20260930) -> pd.DataFrame:
    """Build a frame shaped exactly like the raw UCI export.

    Deterministic for a given seed, so a failure is reproducible from the
    test name alone rather than "it goes red about one run in twenty".
    """
    if n_rows < MIN_SYNTHETIC_ROWS:
        raise ValueError(f"need at least {MIN_SYNTHETIC_ROWS} rows to place the edge cases")

    rng = np.random.default_rng(seed)

    sex = rng.integers(1, 3, size=n_rows)
    education = rng.integers(1, 5, size=n_rows)
    marriage = rng.integers(1, 4, size=n_rows)
    age = rng.integers(schema.AGE_MIN, schema.AGE_MAX + 1, size=n_rows)
    # Always positive: LIMIT_BAL <= 0 is an error the validators must catch,
    # so the baseline frame must not contain it by accident.
    limit = rng.integers(1, 81, size=n_rows) * 10_000

    # The latent trait every other column hangs off. It is also why dropping
    # SEX cannot hide SEX: the trait leaks into the repayment history.
    propensity = rng.normal(0.0, 1.0, n_rows) + 0.20 * (sex == 1) + 0.15 * (education == 3)

    status = np.zeros((n_rows, len(schema.PAY_COLS)), dtype="int64")
    carry = propensity.copy()
    # Column 0 is the most recent month, so walk the columns backwards: the
    # oldest month is generated first and each later one inherits most of it.
    for month in range(len(schema.PAY_COLS) - 1, -1, -1):
        carry = _PERSISTENCE * carry + (1.0 - _PERSISTENCE) * rng.normal(0.0, 1.0, n_rows)
        status[:, month] = np.clip(
            np.round(1.6 * carry - 0.4), schema.PAY_MIN, schema.PAY_MAX
        ).astype("int64")

    # Utilisation persists month to month as well. The lower clip sits just
    # below zero on purpose: negative bills are legitimate (the customer
    # overpaid) and the validators have to keep allowing them.
    utilization = np.clip(
        0.35 + 0.20 * propensity[:, None] + rng.normal(0.0, 0.12, (n_rows, len(schema.BILL_COLS))),
        -0.05,
        1.2,
    )
    bills = (utilization * limit[:, None]).round().astype("int64")
    # A late month is a month where little of the bill was paid -- that is the
    # relationship payment_ratio_* exists to measure, and it has to be in the
    # data for a test of it to mean anything.
    paid_share = np.clip(
        0.9 - 0.30 * np.maximum(status, 0) + rng.normal(0.0, 0.15, status.shape), 0.0, 1.2
    )
    payments = np.maximum(bills * paid_share, 0.0).round().astype("int64")

    # Pinned by quantile rather than sampled, so a 30-row fixture has the same
    # class balance as a 30,000-row one and a test does not go red on n alone.
    risk = propensity + 0.30 * status[:, 0] + 0.60 * utilization.mean(axis=1)
    target = (risk > np.quantile(risk, 1.0 - schema.BASE_POSITIVE_RATE)).astype("int64")

    data: dict[str, np.ndarray] = {
        schema.ID_COL: np.arange(1, n_rows + 1, dtype="int64"),
        schema.LIMIT_BAL: limit,
        schema.SEX: sex,
        schema.EDUCATION: education,
        schema.MARRIAGE: marriage,
        schema.AGE: age,
    }
    for index, column in enumerate(RAW_PAY_STATUS_COLS):
        data[column] = status[:, index]
    for index, column in enumerate(schema.BILL_COLS):
        data[column] = bills[:, index]
    for index, column in enumerate(schema.PAY_AMT_COLS):
        data[column] = payments[:, index]
    data[schema.TARGET_RAW] = target

    frame = pd.DataFrame(data, columns=list(schema.RAW_COLUMNS)).astype("int64")

    # Defects measured in the published file, reproduced so the cleaning and
    # warning paths are exercised without downloading 5.5 MB.
    frame.loc[0, schema.EDUCATION] = 0
    frame.loc[1, schema.EDUCATION] = 5
    frame.loc[2, schema.EDUCATION] = 6
    frame.loc[3, schema.MARRIAGE] = 0

    # Boundary values. AGE_MIN and AGE_MAX also guarantee both AGE_GROUP
    # values exist, which several tests rely on.
    frame.loc[0, schema.AGE] = schema.AGE_MIN
    frame.loc[1, schema.AGE] = schema.AGE_MAX
    frame.loc[4, list(RAW_PAY_STATUS_COLS)] = schema.PAY_MAX
    frame.loc[5, list(RAW_PAY_STATUS_COLS)] = schema.PAY_MIN
    # An account with no activity at all: every guarded division in the
    # feature builder has to survive this row.
    frame.loc[5, list(schema.BILL_COLS)] = 0
    frame.loc[5, list(schema.PAY_AMT_COLS)] = 0

    return frame


@pytest.fixture
def make_raw_frame() -> Callable[..., pd.DataFrame]:
    """The raw-frame builder itself, for tests that need a different size."""
    return make_synthetic_raw


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """200 rows shaped like the raw UCI export, defects included."""
    return make_synthetic_raw()


@pytest.fixture
def synthetic_clean_df(synthetic_raw_df: pd.DataFrame) -> pd.DataFrame:
    """The same rows after the real cleaning step, not a hand-written copy of it."""
    return clean(synthetic_raw_df)
