"""The single source of feature construction, for training and for serving.

Train/serve skew is the failure this module exists to prevent. It is the most
common way a production ML system goes wrong and the hardest to notice: the
service stays up, latency is flat, every dashboard is green, and the model is
simply answering a different question than the one it was trained on. The
defence here is architectural rather than procedural -- ``models/train.py``
and ``serving/routes.py`` both import :func:`build_features`, and
:func:`build_features_from_record` (the one-row path the API uses) delegates
to it rather than reimplementing it, so the two cannot drift apart. A unit
test asserts the two paths agree column by column.

Three rules hold for the builders below:

* No fitted state, no globals, no I/O. Given the same frame you get the same
  features, in any process, in any order. (:func:`main` is a thin CLI wrapper
  around them and is the only thing here that touches a file.)
* Every division is guarded. A NaN reaching the model is a silent wrong
  answer, not a crash, so the guards are explicit and the output is checked
  for finiteness before it is returned.
* Every feature is named for a human. These strings become SHAP axis labels
  in an explanation shown to a risk officer, and eventually the wording of an
  adverse-action notice.

Code normalisation lives here too, in :func:`normalize_codes`, and that is
deliberate. The undocumented ``EDUCATION`` codes 0/5/6 and ``MARRIAGE`` code 0
used to be folded in the cleaning step alone, which the serving path never
runs -- so a request carrying ``EDUCATION=5`` reached the model as "two steps
past graduate school", extrapolating every tree split and the logistic
coefficient into a probability nobody had ever validated. Folding inside
:func:`build_features` puts the normalisation on both sides of the contract.

Month numbering follows the raw file: ``m1`` is the most recent month
(September 2005) and ``m6`` the oldest (April 2005). Trends are fitted in
chronological order, so a positive trend means "rising over time".

Run it directly to preflight one or more split parquets:

    python -m credit_risk.features.build data/processed/train.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from credit_risk import schema

log = logging.getLogger(__name__)

N_MONTHS: Final = 6

# Columns build_features needs. ID, AGE_GROUP and the target are deliberately
# absent: ID is an identifier, AGE_GROUP is a fairness slice, and the target
# is what we are predicting.
REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    schema.LIMIT_BAL,
    schema.SEX,
    schema.EDUCATION,
    schema.MARRIAGE,
    schema.AGE,
    *schema.PAY_COLS,
    *schema.BILL_COLS,
    *schema.PAY_AMT_COLS,
)

# A customer who paid double last month's bill is already as good as this
# feature can express. Without the cap, a 30 NT$ bill settled with a 30,000
# NT$ payment produces a ratio of 1000 that dominates every tree split and
# wrecks the scaling of the logistic baseline.
PAYMENT_RATIO_CAP: Final = 2.0

# PAY_* semantics: -2 no consumption, -1 paid in full, 0 revolving credit,
# 1..8 months overdue. "Delinquent" therefore starts at 1, not at 0.
DELINQUENT_FROM: Final = 1

# Sentinel for "no delinquency in the six-month window". N_MONTHS + 1 keeps
# the feature monotone -- larger means longer since trouble -- where a 0 or a
# NaN would read as "trouble right now".
NO_DELINQUENCY_SENTINEL: Final = N_MONTHS + 1

_UTILIZATION_MONTHLY: Final[tuple[str, ...]] = tuple(f"utilization_m{i}" for i in range(1, 7))
_PAYMENT_RATIO_MONTHLY: Final[tuple[str, ...]] = tuple(f"payment_ratio_m{i}" for i in range(1, 6))

#: The model input contract: exact names, exact order. Anything that consumes
#: a model -- training, serving, SHAP, LIME, the fairness report -- reads this.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    # Raw signal, kept as-is. SEX stays in by default; dropping it is the
    # "fairness through unawareness" experiment in fairness/mitigation.py,
    # and it is an experiment precisely because it does not work.
    schema.LIMIT_BAL,
    schema.AGE,
    schema.SEX,
    schema.EDUCATION,
    schema.MARRIAGE,
    *schema.PAY_COLS,
    # Credit utilisation.
    *_UTILIZATION_MONTHLY,
    "utilization_mean",
    "utilization_max",
    "utilization_trend",
    "available_credit",
    # Repayment behaviour.
    *_PAYMENT_RATIO_MONTHLY,
    "payment_ratio_mean",
    "payment_ratio_min",
    "payment_ratio_trend",
    # Delinquency shape.
    "months_delinquent",
    "max_consecutive_delinquent",
    "months_since_last_delinquency",
    "worst_pay_status",
    # Volume.
    "total_bill_amt",
    "mean_bill_amt",
    "total_pay_amt",
    "mean_pay_amt",
    "payment_to_limit_ratio",
)


# EDUCATION and MARRIAGE enter the matrix as plain numeric columns, so an
# out-of-dictionary code is not a label the model has never seen -- it is a
# larger number on an axis the model already splits on. Folding them is the
# difference between "other" and "two grades beyond graduate school".
_CODE_FOLDS: Final[tuple[tuple[str, tuple[int, ...], int], ...]] = (
    (schema.EDUCATION, schema.UNDOCUMENTED_EDUCATION, schema.EDUCATION_OTHER),
    (schema.MARRIAGE, schema.UNDOCUMENTED_MARRIAGE, schema.MARRIAGE_OTHER),
)


class FeatureBuildError(ValueError):
    """The input could not be turned into a valid feature matrix."""


def normalize_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Fold the undocumented categorical codes into the documented "other".

    ``EDUCATION`` 0/5/6 become 4 and ``MARRIAGE`` 0 becomes 3, which is what
    the published dictionary means by "others" -- 345 and 54 rows of the real
    file respectively, and any request the API is willing to accept.

    Returns a copy. :func:`build_features` calls it so the serving path cannot
    skip it, and ``data.split.clean`` calls this same function rather than
    keeping a second copy of the mapping, because two copies of a mapping is
    how the training data and the live request stop meaning the same thing.

    Columns that are absent are left alone; reporting them is
    :func:`build_features`' job and it names all of them at once.
    """
    out = df.copy()
    for column, undocumented, other in _CODE_FOLDS:
        if column not in out.columns:
            continue
        # np.where rather than Series.replace: replace() on an integer column
        # emits a downcasting FutureWarning in pandas 2.2 and its result dtype
        # depends on what was replaced, which is not a thing to leave to
        # chance when a split boundary is riding on it.
        original = out[column].dtype
        folded = np.where(out[column].isin(undocumented), other, out[column])
        out[column] = folded.astype(original)
    return out


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, fallback: float) -> np.ndarray:
    """Element-wise division that never yields NaN or inf.

    The denominator is replaced by 1.0 wherever it is non-positive *before*
    dividing, so numpy never evaluates the invalid operation at all; skipping
    that and masking afterwards leaves inf/NaN in the array and prints a
    RuntimeWarning per batch.
    """
    usable = denominator > 0
    safe = np.where(usable, denominator, 1.0)
    return np.where(usable, numerator / safe, fallback)


def _slope(chronological: np.ndarray) -> np.ndarray:
    """Least-squares slope per row against evenly spaced months.

    Closed form rather than polyfit: the x values are fixed, so the whole
    thing is one dot product and stays vectorised over 20,000 rows.
    """
    deviation = np.arange(chronological.shape[1], dtype="float64")
    deviation -= deviation.mean()
    return (chronological * deviation).sum(axis=1) / float((deviation**2).sum())


def _max_consecutive(flags: np.ndarray) -> np.ndarray:
    """Longest run of True per row. Six iterations over columns, not rows."""
    run = np.zeros(flags.shape[0], dtype="float64")
    best = np.zeros(flags.shape[0], dtype="float64")
    for column in range(flags.shape[1]):
        run = np.where(flags[:, column], run + 1.0, 0.0)
        best = np.maximum(best, run)
    return best


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned account rows into the model input matrix.

    Pure and vectorised: no fitted state, nothing cached, nothing mutated.
    The returned frame has exactly :data:`FEATURE_NAMES` as its columns, in
    that order, all float64, and shares the caller's index so predictions can
    be joined back to accounts.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise FeatureBuildError(f"input is missing required columns: {missing}")

    # First, not last: everything below reads EDUCATION and MARRIAGE as
    # numbers, so the fold has to happen before any of it. Doing this here
    # rather than only in the cleaning step is what keeps a live request with
    # EDUCATION=5 from being scored on an axis position the training data
    # never contained.
    df = normalize_codes(df)

    limit = df[schema.LIMIT_BAL].to_numpy(dtype="float64")
    bills = df[list(schema.BILL_COLS)].to_numpy(dtype="float64")
    payments = df[list(schema.PAY_AMT_COLS)].to_numpy(dtype="float64")
    status = df[list(schema.PAY_COLS)].to_numpy(dtype="float64")

    # --- utilisation ------------------------------------------------------
    # A zero or negative limit is not in the cleaned data, but the API accepts
    # whatever a caller sends; 0.0 ("no credit in use") is the honest answer
    # for an account with no credit line.
    utilization = _safe_ratio(bills, np.repeat(limit[:, None], N_MONTHS, axis=1), 0.0)
    utilization_chronological = utilization[:, ::-1]

    # --- repayment --------------------------------------------------------
    # PAY_AMT_i settles BILL_AMT_{i+1}: the payment made in September clears
    # the August statement. Pairing PAY_AMT_i with BILL_AMT_i instead would
    # look reasonable and be off by one month for every account.
    paid = payments[:, : N_MONTHS - 1]
    previously_owed = bills[:, 1:]
    # Nothing owed means nothing to miss, so the ratio is 1.0 rather than 0.0,
    # which would otherwise read as "paid none of the bill".
    payment_ratio = np.clip(_safe_ratio(paid, previously_owed, 1.0), 0.0, PAYMENT_RATIO_CAP)

    # --- delinquency ------------------------------------------------------
    delinquent = status >= DELINQUENT_FROM
    any_delinquent = delinquent.any(axis=1)
    # argmax on a boolean row gives the first True, i.e. the most recent
    # delinquent month, because column 0 is the most recent month.
    months_since = np.where(
        any_delinquent,
        np.argmax(delinquent, axis=1) + 1,
        NO_DELINQUENCY_SENTINEL,
    ).astype("float64")

    total_bill = bills.sum(axis=1)
    total_pay = payments.sum(axis=1)

    built: dict[str, np.ndarray] = {
        schema.LIMIT_BAL: limit,
        schema.AGE: df[schema.AGE].to_numpy(dtype="float64"),
        schema.SEX: df[schema.SEX].to_numpy(dtype="float64"),
        schema.EDUCATION: df[schema.EDUCATION].to_numpy(dtype="float64"),
        schema.MARRIAGE: df[schema.MARRIAGE].to_numpy(dtype="float64"),
        "utilization_mean": utilization.mean(axis=1),
        "utilization_max": utilization.max(axis=1),
        "utilization_trend": _slope(utilization_chronological),
        "available_credit": limit - bills[:, 0],
        "payment_ratio_mean": payment_ratio.mean(axis=1),
        "payment_ratio_min": payment_ratio.min(axis=1),
        "payment_ratio_trend": _slope(payment_ratio[:, ::-1]),
        "months_delinquent": delinquent.sum(axis=1).astype("float64"),
        "max_consecutive_delinquent": _max_consecutive(delinquent[:, ::-1]),
        "months_since_last_delinquency": months_since,
        "worst_pay_status": status.max(axis=1),
        "total_bill_amt": total_bill,
        "mean_bill_amt": total_bill / N_MONTHS,
        "total_pay_amt": total_pay,
        "mean_pay_amt": total_pay / N_MONTHS,
        "payment_to_limit_ratio": _safe_ratio(total_pay, limit, 0.0),
    }
    for index, name in enumerate(schema.PAY_COLS):
        built[name] = status[:, index]
    for index, name in enumerate(_UTILIZATION_MONTHLY):
        built[name] = utilization[:, index]
    for index, name in enumerate(_PAYMENT_RATIO_MONTHLY):
        built[name] = payment_ratio[:, index]

    if set(built) != set(FEATURE_NAMES):
        # The published contract and the code that fills it drifting apart is
        # exactly the class of bug this module exists to make impossible.
        raise FeatureBuildError(
            f"FEATURE_NAMES and the built columns disagree: "
            f"only in contract={sorted(set(FEATURE_NAMES) - set(built))}, "
            f"only in build={sorted(set(built) - set(FEATURE_NAMES))}"
        )

    features = pd.DataFrame(built, index=df.index, columns=list(FEATURE_NAMES)).astype("float64")

    # Checked, not assumed. A non-finite value here would reach predict_proba
    # and come back as a confident-looking number with no meaning behind it.
    values = features.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        broken = [
            name
            for position, name in enumerate(FEATURE_NAMES)
            if not np.isfinite(values[:, position]).all()
        ]
        raise FeatureBuildError(f"non-finite values produced in: {broken}")

    return features


def build_features_from_record(record: Mapping[str, Any]) -> pd.DataFrame:
    """Build the one-row feature matrix for a single API request.

    Delegates to :func:`build_features` rather than computing anything of its
    own. That is the whole point: the serving path has no arithmetic in it, so
    it cannot disagree with the training path.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in record]
    if missing:
        raise FeatureBuildError(f"record is missing required fields: {missing}")

    # Column-projected, not passed through: an extra key such as ID would
    # otherwise widen the frame and change nothing visibly until something
    # downstream indexes by position.
    row = {column: record[column] for column in REQUIRED_INPUT_COLUMNS}
    return build_features(pd.DataFrame([row]))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m credit_risk.features.build <parquet>...``.

    A preflight check, not a materialisation step, and the distinction is the
    point. ``models/train.py`` builds its own matrix from the split it loads --
    that is exactly what makes this module the single source -- so writing a
    feature parquet here would leave a file nothing reads and a second copy of
    the matrix to keep in step with the first. What the DAG buys instead is
    failing in seconds when the feature contract is broken, rather than after
    a five-minute hyperparameter search that was doomed at its first row.
    """
    parser = argparse.ArgumentParser(description="Preflight the feature matrix for split parquets")
    parser.add_argument("parquet", nargs="+", type=Path, help="cleaned split parquet(s)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    summary: dict[str, dict[str, int]] = {}
    for path in args.parquet:
        features = build_features(pd.read_parquet(path))
        summary[path.stem] = {"n_rows": int(len(features)), "n_features": len(features.columns)}
        log.info("%s: %d rows x %d features", path, len(features), features.shape[1])

    # stdout is the contract for subprocess callers; logs go to stderr.
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
