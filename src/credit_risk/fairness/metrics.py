"""Measuring whether the model treats groups differently.

Everything here operates on *decisions*, not scores, because a probability
harms nobody -- a phone call, a lowered limit or a restructuring offer does.
The decision rule in production is "top k by score", so that is the rule these
metrics are fed.

The gate thresholds live in `schema`, not here. This module reports; policy is
a constant somebody has to change on purpose, in a reviewed commit.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, precision_score

from credit_risk import schema

GROUP_METRIC_COLUMNS: tuple[str, ...] = (
    "n",
    "selection_rate",
    "tpr",
    "fpr",
    "precision",
    "accuracy",
)


def _aligned(
    y_true: ArrayLike, y_pred: ArrayLike, sensitive: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    d = np.asarray(y_pred).astype(int).ravel()
    s = np.asarray(sensitive).ravel()
    if not (y.shape[0] == d.shape[0] == s.shape[0]):
        raise ValueError(
            f"length mismatch: y_true={y.shape[0]} y_pred={d.shape[0]} sensitive={s.shape[0]}"
        )
    if y.size == 0:
        raise ValueError("cannot compute fairness metrics on an empty set")
    return y, d, s


def group_report(y_true: ArrayLike, y_pred: ArrayLike, sensitive: ArrayLike) -> pd.DataFrame:
    """Per-group rates, indexed by group value.

    `n` is in the table because a 0.30 selection rate over 41 accounts and the
    same rate over 18,000 are not the same finding, and a parity difference
    computed on a handful of rows is noise wearing a suit.
    """
    y, d, s = _aligned(y_true, y_pred, sensitive)
    frame = MetricFrame(
        metrics={
            "n": count,
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            # zero_division=0: a group nobody was selected from has undefined
            # precision, and sklearn's default warning would fire once per
            # fold during cross-validation.
            "precision": partial(precision_score, zero_division=0),
            "accuracy": accuracy_score,
        },
        y_true=y,
        y_pred=d,
        sensitive_features=s,
    )
    report = frame.by_group.copy()
    report.index.name = "group"
    report["n"] = report["n"].astype(int)
    return report[list(GROUP_METRIC_COLUMNS)]


def fairness_summary(
    y_true: ArrayLike, y_pred: ArrayLike, sensitive: ArrayLike
) -> dict[str, float]:
    """The scalar summary the gate reads and MLflow stores.

    `selection_rate_gap` duplicates fairlearn's between-groups demographic
    parity difference by design. If the two ever disagree, the library upgrade
    that changed the definition under us is caught by a unit test rather than
    by a compliance officer.

    `n_smallest_group` travels with the gaps because an equalized-odds
    difference is a max over per-group TPR and FPR differences, and on a group
    of a few dozen accounts one extra true positive moves it by several points.
    A gap read without the size of the group behind it cannot be weighed.
    """
    y, d, s = _aligned(y_true, y_pred, sensitive)
    report = group_report(y, d, s)
    rates = report["selection_rate"].to_numpy(dtype=float)
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(y, d, sensitive_features=s)
        ),
        "demographic_parity_ratio": float(demographic_parity_ratio(y, d, sensitive_features=s)),
        "equalized_odds_difference": float(equalized_odds_difference(y, d, sensitive_features=s)),
        "selection_rate_gap": float(rates.max() - rates.min()),
        "selection_rate_min": float(rates.min()),
        "selection_rate_max": float(rates.max()),
        "n_groups": float(rates.size),
        "n_smallest_group": float(report["n"].min()),
    }


def passes_gates(summary: dict[str, float]) -> tuple[bool, list[str]]:
    """Policy as code: the two thresholds that stop a model being registered.

    This is not a report. A candidate that breaches either limit is not
    promoted, no matter how good its PR-AUC is, and the reasons come back as
    strings so the refusal can be written to the run and read by a human later.
    """
    reasons: list[str] = []
    dp = summary.get("demographic_parity_difference")
    eo = summary.get("equalized_odds_difference")

    if dp is None or not np.isfinite(dp):
        reasons.append("demographic_parity_difference is missing or not finite")
    elif dp > schema.MAX_DEMOGRAPHIC_PARITY_DIFF:
        reasons.append(
            f"demographic_parity_difference {dp:.4f} exceeds "
            f"{schema.MAX_DEMOGRAPHIC_PARITY_DIFF:.2f}"
        )

    if eo is None or not np.isfinite(eo):
        reasons.append("equalized_odds_difference is missing or not finite")
    elif eo > schema.MAX_EQUALIZED_ODDS_DIFF:
        reasons.append(
            f"equalized_odds_difference {eo:.4f} exceeds {schema.MAX_EQUALIZED_ODDS_DIFF:.2f}"
        )

    return (not reasons), reasons


def calibration_by_group(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    sensitive: ArrayLike,
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Predicted probability against observed default rate, per group and bin.

    Parity and calibration are different properties and a model can hold one
    while breaking the other: equalising selection rates by shifting a group's
    threshold leaves its probabilities exactly as wrong as they were. Expected
    loss is computed from those probabilities, so a group whose 0.6 really
    means 0.4 is being priced incorrectly even when the fairness dashboard is
    green.
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()
    s = np.asarray(sensitive).ravel()
    if not (y.shape[0] == p.shape[0] == s.shape[0]):
        raise ValueError(
            f"length mismatch: y_true={y.shape[0]} y_proba={p.shape[0]} sensitive={s.shape[0]}"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=True` with a leading clip so a probability of exactly 0.0 lands in
    # the first bin instead of its own phantom bin zero.
    bin_index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for group in pd.unique(pd.Series(s)):
        in_group = s == group
        for b in range(n_bins):
            cell = in_group & (bin_index == b)
            n_cell = int(cell.sum())
            if n_cell == 0:
                continue
            mean_predicted = float(p[cell].mean())
            observed = float(y[cell].mean())
            rows.append(
                {
                    "group": group,
                    "bin": b,
                    "bin_lower": float(edges[b]),
                    "bin_upper": float(edges[b + 1]),
                    "n": n_cell,
                    "mean_predicted": mean_predicted,
                    "observed_rate": observed,
                    "gap": mean_predicted - observed,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "group",
            "bin",
            "bin_lower",
            "bin_upper",
            "n",
            "mean_predicted",
            "observed_rate",
            "gap",
        ],
    )


def calibration_error_by_group(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    sensitive: ArrayLike,
    *,
    n_bins: int = 5,
) -> dict[str, float]:
    """Expected calibration error per group, keyed by `str(group)`.

    Keys are strings because these end up as MLflow metric names, and MLflow
    will happily accept `1` and `'1'` as two different metrics.
    """
    table = calibration_by_group(y_true, y_proba, sensitive, n_bins=n_bins)
    if table.empty:
        return {}
    out: dict[str, float] = {}
    for group, rows in table.groupby("group", sort=True):
        weights = rows["n"].to_numpy(dtype=float)
        out[str(group)] = float(
            np.average(np.abs(rows["gap"].to_numpy(dtype=float)), weights=weights)
        )
    return out


def flatten_summary(summary: dict[str, float], prefix: str = "fair") -> dict[str, float]:
    """Prefix keys so several protected attributes can share one MLflow run."""
    return {f"{prefix}_{key}": float(value) for key, value in summary.items()}


# Pairs reported jointly. A marginal on SEX and a marginal on AGE_GROUP say
# nothing about young women; the joint does. Reported, not gated: on the
# 5,000-row test split each of the four cells holds roughly 1,000-1,600
# accounts, enough to read a rate and too few to hold a model back on.
INTERSECTIONS: tuple[tuple[str, str], ...] = ((schema.SEX, schema.AGE_GROUP),)


def intersection_key(left: str, right: str) -> str:
    """The name a joint attribute is reported under, e.g. `SEX_x_AGE_GROUP`."""
    return f"{left}_x_{right}"


def summaries_for_attributes(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    frame: pd.DataFrame,
    *,
    attributes: Sequence[str] = schema.PROTECTED_ATTRIBUTES,
    intersections: Sequence[tuple[str, str]] = INTERSECTIONS,
) -> dict[str, dict[str, float]]:
    """One summary per protected attribute present in `frame`, plus the joints.

    SEX is the gate; EDUCATION, MARRIAGE and AGE_GROUP are reported because a
    model that passes on the attribute you chose to police and fails on the
    other three has not been made fair, only audited narrowly. Each pair in
    `intersections` is summarised as one attribute whose groups are the
    combinations (`1_young`, `2_older`, ...), because two marginals that both
    look fine can hide a cell that does not.
    """
    out: dict[str, dict[str, float]] = {}
    for attribute in attributes:
        if attribute not in frame.columns:
            continue
        out[attribute] = fairness_summary(y_true, y_pred, frame[attribute])
    for left, right in intersections:
        if left not in frame.columns or right not in frame.columns:
            continue
        joint = frame[left].astype(str) + "_" + frame[right].astype(str)
        out[intersection_key(left, right)] = fairness_summary(y_true, y_pred, joint)
    return out
