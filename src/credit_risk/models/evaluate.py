"""Scoring a candidate model the way the business will actually use it.

Two decisions are baked into this module and both are deliberate.

First, PR-AUC is the primary metric, not ROC-AUC. The positive class is 22%
of the portfolio, and ROC-AUC counts every true negative it gets right --
of which there are four times as many. A model can gain several points of
ROC-AUC by better ordering accounts that were never going to default, which
is worth nothing to the risk team. Precision-recall only ever looks at the
positives, so it moves when the model gets better at the job it was hired
for. ROC-AUC is still reported, purely so the numbers can be compared with
the published literature on this dataset.

Second, every business number is defined against the intervention capacity,
not against p > 0.5. The risk team can call about 3,000 people a month. The
question is never "is this account more likely than not to default" -- it is
"is this account in the worst 3,000". `threshold_for_capacity` is where that
sentence becomes a float.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_risk.config import settings

logger = logging.getLogger(__name__)


def _as_arrays(y_true: ArrayLike, y_proba: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays and refuse mismatched lengths early."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_proba).astype(float).ravel()
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true has {y.shape[0]} rows, y_proba has {p.shape[0]}")
    if y.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")
    return y, p


def capacity_count(n: int, capacity_fraction: float) -> int:
    """How many accounts the risk team can act on, as a whole number.

    Rounded, then clamped to at least one: a capacity of zero accounts makes
    precision@k a division by zero, and a pipeline that silently reports 0.0
    is worse than one that reports the single riskiest account.
    """
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError(f"capacity_fraction must be in (0, 1], got {capacity_fraction}")
    return int(min(n, max(1, round(n * capacity_fraction))))


def top_k_mask(y_proba: ArrayLike, capacity_fraction: float) -> np.ndarray:
    """Boolean mask selecting exactly k accounts, ties or no ties.

    Thresholding with `>=` selects more than k when scores tie -- rare with a
    gradient booster, routine with a coarse model, and it silently inflates
    recall@k. Ranking first makes the selected count exact by construction.
    """
    p = np.asarray(y_proba, dtype=float).ravel()
    k = capacity_count(p.size, capacity_fraction)
    # Stable sort so equal scores break by row order and two runs on the
    # same data produce the same intervention list.
    order = np.argsort(-p, kind="stable")
    mask = np.zeros(p.size, dtype=bool)
    mask[order[:k]] = True
    return mask


def threshold_for_capacity(y_proba: ArrayLike, capacity_fraction: float) -> float:
    """The probability cutoff that admits the top `capacity_fraction` of accounts.

    This is the number the serving layer carries: it turns "we can call 3,000
    people" into something a request handler can compare against.
    """
    p = np.asarray(y_proba, dtype=float).ravel()
    k = capacity_count(p.size, capacity_fraction)
    return float(np.sort(p)[-k])


def expected_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    cost_false_negative: float | None = None,
    cost_false_positive: float | None = None,
) -> float:
    """Money lost by this decision rule, in NT$.

    A missed default costs the outstanding balance; an unnecessary call costs
    an agent's time. The two are three orders of magnitude apart, which is the
    whole reason accuracy is a useless metric here.
    """
    y = np.asarray(y_true).astype(int).ravel()
    d = np.asarray(y_pred).astype(int).ravel()
    if y.shape[0] != d.shape[0]:
        raise ValueError(f"y_true has {y.shape[0]} rows, y_pred has {d.shape[0]}")
    c_fn = settings.cost_false_negative if cost_false_negative is None else cost_false_negative
    c_fp = settings.cost_false_positive if cost_false_positive is None else cost_false_positive
    false_negatives = int(np.sum((y == 1) & (d == 0)))
    false_positives = int(np.sum((y == 0) & (d == 1)))
    return float(false_negatives * c_fn + false_positives * c_fp)


def evaluate(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    capacity_fraction: float | None = None,
) -> dict[str, float]:
    """Every number the gate, the model card and the README need, in one dict.

    Flat and float-valued on purpose: this goes straight into
    `mlflow.log_metrics`, which rejects anything else.
    """
    capacity = (
        settings.intervention_capacity_fraction if capacity_fraction is None else capacity_fraction
    )
    y, p = _as_arrays(y_true, y_proba)
    n = y.size
    positives = int(y.sum())
    if positives == 0 or positives == n:
        raise ValueError("y_true must contain both classes to be evaluated")

    selected = top_k_mask(p, capacity)
    k = int(selected.sum())
    hits = int(y[selected].sum())
    # The realised fraction, not the requested one: with n = 997 and a 10%
    # capacity the two differ, and recall/lift must agree with each other.
    selected_fraction = k / n

    recall_at_k = hits / positives
    precision_at_k = hits / k
    base_rate = positives / n

    cost = expected_cost(y, selected.astype(int))
    # Doing nothing is not free: every default goes unflagged. Reporting the
    # difference stops "expected_cost = NT$18M" reading like a failure.
    cost_do_nothing = float(positives * settings.cost_false_negative)

    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "recall_at_k": float(recall_at_k),
        "precision_at_k": float(precision_at_k),
        "lift_at_k": float(recall_at_k / selected_fraction),
        "threshold_at_k": threshold_for_capacity(p, capacity),
        "expected_cost": cost,
        "expected_cost_do_nothing": cost_do_nothing,
        "avoided_cost": cost_do_nothing - cost,
        "capacity_fraction": float(selected_fraction),
        "base_rate": float(base_rate),
        "n_scored": float(n),
        "n_positive": float(positives),
    }


def cost_matrix_report(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    capacity_fractions: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
) -> pd.DataFrame:
    """Confusion counts and money at several capacities -- the README table.

    The point of sweeping capacity is that the optimal one is a budget
    decision, not a modelling one. Somebody in the room has to choose how many
    agents to staff, and this is the table they choose from.

    `contact_cost` is the whole call-centre bill for k accounts; `expected_cost`
    charges only the false alarms. They are different questions and have been
    confused before, so both appear with their own names.
    """
    y, p = _as_arrays(y_true, y_proba)
    positives = int(y.sum())
    if positives == 0:
        # recall@k is tp/positives and there is no honest value for it here.
        # `_as_arrays` only checks length, and `evaluate`'s both-classes guard
        # is on the other function -- so without this, an all-negative scoring
        # window (a monitoring slice, a small batch) reaches the division and
        # comes back as ZeroDivisionError from inside a reporting call.
        raise ValueError("y_true must contain at least one positive to report recall@k")
    rows = []
    for capacity in capacity_fractions:
        selected = top_k_mask(p, capacity)
        k = int(selected.sum())
        tp = int(y[selected].sum())
        fp = k - tp
        fn = positives - tp
        tn = y.size - k - fn
        cost = expected_cost(y, selected.astype(int))
        rows.append(
            {
                "capacity_fraction": round(k / y.size, 4),
                "accounts_contacted": k,
                "threshold": round(threshold_for_capacity(p, capacity), 4),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision_at_k": round(tp / k, 4),
                "recall_at_k": round(tp / positives, 4),
                "contact_cost": float(k * settings.cost_false_positive),
                "missed_default_cost": float(fn * settings.cost_false_negative),
                "expected_cost": cost,
                "avoided_cost": float(positives * settings.cost_false_negative) - cost,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m credit_risk.models.evaluate`.

    The DAG runs this as its own task rather than folding it into training,
    because "the model trained" and "the model is allowed to ship" are two
    different facts and a pipeline that conflates them cannot show you which
    one failed. It reads the hand-off artefact training wrote, re-applies both
    gates, and exits non-zero when the candidate is refused -- so the Airflow
    task turns red for the same reason a human reading the log would.
    """
    # Imported inside the function, unlike everything else in this module:
    # `models.train` imports this module at the top, so a module-level import
    # of it here is a cycle. Nothing above `main` needs either name.
    from credit_risk.models.registry import evaluate_gate, record_refusal
    from credit_risk.models.train import load_training_result, training_result_path

    parser = argparse.ArgumentParser(description="Apply the performance and fairness gates")
    parser.add_argument("--result", type=Path, default=None, help="path to training_result.json")
    parser.add_argument("--out", type=Path, default=None, help="where to write the gate decision")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )

    result = load_training_result(args.result)
    reasons = evaluate_gate(result["metrics"], result["fairness"])
    decision = {
        "run_id": result["run_id"],
        "model": result["model"],
        "passed": not reasons,
        "reasons": reasons,
        "pr_auc": result["metrics"].get("pr_auc"),
        "demographic_parity_difference": result["fairness"].get("demographic_parity_difference"),
        "equalized_odds_difference": result["fairness"].get("equalized_odds_difference"),
    }

    out = args.out or (
        training_result_path(args.result.parent if args.result else None).parent
        / "gate_decision.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))

    if reasons:
        # The verdict first: an unreachable tracking server is retried for
        # minutes, and the log must say why the task failed before that.
        print(f"\nGATE REFUSED: {'; '.join(reasons)}", file=sys.stderr)
        # Then onto the run as well as in the file: this exit fails the DAG, so
        # register_model, the other step that tags refusals, never runs.
        record_refusal(result.get("run_id"), reasons, tracking_uri=result.get("tracking_uri"))
        return 2
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
