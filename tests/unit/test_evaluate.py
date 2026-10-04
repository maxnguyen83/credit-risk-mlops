"""Arithmetic checks on the business metrics.

Every number in here was worked out by hand first. That is the point: these
functions decide how many people get a phone call and how much money the bank
is told it saved, so "the library returned something plausible" is not enough.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.config import settings
from credit_risk.models.evaluate import (
    capacity_count,
    cost_matrix_report,
    evaluate,
    expected_cost,
    threshold_for_capacity,
    top_k_mask,
)


def test_capacity_count_rounds_and_never_returns_zero():
    assert capacity_count(1000, 0.10) == 100
    assert capacity_count(997, 0.10) == 100  # 99.7 rounds to 100
    # Five accounts at 10% capacity is half an account. One is the only
    # answer that keeps precision@k defined.
    assert capacity_count(5, 0.10) == 1
    assert capacity_count(10, 1.0) == 10


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_capacity_count_rejects_impossible_fractions(bad):
    with pytest.raises(ValueError, match="capacity_fraction"):
        capacity_count(100, bad)


def test_recall_at_k_matches_the_hand_computation():
    # 10 accounts, 4 of them default, scores already in descending order.
    # At 30% capacity the team calls 3 people, and the top 3 are all
    # defaulters: recall = 3/4, precision = 3/3, lift = 0.75 / 0.3 = 2.5.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.99, 0.95, 0.90, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10])

    metrics = evaluate(y_true, y_proba, capacity_fraction=0.30)

    assert metrics["recall_at_k"] == pytest.approx(0.75)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["lift_at_k"] == pytest.approx(2.5)
    assert metrics["base_rate"] == pytest.approx(0.4)


def test_threshold_for_capacity_selects_exactly_k_accounts():
    y_proba = np.linspace(0.01, 0.99, 100)
    threshold = threshold_for_capacity(y_proba, 0.10)

    assert int((y_proba >= threshold).sum()) == 10
    # The cutoff is a real score from the population, not an interpolation.
    assert threshold in set(y_proba.tolist())


def test_top_k_mask_selects_exactly_k_even_when_every_score_ties():
    # A degenerate model that scores everyone identically still has to produce
    # a list of exactly k names; thresholding with >= would return all 50.
    y_proba = np.full(50, 0.22)
    mask = top_k_mask(y_proba, 0.10)
    assert int(mask.sum()) == 5


def test_perfect_ranking_gives_lift_of_one_over_capacity():
    y_true = np.zeros(100, dtype=int)
    y_true[:10] = 1
    y_proba = np.linspace(1.0, 0.0, 100)  # perfectly ordered

    metrics = evaluate(y_true, y_proba, capacity_fraction=0.10)

    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["lift_at_k"] == pytest.approx(1 / 0.10)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_expected_cost_is_exact_on_a_tiny_case():
    # y:  1 1 0 0   ->  two defaults, two good accounts
    # d:  0 1 1 0   ->  one missed default, one wasted call
    y_true = [1, 1, 0, 0]
    y_pred = [0, 1, 1, 0]

    assert expected_cost(
        y_true, y_pred, cost_false_negative=10.0, cost_false_positive=1.0
    ) == pytest.approx(11.0)
    # And with the configured NT$ costs: 30,000 + 500.
    assert expected_cost(y_true, y_pred) == pytest.approx(
        settings.cost_false_negative + settings.cost_false_positive
    )


def test_expected_cost_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="rows"):
        expected_cost([1, 0, 1], [1, 0])


def test_avoided_cost_is_the_gap_against_doing_nothing():
    y_true = np.zeros(100, dtype=int)
    y_true[:10] = 1
    y_proba = np.linspace(1.0, 0.0, 100)

    metrics = evaluate(y_true, y_proba, capacity_fraction=0.10)

    # A perfect model catches all ten defaults and wastes no calls.
    assert metrics["expected_cost"] == pytest.approx(0.0)
    assert metrics["expected_cost_do_nothing"] == pytest.approx(10 * settings.cost_false_negative)
    assert metrics["avoided_cost"] == pytest.approx(metrics["expected_cost_do_nothing"])


def test_evaluate_refuses_input_it_cannot_score():
    with pytest.raises(ValueError, match="both classes"):
        evaluate(np.zeros(20, dtype=int), np.linspace(0, 1, 20))
    with pytest.raises(ValueError, match="rows"):
        evaluate([1, 0, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [])


def test_evaluate_defaults_to_the_configured_capacity():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.22, 1000)
    y_proba = rng.uniform(size=1000)

    metrics = evaluate(y_true, y_proba)

    assert metrics["capacity_fraction"] == pytest.approx(
        settings.intervention_capacity_fraction, abs=0.001
    )
    assert 0.0 <= metrics["brier"] <= 1.0


def test_cost_matrix_report_is_internally_consistent():
    rng = np.random.default_rng(7)
    y_true = rng.binomial(1, 0.22, 500)
    y_proba = np.clip(0.2 + 0.4 * y_true + rng.normal(scale=0.2, size=500), 0.01, 0.99)

    table = cost_matrix_report(y_true, y_proba, capacity_fractions=(0.05, 0.10, 0.20))

    assert list(table["accounts_contacted"]) == [25, 50, 100]
    for _, row in table.iterrows():
        counts = row[["true_positive", "false_positive", "false_negative", "true_negative"]]
        assert counts.sum() == 500
        assert row["contact_cost"] == pytest.approx(
            row["accounts_contacted"] * settings.cost_false_positive
        )
        assert row["expected_cost"] == pytest.approx(
            row["false_negative"] * settings.cost_false_negative
            + row["false_positive"] * settings.cost_false_positive
        )
    # More calls can only ever find more defaults.
    assert list(table["recall_at_k"]) == sorted(table["recall_at_k"])


def test_cost_matrix_report_refuses_a_window_with_no_defaults_in_it():
    # recall@k is tp/positives. An all-negative scoring window is an ordinary
    # thing for a monitoring slice to contain, and before the guard it reached
    # the division and came back as ZeroDivisionError from inside a reporting
    # call -- an exception nobody upstream would think to catch.
    with pytest.raises(ValueError, match="at least one positive"):
        cost_matrix_report(np.zeros(50, dtype=int), np.linspace(0, 1, 50))

    # One positive is enough; the guard is about zero, not about small.
    y_true = np.zeros(50, dtype=int)
    y_true[0] = 1
    assert cost_matrix_report(y_true, np.linspace(1, 0, 50))["recall_at_k"].iloc[0] == 1.0
