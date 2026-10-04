"""Behavioural tests that turn the fairness claim into something falsifiable.

A model card that says "we mitigated bias by removing the protected attribute"
is an assertion. These two tests are the difference between asserting it and
knowing it:

DIRECTIONAL -- more months delinquent must not make an account look safer. This
is the one property a risk officer would notice being wrong, and a gradient
booster has no built-in guarantee of it: nothing in the objective forbids a
split that lowers risk as delinquency rises. Left untested it would be a
plausible-sounding claim about a model that had never been asked.

PROTECTED-ATTRIBUTE INVARIANCE -- flipping SEX on an account must not move its
probability. For the mitigated model that holds *exactly*, because the column
never reaches the model at all, so the test asserts the feature frames are
identical rather than pretending a tolerance was measured. It fails the moment
somebody puts a protected column, or something derived from one, back into
FEATURE_NAMES. The baseline is checked too, and it moves by up to several
points, which is what makes the mitigated result worth anything: without that
control the invariance test would pass on a model that ignored every feature.

The last test measures what a group-aware threshold policy actually costs, in
accounts whose decision depends on their sex. That number belongs in ETHICS.md,
not in a footnote.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk import schema
from credit_risk.fairness.mitigation import (
    apply_group_thresholds,
    drop_protected,
    fit_threshold_optimizer,
    group_thresholds,
)
from credit_risk.models.train import feature_frame, make_lightgbm
from tests.model.test_performance import synthetic_split

# The fixture is defined next door so the performance numbers and the
# behavioural ones are measured on identically distributed data.

DELINQUENCY_SWEEP = (-1, 0, 1, 2, 3, 4)
MAX_PROBABILITY_SHIFT_ON_FLIP = 0.02

# Every test in this file fits a booster through the module fixtures, and
# pyproject defines `slow` as exactly that: "tests that need the full docker
# stack or train a model". CI's fast job runs `-m "not slow"`, so leaving these
# unmarked put two LightGBM fits on the critical path of every push.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def data():
    return synthetic_split(n=6000)


@pytest.fixture(scope="module")
def baseline(data):
    """The candidate as trained: every feature, protected attributes included."""
    train_df, _ = data
    return make_lightgbm(n_estimators=150).fit(feature_frame(train_df), train_df[schema.TARGET])


@pytest.fixture(scope="module")
def mitigated(data):
    """The unawareness-mitigated candidate: protected columns deleted."""
    train_df, _ = data
    return make_lightgbm(n_estimators=150).fit(
        drop_protected(feature_frame(train_df)), train_df[schema.TARGET]
    )


def _with_delinquency(frame, months: int):
    """Same accounts, every repayment column set to `months`."""
    perturbed = frame.copy()
    for column in schema.PAY_COLS:
        perturbed[column] = months
    return perturbed


def _flip_sex(frame):
    """1 -> 2 and 2 -> 1, leaving everything else identical."""
    flipped = frame.copy()
    flipped[schema.SEX] = 3 - flipped[schema.SEX]
    return flipped


# ---------------------------------------------------------- directional


def test_more_delinquency_never_lowers_the_average_predicted_risk(baseline, data):
    _, test_df = data

    means = [
        float(
            baseline.predict_proba(feature_frame(_with_delinquency(test_df, months)))[:, 1].mean()
        )
        for months in DELINQUENCY_SWEEP
    ]

    # Strictly increasing across the sweep. Population level, because a tree
    # ensemble is not monotone by construction -- if this ever fails on real
    # data the fix is LightGBM's `monotone_constraints`, not a looser bound.
    assert means == sorted(means)
    assert all(later > earlier for earlier, later in zip(means, means[1:], strict=False))


def test_every_account_looks_riskier_when_it_falls_four_months_behind(baseline, data):
    _, test_df = data
    paid_up = baseline.predict_proba(feature_frame(_with_delinquency(test_df, -1)))[:, 1]
    behind = baseline.predict_proba(feature_frame(_with_delinquency(test_df, 4)))[:, 1]

    # Per-account, not on average: an account that went from "paid in full" to
    # "four months late" and came out safer would be indefensible to explain.
    assert (behind > paid_up).all()
    assert float((behind - paid_up).min()) > 0.01


# ------------------------------------------------- protected invariance


def test_flipping_sex_cannot_move_the_mitigated_probability_at_all(mitigated, data):
    """The mitigated invariance is structural, and the assertion says so.

    An earlier version compared the two probability vectors and asserted a
    0.02 bound. `drop_protected` deletes SEX before the model is ever called,
    so the two feature frames are byte-identical and the shift was exactly
    0.0: the bound was arithmetic on a constant and would have passed against
    a model that ignored every feature. What is worth pinning is the contract
    underneath it -- no protected column, and nothing derived from one,
    survives into the matrix -- so that is what is asserted, exactly.
    """
    _, test_df = data
    original_features = drop_protected(feature_frame(test_df))
    flipped_features = drop_protected(feature_frame(_flip_sex(test_df)))

    assert schema.SEX not in original_features.columns
    # The load-bearing line. It starts failing the day somebody adds a feature
    # computed from SEX under a name `protected_columns` does not match --
    # `is_female`, say -- at which point the frames stop being identical and
    # "the mitigated model cannot see sex" stops being true.
    assert original_features.equals(flipped_features)

    original = mitigated.predict_proba(original_features)[:, 1]
    flipped = mitigated.predict_proba(flipped_features)[:, 1]
    assert float(np.abs(flipped - original).max()) == 0.0


def test_the_unmitigated_baseline_does_move_when_sex_is_flipped(baseline, data):
    _, test_df = data
    original = baseline.predict_proba(feature_frame(test_df))[:, 1]
    flipped = baseline.predict_proba(feature_frame(_flip_sex(test_df)))[:, 1]

    shift = np.abs(flipped - original)
    # The control. Without it the invariance test above could be passing on a
    # model that had learned nothing at all.
    assert float(shift.max()) > MAX_PROBABILITY_SHIFT_ON_FLIP
    assert float((shift > 0).mean()) > 0.5


# ------------------------------------------- the price of group thresholds


def test_group_aware_thresholds_move_only_the_accounts_between_the_cutoffs(mitigated, data):
    train_df, test_df = data
    X_train = drop_protected(feature_frame(train_df))
    X_test = drop_protected(feature_frame(test_df))

    optimizer = fit_threshold_optimizer(
        mitigated, X_train, train_df[schema.TARGET], train_df[schema.SEX]
    )
    thresholds = group_thresholds(optimizer)
    assert set(thresholds) == {"1", "2"}

    proba = mitigated.predict_proba(X_test)[:, 1]
    as_is = apply_group_thresholds(proba, test_df[schema.SEX], thresholds)
    as_if_flipped = apply_group_thresholds(proba, 3 - test_df[schema.SEX], thresholds)

    flipped_decision = as_is != as_if_flipped
    lower, upper = min(thresholds.values()), max(thresholds.values())
    # Half-open the same way `apply_group_thresholds` is: a score equal to the
    # lower cutoff clears it and one equal to the upper cutoff clears that too,
    # so the accounts whose decision depends on their sex are [lower, upper).
    between_cutoffs = (proba >= lower) & (proba < upper)

    # The score is blind to sex, so the only channel left through which sex can
    # change a decision is the threshold -- and it reaches exactly the accounts
    # sitting between the two cutoffs. Naming that set is what lets the ethics
    # discussion be about a measured number of people rather than a principle.
    assert np.array_equal(flipped_decision, between_cutoffs)
    assert float(flipped_decision.mean()) < 0.10
