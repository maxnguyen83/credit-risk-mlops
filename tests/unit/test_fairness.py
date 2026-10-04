"""Checks on the fairness measurements and the three mitigations.

The frames below are hand-built so the expected answers are known before
fairlearn is called. A fairness number nobody can derive by hand is a number
nobody can defend in a review, and these are the numbers that decide whether a
model is allowed to serve traffic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.fairness.metrics import (
    calibration_by_group,
    calibration_error_by_group,
    fairness_summary,
    flatten_summary,
    group_report,
    intersection_key,
    passes_gates,
    summaries_for_attributes,
)
from credit_risk.fairness.mitigation import (
    STRATEGIES,
    apply_group_thresholds,
    drop_protected,
    fit_threshold_optimizer,
    group_thresholds,
    group_thresholds_path,
    load_group_thresholds,
    protected_columns,
    reweigh,
    save_group_thresholds,
    tradeoff_curve,
    unawareness_probe,
)

# Two groups of ten. Group 1 is selected six times, group 2 four times, so the
# selection-rate gap is exactly 0.20 before any library gets involved.
# Group 1: TPR 4/4, FPR 2/6.  Group 2: TPR 4/4, FPR 0/6.  eo_diff = 1/3.
_Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 2)
_Y_PRED = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
_SENSITIVE = np.array([1] * 10 + [2] * 10)


def test_group_report_reproduces_the_rates_by_hand():
    report = group_report(_Y_TRUE, _Y_PRED, _SENSITIVE)

    assert list(report.columns) == [
        "n",
        "selection_rate",
        "tpr",
        "fpr",
        "precision",
        "accuracy",
    ]
    assert report.loc[1, "n"] == 10
    assert report.loc[1, "selection_rate"] == pytest.approx(0.6)
    assert report.loc[2, "selection_rate"] == pytest.approx(0.4)
    assert report.loc[1, "fpr"] == pytest.approx(2 / 6)
    assert report.loc[2, "fpr"] == pytest.approx(0.0)
    assert report.loc[1, "precision"] == pytest.approx(4 / 6)
    assert report.loc[2, "accuracy"] == pytest.approx(1.0)


def test_fairness_summary_returns_the_known_gap():
    summary = fairness_summary(_Y_TRUE, _Y_PRED, _SENSITIVE)

    assert summary["demographic_parity_difference"] == pytest.approx(0.2, abs=1e-9)
    # Computed two ways on purpose -- see the docstring in metrics.py.
    assert summary["selection_rate_gap"] == pytest.approx(0.2, abs=1e-9)
    assert summary["demographic_parity_ratio"] == pytest.approx(0.4 / 0.6)
    assert summary["equalized_odds_difference"] == pytest.approx(1 / 3)
    assert summary["n_groups"] == 2
    assert summary["n_smallest_group"] == 10


def test_fairness_summary_carries_the_size_of_the_smallest_group():
    # 17 accounts in one group and 3 in the other: the gap is computed either
    # way, and the summary has to say it rests on three people.
    sensitive = np.array([1] * 17 + [2] * 3)
    summary = fairness_summary(_Y_TRUE, _Y_PRED, sensitive)

    assert summary["n_smallest_group"] == 3
    # The gate never reads it -- size informs the reader, it is not a policy.
    assert passes_gates(summary) == passes_gates(
        {key: value for key, value in summary.items() if key != "n_smallest_group"}
    )


def test_passes_gates_rejects_the_known_gap_and_names_both_reasons():
    ok, reasons = passes_gates(fairness_summary(_Y_TRUE, _Y_PRED, _SENSITIVE))

    assert ok is False
    joined = " ".join(reasons)
    assert "demographic_parity_difference" in joined
    assert "equalized_odds_difference" in joined
    assert f"{schema.MAX_DEMOGRAPHIC_PARITY_DIFF:.2f}" in joined


def test_passes_gates_accepts_a_model_inside_both_limits():
    ok, reasons = passes_gates(
        {"demographic_parity_difference": 0.01, "equalized_odds_difference": 0.02}
    )
    assert ok is True
    assert reasons == []


def test_passes_gates_treats_missing_or_nan_metrics_as_failures():
    ok, reasons = passes_gates({})
    assert ok is False
    assert len(reasons) == 2

    ok, reasons = passes_gates(
        {"demographic_parity_difference": float("nan"), "equalized_odds_difference": 0.01}
    )
    assert ok is False
    assert "not finite" in reasons[0]


def test_fairness_summary_rejects_ragged_input():
    with pytest.raises(ValueError, match="length mismatch"):
        fairness_summary([1, 0, 1], [1, 0], [1, 1, 2])
    with pytest.raises(ValueError, match="empty"):
        fairness_summary([], [], [])


def test_flatten_and_multi_attribute_summaries():
    frame = pd.DataFrame(
        {schema.SEX: _SENSITIVE, schema.EDUCATION: np.tile([1, 2], 10), "unrelated": 0}
    )
    summaries = summaries_for_attributes(_Y_TRUE, _Y_PRED, frame)

    # MARRIAGE and AGE_GROUP are absent from the frame and simply skipped.
    assert set(summaries) == {schema.SEX, schema.EDUCATION}
    flat = flatten_summary(summaries[schema.SEX], prefix="fair_sex")
    assert flat["fair_sex_demographic_parity_difference"] == pytest.approx(0.2, abs=1e-9)


def test_sex_by_age_group_is_reported_as_one_joint_attribute():
    # Each marginal shows half a group selected and a 0.5 gap. Only the joint
    # shows where it sits: young men are always selected and nobody else is.
    frame = pd.DataFrame({schema.SEX: [1, 1, 2, 2] * 5, schema.AGE_GROUP: ["young", "older"] * 10})
    y_true = np.array([1, 0, 1, 0] * 5)
    y_pred = np.array([1, 0, 0, 0] * 5)

    summaries = summaries_for_attributes(y_true, y_pred, frame)
    key = intersection_key(schema.SEX, schema.AGE_GROUP)

    assert key == "SEX_x_AGE_GROUP"
    assert set(summaries) == {schema.SEX, schema.AGE_GROUP, key}
    joint = summaries[key]
    assert joint["n_groups"] == 4
    assert joint["n_smallest_group"] == 5
    assert joint["selection_rate_max"] == pytest.approx(1.0)
    assert joint["selection_rate_min"] == pytest.approx(0.0)
    # Both marginals are capped at 0.5 by construction; the joint gap is not.
    assert summaries[schema.SEX]["selection_rate_gap"] == pytest.approx(0.5)
    assert joint["selection_rate_gap"] == pytest.approx(1.0)
    # And it reaches MLflow under a name the experiment list can filter on.
    flat = flatten_summary(joint, prefix=f"by_{key}")
    assert "by_SEX_x_AGE_GROUP_demographic_parity_difference" in flat


def test_an_intersection_is_skipped_when_either_side_is_missing():
    frame = pd.DataFrame({schema.SEX: _SENSITIVE})
    summaries = summaries_for_attributes(_Y_TRUE, _Y_PRED, frame)

    assert set(summaries) == {schema.SEX}
    assert set(summaries_for_attributes(_Y_TRUE, _Y_PRED, frame, intersections=())) == {schema.SEX}


# ------------------------------------------------------------ calibration


def test_calibration_by_group_bins_and_reports_the_gap():
    # Group 1 is well calibrated, group 2 is told 0.9 and defaults half the
    # time -- the exact failure demographic parity cannot see.
    y_true = np.array([1, 1, 0, 0] + [1, 0, 1, 0])
    y_proba = np.array([0.9, 0.9, 0.1, 0.1] + [0.9, 0.9, 0.9, 0.9])
    sensitive = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    table = calibration_by_group(y_true, y_proba, sensitive, n_bins=5)

    assert set(table.columns) == {
        "group",
        "bin",
        "bin_lower",
        "bin_upper",
        "n",
        "mean_predicted",
        "observed_rate",
        "gap",
    }
    assert table["n"].sum() == 8
    errors = calibration_error_by_group(y_true, y_proba, sensitive, n_bins=5)
    assert errors["1"] == pytest.approx(0.1)  # 0.9 vs 1.0 and 0.1 vs 0.0
    assert errors["2"] == pytest.approx(0.4)  # promised 0.9, delivered 0.5


def test_calibration_error_by_group_is_empty_when_there_is_nothing_to_bin():
    assert calibration_error_by_group([], [], []) == {}


def test_calibration_by_group_validates_its_arguments():
    with pytest.raises(ValueError, match="length mismatch"):
        calibration_by_group([1, 0], [0.5, 0.5, 0.5], [1, 2])
    with pytest.raises(ValueError, match="n_bins"):
        calibration_by_group([1, 0], [0.5, 0.5], [1, 2], n_bins=0)
    assert calibration_error_by_group([1, 0], [0.5, 0.5], [1, 2], n_bins=1) != {}


# ------------------------------------------------------------- reweighing


def _imbalanced_sample(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Group 1 defaults twice as often as group 2 -- the base rate to remove."""
    rng = np.random.default_rng(seed)
    sensitive = np.repeat([1, 2], [400, 600])
    y = np.concatenate([rng.binomial(1, 0.40, 400), rng.binomial(1, 0.20, 600)])
    return y, sensitive


def test_reweigh_preserves_the_effective_sample_size():
    y, sensitive = _imbalanced_sample()
    weights = reweigh(y, sensitive)

    assert weights.shape == y.shape
    assert weights.sum() == pytest.approx(len(y))
    assert np.all(weights > 0)


def test_reweigh_makes_group_and_label_independent():
    y, sensitive = _imbalanced_sample()
    weights = reweigh(y, sensitive)
    n = len(y)

    for group in np.unique(sensitive):
        p_group = (sensitive == group).mean()
        for label in np.unique(y):
            p_label = (y == label).mean()
            cell = (sensitive == group) & (y == label)
            # This is the definition of the weights, restated as the property
            # they were computed to produce: P(g, c) == P(g) P(c).
            assert weights[cell].sum() / n == pytest.approx(p_group * p_label)


def test_reweigh_handles_an_empty_cell_without_dividing_by_zero():
    # Group 2 never defaults, so the (2, default) cell is empty. The weights
    # stay finite and the total falls short of n by exactly the mass that cell
    # would have carried: n_group * n_label / n = 2 * 1 / 4 = 0.5.
    y = np.array([1, 0, 0, 0])
    sensitive = np.array([1, 1, 2, 2])
    weights = reweigh(y, sensitive)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(4.0 - 0.5)


def test_reweigh_rejects_bad_input():
    with pytest.raises(ValueError, match="rows"):
        reweigh([1, 0, 1], [1, 2])
    with pytest.raises(ValueError, match="empty"):
        reweigh([], [])


# ------------------------------------------------------------ unawareness


def _proxy_frame(n: int = 600, seed: int = 11) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame in which LIMIT_BAL leaks the protected attribute."""
    rng = np.random.default_rng(seed)
    sensitive = rng.integers(1, 3, n)
    return (
        pd.DataFrame(
            {
                schema.SEX: sensitive,
                schema.LIMIT_BAL: 100_000 + 80_000 * (sensitive == 1) + rng.normal(0, 20_000, n),
                "PAY_1": rng.integers(-1, 3, n),
                "util_mean": rng.uniform(0, 1, n),
            }
        ),
        sensitive,
    )


def test_protected_columns_matches_raw_names_and_one_hot_prefixes():
    columns = ["SEX", "SEX_female", "AGE_GROUP", "EDUCATION_2", "LIMIT_BAL", "PAY_1", "SEXTANT"]
    found = protected_columns(columns)
    assert found == ["SEX", "SEX_female", "AGE_GROUP", "EDUCATION_2"]
    assert "SEXTANT" not in found  # prefix match requires the underscore


def test_unawareness_probe_recovers_the_attribute_from_a_proxy():
    X, sensitive = _proxy_frame()
    result = unawareness_probe(X, sensitive)

    assert schema.SEX in result["dropped_columns"]
    assert schema.SEX not in drop_protected(X).columns
    # The negative result the strategy exists to produce: SEX is gone from the
    # feature set and a linear model still finds it.
    assert result["roc_auc"] > 0.85
    assert result["top_proxies"][0]["feature"] == schema.LIMIT_BAL
    assert result["n_groups"] == 2


def test_unawareness_probe_handles_an_attribute_with_more_than_two_groups():
    # EDUCATION has four codes, so the probe has to score one-vs-rest instead
    # of reading a single positive-class column.
    rng = np.random.default_rng(13)
    n = 900
    education = rng.integers(1, 5, n)
    X = pd.DataFrame(
        {
            schema.EDUCATION: education,
            schema.LIMIT_BAL: 50_000 * education + rng.normal(0, 20_000, n),
            "PAY_1": rng.integers(-1, 3, n),
        }
    )
    result = unawareness_probe(X, education)

    assert result["n_groups"] == 4
    assert result["roc_auc"] > 0.75


def test_unawareness_probe_is_near_chance_when_nothing_leaks():
    rng = np.random.default_rng(5)
    n = 600
    sensitive = rng.integers(1, 3, n)
    X = pd.DataFrame(
        {
            schema.SEX: sensitive,  # the only informative column, and it is dropped
            schema.LIMIT_BAL: rng.normal(size=n),
            "PAY_1": rng.normal(size=n),
        }
    )
    result = unawareness_probe(X, sensitive)
    assert 0.35 < result["roc_auc"] < 0.65


def test_unawareness_probe_refuses_degenerate_input():
    X, sensitive = _proxy_frame(n=100)
    with pytest.raises(ValueError, match="nothing left to probe"):
        unawareness_probe(X[[schema.SEX]], sensitive)
    with pytest.raises(ValueError, match="single value"):
        unawareness_probe(X, np.ones(100, dtype=int))


# ----------------------------------------------------- threshold optimizer


def _fitted_baseline(n: int = 800, seed: int = 17):
    rng = np.random.default_rng(seed)
    sensitive = rng.integers(1, 3, n)
    risk = rng.normal(size=n) + 0.6 * (sensitive == 1)
    X = pd.DataFrame({schema.SEX: sensitive, "risk": risk, "noise": rng.normal(size=n)})
    y = (risk + rng.normal(scale=0.7, size=n) > 0.5).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return model, X, y, sensitive


def test_group_thresholds_exports_one_cutoff_per_group():
    model, X, y, sensitive = _fitted_baseline()
    optimizer = fit_threshold_optimizer(model, X, y, sensitive)

    thresholds = group_thresholds(optimizer)

    assert set(thresholds) == {"1", "2"}
    assert all(0.0 <= value <= 1.0 for value in thresholds.values())

    # And the exported cutoffs reproduce fairlearn's own decisions closely --
    # they differ only where the randomised leg would have fired.
    proba = model.predict_proba(X)[:, 1]
    ours = apply_group_thresholds(proba, sensitive, thresholds)
    theirs = np.asarray(optimizer.predict(X, sensitive_features=sensitive, random_state=0))
    assert (ours == theirs).mean() > 0.9


def test_group_thresholds_default_next_to_the_processed_splits():
    # Resolved, not written: the default lands in the one directory the
    # pipeline already owns, and no test should create it as a side effect.
    default = group_thresholds_path()
    assert default.name == "group_thresholds.json"
    assert default.parent == Path(settings.processed_dir)


def test_group_thresholds_round_trip_through_the_exported_file(tmp_path):
    path = tmp_path / "group_thresholds.json"
    save_group_thresholds({"1": 0.4931, "2": 0.4887}, path)

    # Called with no optimizer, exactly as the serving layer calls it.
    assert group_thresholds(path=path) == {"1": 0.4931, "2": 0.4887}
    assert "0.4931" in path.read_text()  # readable by a human, not pickled
    # The attribute travels with the numbers so a reader can tell what the
    # keys mean. Without it "1" and "2" are just as plausibly MARRIAGE codes.
    assert json.loads(path.read_text())["attribute"] == schema.SEX


def test_thresholds_fitted_on_another_attribute_are_refused_not_misapplied(tmp_path, caplog):
    # MARRIAGE codes are 1, 2, 3 and SEX codes are 1, 2 -- so two of the three
    # keys match by pure coincidence. Reading this file as a SEX policy would
    # apply a marital cutoff to every man and woman in the portfolio and say
    # nothing about it.
    path = tmp_path / "group_thresholds.json"
    save_group_thresholds({"1": 0.31, "2": 0.62, "3": 0.44}, path, attribute=schema.MARRIAGE)

    with caplog.at_level("WARNING"):
        assert load_group_thresholds(path, attribute=schema.SEX) == {}
    assert "were fitted on MARRIAGE" in caplog.text

    # Read as what it is, it loads.
    assert load_group_thresholds(path, attribute=schema.MARRIAGE)["3"] == pytest.approx(0.44)


def test_a_flat_thresholds_file_still_loads(tmp_path):
    # The shape written before the attribute was recorded. Serving reads this
    # file off a mounted volume that outlives any one training run, so the
    # reader has to survive finding the previous format there.
    path = tmp_path / "group_thresholds.json"
    path.write_text(json.dumps({"1": 0.51, "2": 0.47}))

    assert load_group_thresholds(path) == {"1": 0.51, "2": 0.47}


def test_loading_thresholds_degrades_instead_of_raising(tmp_path):
    # Never written: the group-aware policy has simply never been fitted.
    assert load_group_thresholds(tmp_path / "absent.json") == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    # A half-written file must not take /predict down with it.
    assert load_group_thresholds(corrupt) == {}


def test_group_thresholds_refuses_an_unfitted_optimizer():
    from fairlearn.postprocessing import ThresholdOptimizer

    unfitted = ThresholdOptimizer(
        estimator=LogisticRegression(), constraints="equalized_odds", prefit=True
    )
    with pytest.raises(ValueError, match="not fitted"):
        group_thresholds(unfitted)


def test_apply_group_thresholds_falls_back_for_an_unseen_group():
    proba = np.array([0.1, 0.9, 0.9])
    sensitive = np.array([1, 1, 99])
    decisions = apply_group_thresholds(proba, sensitive, {"1": 0.5}, default_threshold=0.95)

    assert list(decisions) == [0, 1, 0]  # the unseen group used the fallback
    with pytest.raises(ValueError, match="rows"):
        apply_group_thresholds([0.1, 0.2], [1], {})


def test_a_score_exactly_on_the_cutoff_is_selected_like_serving_selects_it():
    # `serving.routes.decide` is `probability >= threshold` and
    # `evaluate.threshold_for_capacity` returns the k-th largest score, so a
    # cutoff lifted from either one admits the account sitting on it. This
    # function used to use `>`, which selected k-1 accounts offline and k in
    # production -- a disagreement that only ever shows up on a tie, and the
    # default cutoff is 0.5, a number somebody typed.
    on_the_line = apply_group_thresholds([0.5, 0.5], [1, 2], {"1": 0.5}, default_threshold=0.5)
    assert list(on_the_line) == [1, 1]

    from credit_risk.models.evaluate import threshold_for_capacity

    proba = np.linspace(0.01, 0.99, 100)
    cutoff = threshold_for_capacity(proba, 0.10)
    selected = apply_group_thresholds(proba, np.ones(100, dtype=int), {}, default_threshold=cutoff)
    assert int(selected.sum()) == 10


def test_a_non_finite_cutoff_is_clamped_so_the_file_stays_valid_json(tmp_path):
    # fairlearn's ThresholdOperation documents its threshold as possibly
    # +/-inf, and its ROC sweep seeds the first point with inf. json.dumps
    # writes that as `Infinity`, which Python reads back and every other JSON
    # reader rejects -- including whatever an auditor opens the file with.
    class _Operation:
        threshold = float("inf")

    class _Thresholder:
        interpolation_dict = {1: {"p0": 1.0, "p1": 0.0, "operation0": _Operation()}}

    class _Optimizer:
        interpolated_thresholder_ = _Thresholder()

    thresholds = group_thresholds(_Optimizer())
    assert thresholds == {"1": 1.0}

    path = save_group_thresholds(thresholds, tmp_path / "group_thresholds.json")
    assert "Infinity" not in path.read_text()
    assert json.loads(path.read_text())["thresholds"] == {"1": 1.0}


# --------------------------------------------------------- tradeoff table


def test_tradeoff_curve_scores_all_four_strategies_identically():
    rng = np.random.default_rng(23)
    n = 1200
    sensitive = rng.integers(1, 3, n)
    risk = rng.normal(size=n) + 0.7 * (sensitive == 1)
    X = pd.DataFrame({schema.SEX: sensitive, "risk": risk, "noise": rng.normal(size=n)})
    y = (risk + rng.normal(scale=0.8, size=n) > 0.6).astype(int)
    split = 800

    table = tradeoff_curve(
        X.iloc[:split],
        y[:split],
        sensitive[:split],
        X.iloc[split:],
        y[split:],
        sensitive[split:],
        capacity_fraction=0.20,
    )

    assert list(table["strategy"]) == list(STRATEGIES)
    assert {"pr_auc", "dp_diff", "eo_diff", "recall_at_k", "selection_rate"} <= set(table.columns)
    assert table.notna().all().all()

    rows = table.set_index("strategy")
    # Post-processing cannot reorder anything, so its PR-AUC is the baseline's
    # to the last digit. The price shows up in the selection rate instead.
    assert rows.loc["threshold_optimizer", "pr_auc"] == pytest.approx(
        rows.loc["baseline", "pr_auc"]
    )
    # The three score-based strategies all contact the same number of people.
    assert rows.loc["baseline", "selection_rate"] == pytest.approx(0.2, abs=0.005)
    assert rows.loc["unawareness", "selection_rate"] == pytest.approx(0.2, abs=0.005)
    assert rows.loc["reweighing", "selection_rate"] == pytest.approx(0.2, abs=0.005)


def test_tradeoff_curve_says_so_when_there_is_no_protected_column_to_drop(caplog):
    # A feature frame that never exposed the attribute makes the unawareness
    # row a duplicate of the baseline. That is a real result, and it has to be
    # announced rather than quietly printed as a second data point.
    rng = np.random.default_rng(31)
    n = 800
    sensitive = rng.integers(1, 3, n)
    risk = rng.normal(size=n) + 0.7 * (sensitive == 1)
    X = pd.DataFrame({"risk": risk, "noise": rng.normal(size=n)})
    y = (risk + rng.normal(scale=0.8, size=n) > 0.6).astype(int)

    with caplog.at_level("WARNING"):
        table = tradeoff_curve(
            X.iloc[:500],
            y[:500],
            sensitive[:500],
            X.iloc[500:],
            y[500:],
            sensitive[500:],
            capacity_fraction=0.20,
        )

    assert "no protected-derived columns" in caplog.text
    rows = table.set_index("strategy")
    assert rows.loc["unawareness", "pr_auc"] == pytest.approx(rows.loc["baseline", "pr_auc"])


def test_tradeoff_curve_drops_the_post_processing_row_it_cannot_fit(caplog):
    # fairlearn refuses a group that carries a single label. MARRIAGE=3 is 323
    # rows in 30,000, so one thin batch is all it takes -- and it happens after
    # the grid search has already been paid for. The other three strategies are
    # still worth reporting, so the row goes and the run does not.
    rng = np.random.default_rng(41)
    n = 600
    # Group 3 has to land inside the first 400 rows: those are the ones the
    # optimizer is fitted on, and a group that only appears at scoring time is
    # a different failure (the fallback cutoff), tested elsewhere.
    sensitive = np.array([1] * 190 + [2] * 190 + [3] * 20 + [1] * 100 + [2] * 90 + [3] * 10)
    risk = rng.normal(size=n) + 0.7 * (sensitive == 1)
    X = pd.DataFrame({schema.SEX: sensitive, "risk": risk, "noise": rng.normal(size=n)})
    y = (risk + rng.normal(scale=0.8, size=n) > 0.6).astype(int)
    y[sensitive == 3] = 0  # the degenerate group: one label, nothing to trade off

    with caplog.at_level("WARNING"):
        table = tradeoff_curve(
            X.iloc[:400],
            y[:400],
            sensitive[:400],
            X.iloc[400:],
            y[400:],
            sensitive[400:],
            capacity_fraction=0.20,
        )

    assert "threshold_optimizer could not be fitted" in caplog.text
    assert list(table["strategy"]) == ["baseline", "unawareness", "reweighing"]
