"""Tests for the serving metrics module.

Two of these are load-bearing rather than decorative. `test_window_is_bounded`
pins the memory behaviour of a process that stays up for days, and
`test_series_count_stays_bounded_under_traffic` turns the cardinality rule from
a comment into something CI fails on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY

from credit_risk.config import settings
from credit_risk.serving import metrics as m

# Every metric name an alert rule or dashboard panel queries. If serving stops
# declaring one of these, the rule silently never fires -- which looks exactly
# like a healthy system.
ALERTED_METRICS = (
    "credit_predictions_total",
    "credit_errors_total",
    "credit_prediction_latency_seconds",
    "credit_explain_duration_seconds",
    "credit_model_loaded",
    "credit_model_info",
    "credit_high_risk_share",
    "credit_baseline_high_risk_share",
    "credit_selection_rate",
    "credit_feature_psi",
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each case starts from an empty window, baseline and label set."""
    m.set_baseline({})
    m.reset_windows()
    m.PREDICTIONS.clear()
    m.ERRORS.clear()
    m.MODEL_INFO.clear()
    m.BASELINE_HIGH_RISK_SHARE.set(0.0)
    yield
    m.set_baseline({})
    m.reset_windows()
    m.BASELINE_HIGH_RISK_SHARE.set(0.0)


def _gauge(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


# ------------------------------------------------------------------- psi


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=2_000).tolist()
    assert m.compute_psi(sample, sample) == 0.0


def test_psi_stays_below_the_stable_threshold_for_resampled_noise():
    # The index has to survive sampling noise, otherwise FeatureDriftHigh fires
    # every time traffic is quiet and the alert gets muted for good.
    rng = np.random.default_rng(1)
    reference = rng.normal(size=5_000).tolist()
    current = rng.normal(size=5_000).tolist()
    assert m.compute_psi(reference, current) < m.PSI_STABLE


def test_psi_flags_a_clearly_shifted_distribution():
    rng = np.random.default_rng(2)
    reference = rng.normal(loc=0.0, scale=1.0, size=5_000).tolist()
    current = rng.normal(loc=1.5, scale=1.0, size=5_000).tolist()
    assert m.compute_psi(reference, current) > m.PSI_SIGNIFICANT


def test_psi_is_finite_when_bins_empty_out():
    # Every serving value lands in the top bin, so nine of ten bins are empty.
    # Without epsilon smoothing this is ln(0) and the gauge pins at -inf/+inf.
    rng = np.random.default_rng(3)
    reference = rng.uniform(0.0, 1.0, size=2_000).tolist()
    current = [0.999] * 500
    psi = m.compute_psi(reference, current)
    assert np.isfinite(psi)
    assert psi > m.PSI_SIGNIFICANT


def test_psi_of_a_constant_reference_column_is_zero():
    # A column with no spread in training cannot evidence drift; it must report
    # 0.0 rather than blow up on degenerate quantile edges.
    assert m.compute_psi([5.0] * 200, [7.0] * 200) == 0.0


def test_psi_handles_an_empty_current_sample():
    psi = m.compute_psi([float(i) for i in range(200)], [])
    assert np.isfinite(psi)


# ---------------------------------------------------------- share gauges


def test_share_and_selection_rates_are_exact_over_a_known_sequence():
    # Five men, three of whom are selected; five women, one of whom is.
    # Four of the ten probabilities sit at or above the 0.5 threshold.
    sequence = [
        (0.90, "intervene", "1"),
        (0.80, "intervene", "1"),
        (0.70, "intervene", "1"),
        (0.20, "monitor", "1"),
        (0.10, "monitor", "1"),
        (0.90, "intervene", "2"),
        (0.10, "monitor", "2"),
        (0.10, "monitor", "2"),
        (0.10, "monitor", "2"),
        (0.10, "monitor", "2"),
    ]
    # Repeated until both groups clear SELECTION_RATE_MIN_SAMPLES: the ratios
    # are what is under test, and below the floor the gauge is not published.
    repeats = -(-m.SELECTION_RATE_MIN_SAMPLES // 5)
    for _ in range(repeats):
        for probability, decision, group in sequence:
            m.observe_prediction(probability, decision, group)

    assert _gauge("credit_high_risk_share") == pytest.approx(0.4)
    assert _gauge("credit_selection_rate", group="1") == pytest.approx(0.6)
    assert _gauge("credit_selection_rate", group="2") == pytest.approx(0.2)
    assert _gauge("credit_predictions_total", decision="intervene") == pytest.approx(4 * repeats)
    assert _gauge("credit_predictions_total", decision="monitor") == pytest.approx(6 * repeats)


def test_window_is_bounded():
    window = settings.risk_share_window
    for _ in range(window * 10):
        m.observe_prediction(0.9, "intervene", "1")

    assert len(m._window) == window

    # And the gauge follows the window rather than all history: replacing every
    # slot with a low score has to drive the share to zero.
    for _ in range(window):
        m.observe_prediction(0.1, "monitor", "1")
    assert _gauge("credit_high_risk_share") == pytest.approx(0.0)


def test_high_risk_share_is_nan_until_the_window_holds_enough_requests():
    # Idle, and then a handful of requests: neither is a measurement. A 0.0
    # here read as a 12-point drift from a 0.124 baseline and kept
    # HighRiskShareShift firing on an API nobody was calling; a 1.0 after one
    # request would do the same in the other direction.
    share = _gauge("credit_high_risk_share")
    assert share is not None and math.isnan(share)

    for _ in range(m.HIGH_RISK_SHARE_MIN_SAMPLES - 1):
        m.observe_prediction(0.9, "intervene", "1")
    share = _gauge("credit_high_risk_share")
    assert share is not None and math.isnan(share)

    m.observe_prediction(0.9, "intervene", "1")
    assert _gauge("credit_high_risk_share") == pytest.approx(1.0)


def test_the_exposition_carries_nan_for_an_idle_window():
    # NaN is what makes the alert silent: PromQL drops every sample for which
    # `abs(NaN - baseline) > 0.10` is evaluated, so the rule has nothing to fire on.
    payload, _ = m.render()
    assert b"credit_high_risk_share NaN" in payload


def test_group_that_leaves_the_window_loses_its_series():
    window = settings.risk_share_window
    for _ in range(window // 2):
        m.observe_prediction(0.9, "intervene", "1")
        m.observe_prediction(0.1, "monitor", "2")
    assert _gauge("credit_selection_rate", group="2") is not None

    # Push group 2 entirely out of the window. A frozen stale gauge here would
    # keep FairnessGapExceeded firing against traffic that no longer exists.
    for _ in range(window):
        m.observe_prediction(0.9, "intervene", "1")
    assert _gauge("credit_selection_rate", group="2") is None
    assert _gauge("credit_selection_rate", group="1") == pytest.approx(1.0)


def test_reset_windows_clears_every_rolling_structure():
    m.set_baseline(_reference(2))
    for _ in range(m.SELECTION_RATE_MIN_SAMPLES):
        m.observe_prediction(0.9, "intervene", "1")
    for _ in range(m.PSI_REFRESH_EVERY):
        m.observe_features({"f0": 0.0, "f1": 0.0})
    assert _gauge("credit_selection_rate", group="1") is not None
    assert _gauge("credit_feature_psi", feature="f0") is not None

    m.reset_windows()
    assert _gauge("credit_selection_rate", group="1") is None
    assert _gauge("credit_feature_psi", feature="f0") is None
    share = _gauge("credit_high_risk_share")
    assert share is not None and math.isnan(share)
    assert len(m._window) == 0
    assert all(len(w) == 0 for w in m._feature_windows.values())


# ----------------------------------------------------------- feature psi


def _reference(n_features: int, n_values: int = 400) -> dict[str, list[float]]:
    rng = np.random.default_rng(7)
    return {f"f{i}": rng.normal(size=n_values).tolist() for i in range(n_features)}


def test_observe_features_is_a_noop_without_a_baseline():
    for _ in range(m.PSI_REFRESH_EVERY * 2):
        m.observe_features({"f0": 1.0})
    assert _gauge("credit_feature_psi", feature="f0") is None


def test_observe_features_publishes_psi_against_the_baseline():
    m.set_baseline(_reference(2))
    rng = np.random.default_rng(11)
    for _ in range(m.PSI_REFRESH_EVERY * 3):
        # f0 stays on the training distribution, f1 is shifted hard.
        m.observe_features({"f0": float(rng.normal()), "f1": float(rng.normal() + 4.0)})

    stable = _gauge("credit_feature_psi", feature="f0")
    shifted = _gauge("credit_feature_psi", feature="f1")
    assert stable is not None and shifted is not None
    assert shifted > m.PSI_SIGNIFICANT
    assert shifted > stable


def test_observe_features_tolerates_a_missing_column():
    m.set_baseline(_reference(2))
    for _ in range(m.PSI_REFRESH_EVERY * 2):
        m.observe_features({"f0": 0.0})
    assert _gauge("credit_feature_psi", feature="f0") is not None
    assert _gauge("credit_feature_psi", feature="f1") is None


def test_set_baseline_refuses_more_features_than_the_label_cap():
    # Truncating would pick twelve of the 38 engineered columns by dict
    # insertion order -- a choice nobody made and nobody can see.
    with pytest.raises(ValueError, match="capped at"):
        m.set_baseline(_reference(m.MAX_PSI_FEATURES + 5))


def test_set_baseline_accepts_exactly_the_label_cap():
    m.set_baseline(_reference(m.MAX_PSI_FEATURES))
    assert len(m._baseline) == m.MAX_PSI_FEATURES


def test_set_baseline_skips_columns_with_too_little_history():
    m.set_baseline({"tiny": [1.0, 2.0, 3.0], "ok": list(range(m.PSI_MIN_SAMPLES * 2))})
    assert "tiny" not in m._baseline
    assert "ok" in m._baseline


def test_set_baseline_drops_features_that_disappear():
    m.set_baseline(_reference(2))
    for _ in range(m.PSI_REFRESH_EVERY):
        m.observe_features({"f0": 0.0, "f1": 0.0})
    assert _gauge("credit_feature_psi", feature="f1") is not None

    m.set_baseline({"f0": _reference(1)["f0"]})
    assert _gauge("credit_feature_psi", feature="f1") is None


# ------------------------------------------------------- fairness floor


def test_a_group_below_the_sample_floor_is_not_published():
    # The failure this prevents: one request with an unparseable SEX field
    # creates a `sex_unknown` series pinned at exactly 0.0 or 1.0, and
    # max() - min() clears the 0.05 parity gate on a sample of one.
    for _ in range(m.SELECTION_RATE_MIN_SAMPLES * 2):
        m.observe_prediction(0.9, "intervene", "sex_male")
    m.observe_prediction(0.1, "monitor", "sex_unknown")

    assert _gauge("credit_selection_rate", group="sex_male") == pytest.approx(1.0)
    assert _gauge("credit_selection_rate", group="sex_unknown") is None


def test_a_group_that_shrinks_below_the_floor_loses_its_series():
    for _ in range(m.SELECTION_RATE_MIN_SAMPLES * 2):
        m.observe_prediction(0.9, "intervene", "sex_male")
        m.observe_prediction(0.1, "monitor", "sex_female")
    assert _gauge("credit_selection_rate", group="sex_female") is not None

    # Squeeze the women down to a handful of slots without evicting them
    # entirely. A frozen last value here is the stale-gauge failure again.
    for _ in range(settings.risk_share_window):
        m.observe_prediction(0.9, "intervene", "sex_male")
    m.observe_prediction(0.1, "monitor", "sex_female")
    assert _gauge("credit_selection_rate", group="sex_female") is None


# ------------------------------------------------------------ exposition


def test_render_returns_prometheus_content_type_and_every_metric_name():
    m.observe_prediction(0.9, "intervene", "1")
    m.set_baseline_high_risk_share(0.11)
    m.MODEL_LOADED.set(1)
    m.set_model_info(version="7", algo="lightgbm", trained_at="2026-09-30", git_sha="abc1234")
    m.LATENCY.labels(endpoint="/api/v1/predict").observe(0.01)
    m.EXPLAIN_DURATION.labels(method="shap").observe(0.2)
    m.ERRORS.labels(reason="validation_error").inc()

    payload, content_type = m.render()
    assert content_type == CONTENT_TYPE_LATEST
    assert isinstance(payload, bytes)
    for name in ALERTED_METRICS:
        assert name in payload.decode(), f"missing metric: {name}"

    # The bucket series is what histogram_quantile reads; the family name alone
    # is not enough for SlowPredictions to evaluate.
    assert "credit_prediction_latency_seconds_bucket" in payload.decode()


def test_baseline_high_risk_share_is_published_and_replaced():
    # HighRiskShareShift subtracts this gauge from the live share, so it has to
    # be a single value that a model swap overwrites rather than accumulates.
    m.set_baseline_high_risk_share(0.11)
    assert _gauge("credit_baseline_high_risk_share") == pytest.approx(0.11)
    m.set_baseline_high_risk_share(0.09)
    assert _gauge("credit_baseline_high_risk_share") == pytest.approx(0.09)


def test_set_model_info_replaces_the_previous_version():
    m.set_model_info(version="6", algo="lightgbm", trained_at="2026-09-01", git_sha="aaa")
    m.set_model_info(version="7", algo="lightgbm", trained_at="2026-09-30", git_sha="bbb")

    old = _gauge(
        "credit_model_info", version="6", algo="lightgbm", trained_at="2026-09-01", git_sha="aaa"
    )
    new = _gauge(
        "credit_model_info", version="7", algo="lightgbm", trained_at="2026-09-30", git_sha="bbb"
    )
    assert old is None
    assert new == pytest.approx(1.0)


def test_series_count_stays_bounded_under_traffic():
    m.set_baseline(_reference(8))
    rng = np.random.default_rng(13)

    after_warmup = 0
    for i in range(1_000):
        probability = float(rng.uniform())
        decision = "intervene" if probability >= settings.decision_threshold else "monitor"
        group = "1" if i % 2 == 0 else "2"
        m.observe_prediction(probability, decision, group)
        m.observe_features({f"f{j}": float(rng.normal()) for j in range(8)})
        if i == 199:
            after_warmup = m.series_count()

    # The bound the Lab 04 incident bought us: 1,000 requests with 1,000
    # distinct payloads must not add a single series beyond warm-up.
    assert m.series_count() == after_warmup
    assert m.series_count() < 60
