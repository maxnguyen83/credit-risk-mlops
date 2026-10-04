"""Prometheus instrumentation for the credit-risk API.

Every metric name that an alert rule or a dashboard panel references is declared
here and nowhere else. A metric defined in two modules is a metric that silently
stops existing the moment only one of the two import paths runs.

Cardinality is the binding constraint, not storage. In the Lab 04 stack a single
per-request label turned 23 time series into 639 and made Prometheus the slowest
process in the compose file. The rule that came out of that is enforced by the
shape of this module rather than by discipline: no function here accepts an
account id, a request id, or anything else drawn from a request body, and the
only open-ended label -- `feature` on the PSI gauge -- is bounded by the baseline
captured at training time, not by what a caller happens to send.
`series_count()` exists so a test can assert that bound instead of trusting it.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Final

import numpy as np
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from credit_risk.config import settings

# --------------------------------------------------------------- tuning

# Prometheus stores one time series per bucket, so a bucket list is a cardinality
# decision as much as a resolution decision. These ten straddle the 100 ms p95
# target with room either side; anything finer buys nothing histogram_quantile
# can actually use at this traffic volume.
LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

# SHAP is an order of magnitude slower than a predict call and the SLO is 500 ms,
# so the interesting resolution sits an order of magnitude higher than above.
EXPLAIN_BUCKETS: Final[tuple[float, ...]] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

# Every metric declared here shares this prefix; series_count() uses it to count
# what this module owns rather than the interpreter stats Python ships for free.
METRIC_PREFIX: Final = "credit_"

PSI_BINS: Final = 10

# PSI over ten bins needs more history than the risk-share gauge does: with a
# 200-observation window a bin holds ~20 values and the index rattles across 0.25
# on sampling noise alone. Five windows is the cheapest fix that still bounds the
# structure -- and bounded is the point, this process stays up for days.
PSI_WINDOW: Final = settings.risk_share_window * 5

# Recomputing ten histograms on every request is work no scrape asked for. Drift
# moves over minutes; refreshing every 50 observations is comfortably inside that.
PSI_REFRESH_EVERY: Final = 50

# Below this many samples a PSI number is noise wearing a measurement's clothes.
PSI_MIN_SAMPLES: Final = 50

# An empty bin makes ln(a/b) infinite, which pins the gauge at +Inf and leaves
# FeatureDriftHigh firing forever with no traffic pattern able to clear it.
PSI_EPSILON: Final = 1e-6

# The standard reading of the index, for whoever is looking at this at 2am:
#   < 0.10      stable             -- no action
#   0.10..0.25  moderate shift     -- worth a look at the next model review
#   > 0.25      significant shift  -- investigate; the training distribution is gone
PSI_STABLE: Final = 0.10
PSI_SIGNIFICANT: Final = 0.25

# Backstop against a caller handing over every engineered feature. The training
# job decides what is worth baselining; this only bounds the blast radius if it
# decides badly.
MAX_PSI_FEATURES: Final = 12

# A selection rate computed over a handful of requests is not a measurement.
# One request from a group publishes exactly 0.0 or 1.0, which clears the 0.05
# parity gate by itself -- and it takes only one malformed SEX field to create
# such a group, because the serving layer buckets that as `sex_unknown`. This is
# the same discipline PSI_MIN_SAMPLES already applies to the drift gauge.
#
# 30 of a 200-slot window does not make a minority group's rate precise: at
# p = 0.15 the sampling s.d. is still about 0.065. What it removes is the 0-or-1
# pathology and the worst of the small-sample noise; the 15-minute `for:` on
# FairnessGapExceeded absorbs the rest, because a noise excursion has to survive
# roughly ninety complete turnovers of the window to reach a human.
SELECTION_RATE_MIN_SAMPLES: Final = 30

# The same discipline for the high-risk share. Below this many scored requests
# the share is published as NaN, which every PromQL comparison treats as false,
# so an API that has just started -- or has been restarted and seen no traffic
# since -- cannot read as a 10-point drift from its baseline. Fifty is the PSI
# floor too; at a 0.12 share it still leaves a sampling s.d. near 0.05, but it
# removes the 0-out-of-0 and 1-out-of-1 readings outright.
HIGH_RISK_SHARE_MIN_SAMPLES: Final = 50

# The decision string the serving layer emits for an account that goes onto the
# intervention list. Selection rate is measured on this rather than on the
# probability because this is the thing that actually happens to a customer.
INTERVENE: Final = "intervene"

# ---------------------------------------------------------------- metrics

PREDICTIONS = Counter(
    "credit_predictions_total",
    "Predictions served, by decision.",
    ["decision"],
)

ERRORS = Counter(
    "credit_errors_total",
    "Requests that failed, by reason. Every error branch increments exactly one.",
    ["reason"],
)

LATENCY = Histogram(
    "credit_prediction_latency_seconds",
    "Wall-clock time to serve a request, by endpoint.",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)

EXPLAIN_DURATION = Histogram(
    "credit_explain_duration_seconds",
    "Time to produce one explanation, by method (shap or lime).",
    ["method"],
    buckets=EXPLAIN_BUCKETS,
)

# A process can answer /health on port 8000 and still be useless. This gauge is
# the difference between "the port is open" and "the service does its job".
MODEL_LOADED = Gauge(
    "credit_model_loaded",
    "1 when a model is loaded and serving, 0 when the API is up but degraded.",
)

MODEL_INFO = Gauge(
    "credit_model_info",
    "Always 1. The labels carry which model is serving; read them during an incident.",
    ["version", "algo", "trained_at", "git_sha"],
)

HIGH_RISK_SHARE = Gauge(
    "credit_high_risk_share",
    "Share of the recent window scored at or above the decision threshold.",
)
# A Gauge starts at 0.0, and 0.0 is a reading: |0 - 0.124| > 0.10 kept
# HighRiskShareShift firing on an API that had not served a single request.
HIGH_RISK_SHARE.set(float("nan"))

# What the SERVING model predicted positive at its own threshold, measured on
# the evaluation set. Not the label prevalence: 22.12% of accounts do default,
# but a model at the project's PR-AUC target flags far fewer than that at a 0.5
# cutoff, and under the 10%-of-portfolio capacity policy it flags fewer still.
# Comparing the live share against prevalence therefore reports a drift of 10+
# points on a model that is behaving exactly as it did at registration, which is
# an alert that fires on day one and is muted by day two.
BASELINE_HIGH_RISK_SHARE = Gauge(
    "credit_baseline_high_risk_share",
    "Share the serving model scored at or above its threshold on the evaluation set.",
)

SELECTION_RATE = Gauge(
    "credit_selection_rate",
    "Share of recent requests in each protected group selected for intervention.",
    ["group"],
)

FEATURE_PSI = Gauge(
    "credit_feature_psi",
    "Population Stability Index of a feature against its training baseline.",
    ["feature"],
)

# ------------------------------------------------------------------ state

# Uvicorn runs handlers across a thread pool. deque.append is atomic but the
# recompute walks the whole deque, and a concurrent append during that walk
# raises RuntimeError in the middle of a request. One lock, held briefly.
_lock = threading.Lock()

# The cutoff the high-risk share counts against. The lifespan installs the
# serving model's own threshold here, the same one /predict decides at and the
# baseline gauge is scored at, so the live share and its baseline always
# describe one cutoff. The configured constant is only the value before a model
# has loaded.
_decision_threshold: float = settings.decision_threshold

# (probability, decision, group) for the most recent requests. maxlen is the
# whole point: an unbounded structure in a process that runs for days is a leak
# with a metrics label on it.
_window: deque[tuple[float, str, str]] = deque(maxlen=settings.risk_share_window)

# Per-group windows, not one shared window sliced by group. Under a skewed
# arrival mix the minority group falls below SELECTION_RATE_MIN_SAMPLES in a
# shared window and its series disappears -- so max() - min() collapses to a
# single group, reads 0, and FairnessGapExceeded goes quiet at exactly the
# moment the population shifted. Giving each group its own bounded deque means a
# group's rate ages on its own traffic: it stays comparable when the mix moves,
# and still vanishes when that group genuinely stops arriving.
_group_windows: dict[str, deque[str]] = {}

_feature_windows: dict[str, deque[float]] = {}
_baseline: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_feature_obs = 0
_known_groups: set[str] = set()


# -------------------------------------------------------------------- psi


def _bin_edges(reference: Sequence[float], bins: int = PSI_BINS) -> np.ndarray:
    """Quantile bin edges from a reference sample, with the outer edges opened."""
    arr = np.asarray(reference, dtype=float)
    edges = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant reference column collapses to a single bin, where PSI can
        # only ever be 0. Returning one open bin keeps the caller branch-free.
        return np.array([-np.inf, np.inf])
    # Opened deliberately: a serving value below the training minimum belongs in
    # the first bin, not outside the histogram where it stops counting at all.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _proportions(values: Sequence[float], edges: np.ndarray) -> np.ndarray:
    """Share of `values` in each bin, with empty bins smoothed to PSI_EPSILON."""
    arr = np.asarray(values, dtype=float)
    # searchsorted rather than np.histogram: the outer edges are infinite, which
    # np.histogram rejects, and clipping keeps out-of-range values in the end
    # bins instead of silently dropping them out of the denominator.
    idx = np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, edges.size - 2)
    counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
    total = float(counts.sum())
    if total == 0.0:
        return np.full(edges.size - 1, 1.0 / (edges.size - 1))
    return np.maximum(counts / total, PSI_EPSILON)


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    """sum over bins of (actual - expected) * ln(actual / expected)."""
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_psi(
    reference: Sequence[float], current: Sequence[float], bins: int = PSI_BINS
) -> float:
    """Population Stability Index of `current` measured against `reference`.

    Reads on the standard scale: < 0.10 stable, 0.10-0.25 moderate shift,
    > 0.25 significant shift. Identical samples give exactly 0.0.
    """
    edges = _bin_edges(reference, bins)
    return _psi_from_proportions(_proportions(reference, edges), _proportions(current, edges))


def set_baseline(reference: dict[str, list[float]]) -> None:
    """Install the training-time distributions that serving traffic is compared against.

    Called once at model load, with the columns the training job deliberately
    chose to watch. Their names become the entire label set of
    `credit_feature_psi` from that point on, which is how the drift gauge stays
    bounded by training configuration instead of by request content.

    Raises ValueError for more than MAX_PSI_FEATURES columns. Silently keeping
    the first twelve was worse than it looks: the engineered frame has 38, so
    the cut was always happening, and it was made by dict insertion order -- a
    choice nobody took, nobody could see on a dashboard, and that moves the
    moment the feature builder reorders its output. A caller on the serving path
    should log this and carry on: an API with no drift gauge still scores
    accounts, and refusing to boot over a monitoring detail is the larger outage.
    """
    if len(reference) > MAX_PSI_FEATURES:
        raise ValueError(
            f"set_baseline received {len(reference)} features but the PSI label set is "
            f"capped at {MAX_PSI_FEATURES}. Pass the columns worth watching, not the "
            "whole feature frame."
        )
    with _lock:
        _baseline.clear()
        _feature_windows.clear()
        # Removing every child rather than overwriting: a feature dropped from
        # the new baseline would otherwise keep exporting its last PSI forever,
        # and nothing on the dashboard would say the number had stopped moving.
        FEATURE_PSI.clear()
        for feature, values in reference.items():
            if len(values) < PSI_MIN_SAMPLES:
                continue
            edges = _bin_edges(values)
            _baseline[feature] = (edges, _proportions(values, edges))
            _feature_windows[feature] = deque(maxlen=PSI_WINDOW)


def _refresh_psi() -> None:
    """Recompute PSI for every tracked feature. Caller holds `_lock`."""
    for feature, (edges, expected) in _baseline.items():
        window = _feature_windows[feature]
        if len(window) < PSI_MIN_SAMPLES:
            continue
        FEATURE_PSI.labels(feature=feature).set(
            _psi_from_proportions(expected, _proportions(window, edges))
        )


def observe_features(row: dict[str, float]) -> None:
    """Feed one served row into the drift windows and refresh PSI periodically."""
    if not _baseline:
        # No baseline, no PSI -- and no labels either, which is also how the
        # feature label set stays empty until training says otherwise.
        return
    global _feature_obs
    with _lock:
        for feature, window in _feature_windows.items():
            value = row.get(feature)
            if value is None:
                continue
            window.append(float(value))
        _feature_obs += 1
        if _feature_obs % PSI_REFRESH_EVERY == 0:
            _refresh_psi()


# ------------------------------------------------------------ predictions


def _refresh_share_gauges() -> None:
    """Recompute the rolling share gauges from `_window`. Caller holds `_lock`."""
    global _known_groups
    total = len(_window)
    if total == 0:
        HIGH_RISK_SHARE.set(float("nan"))
        return

    # High-risk share tracks the model's output distribution, read against
    # BASELINE_HIGH_RISK_SHARE -- what this model flagged at registration -- and
    # deliberately not against the 22.12% label prevalence, which is how many
    # accounts default rather than how many the model picks. Selection rate
    # tracks what the policy then did. The two agree under one global threshold
    # and separate the moment group-aware thresholds are switched on, which is
    # exactly when you want to see both.
    threshold = _decision_threshold
    if total < HIGH_RISK_SHARE_MIN_SAMPLES:
        HIGH_RISK_SHARE.set(float("nan"))
    else:
        HIGH_RISK_SHARE.set(
            sum(1 for probability, _, _ in _window if probability >= threshold) / total
        )

    # Groups under the floor are not published at all, rather than published
    # with a wide error bar nothing downstream can read. FairnessGapExceeded is
    # max() - min() over whatever series exist, so a group represented by two
    # requests would set the minimum to 0.0 and the alert would describe a
    # discrimination incident that is really a sample size.
    #
    # Two conditions, and they guard different failures. The per-group window
    # supplies the RATE, so a shifted arrival mix cannot erase a minority
    # group's series. Presence in the shared window decides whether the group is
    # published at all, so a group that has genuinely stopped arriving still
    # ages out -- a frozen gauge would keep FairnessGapExceeded firing against
    # traffic that no longer exists, and the on-call cannot tell that from a
    # real gap.
    still_arriving = {group for _, _, group in _window}
    published = {
        group
        for group, decisions in _group_windows.items()
        if len(decisions) >= SELECTION_RATE_MIN_SAMPLES and group in still_arriving
    }
    for group in published:
        decisions = _group_windows[group]
        SELECTION_RATE.labels(group=group).set(
            sum(1 for decision in decisions if decision == INTERVENE) / len(decisions)
        )

    # A group that ages out of the window -- or shrinks below the floor -- has to
    # lose its series, not freeze at its last value: a stale gauge keeps
    # FairnessGapExceeded firing against a group nobody has sent a request for in
    # an hour, and the on-call has no way to tell that from a real gap.
    for gone in _known_groups - published:
        SELECTION_RATE.remove(gone)
    _known_groups = published

    # Drop the private window too once a group has left the shared one. Without
    # this the dict is an unbounded structure in a long-lived process -- the
    # same leak the bounded deques exist to prevent, one level up.
    for gone in [group for group in _group_windows if group not in still_arriving]:
        del _group_windows[gone]


def observe_prediction(probability: float, decision: str, group: str) -> None:
    """Record one served prediction and refresh the rolling share gauges.

    `group` is a protected-attribute bucket (a SEX code), never an identifier.

    This function is the sole owner of `credit_predictions_total`. A handler that
    also increments it doubles every rate, throughput and error-ratio number
    derived from it -- and because both halves move together nothing looks wrong;
    the service simply reports twice the traffic it is serving, and HighErrorRate
    needs twice the real error rate to fire.
    """
    PREDICTIONS.labels(decision=decision).inc()
    with _lock:
        _window.append((float(probability), decision, group))
        window = _group_windows.get(group)
        if window is None:
            window = _group_windows[group] = deque(maxlen=settings.risk_share_window)
        window.append(decision)
        _refresh_share_gauges()


def set_decision_threshold(threshold: float) -> None:
    """Count the high-risk share against `threshold` from now on.

    Recomputed at once, so a model swap cannot leave the gauge describing the
    previous cutoff until the next request happens to arrive.
    """
    global _decision_threshold
    with _lock:
        _decision_threshold = float(threshold)
        _refresh_share_gauges()


def decision_threshold() -> float:
    """The cutoff the high-risk share is currently counted against."""
    return _decision_threshold


def set_baseline_high_risk_share(share: float) -> None:
    """Publish the predicted-positive rate this model produced at registration.

    HighRiskShareShift compares the live window against this rather than against
    the 22.12% label prevalence, because the two answer different questions:
    prevalence is how many accounts default, this is how many the model flags.
    They coincide only for a near-perfect classifier. Installed from registry
    metadata in the same place the PSI baseline is, so a model swap moves both.
    """
    BASELINE_HIGH_RISK_SHARE.set(float(share))


def set_model_info(version: str, algo: str, trained_at: str, git_sha: str) -> None:
    """Publish which model is serving, replacing any previously reported one."""
    # Cleared first: after a hot reload the old version would keep exporting 1
    # alongside the new one, and an incident review would find two models both
    # claiming to be in production.
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=version, algo=algo, trained_at=trained_at, git_sha=git_sha).set(1)


# ----------------------------------------------------------- exposition


def render() -> tuple[bytes, str]:
    """The /metrics payload and the content type Prometheus expects with it."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def series_count() -> int:
    """Distinct label combinations exported across the `credit_*` metrics.

    Histogram buckets are counted once per label set rather than once per bucket:
    the bucket lists are fixed at import time and cannot grow with traffic,
    whereas a label value can. This is the number that went from 23 to 639 in
    Lab 04 the day someone labelled by account id, so this is the number a test
    holds down.
    """
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for metric in REGISTRY.collect():
        if not metric.name.startswith(METRIC_PREFIX):
            continue
        for sample in metric.samples:
            labels = tuple(sorted((k, v) for k, v in sample.labels.items() if k != "le"))
            seen.add((metric.name, labels))
    return len(seen)


def reset_windows() -> None:
    """Drop all rolling state and the gauges derived from it.

    Only tests call this. A long-lived server has no reason to forget its window;
    it exists so one test case cannot leak its window into the next.
    """
    global _feature_obs, _known_groups
    with _lock:
        _window.clear()
        _group_windows.clear()
        for window in _feature_windows.values():
            window.clear()
        _feature_obs = 0
        _known_groups = set()
        SELECTION_RATE.clear()
        FEATURE_PSI.clear()
        # Re-derive rather than hand-zero: one function decides what an empty
        # window means, so the reset path cannot drift from the serving path.
        _refresh_share_gauges()
