"""The versioned HTTP surface: score, explain, and report on fairness.

Every handler follows the same spine -- validate, build features through the
one shared builder, predict, apply the threshold policy, record metrics,
answer -- so a reader who understands /predict understands all four.

Handlers are plain `def`, not `async def`. Scoring and SHAP are CPU-bound and
blocking; declared synchronously FastAPI runs them in a worker thread instead
of parking the event loop, which is the difference between a p95 of 40 ms and
a p95 that tracks the slowest request in flight.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from functools import lru_cache
from time import perf_counter
from types import MappingProxyType
from typing import Any, Final
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Request

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.explain import lime_explainer, shap_explainer
from credit_risk.explain.shap_explainer import ExplainerUnavailable
from credit_risk.serving import metrics as m
from credit_risk.serving import model_loader
from credit_risk.serving.models import (
    Agreement,
    ApiError,
    BatchPredictRequest,
    BatchPredictResponse,
    CreditApplication,
    Decision,
    ErrorResponse,
    ExplainResponse,
    FairnessReportResponse,
    GroupFairness,
    LimeExplanation,
    PredictResponse,
    RiskBand,
    ShapExplanation,
    utc_now_iso,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Pinned to 3: the published response schema names the field `top3_overlap`,
# and the adverse-action drafts in the spec carry three reasons.
TOP_K: Final = 3

# The full contribution list is one entry per engineered feature. Ten is what a
# risk officer reads; the rest is noise in a JSON payload.
EXPLAIN_TOP_FEATURES: Final = 10

# Anything at or above this is "high" regardless of the decision threshold;
# below the portfolio base rate is "low". Anchoring the middle band on the
# measured base rate means "medium" says "worse than the average account in
# this book" rather than naming an arbitrary third of the number line.
HIGH_RISK_PROBABILITY: Final = 0.50

GROUP_AWARE_POLICY: Final = "group_aware_equalized_odds"

ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    422: {"model": ErrorResponse, "description": "The record did not match the schema."},
    500: {"model": ErrorResponse, "description": "Unexpected failure; see request_id in the logs."},
    503: {"model": ErrorResponse, "description": "No model is loaded; the service is degraded."},
}


# --------------------------------------------------------------- helpers


def request_id(request: Request) -> str:
    """The id that ties a response, a log line and an alert together."""
    existing = getattr(request.state, "request_id", None)
    return str(existing) if existing else uuid4().hex


def risk_band(probability: float) -> RiskBand:
    """Triage band for a probability."""
    if probability >= HIGH_RISK_PROBABILITY:
        return "high"
    if probability >= schema.BASE_POSITIVE_RATE:
        return "medium"
    return "low"


def decide(probability: float, threshold: float) -> Decision:
    """The only place a probability becomes an action."""
    return "intervene" if probability >= threshold else "monitor"


def group_label(record: Mapping[str, Any]) -> str:
    """The protected-group label used on metrics.

    Deliberately coarse. Labelling by account_id would leak a customer into the
    monitoring stack and blow up Prometheus cardinality at the same time -- one
    bug, two incident reports.
    """
    try:
        code = int(record[schema.SEX])
    except (KeyError, TypeError, ValueError):
        return "sex_unknown"
    return f"sex_{schema.SEX_CODES.get(code, 'unknown')}"


# Serving cannot refit a ThresholdOptimizer -- that needs labels and fairlearn
# -- so the fitted per-group cutoffs have to travel as data. With no mapping
# available every group is judged by the base cutoff, which is the safe default
# rather than a policy applied to half the portfolio.
_INJECTED_THRESHOLDS: dict[Any, float] | None = None


def set_group_thresholds(thresholds: Mapping[Any, float] | None) -> None:
    """Install the per-group cutoffs directly; None clears them."""
    global _INJECTED_THRESHOLDS
    _INJECTED_THRESHOLDS = None if thresholds is None else dict(thresholds)
    reset_threshold_cache()


@lru_cache(maxsize=1)
def group_threshold_map() -> Mapping[Any, float]:
    """Per-group cutoffs fitted by the mitigation step, if there are any.

    Cached because the thresholds change only when a model does -- the lifespan
    clears it. The read itself belongs to `fairness.mitigation`, which owns both
    the filename and the code that writes it; a second copy of that path here
    would be one rename away from serving falling back to the base cutoff with
    nothing in the logs to say the handoff had been missed.
    """
    if _INJECTED_THRESHOLDS is not None:
        return MappingProxyType(dict(_INJECTED_THRESHOLDS))

    # Imported inside the function: mitigation pulls in fairlearn, and the
    # default policy never asks for a per-group cutoff at all.
    from credit_risk.fairness.mitigation import load_group_thresholds

    return MappingProxyType(load_group_thresholds())


def reset_threshold_cache() -> None:
    """Forget the fitted thresholds. Called whenever the served model changes."""
    group_threshold_map.cache_clear()


def base_threshold() -> float:
    """The one cutoff for everyone: the loaded version's own, or the configured fallback.

    Read off the model holder rather than settings. Registration tags each
    version with the cutoff that admits the intervention capacity on its
    evaluation split, so the number a decision is measured against travels with
    the model it was computed for; `/health` says which of the two this is.
    """
    return model_loader.MODEL.threshold


def applied_threshold(record: Mapping[str, Any]) -> tuple[float, str]:
    """The cutoff this record is judged against, and the policy that chose it."""
    policy = settings.threshold_policy
    if policy != GROUP_AWARE_POLICY:
        return base_threshold(), policy

    thresholds = group_threshold_map()
    key: Any = record.get(schema.SEX)
    candidates: list[Any] = [key, str(key)]
    with suppress(KeyError, TypeError, ValueError):
        candidates.append(schema.SEX_CODES[int(key)])
    for candidate in candidates:
        if candidate in thresholds:
            return float(thresholds[candidate]), policy

    # Falling back to one cutoff for everyone is the safe failure: it is the
    # status quo, not a new and unreviewed form of differential treatment.
    log.warning("no group threshold for the requested group; falling back to the base cutoff")
    return base_threshold(), "base"


def feature_row(frame: pd.DataFrame, index: int = 0) -> dict[str, float]:
    """One row of features as plain floats, for the PSI bookkeeping."""
    row: dict[str, float] = {}
    for name, value in frame.iloc[index].items():
        try:
            row[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return row


def require_model() -> Any:
    """The loaded estimator, or a 503 that says why there isn't one."""
    holder = model_loader.MODEL
    if not holder.is_loaded:
        m.ERRORS.labels(reason="model_not_loaded").inc()
        raise ApiError(
            503,
            "model_not_loaded",
            "No model is loaded; the service is running degraded.",
            holder.last_error,
        )
    return holder.model


def observed_selection_rates() -> dict[str, float]:
    """Read the live selection-rate gauge back out, per group.

    Reported from the same gauge Prometheus scrapes rather than from a second
    counter, so /fairness/report and the FairnessGapExceeded alert can never
    disagree about what the service has been doing.
    """
    rates: dict[str, float] = {}
    for metric in m.SELECTION_RATE.collect():
        for sample in metric.samples:
            group = sample.labels.get("group")
            if group is not None:
                rates[str(group)] = float(sample.value)
    return rates


def _record_prediction(
    record: Mapping[str, Any], frame: pd.DataFrame, index: int, probability: float
) -> tuple[Decision, RiskBand, float, str]:
    """Apply the policy to one row and publish what happened. Returns the outcome."""
    threshold, policy = applied_threshold(record)
    decision = decide(probability, threshold)
    band = risk_band(probability)
    group = group_label(record)

    # credit_predictions_total is incremented by observe_prediction and nowhere
    # else. Counting it here as well doubled it, and HighErrorRate divides by
    # it -- a 10% error rate read as 5% and never crossed the 5% threshold.
    m.observe_prediction(probability, decision, group)
    m.observe_features(feature_row(frame, index))
    return decision, band, threshold, policy


# ---------------------------------------------------------------- routes


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=ERROR_RESPONSES,
    tags=["inference"],
    summary="Score one account for default next month",
)
def predict(payload: CreditApplication, request: Request) -> PredictResponse:
    """Score a single account and return the decision with its policy."""
    started = perf_counter()
    rid = request_id(request)
    model = require_model()
    record = payload.model_dump(exclude={"account_id"})

    try:
        frame = model_loader.to_feature_frame(record)
        probability = float(model_loader.predict_probability(model, frame)[0])
    except Exception as exc:
        m.ERRORS.labels(reason="prediction_failed").inc()
        raise ApiError(500, "prediction_failed", "Could not score this record.", str(exc)) from exc

    decision, band, threshold, policy = _record_prediction(record, frame, 0, probability)
    elapsed = perf_counter() - started
    m.LATENCY.labels(endpoint="/predict").observe(elapsed)

    # The payload is credit data and never reaches a log at INFO. The
    # request_id plus the derived outcome is enough to reconstruct a decision
    # from the model and the stored record, without copying a customer into
    # every log sink the stack ships to.
    log.info(
        "predict request_id=%s decision=%s band=%s ms=%.1f model=%s",
        rid,
        decision,
        band,
        elapsed * 1000.0,
        model_loader.MODEL.version,
    )

    return PredictResponse(
        account_id=payload.account_id,
        default_probability=probability,
        decision=decision,
        risk_band=band,
        threshold_used=threshold,
        threshold_policy=policy,
        model_name=settings.model_name,
        model_version=model_loader.MODEL.version,
        request_id=rid,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses=ERROR_RESPONSES,
    tags=["inference"],
    summary="Score a portfolio slice in one call",
)
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Score up to `max_batch_size` accounts; the nightly core-banking path."""
    started = perf_counter()
    rid = request_id(request)
    model = require_model()
    records = [item.model_dump(exclude={"account_id"}) for item in payload.applications]

    try:
        frame = model_loader.to_feature_frame_many(records)
        probabilities = model_loader.predict_probability(model, frame)
    except Exception as exc:
        m.ERRORS.labels(reason="prediction_failed").inc()
        raise ApiError(500, "prediction_failed", "Could not score this batch.", str(exc)) from exc

    served_at = utc_now_iso()
    predictions: list[PredictResponse] = []
    for index, (application, record) in enumerate(zip(payload.applications, records, strict=True)):
        probability = float(probabilities[index])
        decision, band, threshold, policy = _record_prediction(record, frame, index, probability)
        predictions.append(
            PredictResponse(
                account_id=application.account_id,
                default_probability=probability,
                decision=decision,
                risk_band=band,
                threshold_used=threshold,
                threshold_policy=policy,
                model_name=settings.model_name,
                model_version=model_loader.MODEL.version,
                served_at=served_at,
                request_id=rid,
            )
        )

    elapsed = perf_counter() - started
    m.LATENCY.labels(endpoint="/predict/batch").observe(elapsed)
    intervene_count = sum(1 for item in predictions if item.decision == "intervene")
    high_risk_count = sum(1 for item in predictions if item.risk_band == "high")
    log.info(
        "batch request_id=%s n=%d intervene=%d ms=%.1f model=%s",
        rid,
        len(predictions),
        intervene_count,
        elapsed * 1000.0,
        model_loader.MODEL.version,
    )

    return BatchPredictResponse(
        count=len(predictions),
        intervene_count=intervene_count,
        high_risk_count=high_risk_count,
        predictions=predictions,
        model_name=settings.model_name,
        model_version=model_loader.MODEL.version,
        served_at=served_at,
        request_id=rid,
    )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    responses=ERROR_RESPONSES,
    tags=["inference"],
    summary="SHAP and LIME reasons behind one score",
)
def explain(payload: CreditApplication, request: Request) -> ExplainResponse:
    """Explain one decision with both methods and state how far they agree."""
    started = perf_counter()
    rid = request_id(request)
    model = require_model()
    record = payload.model_dump(exclude={"account_id"})

    try:
        frame = model_loader.to_feature_frame(record)
        probability = float(model_loader.predict_probability(model, frame)[0])
    except Exception as exc:
        m.ERRORS.labels(reason="prediction_failed").inc()
        raise ApiError(500, "prediction_failed", "Could not score this record.", str(exc)) from exc

    try:
        mark = perf_counter()
        shap_result = shap_explainer.explain_local(
            record, model=model, features=frame, version=model_loader.MODEL.version
        )
        m.EXPLAIN_DURATION.labels(method="shap").observe(perf_counter() - mark)

        mark = perf_counter()
        lime_result = lime_explainer.explain_local(record, model=model, features=frame)
        m.EXPLAIN_DURATION.labels(method="lime").observe(perf_counter() - mark)
    except ExplainerUnavailable as exc:
        m.ERRORS.labels(reason="explainer_unavailable").inc()
        raise ApiError(
            503,
            "explainer_unavailable",
            "Explanations are not available for the model that is loaded.",
            str(exc),
        ) from exc
    except Exception as exc:
        m.ERRORS.labels(reason="explain_failed").inc()
        raise ApiError(500, "explain_failed", "Could not explain this record.", str(exc)) from exc

    overlap = lime_explainer.agreement(shap_result, lime_result, k=TOP_K)
    reasons = shap_explainer.top_reasons(shap_result["contributions"], k=TOP_K)

    # Same policy path /predict uses, so an explanation can never describe a
    # decision the scoring endpoint would not have made.
    threshold, policy = applied_threshold(record)
    decision = decide(probability, threshold)
    band = risk_band(probability)

    elapsed = perf_counter() - started
    m.LATENCY.labels(endpoint="/explain").observe(elapsed)
    log.info(
        "explain request_id=%s ms=%.1f model=%s",
        rid,
        elapsed * 1000.0,
        model_loader.MODEL.version,
    )

    return ExplainResponse(
        account_id=payload.account_id,
        default_probability=probability,
        decision=decision,
        risk_band=band,
        threshold_used=threshold,
        threshold_policy=policy,
        shap=ShapExplanation(
            base_value=shap_result["base_value"],
            contributions=shap_result["contributions"][:EXPLAIN_TOP_FEATURES],
        ),
        lime=LimeExplanation(contributions=lime_result["contributions"]),
        top_reasons=reasons,
        agreement=Agreement(
            top3_overlap=int(overlap[f"top{TOP_K}_overlap"]),
            note=str(overlap["note"]),
        ),
        model_name=settings.model_name,
        model_version=model_loader.MODEL.version,
        request_id=rid,
    )


@router.get(
    "/fairness/report",
    response_model=FairnessReportResponse,
    responses=ERROR_RESPONSES,
    tags=["responsible-ai"],
    summary="Fairness of the model that is serving right now",
)
def fairness_report(request: Request) -> FairnessReportResponse:
    """Live selection rates per protected group, against the registration gates."""
    started = perf_counter()
    rid = request_id(request)
    require_model()

    rates = observed_selection_rates()
    groups: list[GroupFairness] = []
    for code in sorted(schema.SEX_CODES):
        record = {schema.SEX: code}
        label = group_label(record)
        threshold, _ = applied_threshold(record)
        groups.append(
            GroupFairness(group=label, threshold=threshold, selection_rate=rates.get(label))
        )
    known = {item.group for item in groups}
    for label, rate in sorted(rates.items()):
        if label not in known:
            groups.append(
                GroupFairness(group=label, threshold=base_threshold(), selection_rate=rate)
            )

    observed = [item.selection_rate for item in groups if item.selection_rate is not None]
    dp_diff = max(observed) - min(observed) if len(observed) >= 2 else None
    within_gate = dp_diff <= schema.MAX_DEMOGRAPHIC_PARITY_DIFF if dp_diff is not None else None

    # Equalized odds needs realised outcomes, which arrive a month after the
    # decision. Reporting the offline gate here and the live parity gap beside
    # it is honest; computing an "online eo_diff" from nothing would not be.
    note = (
        "Selection rates are measured over the service's recent request window, not "
        "over a labelled test set. Equalized odds cannot be computed online because "
        f"outcomes are only known a month later; the registered model passed a "
        f"{schema.MAX_EQUALIZED_ODDS_DIFF} gate offline."
    )

    m.LATENCY.labels(endpoint="/fairness/report").observe(perf_counter() - started)
    return FairnessReportResponse(
        protected_attribute=schema.PRIMARY_PROTECTED,
        threshold_policy=settings.threshold_policy,
        groups=groups,
        demographic_parity_difference=dp_diff,
        max_demographic_parity_difference=schema.MAX_DEMOGRAPHIC_PARITY_DIFF,
        max_equalized_odds_difference=schema.MAX_EQUALIZED_ODDS_DIFF,
        within_gate=within_gate,
        note=note,
        model_name=settings.model_name,
        model_version=model_loader.MODEL.version,
        request_id=rid,
    )
