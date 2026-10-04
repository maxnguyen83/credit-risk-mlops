"""The FastAPI application: boot, cross-cutting concerns, and the ops endpoints.

Three things live here and nowhere else: the lifespan that decides what this
process is serving, the request-id that stitches a response to a log line, and
the exception handlers that guarantee every failure leaves the building in the
same shape.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Final
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.explain import lime_explainer, shap_explainer
from credit_risk.features.build import FEATURE_NAMES
from credit_risk.serving import metrics as m
from credit_risk.serving import model_loader, routes
from credit_risk.serving.models import ApiError, ErrorResponse, HealthResponse, VersionResponse

log = logging.getLogger(__name__)

_STARTED_AT = time.monotonic()

TAGS_METADATA: Final[list[dict[str, Any]]] = [
    {"name": "inference", "description": "Score accounts and explain the scores."},
    {
        "name": "responsible-ai",
        "description": "Fairness of the model that is actually serving traffic.",
    },
    {"name": "ops", "description": "Health, version and Prometheus exposition."},
]

DESCRIPTION: Final = (
    "Early-warning scoring for an existing credit-card portfolio. Ranks accounts "
    "for the monthly intervention list; it does not approve or decline anyone, and "
    "a human makes the final call on every account it flags."
)

# The columns credit_feature_psi watches, and the whole label set of that gauge.
# Chosen rather than taken off the front of FEATURE_NAMES: the front of that
# tuple is LIMIT_BAL, the demographics and PAY_1..PAY_6, and a covariate shift
# lands on the monetary aggregates. Baselining the repayment codes would leave
# the gauge flat through exactly the scenario FeatureDriftHigh exists to catch,
# because a -2..8 ordinal cannot move the way a balance can.
#
# Eleven names, against the MAX_PSI_FEATURES cap of twelve, so adding one more
# is a decision someone makes rather than a silent truncation.
DRIFT_WATCH_FEATURES: Final[tuple[str, ...]] = (
    schema.LIMIT_BAL,
    "available_credit",
    "utilization_mean",
    "utilization_max",
    "payment_ratio_mean",
    "payment_ratio_min",
    "payment_to_limit_ratio",
    "total_bill_amt",
    "mean_bill_amt",
    "total_pay_amt",
    "mean_pay_amt",
)


def prime_reference_data() -> bool:
    """Install the training sample that /explain and the drift gauge both need.

    One sample serves both on purpose: the rows LIME perturbs around are the
    rows credit_feature_psi measures against, so the two can never end up
    describing different populations.

    Never raises, and the return value is for tests. Without the processed
    splits reachable there is no sample to be had, and a 503 from /explain plus
    an empty PSI gauge is a far better failure than a container that will not
    boot -- but it is a failure, so it is logged at WARNING with the reason,
    which is the only thing that distinguishes it from "nobody sent traffic".
    """
    try:
        frame = pd.DataFrame(lime_explainer.background(), columns=list(FEATURE_NAMES))
        m.set_baseline({name: frame[name].tolist() for name in DRIFT_WATCH_FEATURES})
    except Exception as exc:
        log.warning(
            "no reference sample (%s: %s): /explain will answer 503 and credit_feature_psi "
            "will export no series, so FeatureDriftHigh cannot fire",
            type(exc).__name__,
            exc,
        )
        return False

    log.info(
        "drift baseline installed over %d rows and %d features",
        len(frame),
        len(DRIFT_WATCH_FEATURES),
    )
    prime_high_risk_baseline(frame)
    return True


def prime_high_risk_baseline(frame: pd.DataFrame) -> float | None:
    """Score the reference sample and publish the share this model flags.

    HighRiskShareShift compares the live window against this number, so while it
    is zero the alert is armed against nothing and can never fire -- a silent
    hole that looks identical to "no drift has happened". Smoke tests check the
    gauge for exactly that reason.

    Measured rather than configured: the baseline is what *this* model version
    flags on the population it was fitted for, so promoting a differently
    calibrated version moves it automatically. A constant in settings would
    quietly describe the previous model.
    """
    model = model_loader.MODEL.model
    if model is None:
        return None
    try:
        proba = model_loader.predict_probability(model, frame)
        # The cutoff this model decides at, so the baseline describes the
        # decisions the live share is compared with.
        threshold = model_loader.MODEL.threshold
        share = float((proba >= threshold).mean())
    except Exception as exc:  # noqa: BLE001 - a missing baseline must not block boot
        log.warning(
            "could not score the reference sample (%s: %s): credit_baseline_high_risk_share "
            "stays 0 and HighRiskShareShift cannot fire",
            type(exc).__name__,
            exc,
        )
        return None

    m.set_baseline_high_risk_share(share)
    log.info("baseline high-risk share %.4f over %d reference rows", share, len(frame))
    return share


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the production model at boot and publish what happened either way."""
    loaded = model_loader.MODEL.load()
    routes.reset_threshold_cache()
    m.MODEL_LOADED.set(1.0 if loaded else 0.0)
    m.set_decision_threshold(model_loader.MODEL.threshold)

    if loaded:
        # set_model_info rather than MODEL_INFO.labels(...): it clears the gauge
        # first. Written directly, a reload leaves the previous version also
        # exporting 1, and credit_model_info is the one series an incident
        # review reads to answer "which model is running".
        m.set_model_info(
            version=model_loader.MODEL.version,
            algo=model_loader.MODEL.algo,
            trained_at=model_loader.MODEL.trained_at,
            git_sha=settings.git_sha,
        )
        prime_reference_data()
        log.info(
            "serving %s version=%s algo=%s threshold=%.4f threshold_source=%s",
            settings.model_name,
            model_loader.MODEL.version,
            model_loader.MODEL.algo,
            model_loader.MODEL.threshold,
            model_loader.MODEL.threshold_source,
        )
    else:
        # Boot regardless. A process that refuses to start cannot tell anyone
        # why it is unhealthy; a process that starts DEGRADED can answer
        # /health with the exception text and drive the ModelNotLoaded alert.
        log.warning("starting DEGRADED: %s", model_loader.MODEL.last_error)

    yield

    model_loader.MODEL.unload()
    m.set_decision_threshold(model_loader.MODEL.threshold)
    # The cached explainers hold the estimator that has just gone away. Dropped
    # here because CPython reuses the address of a freed object: a reload that
    # lands the new estimator on the old address would otherwise be handed back
    # an explainer built for the model that is no longer serving, with no error.
    shap_explainer.reset()
    lime_explainer.reset()
    m.MODEL_LOADED.set(0.0)
    m.MODEL_INFO.clear()


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give every request an id, honouring one supplied upstream."""
    rid = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    detail: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    rid = str(getattr(request.state, "request_id", "unknown"))
    body = ErrorResponse(code=code, message=message, detail=detail, request_id=rid)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers={**(headers or {}), "X-Request-ID": rid},
    )


def _summarise(exc: RequestValidationError) -> str:
    """Which fields failed and why -- never what was in them.

    FastAPI's default 422 body echoes the offending `input` back to the caller.
    On a credit API that puts a customer's balances into every client log and
    error tracker that touches the response, so the body is rebuilt here.
    """
    parts: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        parts.append(f"{location or 'body'}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """Render a failure the routes already classified and counted."""
    return _error(request, exc.status_code, exc.code, exc.message, exc.detail)


# Routing failures, raised by Starlette before any route runs. Without this
# handler they leave as FastAPI's default {"detail": "Not Found"}: a second error
# shape that every client of this API would have to parse.
_HTTP_ERRORS: Final[dict[int, tuple[str, str]]] = {
    404: ("not_found", "No such endpoint. The scoring API lives under /api/v1; see /docs."),
    405: ("method_not_allowed", "This endpoint exists but does not accept that method."),
}


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """404, 405 and any other HTTP error leave in the same shape as every failure.

    Not counted in credit_errors_total: a client probing a path that does not
    exist is not the service failing, and counting it would let a scanner drive
    HighErrorRate.
    """
    code, message = _HTTP_ERRORS.get(exc.status_code, ("http_error", str(exc.detail)))
    return _error(request, exc.status_code, code, message, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A rejected payload is both a log line and a metric."""
    m.ERRORS.labels(reason="validation").inc()
    return _error(
        request,
        422,
        "validation_error",
        "The request did not match the documented schema.",
        _summarise(exc),
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: no bare stack trace ever reaches a client."""
    rid = str(getattr(request.state, "request_id", "unknown"))
    m.ERRORS.labels(reason="internal").inc()
    log.exception("unhandled error request_id=%s", rid)
    return _error(
        request,
        500,
        "internal_error",
        "The service failed to process this request.",
        type(exc).__name__,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Health, plus which model is answering",
)
def health() -> HealthResponse:
    """Report the served model, not merely that the port is open.

    Returns 200 even when degraded, on purpose: a 503 here puts the container
    into a restart loop that destroys the very logs explaining the problem.
    `credit_model_loaded == 0` and the ModelNotLoaded alert are what wake
    someone up; this endpoint's job is to say why.

    It also says which cutoff decisions are made at and where it came from.
    `threshold_source: "fallback"` is a working service deciding at a configured
    constant because the loaded version carries no threshold tag.
    """
    holder = model_loader.MODEL
    return HealthResponse(
        status="ok" if holder.is_loaded else "degraded",
        model_loaded=holder.is_loaded,
        model_name=settings.model_name,
        model_version=holder.version,
        run_id=holder.run_id,
        algo=holder.algo,
        trained_at=holder.trained_at,
        api_version=settings.api_version,
        threshold_policy=settings.threshold_policy,
        threshold=holder.threshold,
        threshold_source=holder.threshold_source,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
        detail=holder.last_error,
    )


@app.get(
    "/metrics",
    tags=["ops"],
    summary="Prometheus exposition",
    response_class=Response,
)
def metrics() -> Response:
    """Hand the registry straight to Prometheus."""
    payload, content_type = m.render()
    return Response(content=payload, media_type=content_type)


@app.get(
    "/version",
    response_model=VersionResponse,
    tags=["ops"],
    summary="What is deployed right now",
)
def version() -> VersionResponse:
    """API build and model build in one place, for incident reviews."""
    holder = model_loader.MODEL
    return VersionResponse(
        api_title=settings.api_title,
        api_version=settings.api_version,
        model_name=settings.model_name,
        model_version=holder.version,
        algo=holder.algo,
        trained_at=holder.trained_at,
        git_sha=settings.git_sha,
        threshold_policy=settings.threshold_policy,
    )


app.include_router(routes.router)
