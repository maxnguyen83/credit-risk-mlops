"""Request and response contracts for the serving API.

These classes are the published interface: Swagger renders their descriptions,
the smoke test asserts their field names, and a downstream core-banking job
parses their JSON. Changing a field name here is an API break, not a rename.

Bounds come from `schema.py` rather than from literals, so a correction to the
data contract cannot leave the API validating against the old one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from credit_risk import schema
from credit_risk.config import settings

Decision = Literal["intervene", "monitor"]
RiskBand = Literal["low", "medium", "high"]


def utc_now_iso() -> str:
    """Timestamp in the shape an auditor expects: UTC, whole seconds, trailing Z."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# The six statement columns run newest-first: index 1 is the most recent month.
_WHEN: Final[tuple[str, ...]] = (
    "last month",
    "two months ago",
    "three months ago",
    "four months ago",
    "five months ago",
    "six months ago",
)

_PAY_DESC: Final = (
    "Repayment status {when}: -2 no consumption, -1 paid in full, 0 revolving "
    "credit, 1-8 months overdue."
)
_BILL_DESC: Final = (
    "Statement balance {when}, in NT$. Negative is legitimate -- it means the "
    "account was in credit after an overpayment."
)
_PAY_AMT_DESC: Final = "Amount actually repaid {when}, in NT$."

# A real account shape, not zeros: utilisation near the limit, two months in
# arrears, repayments trailing the statements. Someone reading /docs should be
# able to paste this and get an interesting answer back.
EXAMPLE_APPLICATION: Final[dict[str, Any]] = {
    "account_id": "A-10023",
    "LIMIT_BAL": 120000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 39,
    "PAY_1": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 113500.0,
    "BILL_AMT2": 110200.0,
    "BILL_AMT3": 107800.0,
    "BILL_AMT4": 104300.0,
    "BILL_AMT5": 99900.0,
    "BILL_AMT6": 95100.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 4000.0,
    "PAY_AMT3": 3800.0,
    "PAY_AMT4": 5000.0,
    "PAY_AMT5": 4500.0,
    "PAY_AMT6": 4200.0,
}


# Response examples are captured, not composed: each one is what this API
# returned for docs/examples/high_risk.json with registry version 2 loaded.
# Only served_at and request_id change from one call to the next. The two
# contribution lists are cut to their first three entries; the API returns ten.
EXAMPLE_PREDICTION: Final[dict[str, Any]] = {
    "account_id": None,
    "default_probability": 0.7369284676988783,
    "decision": "intervene",
    "risk_band": "high",
    "threshold_used": 0.5,
    "threshold_policy": "base",
    "model_name": "credit-risk",
    "model_version": "2",
    "served_at": "2026-10-03T04:18:41Z",
    "request_id": "b6efbb2d31d946269a3643ac296bf048",
}

EXAMPLE_EXPLANATION: Final[dict[str, Any]] = {
    "account_id": None,
    "default_probability": 0.7369284676988783,
    "decision": "intervene",
    "risk_band": "high",
    "threshold_used": 0.5,
    "threshold_policy": "base",
    "shap": {
        "base_value": -1.515763556521393,
        "contributions": [
            {"feature": "months_since_last_delinquency", "value": 1.0, "shap": 0.8741883328146505},
            {"feature": "PAY_1", "value": 2.0, "shap": 0.5591900821099798},
            {"feature": "total_bill_amt", "value": 7704.0, "shap": 0.276083116432872},
        ],
    },
    "lime": {
        "contributions": [
            {"feature": "months_since_last_delinquency <= 2.00", "weight": 0.11969707657229522},
            {"feature": "total_pay_amt <= 6234.50", "weight": 0.08349350240368114},
            {"feature": "utilization_m5 <= 0.01", "weight": 0.06519232350168992},
        ]
    },
    "top_reasons": [
        "The time since the last missed payment (1) increased the estimated risk of default.",
        "The repayment status last month (2) increased the estimated risk of default.",
        "The total billed over six months (7704) increased the estimated risk of default.",
    ],
    "agreement": {"top3_overlap": 1, "note": "SHAP and LIME agree on 1 of the top 3 drivers."},
    "model_name": "credit-risk",
    "model_version": "2",
    "served_at": "2026-10-03T04:18:42Z",
    "request_id": "767f002a55a746e8bbb9a631d8ac27a5",
}

# The report as an idle service gives it: no group has enough recent requests
# for a selection rate, so the rates, the gap and the verdict are all null
# rather than a misleading 0.
EXAMPLE_FAIRNESS_REPORT: Final[dict[str, Any]] = {
    "protected_attribute": "SEX",
    "threshold_policy": "base",
    "groups": [
        {"group": "sex_male", "threshold": 0.5, "selection_rate": None},
        {"group": "sex_female", "threshold": 0.5, "selection_rate": None},
    ],
    "demographic_parity_difference": None,
    "max_demographic_parity_difference": 0.05,
    "max_equalized_odds_difference": 0.08,
    "within_gate": None,
    "note": (
        "Selection rates are measured over the service's recent request window, not "
        "over a labelled test set. Equalized odds cannot be computed online because "
        "outcomes are only known a month later; the registered model passed a 0.08 "
        "gate offline."
    ),
    "model_name": "credit-risk",
    "model_version": "2",
    "generated_at": "2026-10-03T04:18:41Z",
    "request_id": "abb82478fdce45ed98805075fe040e62",
}

# Captured like the others, except `run_id`, `threshold` and `threshold_source`,
# which were added afterwards and show what this code reports for version 2:
# its run id is the one the registry records, and version 2 was registered
# before versions carried a threshold tag, so until the tag is filled in
# (`python -m credit_risk.models.registry tag-threshold --version 2`) it decides
# at the configured 0.5 and says so.
EXAMPLE_HEALTH: Final[dict[str, Any]] = {
    "status": "ok",
    "model_loaded": True,
    "model_name": "credit-risk",
    "model_version": "2",
    "run_id": "3914189dcbc645808759fd94c2900e5f",
    "algo": "LGBMClassifier",
    "trained_at": "unknown",
    "api_version": "1.0.0",
    "threshold_policy": "base",
    "threshold": 0.5,
    "threshold_source": "fallback",
    "uptime_seconds": 0.621,
    "detail": None,
}


class _ApiModel(BaseModel):
    """Base for responses that report which model answered.

    `model_name` and `model_version` collide with pydantic's reserved `model_`
    namespace. The field names are fixed by the published contract, so the
    namespace guard is the thing that gives way -- not the API.
    """

    model_config = ConfigDict(protected_namespaces=())


class CreditApplication(BaseModel):
    """One cardholder-month, in the shape the core-banking extract emits."""

    # extra="forbid": a core-banking job that still sends the raw `PAY_0`
    # column should be told so, not silently scored on a missing feature.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_APPLICATION]},
    )

    account_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Opaque account reference, echoed back on the response. Never a model "
            "feature and never a metric label."
        ),
    )
    LIMIT_BAL: float = Field(
        ...,
        gt=0,
        description="Approved credit limit in NT$, including any supplementary cards.",
    )
    SEX: int = Field(
        ...,
        ge=1,
        le=2,
        description=(
            "1 = male, 2 = female. Protected attribute: measured and monitored, "
            "never offered as a reason for a decision."
        ),
    )
    EDUCATION: int = Field(
        ...,
        ge=0,
        le=6,
        description=(
            "1 = graduate school, 2 = university, 3 = high school, 4 = other. "
            "Codes 0, 5 and 6 occur in the source extract and are folded into 'other'."
        ),
    )
    MARRIAGE: int = Field(
        ...,
        ge=0,
        le=3,
        description=(
            "1 = married, 2 = single, 3 = other. Code 0 occurs in the source "
            "extract and is folded into 'other'."
        ),
    )
    AGE: int = Field(
        ...,
        ge=schema.AGE_MIN,
        le=schema.AGE_MAX,
        description=(
            f"Age in years. Outside {schema.AGE_MIN}-{schema.AGE_MAX} the model has "
            "seen no examples, so the record is refused rather than extrapolated."
        ),
    )
    PAY_1: int = Field(
        ..., ge=schema.PAY_MIN, le=schema.PAY_MAX, description=_PAY_DESC.format(when=_WHEN[0])
    )
    PAY_2: int = Field(
        ..., ge=schema.PAY_MIN, le=schema.PAY_MAX, description=_PAY_DESC.format(when=_WHEN[1])
    )
    PAY_3: int = Field(
        ..., ge=schema.PAY_MIN, le=schema.PAY_MAX, description=_PAY_DESC.format(when=_WHEN[2])
    )
    PAY_4: int = Field(
        ..., ge=schema.PAY_MIN, le=schema.PAY_MAX, description=_PAY_DESC.format(when=_WHEN[3])
    )
    PAY_5: int = Field(
        ..., ge=schema.PAY_MIN, le=schema.PAY_MAX, description=_PAY_DESC.format(when=_WHEN[4])
    )
    PAY_6: int = Field(
        ..., ge=schema.PAY_MIN, le=schema.PAY_MAX, description=_PAY_DESC.format(when=_WHEN[5])
    )
    BILL_AMT1: float = Field(..., description=_BILL_DESC.format(when=_WHEN[0]))
    BILL_AMT2: float = Field(..., description=_BILL_DESC.format(when=_WHEN[1]))
    BILL_AMT3: float = Field(..., description=_BILL_DESC.format(when=_WHEN[2]))
    BILL_AMT4: float = Field(..., description=_BILL_DESC.format(when=_WHEN[3]))
    BILL_AMT5: float = Field(..., description=_BILL_DESC.format(when=_WHEN[4]))
    BILL_AMT6: float = Field(..., description=_BILL_DESC.format(when=_WHEN[5]))
    PAY_AMT1: float = Field(..., ge=0, description=_PAY_AMT_DESC.format(when=_WHEN[0]))
    PAY_AMT2: float = Field(..., ge=0, description=_PAY_AMT_DESC.format(when=_WHEN[1]))
    PAY_AMT3: float = Field(..., ge=0, description=_PAY_AMT_DESC.format(when=_WHEN[2]))
    PAY_AMT4: float = Field(..., ge=0, description=_PAY_AMT_DESC.format(when=_WHEN[3]))
    PAY_AMT5: float = Field(..., ge=0, description=_PAY_AMT_DESC.format(when=_WHEN[4]))
    PAY_AMT6: float = Field(..., ge=0, description=_PAY_AMT_DESC.format(when=_WHEN[5]))


class PredictResponse(_ApiModel):
    """The scoring answer, including the policy that produced the decision."""

    model_config = ConfigDict(
        protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_PREDICTION]}
    )

    account_id: str | None = Field(default=None, description="Echoed from the request.")
    default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated P(default next month)."
    )
    decision: Decision = Field(
        ..., description="'intervene' puts the account on the monthly outreach list."
    )
    risk_band: RiskBand = Field(
        ..., description="Band relative to the portfolio base rate, for triage."
    )
    threshold_used: float = Field(..., description="The cutoff this record was judged against.")
    threshold_policy: str = Field(
        ...,
        description=(
            "'base' = one cutoff for everyone; 'group_aware_equalized_odds' = the "
            "per-group cutoffs fitted by the mitigation step."
        ),
    )
    model_name: str
    model_version: str
    served_at: str = Field(default_factory=utc_now_iso)
    request_id: str = Field(..., description="Correlates this response with the server logs.")


class BatchPredictRequest(BaseModel):
    """A nightly portfolio slice."""

    applications: list[CreditApplication] = Field(
        ...,
        min_length=1,
        max_length=settings.max_batch_size,
        description=(
            f"Up to {settings.max_batch_size} accounts. The cap is enforced here so an "
            "oversized batch fails fast at validation instead of after the features "
            "have been built."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"applications": [EXAMPLE_APPLICATION]}]}
    )


class BatchPredictResponse(_ApiModel):
    """Scores for the slice, plus the two counts the risk desk actually reads."""

    count: int
    intervene_count: int = Field(..., description="How many accounts crossed the threshold.")
    high_risk_count: int = Field(..., description="How many landed in the 'high' band.")
    predictions: list[PredictResponse]
    model_name: str
    model_version: str
    served_at: str = Field(default_factory=utc_now_iso)
    request_id: str


class ShapContribution(BaseModel):
    feature: str
    value: float | None = Field(default=None, description="The observed feature value.")
    shap: float = Field(..., description="Signed push on the log-odds. Positive = riskier.")


class ShapExplanation(BaseModel):
    base_value: float = Field(..., description="The model's output for an average account.")
    contributions: list[ShapContribution]


class LimeContribution(BaseModel):
    feature: str = Field(..., description="A discretised condition, e.g. 'PAY_1 > 1.00'.")
    weight: float


class LimeExplanation(BaseModel):
    contributions: list[LimeContribution]


class Agreement(BaseModel):
    """How far the two explainers tell the same story."""

    top3_overlap: int = Field(..., ge=0, description="Shared features among each method's top 3.")
    note: str


class ExplainResponse(_ApiModel):
    """SHAP, LIME, their overlap, and a draft adverse-action notice."""

    model_config = ConfigDict(
        protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_EXPLANATION]}
    )

    account_id: str | None = None
    default_probability: float = Field(..., ge=0.0, le=1.0)
    # The decision travels with the explanation. An adverse-action notice that
    # does not say what was decided is a list of numbers, and a reviewer cannot
    # check a reason against a decision they cannot see.
    decision: Decision
    risk_band: RiskBand
    threshold_used: float = Field(..., ge=0.0, le=1.0)
    threshold_policy: str
    shap: ShapExplanation
    lime: LimeExplanation
    top_reasons: list[str] = Field(
        ...,
        description=(
            "Plain-English drivers of the score, largest first. This is a draft "
            "adverse-action notice and requires human review before it is sent."
        ),
    )
    agreement: Agreement
    model_name: str
    model_version: str
    served_at: str = Field(default_factory=utc_now_iso)
    request_id: str


class GroupFairness(BaseModel):
    group: str
    threshold: float = Field(..., description="The cutoff currently applied to this group.")
    selection_rate: float | None = Field(
        default=None,
        description="Share of recent requests from this group that were flagged.",
    )


class FairnessReportResponse(_ApiModel):
    """Fairness of the model that is serving right now, not of an offline run."""

    model_config = ConfigDict(
        protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_FAIRNESS_REPORT]}
    )

    protected_attribute: str
    threshold_policy: str
    groups: list[GroupFairness]
    demographic_parity_difference: float | None = Field(
        default=None, description="Widest observed gap in selection rate. Null if under two groups."
    )
    max_demographic_parity_difference: float = Field(
        ..., description="The registration gate the model passed offline."
    )
    max_equalized_odds_difference: float = Field(
        ..., description="The registration gate the model passed offline."
    )
    within_gate: bool | None = Field(
        default=None,
        description="Whether the live gap is still inside the demographic-parity gate.",
    )
    note: str
    model_name: str
    model_version: str
    generated_at: str = Field(default_factory=utc_now_iso)
    request_id: str


class HealthResponse(_ApiModel):
    """Not 'the port answered' -- which model answered, and why it might not."""

    model_config = ConfigDict(
        protected_namespaces=(), json_schema_extra={"examples": [EXAMPLE_HEALTH]}
    )

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    model_version: str
    run_id: str = Field(..., description="The MLflow run the serving version was trained in.")
    algo: str
    trained_at: str
    api_version: str
    threshold_policy: str
    # Unbounded on purpose: /health reports, it does not validate. A bound here
    # would turn a misconfigured DECISION_THRESHOLD into a 500 from the one
    # endpoint that has to keep answering.
    threshold: float = Field(
        ..., description="The base cutoff: a probability at or above it is 'intervene'."
    )
    threshold_source: Literal["registry", "env", "fallback"] = Field(
        ...,
        description=(
            "'registry' = the threshold_at_k tag on the loaded version, the cutoff that "
            "admits the intervention capacity on its evaluation split; 'env' = "
            "DECISION_THRESHOLD, because THRESHOLD_SOURCE=env; 'fallback' = "
            "DECISION_THRESHOLD, because the loaded version carries no usable tag."
        ),
    )
    uptime_seconds: float
    detail: str | None = Field(default=None, description="Why the model is missing, if it is.")


class VersionResponse(_ApiModel):
    """What is deployed, in the terms an incident review needs."""

    api_title: str
    api_version: str
    model_name: str
    model_version: str
    algo: str
    trained_at: str
    git_sha: str
    threshold_policy: str


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns."""

    code: str = Field(..., description="Stable machine-readable reason, e.g. 'model_not_loaded'.")
    message: str = Field(..., description="One sentence a human can act on.")
    detail: str | None = Field(
        default=None,
        description="Field names and rules that failed. Never the submitted values.",
    )
    request_id: str


class ApiError(Exception):
    """A failure the API already understands; rendered as ErrorResponse by the app.

    The `credit_errors_total{reason}` counter is incremented where the error is
    raised rather than here, because the raise site is the only place that knows
    what actually broke.
    """

    def __init__(
        self, status_code: int, code: str, message: str, detail: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
