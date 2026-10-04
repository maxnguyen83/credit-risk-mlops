"""SHAP explanations: one per decision, plus the beeswarm for the model report.

TreeExplainer is exact for gradient-boosted trees and orders of magnitude
faster than KernelExplainer. That is the reason the production model is a tree
and not a network -- explainability was a requirement before it was a nicety.

The output of `top_reasons` is the draft adverse-action notice. It is a draft:
a human signs the letter, not this module.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import shap

from credit_risk import schema
from credit_risk.features.build import FEATURE_NAMES
from credit_risk.serving.model_loader import to_feature_frame

log = logging.getLogger(__name__)


class ExplainerUnavailable(RuntimeError):
    """This method cannot explain the model that is currently loaded."""


_EXPLAINER: Any | None = None
_EXPLAINER_FOR: str | None = None


def reset() -> None:
    """Forget the cached explainer. Called on shutdown and by tests."""
    global _EXPLAINER, _EXPLAINER_FOR
    _EXPLAINER = None
    _EXPLAINER_FOR = None


def _served_model() -> tuple[Any, str]:
    # Local import: explain/ should stay importable by the training DAG, which
    # has no serving stack and no model holder.
    from credit_risk.serving import model_loader

    return model_loader.MODEL.model, model_loader.MODEL.version


def get_explainer(model: Any = None, version: str | None = None) -> Any:
    """Build a TreeExplainer once per model version, then reuse it.

    Keyed on the version, not on the object: an explainer cached across a
    registry promotion will happily keep explaining the model that is no longer
    serving, and the explanations look entirely reasonable while it does. An
    id() key does not catch that -- CPython reuses the address of a freed
    estimator, so the new model can land exactly where the old one was.

    A caller that hands over the served estimator without naming its version
    therefore gets the served version filled in. The id() key survives only for
    the training DAG, which explains a model that has no registry version yet
    and holds a live reference to it for the length of the call.
    """
    global _EXPLAINER, _EXPLAINER_FOR

    served, served_version = _served_model()
    if model is None:
        model, version = served, served_version
    elif version is None and model is served:
        version = served_version
    if model is None:
        raise ExplainerUnavailable("no model is loaded, so there is nothing to explain")

    key = str(version) if version else f"id:{id(model)}"
    if _EXPLAINER is not None and key == _EXPLAINER_FOR:
        return _EXPLAINER

    try:
        explainer = shap.TreeExplainer(model)
    except Exception as exc:
        raise ExplainerUnavailable(
            f"TreeExplainer does not support {type(model).__name__}: {exc}"
        ) from exc

    _EXPLAINER = explainer
    _EXPLAINER_FOR = key
    return explainer


def shap_matrix(explainer: Any, frame: pd.DataFrame) -> np.ndarray:
    """SHAP values for the positive class as a plain (rows, features) matrix.

    shap returns a list-per-class for some model flavours and a 3-d array for
    others, and which one you get changes with the library version. Collapsing
    it here means no caller has to relearn that.
    """
    raw = explainer.shap_values(frame)
    if isinstance(raw, list):
        raw = raw[-1]
    values = np.asarray(raw, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def base_value(explainer: Any) -> float:
    """The model's output for an average account, as one float."""
    expected = getattr(explainer, "expected_value", 0.0)
    flat = np.asarray(expected, dtype=float).ravel()
    return float(flat[-1]) if flat.size else 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_contributions(frame: pd.DataFrame, values: np.ndarray) -> list[dict[str, Any]]:
    """Per-feature contributions for one row, largest magnitude first.

    Everything is cast through float(): numpy scalars are not JSON-serialisable
    and the failure surfaces as a 500 from the response encoder, a long way
    from the line that produced it.
    """
    row = np.asarray(values, dtype=float).ravel()
    observed = frame.iloc[0]
    contributions: list[dict[str, Any]] = [
        {"feature": str(name), "value": _as_float(observed.get(name)), "shap": float(row[index])}
        for index, name in enumerate(frame.columns)
        if index < row.size
    ]
    contributions.sort(key=lambda item: abs(float(item["shap"])), reverse=True)
    return contributions


def explain_local(
    record: Mapping[str, Any],
    *,
    model: Any = None,
    features: pd.DataFrame | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Explain one account. Pass `features` to avoid rebuilding them."""
    frame = features if features is not None else to_feature_frame(record)
    explainer = get_explainer(model, version)
    values = shap_matrix(explainer, frame)
    return {
        "base_value": base_value(explainer),
        "contributions": build_contributions(frame, values[0:1]),
    }


def explain_global(
    X: pd.DataFrame,
    out_path: Path | str,
    *,
    model: Any = None,
    max_display: int = 15,
) -> Path:
    """Write the beeswarm summary plot that goes into the model report."""
    # Agg before pyplot: the API and the DAG both run headless, and the default
    # backend raises on import when there is no display to attach to.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = X.loc[:, list(FEATURE_NAMES)] if set(FEATURE_NAMES).issubset(X.columns) else X
    values = shap_matrix(get_explainer(model), frame)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, frame, show=False, max_display=max_display, plot_type="dot")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("wrote global SHAP summary to %s", path)
    return path


# ------------------------------------------------------- adverse action text

# Feature name -> a noun phrase a customer would recognise. Keyed on the names
# the shared feature builder produces; anything unmapped falls back to a
# de-snake-cased version of the name, which is ugly but never wrong.
_PHRASES: Final[dict[str, str]] = {
    "limit_bal": "the approved credit limit",
    "available_credit": "the credit still available on the account",
    "utilization_mean": "average credit utilisation over six months",
    "utilization_max": "peak credit utilisation over six months",
    "utilization_trend": "the trend in credit utilisation over six months",
    "payment_ratio_mean": "the average share of each statement repaid",
    "payment_ratio_min": "the smallest share of a statement repaid",
    "payment_ratio_trend": "the trend in the share of each statement repaid",
    "payment_to_limit_ratio": "repayments measured against the credit limit",
    "months_delinquent": "the number of months in arrears",
    "max_consecutive_delinquent": "the longest run of consecutive months in arrears",
    "months_since_last_delinquency": "the time since the last missed payment",
    "worst_pay_status": "the worst repayment status in the last six months",
    "total_bill_amt": "the total billed over six months",
    "mean_bill_amt": "the average statement balance",
    "total_pay_amt": "the total repaid over six months",
    "mean_pay_amt": "the average amount repaid",
}

_MONTH_WORDS: Final[tuple[str, ...]] = (
    "last month",
    "two months ago",
    "three months ago",
    "four months ago",
    "five months ago",
    "six months ago",
)

# Protected attributes are model features -- "fairness through unawareness"
# was tested and failed, so dropping them would only hide the proxy. They must
# still never appear in a customer-facing reason: "your gender increased your
# risk" is not a lawful adverse-action notice in any jurisdiction this system
# would run in. They stay in `contributions` for the compliance officer, so
# nothing is hidden from the audit -- only from the letter.
PROTECTED_FEATURES: Final[frozenset[str]] = frozenset(
    name.lower() for name in (*schema.DEMOGRAPHIC_COLS, schema.AGE_GROUP)
)

_PAY_RE: Final = re.compile(r"^pay_([1-6])$")
_BILL_RE: Final = re.compile(r"^bill_amt([1-6])$")
_PAY_AMT_RE: Final = re.compile(r"^pay_amt([1-6])$")
_UTILIZATION_RE: Final = re.compile(r"^utilization_m([1-6])$")
_PAYMENT_RATIO_RE: Final = re.compile(r"^payment_ratio_m([1-6])$")

_PATTERNS: Final[tuple[tuple[Any, str], ...]] = (
    (_PAY_RE, "the repayment status {when}"),
    (_UTILIZATION_RE, "credit utilisation {when}"),
    (_PAYMENT_RATIO_RE, "the share of the statement repaid {when}"),
    (_BILL_RE, "the statement balance {when}"),
    (_PAY_AMT_RE, "the amount repaid {when}"),
)


def phrase_for(feature: str) -> str:
    """A human noun phrase for a feature name."""
    key = feature.strip().lower()
    if key in _PHRASES:
        return _PHRASES[key]
    for pattern, template in _PATTERNS:
        match = pattern.match(key)
        if match:
            return template.format(when=_MONTH_WORDS[int(match.group(1)) - 1])
    return feature.replace("_", " ").strip().lower()


def _format_value(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    if float(number).is_integer() and abs(number) < 1e6:
        return f"{number:.0f}"
    return f"{number:,.2f}"


def _upper_first(text: str) -> str:
    # Not .capitalize(): that lowercases the rest and mangles "NT$" and codes.
    return text[:1].upper() + text[1:]


def top_reasons(contributions: Sequence[Mapping[str, Any]], k: int = 3) -> list[str]:
    """Turn the largest risk-increasing contributions into sentences.

    Only positive contributions qualify: an adverse-action notice explains why
    the answer was adverse, so a feature that pushed the score *down* is not a
    reason for the decision even when its magnitude is large. Protected
    attributes are dropped for the reason given at PROTECTED_FEATURES.
    """
    usable = [
        item
        for item in contributions
        if (_as_float(item.get("shap")) or 0.0) > 0.0
        and str(item.get("feature", "")).lower() not in PROTECTED_FEATURES
    ]
    usable.sort(key=lambda item: abs(_as_float(item.get("shap")) or 0.0), reverse=True)
    if not usable:
        return [
            "No permissible factor increased the estimated risk; this account scores "
            "close to the portfolio average."
        ]

    sentences: list[str] = []
    for item in usable[: max(k, 0)]:
        phrase = phrase_for(str(item.get("feature", "")))
        value = item.get("value")
        if value is None:
            sentences.append(_upper_first(f"{phrase} increased the estimated risk of default."))
        else:
            sentences.append(
                _upper_first(
                    f"{phrase} ({_format_value(value)}) increased the estimated risk of default."
                )
            )
    return sentences
