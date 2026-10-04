"""LIME explanations, and the honest comparison against SHAP.

Two explainers exist here because one is not enough to answer the obvious
challenge: *do your explanations agree?* `agreement()` answers it before it is
asked, with a number rather than a claim.

LIME is a local surrogate fitted on perturbed samples, so it is approximate
where SHAP is exact for this model. It is also fast and it discretises
features into readable conditions, which is why it is worth carrying.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.explain.shap_explainer import ExplainerUnavailable
from credit_risk.features.build import FEATURE_NAMES
from credit_risk.serving.model_loader import to_feature_frame, to_feature_frame_many

log = logging.getLogger(__name__)

CLASS_NAMES: Final[tuple[str, str]] = ("no_default", "default")
POSITIVE_LABEL: Final = 1

# lime's default is 5000 perturbations per explanation. 1000 keeps /explain
# inside its 500 ms p95 budget (the SlowExplanations alert). What the smaller
# sample costs in ranking stability has not been measured: the same record and
# seed always give the same answer (`reseed`, and the reproducibility test in
# tests/integration/test_api.py), but nothing yet compares the top 3 across
# seeds or against a 5000-sample run.
NUM_SAMPLES: Final = 1000

BACKGROUND_ROWS: Final = 500

_BACKGROUND: np.ndarray | None = None
_EXPLAINER: LimeTabularExplainer | None = None


def reset() -> None:
    """Drop the cached explainer and background. Called on shutdown and by tests."""
    global _BACKGROUND, _EXPLAINER
    _BACKGROUND = None
    _EXPLAINER = None


def set_background(data: pd.DataFrame | np.ndarray) -> None:
    """Inject the reference sample LIME perturbs around.

    Exists so the API can be exercised without the processed-data volume
    mounted, and so the training DAG can hand over the exact rows the model
    was fitted on instead of a re-read that might have drifted.
    """
    global _BACKGROUND, _EXPLAINER
    if isinstance(data, pd.DataFrame):
        columns = list(FEATURE_NAMES)
        frame = data.loc[:, columns] if set(columns).issubset(data.columns) else data
        _BACKGROUND = frame.to_numpy(dtype=float)
    else:
        _BACKGROUND = np.asarray(data, dtype=float)
    _EXPLAINER = None


def _background_file(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    root = settings.processed_dir
    candidates = sorted(root.glob("*.parquet"))
    # Glob rather than hardcode: the processed filenames belong to the data
    # track, and a rename there should not silently disable /explain.
    preferred = [item for item in candidates if "train" in item.name]
    chosen = preferred or candidates
    if not chosen:
        raise ExplainerUnavailable(f"no parquet under {root} to draw a LIME background from")
    return chosen[0]


def load_background(path: Path | None = None, n: int = BACKGROUND_ROWS) -> np.ndarray:
    """Read a reference sample off disk, building features if they are absent."""
    file = _background_file(path)
    frame = pd.read_parquet(file)
    if not set(FEATURE_NAMES).issubset(frame.columns):
        frame = to_feature_frame_many(frame.head(n).to_dict(orient="records"))
    frame = frame.loc[:, list(FEATURE_NAMES)]
    if len(frame) > n:
        frame = frame.sample(n=n, random_state=schema.RANDOM_SEED)
    return frame.to_numpy(dtype=float)


def background() -> np.ndarray:
    """The reference sample, loading it from disk on first use."""
    global _BACKGROUND
    if _BACKGROUND is None:
        _BACKGROUND = load_background()
    return _BACKGROUND


def get_explainer() -> LimeTabularExplainer:
    """Build the tabular explainer once over the background sample."""
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = LimeTabularExplainer(
            training_data=background(),
            feature_names=list(FEATURE_NAMES),
            class_names=list(CLASS_NAMES),
            mode="classification",
            discretize_continuous=True,
            random_state=schema.RANDOM_SEED,
        )
    return _EXPLAINER


def predict_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an estimator so LIME can call it with a bare matrix."""
    columns = list(FEATURE_NAMES)

    def predict(data: np.ndarray) -> np.ndarray:
        # Column names, not a bare array: the model was fitted on a DataFrame,
        # and feeding it positional values is how you get a silent mismatch of
        # features. LIME only ever hands back a plain matrix.
        frame = pd.DataFrame(np.asarray(data, dtype=float), columns=columns)
        proba = np.asarray(model.predict_proba(frame), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    return predict


def reseed(explainer: LimeTabularExplainer, seed: int = schema.RANDOM_SEED) -> None:
    """Point every RandomState inside the explainer at one fresh stream.

    LIME hands its constructor seed to three collaborators -- the perturbation
    sampler, the discretizer and the ridge fitter -- and each keeps the
    *object*, which advances on every call. Re-seeding only
    `explainer.random_state` leaves the discretizer drawing from wherever the
    previous request left off, and the same account comes back with subtly
    different weights twice. Reaching in here is the price of a reproducible
    explanation; the attributes are guarded so a lime upgrade degrades to
    "slightly non-deterministic" rather than "500".
    """
    state = np.random.RandomState(seed)
    explainer.random_state = state
    for owner in (getattr(explainer, "discretizer", None), getattr(explainer, "base", None)):
        if owner is not None and hasattr(owner, "random_state"):
            owner.random_state = state


def _served_model() -> Any:
    from credit_risk.serving import model_loader

    return model_loader.MODEL.model


def explain_local(
    record: Mapping[str, Any],
    *,
    model: Any = None,
    features: pd.DataFrame | None = None,
    num_features: int = 10,
) -> dict[str, Any]:
    """Explain one account with LIME. Pass `features` to avoid rebuilding them."""
    estimator = model if model is not None else _served_model()
    if estimator is None:
        raise ExplainerUnavailable("no model is loaded, so there is nothing to explain")

    frame = features if features is not None else to_feature_frame(record)
    explainer = get_explainer()

    # Re-seed per request: without it the second explanation of the same
    # account differs from the first. Stability across *seeds* is a different
    # property, and it is not measured anywhere yet -- see NUM_SAMPLES.
    reseed(explainer)

    explanation = explainer.explain_instance(
        data_row=frame.to_numpy(dtype=float)[0],
        predict_fn=predict_fn(estimator),
        num_features=num_features,
        num_samples=NUM_SAMPLES,
        labels=(POSITIVE_LABEL,),
    )
    contributions = [
        {"feature": str(condition), "weight": float(weight)}
        for condition, weight in explanation.as_list(label=POSITIVE_LABEL)
    ]
    return {"contributions": contributions}


def base_feature(label: str, names: Sequence[str] = FEATURE_NAMES) -> str:
    """Recover the underlying column from a LIME condition string.

    LIME labels a discretised interval ("1.00 < PAY_1 <= 2.00") while SHAP
    labels a column. Without this the two rankings can never overlap and
    `agreement` would report 0 for two explainers that in fact agree.
    """
    matches = [name for name in names if name in label]
    return max(matches, key=len) if matches else label


def _top_features(items: Sequence[Mapping[str, Any]], weight_key: str, k: int) -> list[str]:
    ranked = sorted(
        items,
        key=lambda item: abs(float(item.get(weight_key, 0.0) or 0.0)),
        reverse=True,
    )
    return [str(item.get("feature", "")) for item in ranked[: max(k, 0)]]


def agreement(
    shap_result: Mapping[str, Any], lime_result: Mapping[str, Any], k: int = 3
) -> dict[str, Any]:
    """How much the two explainers' top-k drivers overlap, and what that means.

    The key is interpolated from k; the API pins k=3 because the published
    response schema names the field `top3_overlap`.
    """
    shap_top = _top_features(shap_result.get("contributions", ()), "shap", k)
    lime_top = [
        base_feature(name)
        for name in _top_features(lime_result.get("contributions", ()), "weight", k)
    ]
    overlap = len(set(shap_top) & set(lime_top))

    if overlap == k and k > 0:
        note = f"SHAP and LIME agree on all {k} of the top drivers."
    elif overlap == 0:
        note = (
            f"SHAP and LIME share none of their top {k} drivers; prefer the SHAP "
            "ranking, which is exact for this model, and treat the LIME view as "
            "indicative only."
        )
    else:
        note = f"SHAP and LIME agree on {overlap} of the top {k} drivers."
    return {f"top{k}_overlap": overlap, "note": note}
