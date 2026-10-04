"""The model this process serves, and everything needed to apply it.

The holder is deliberately dumb: it owns the estimator, the facts an on-call
engineer asks for at 2am (which version, which algorithm, trained when, from
which run, deciding at what cutoff and why that one), and the reason it has
none. The pure functions beside it turn a record into a probability -- both the
API and the explainers need that, and neither should keep a private copy of how
it is done.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from credit_risk.config import settings
from credit_risk.features.build import (
    FEATURE_NAMES,
    REQUIRED_INPUT_COLUMNS,
    FeatureBuildError,
    build_features,
    build_features_from_record,
)

log = logging.getLogger(__name__)

UNKNOWN: Final = "unknown"

# Where the cutoff in force came from. `registry`: the `threshold_at_k` tag on
# the loaded version. `env`: DECISION_THRESHOLD, because THRESHOLD_SOURCE=env
# asked for it. `fallback`: DECISION_THRESHOLD, because the registry was asked
# and had no usable tag -- the case that must never look like the first one.
ThresholdSource = Literal["registry", "env", "fallback"]

# How the served version was found. `alias`: `models:/<name>@<MODEL_ALIAS>`, the
# way promotion marks the champion. `stage`: `models:/<name>/<MODEL_STAGE>`,
# the fallback for a registry populated before aliases were used. `unknown`:
# nothing is loaded, or the registry did not say.
ModelRef = Literal["alias", "stage", "unknown"]


@dataclass(frozen=True)
class DecisionThreshold:
    """The base cutoff in force, where it came from, and why if it is a fallback."""

    value: float
    source: ThresholdSource
    reason: str | None = None


def resolve_threshold(
    tag_value: str | None,
    *,
    source: str | None = None,
    configured: float | None = None,
) -> DecisionThreshold:
    """Pick the base cutoff from a version's `threshold_at_k` tag and the settings.

    Pure, so every branch is testable without a registry. A tag that is absent,
    unparseable or outside [0, 1] falls back to the configured constant with
    the reason attached: a broken tag must not take the service down, and it
    must not quietly become 0.5 either.
    """
    wanted = settings.threshold_source if source is None else source
    constant = settings.decision_threshold if configured is None else configured
    if wanted == "env":
        return DecisionThreshold(constant, "env")

    if tag_value is None:
        return DecisionThreshold(
            constant, "fallback", "the loaded version carries no threshold_at_k tag"
        )
    try:
        value = float(tag_value)
    except (TypeError, ValueError):
        return DecisionThreshold(
            constant, "fallback", f"threshold_at_k tag {tag_value!r} is not a number"
        )
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return DecisionThreshold(
            constant, "fallback", f"threshold_at_k tag {tag_value!r} is not a probability"
        )
    return DecisionThreshold(value, "registry")


def _no_model_threshold() -> DecisionThreshold:
    """What applies while nothing is loaded -- reported, though nothing is scored."""
    chosen = resolve_threshold(None)
    if chosen.source == "fallback":
        return DecisionThreshold(chosen.value, "fallback", "no model is loaded")
    return chosen


def _attr(source: Any, key: str) -> Any:
    """Read `key` off a mapping or an object, whichever we were handed."""
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def describe_version(info: Any) -> tuple[str, str, str]:
    """Coerce the registry's answer into (version, algo, trained_at).

    The registry module is owned by another track and may report a bare version
    string, a mapping, or a small metadata object. Accepting all three keeps a
    change on that side from turning a healthy service into a boot crash loop.
    """
    if info is None:
        return UNKNOWN, UNKNOWN, UNKNOWN
    if isinstance(info, str | int | float):
        return str(info), UNKNOWN, UNKNOWN

    version = _attr(info, "version") or _attr(info, "model_version")
    algo = _attr(info, "algo") or _attr(info, "algorithm") or _attr(info, "flavor")
    trained_at = (
        _attr(info, "trained_at") or _attr(info, "created_at") or _attr(info, "creation_timestamp")
    )
    return (
        str(version) if version is not None else UNKNOWN,
        str(algo) if algo is not None else UNKNOWN,
        str(trained_at) if trained_at is not None else UNKNOWN,
    )


@dataclass
class ModelHolder:
    """One process, one production model -- plus why it might be missing."""

    model: Any | None = None
    version: str = UNKNOWN
    algo: str = UNKNOWN
    trained_at: str = UNKNOWN
    run_id: str = UNKNOWN
    model_ref: ModelRef = UNKNOWN
    model_ref_uri: str = UNKNOWN
    decision: DecisionThreshold = field(default_factory=_no_model_threshold)
    last_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        """The base cutoff every 'intervene' decision is measured against."""
        return self.decision.value

    @property
    def threshold_source(self) -> ThresholdSource:
        return self.decision.source

    def install(
        self,
        model: Any,
        *,
        version: str = UNKNOWN,
        algo: str = UNKNOWN,
        trained_at: str = UNKNOWN,
        run_id: str = UNKNOWN,
        threshold: DecisionThreshold | None = None,
        model_ref: ModelRef = UNKNOWN,
        model_ref_uri: str = UNKNOWN,
    ) -> None:
        """Adopt an estimator directly. Used by the loader and by tests.

        No threshold given means the version told us nothing about its cutoff,
        which is the fallback case and is reported as one.
        """
        self.model = model
        self.version = version
        self.algo = algo
        self.trained_at = trained_at
        self.run_id = run_id
        self.model_ref = model_ref
        self.model_ref_uri = model_ref_uri
        self.decision = resolve_threshold(None) if threshold is None else threshold
        self.last_error = None

    def unload(self) -> None:
        """Drop the model but keep `last_error` -- /health still has to explain itself."""
        self.model = None
        self.version = UNKNOWN
        self.algo = UNKNOWN
        self.trained_at = UNKNOWN
        self.run_id = UNKNOWN
        self.model_ref = UNKNOWN
        self.model_ref_uri = UNKNOWN
        self.decision = _no_model_threshold()

    def load(self) -> bool:
        """Pull the champion from the MLflow registry. Never raises.

        The version is the one `models:/<name>@<MODEL_ALIAS>` names, or, when no
        version carries the alias, the one in `MODEL_STAGE`; `model_ref` records
        which. Read once, here: a promotion made later is served after a restart.

        A registry that is down or empty must leave the service running in a
        DEGRADED state rather than aborting the boot: a process that refuses to
        start cannot tell anyone why it is unhealthy, and `docker compose logs`
        on a crash-looping container is a worse debugging surface than
        /health plus credit_model_loaded == 0.
        """
        try:
            # Imported inside the function: mlflow costs seconds to import, and
            # a module-level import would make an unreachable registry a hard
            # import error rather than a reported degradation.
            from credit_risk.models.registry import (
                THRESHOLD_TAG,
                load_production_model,
                production_version_metadata,
            )

            # The version first -- by alias, else by stage -- then the model by
            # that version's own URI, so the tags read below describe the
            # estimator actually loaded even if somebody promotes another
            # version in between.
            metadata = production_version_metadata()
            if metadata is not None:
                loaded = load_production_model(model_uri=metadata.model_uri)
                ref = _model_ref(metadata.resolved_by)
                ref_uri = metadata.resolved_uri or UNKNOWN
                lookup_errors = tuple(metadata.resolution_errors)
            else:
                loaded = load_production_model()
                ref, ref_uri = "stage", settings.model_uri
                lookup_errors = ()
            info: Any = None
            if isinstance(loaded, tuple) and loaded:
                model = loaded[0]
                info = loaded[1] if len(loaded) > 1 else None
            else:
                model = loaded
            if model is None:
                raise RuntimeError(
                    f"registry returned no model for {settings.model_name}@"
                    f"{settings.model_alias} or the {settings.model_stage} stage"
                )
            if info is None:
                info = metadata
            version, algo, trained_at = describe_version(info)
            if algo == UNKNOWN:
                # The registry reports a version string and nothing else. The
                # estimator itself knows what it is, and credit_model_info is
                # far more useful during an incident with "LGBMClassifier" on
                # it than with "unknown".
                algo = type(model).__name__
            tags = metadata.tags if metadata is not None else {}
            run_id = metadata.run_id if metadata is not None and metadata.run_id else UNKNOWN
            threshold = resolve_threshold(tags.get(THRESHOLD_TAG))
        except Exception as exc:
            self.unload()
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("model load failed, serving degraded: %s", self.last_error)
            return False

        self.install(
            model,
            version=version,
            algo=algo,
            trained_at=trained_at,
            run_id=run_id,
            threshold=threshold,
            model_ref=ref,
            model_ref_uri=ref_uri,
        )
        log.info(
            "model loaded version=%s via %s (%s) algo=%s trained_at=%s run_id=%s "
            "threshold=%.4f source=%s",
            version,
            ref,
            ref_uri,
            algo,
            trained_at,
            run_id,
            threshold.value,
            threshold.source,
        )
        if ref == "stage" and lookup_errors:
            # Not "no alias yet": the alias could not be read. The stage keeps
            # the service up, and may not be the version the alias names.
            log.warning(
                "serving version %s from %s because the %r alias could not be read (%s). "
                "Check the registry and restart the API",
                version,
                ref_uri,
                settings.model_alias,
                "; ".join(lookup_errors),
            )
        elif ref == "stage":
            log.warning(
                "serving version %s from %s because no version carries the %r alias. "
                "Set it with `python -m credit_risk.models.registry set-champion "
                "--version %s` and restart the API",
                version,
                ref_uri,
                settings.model_alias,
                version,
            )
        if threshold.source == "fallback":
            # Loud on purpose: the service works, and decides at a cutoff that
            # was typed into configuration rather than computed for this model.
            log.warning(
                "%s; deciding at DECISION_THRESHOLD=%.4f (threshold_source=fallback). "
                "Fill the tag in with `python -m credit_risk.models.registry "
                "tag-threshold --version %s` and restart the API",
                threshold.reason,
                threshold.value,
                version,
            )
        return True


def _model_ref(resolved_by: str | None) -> ModelRef:
    """The registry's "alias"/"stage" as the closed set /health reports."""
    if resolved_by == "alias":
        return "alias"
    if resolved_by == "stage":
        return "stage"
    return UNKNOWN


# One holder per process. Callers reference `model_loader.MODEL` rather than
# importing the object, so a test can swap the whole holder out.
MODEL = ModelHolder()


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the columns back in the order the estimator was fitted on.

    The model was fitted on a DataFrame. Handing it the same values under a
    different column order is how you get a quietly wrong probability from any
    estimator that does not verify feature names -- no exception, no log line.
    """
    if set(FEATURE_NAMES).issubset(frame.columns):
        return frame.loc[:, list(FEATURE_NAMES)]
    return frame


def frame_from_features(features: Any) -> pd.DataFrame:
    """Normalise whatever the feature builder returned into a model-ready frame."""
    if isinstance(features, pd.DataFrame):
        frame = features
    elif isinstance(features, pd.Series):
        frame = features.to_frame().T
    elif isinstance(features, Mapping):
        frame = pd.DataFrame([dict(features)])
    elif isinstance(features, np.ndarray):
        array = features.reshape(1, -1) if features.ndim == 1 else features
        frame = pd.DataFrame(array, columns=list(FEATURE_NAMES))
    else:
        raise TypeError(f"cannot build a feature frame from {type(features).__name__}")
    return order_columns(frame)


def to_feature_frame(record: Mapping[str, Any]) -> pd.DataFrame:
    """One raw record in, one row of model-ready features out.

    This goes through the same `build_features_from_record` the training DAG
    uses. That single call site is the train/serve skew guard; a second,
    serving-only implementation would drift within a sprint.
    """
    return frame_from_features(build_features_from_record(dict(record)))


def to_feature_frame_many(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Every record in one frame, built in one pass.

    Not a loop over the single-record builder. That builder is a thin wrapper
    around a vectorised function, so looping it made a 1,000-account batch pay
    1,000 DataFrame constructions to produce 1,000 single-row frames and then
    concatenate them -- while the docstring claimed the opposite.
    """
    if not records:
        # BatchPredictRequest pins min_length=1, so an empty sequence here means
        # a caller reached past the schema. Saying so beats pandas raising
        # "No objects to concatenate" four frames down the stack.
        raise FeatureBuildError("cannot build features from an empty batch")

    frame = pd.DataFrame([dict(record) for record in records])
    # Projected, exactly as the single-record builder does it: an extra key such
    # as ID widens the frame and changes nothing visible until something
    # downstream indexes by position. A column that is missing rather than extra
    # is left for build_features to name in its own error.
    present = [column for column in REQUIRED_INPUT_COLUMNS if column in frame.columns]
    return order_columns(build_features(frame.loc[:, present]))


def positive_class_index(model: Any) -> int:
    """Which column of predict_proba is P(default).

    Read from `classes_` rather than assuming column 1: an estimator fitted on
    labels in another order would otherwise ship a perfectly inverted score,
    which looks plausible on a dashboard and is catastrophic in a decision.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1
    values = list(np.asarray(classes).ravel())
    for wanted in (1, "1"):
        if wanted in values:
            return values.index(wanted)
    return len(values) - 1


def predict_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """P(default) for every row of `frame`, as a 1-d float array."""
    proba = np.asarray(model.predict_proba(frame), dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, positive_class_index(model)]
