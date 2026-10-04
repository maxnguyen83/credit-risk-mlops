"""Three ways to make the model fairer, and what each one costs.

The brief is "bias analysis with mitigation strategies", so the deliverable is
not a number -- it is a comparison. Each strategy intervenes at a different
point in the pipeline and pays for fairness with a different currency:

    unawareness        pre-processing   lowers the equalized-odds gap, barely
                                        moves demographic parity, and leaves
                                        the attribute recoverable from proxies
    reweighing         pre-processing   no measurable ranking cost on the
                                        reference split (one seed, one split)
    ThresholdOptimizer post-processing  leaves the ranking untouched, selects
                                        more accounts than the capacity, and
                                        costs everything legally -- it applies a
                                        different cutoff per sex, which is
                                        disparate treatment on its face

The measured figures for each row are in MODEL_CARD.md.

`tradeoff_curve` runs all three plus the untouched baseline so the four land
in one table with identical evaluation code. Comparing mitigations measured by
different scripts is how you end up presenting an artefact of the measurement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.fairness.metrics import fairness_summary
from credit_risk.models.evaluate import evaluate, top_k_mask

logger = logging.getLogger(__name__)

EstimatorFactory = Callable[[], BaseEstimator]

STRATEGIES: tuple[str, ...] = (
    "baseline",
    "unawareness",
    "reweighing",
    "threshold_optimizer",
)

# Where training leaves the fitted cutoffs for serving to pick up. A JSON file
# rather than a pickled fairlearn object: the serving image should not have to
# install fairlearn, and a two-line file is something an auditor can read.
#
# It lands next to the processed splits because that is the one directory the
# pipeline already owns and mounts. The API only sees it if that volume is
# mounted into its container; when it is not, `load_group_thresholds` returns
# empty and the single configured threshold applies -- which is also the
# default policy, so the absence is not a silent failure.
GROUP_THRESHOLDS_FILENAME: Final = "group_thresholds.json"


# --------------------------------------------------------------- helpers


def protected_columns(columns: Sequence[str]) -> list[str]:
    """Feature columns derived from a protected attribute.

    Matches on prefix so one-hot output (`SEX_female`) and binned output
    (`AGE_GROUP`) are caught alongside the raw column. A name that merely
    contains the word -- there are none today -- would be missed, and that is
    the safer direction to be wrong in: dropping a column you meant to keep
    silently changes the model, while keeping one you meant to drop shows up
    immediately in the unawareness probe.
    """
    protected = {schema.SEX, schema.EDUCATION, schema.MARRIAGE, schema.AGE, schema.AGE_GROUP}
    return [
        column
        for column in columns
        if column in protected or any(column.startswith(f"{p}_") for p in protected)
    ]


def drop_protected(X: pd.DataFrame) -> pd.DataFrame:
    """The 'fairness through unawareness' transform, in one line."""
    return X.drop(columns=protected_columns(X.columns), errors="ignore")


def _encode_groups(sensitive: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Map group labels to 0..k-1 and return (codes, distinct labels)."""
    s = np.asarray(sensitive).ravel()
    labels = np.unique(s)
    codes = np.searchsorted(labels, s)
    return codes, labels


# ---------------------------------------------------- 1. unawareness probe


def unawareness_probe(
    X: pd.DataFrame,
    sensitive: ArrayLike,
    *,
    test_size: float = 0.25,
    seed: int = schema.RANDOM_SEED,
) -> dict[str, Any]:
    """Predict the protected attribute from the *non*-protected features.

    Whatever this ROC-AUC turns out to be, it is the answer to the question
    "does deleting the column delete the attribute". Anything materially above
    0.5 means no: credit limit, age-correlated balances and repayment behaviour
    reconstruct it. The result is reported, not asserted -- a low number would
    be just as interesting, and anyone proposing "we just drop the column" is
    owed the measurement either way.

    The probe is a linear model on purpose. It is the weakest attacker we could
    have used, so whatever AUC it reaches is a floor -- a gradient booster with
    the same columns does better. Reporting the floor makes the argument
    harder to dismiss.
    """
    features = drop_protected(X)
    if features.shape[1] == 0:
        raise ValueError("every feature is protected-derived; nothing left to probe with")

    codes, labels = _encode_groups(sensitive)
    if labels.size < 2:
        raise ValueError("the protected attribute has a single value; nothing to predict")

    X_train, X_test, g_train, g_test = train_test_split(
        features, codes, test_size=test_size, random_state=seed, stratify=codes
    )
    probe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    probe.fit(X_train, g_train)
    proba = probe.predict_proba(X_test)

    if labels.size == 2:
        auc = float(roc_auc_score(g_test, proba[:, 1]))
    else:
        auc = float(roc_auc_score(g_test, proba, multi_class="ovr", average="macro"))

    coefficients = np.abs(np.asarray(probe.named_steps["clf"].coef_)).mean(axis=0)
    ranking = np.argsort(-coefficients)[:5]
    return {
        "roc_auc": auc,
        "n_features": int(features.shape[1]),
        "n_groups": int(labels.size),
        "dropped_columns": protected_columns(X.columns),
        # Named so the report can say *which* columns leak, instead of
        # gesturing at "proxies" in the abstract.
        "top_proxies": [
            {"feature": str(features.columns[i]), "abs_coef": float(coefficients[i])}
            for i in ranking
        ],
    }


# ------------------------------------------------------- 2. reweighing


def reweigh(y: ArrayLike, sensitive: ArrayLike) -> np.ndarray:
    """Kamiran-Calders pre-processing weights.

        w(g, c) = P(group = g) * P(label = c) / P(group = g, label = c)

    Cells that are over-represented relative to independence get weights below
    one, under-represented cells above one. After weighting, group and label
    are statistically independent in the training sample -- which is exactly
    demographic parity imposed on the data rather than on the decisions.

    Two properties worth knowing. No row is ever dropped, and the weights sum
    to n -- so the effective sample size is unchanged -- provided every
    (group, label) cell has at least one row in it. An empty cell leaves the
    total short by exactly the mass that cell would have carried, which is the
    honest answer: there are no rows there to carry it. Both properties are
    asserted in the unit tests, because a reweighing bug surfaces as "the model
    got a bit worse" rather than as an exception.
    """
    y_arr = np.asarray(y).astype(int).ravel()
    s_arr = np.asarray(sensitive).ravel()
    if y_arr.shape[0] != s_arr.shape[0]:
        raise ValueError(f"y has {y_arr.shape[0]} rows, sensitive has {s_arr.shape[0]}")
    n = y_arr.shape[0]
    if n == 0:
        raise ValueError("cannot compute weights for an empty sample")

    weights = np.ones(n, dtype=float)
    for group in np.unique(s_arr):
        in_group = s_arr == group
        p_group = in_group.mean()
        for label in np.unique(y_arr):
            has_label = y_arr == label
            cell = in_group & has_label
            n_cell = int(cell.sum())
            if n_cell == 0:
                # An empty cell has no rows to weight. It also means the joint
                # distribution cannot be equalised for that group; the caller
                # sees it as an unchanged parity difference, not as a crash.
                continue
            weights[cell] = p_group * has_label.mean() / (n_cell / n)
    return weights


# ------------------------------------------------- 3. threshold optimizer


def fit_threshold_optimizer(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    sensitive: ArrayLike,
    *,
    constraints: str = "equalized_odds",
) -> ThresholdOptimizer:
    """Fit per-group cutoffs on an already-trained estimator.

    `prefit=True` because post-processing must not retrain the model: the whole
    claim of this strategy is that the ranking is untouched and only the
    decision boundary moves. Refitting would confound the two.
    """
    optimizer = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraints,
        prefit=True,
        # Explicit rather than "auto": auto silently falls back to `predict`
        # for estimators it cannot introspect, which would threshold hard
        # labels and produce cutoffs of 0.5 for every group.
        predict_method="predict_proba",
    )
    optimizer.fit(X, np.asarray(y).astype(int).ravel(), sensitive_features=np.asarray(sensitive))
    return optimizer


def group_thresholds_path(path: Path | str | None = None) -> Path:
    """Resolve where the exported cutoffs live."""
    if path is not None:
        return Path(path)
    return Path(settings.processed_dir) / GROUP_THRESHOLDS_FILENAME


def save_group_thresholds(
    thresholds: dict[str, float],
    path: Path | str | None = None,
    *,
    attribute: str = schema.PRIMARY_PROTECTED,
) -> Path:
    """Write the cutoffs where the serving container will look for them.

    The attribute name is written alongside the numbers. `train_all` takes the
    protected attribute as a parameter, so nothing stops a run fitting cutoffs
    on MARRIAGE -- whose codes are 1, 2, 3 -- and exporting them to the file
    serving looks up by SEX, whose codes are 1 and 2. Two of the three keys
    would then match by pure coincidence and the API would apply a marital
    policy to men and women with nothing in the log to say it had happened.
    Recording the attribute is what lets the reader refuse the file instead.
    """
    target = group_thresholds_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"attribute": attribute, "thresholds": thresholds}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("wrote group thresholds for %s to %s", attribute, target)
    return target


def load_group_thresholds(
    path: Path | str | None = None, *, attribute: str = schema.PRIMARY_PROTECTED
) -> dict[str, float]:
    """Read the exported cutoffs, or return an empty mapping if there are none.

    Empty rather than an exception: no fitted thresholds means the group-aware
    policy has never been run, and the correct behaviour then is for serving to
    fall back to the single configured threshold, not to return 500s.

    Empty is also the answer when the file was fitted on a different protected
    attribute than the caller keys its lookups by -- see `save_group_thresholds`
    for how that happens. Falling back to one cutoff for everyone is the status
    quo; applying another attribute's cutoffs is a new and unreviewed form of
    differential treatment, so the mismatch loses.
    """
    target = group_thresholds_path(path)
    if not target.exists():
        logger.info("no exported group thresholds at %s; the base threshold applies", target)
        return {}
    try:
        raw = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read group thresholds from %s: %s", target, exc)
        return {}
    if isinstance(raw, dict) and "thresholds" in raw:
        recorded = str(raw.get("attribute", attribute))
        if recorded != attribute:
            logger.warning(
                "group thresholds in %s were fitted on %s but are being read as %s; "
                "ignoring them, the base threshold applies",
                target,
                recorded,
                attribute,
            )
            return {}
        raw = raw["thresholds"]
    return {str(group): float(value) for group, value in raw.items()}


def group_thresholds(
    optimizer: ThresholdOptimizer | None = None, *, path: Path | str | None = None
) -> dict[str, float]:
    """The per-group cutoff, extracted so serving need not import fairlearn.

    With an optimizer, the cutoffs are read out of the fitted object. With
    none -- which is how the serving layer calls it -- they are read from the
    file the training run exported. Serving has no fairlearn object and no
    training data, so the file is the only thing it can be handed.

    fairlearn's solution is a *randomised* rule: with probability p0 it applies
    one threshold and with probability p1 another, which is how it hits the
    equalised-odds point exactly. We export the dominant leg and throw the coin
    away. That costs a little exactness and buys determinism -- a customer who
    asks why they were called deserves a reason, not a coin flip, and a
    randomised decision cannot be reproduced in an audit.
    """
    if optimizer is None:
        return load_group_thresholds(path)

    thresholder = getattr(optimizer, "interpolated_thresholder_", None)
    if thresholder is None:
        raise ValueError("optimizer is not fitted; call fit_threshold_optimizer first")

    thresholds: dict[str, float] = {}
    for group, params in thresholder.interpolation_dict.items():
        p0 = float(params.get("p0", 0.0))
        p1 = float(params.get("p1", 0.0))
        operation = params.get("operation0") if p0 >= p1 else params.get("operation1")
        if operation is None:  # pragma: no cover - fairlearn always sets one
            continue
        p_ignore = float(params.get("p_ignore", 0.0))
        if p_ignore > 0.01:
            logger.warning(
                "group %s ignores its score %.1f%% of the time under equalized odds; "
                "the exported threshold does not reproduce that behaviour",
                group,
                100 * p_ignore,
            )
        value = float(operation.threshold)
        if not np.isfinite(value):
            # fairlearn seeds its ROC sweep with +/-inf and hands one back when
            # a group's winning leg is "never select" or "always select".
            # json.dumps writes that as `Infinity`, which is not JSON -- jq and
            # every other reader rejects the file the API loads at startup, so
            # the whole group-aware policy silently reverts to one cutoff.
            logger.warning(
                "group %s got a non-finite cutoff (%s); clamping it into [0, 1]", group, value
            )
        thresholds[str(group)] = float(np.clip(value, 0.0, 1.0))
    return thresholds


def apply_group_thresholds(
    y_proba: ArrayLike,
    sensitive: ArrayLike,
    thresholds: dict[str, float],
    *,
    default_threshold: float | None = None,
) -> np.ndarray:
    """Turn scores into decisions using one cutoff per group.

    A group absent from `thresholds` -- a value that never appeared in
    training -- falls back to the configured global threshold rather than
    being silently rejected or silently approved.

    `>=`, not `>`, and the choice is not cosmetic. `serving.routes.decide`
    compares with `>=`, and `evaluate.threshold_for_capacity` returns the k-th
    *largest* score, so a cutoff taken from there admits exactly k accounts
    under `>=` and k-1 under `>`. Measuring the mitigation offline with one
    operator while production uses the other puts every tie on the wrong side
    of the line, and ties are not rare: the fallback cutoff is a round number
    somebody typed into configuration.
    """
    p = np.asarray(y_proba, dtype=float).ravel()
    s = np.asarray(sensitive).ravel()
    if p.shape[0] != s.shape[0]:
        raise ValueError(f"y_proba has {p.shape[0]} rows, sensitive has {s.shape[0]}")
    fallback = settings.decision_threshold if default_threshold is None else default_threshold
    cutoffs = np.array([thresholds.get(str(group), fallback) for group in s], dtype=float)
    return (p >= cutoffs).astype(int)


# ------------------------------------------------------- comparison table


def _fit_with_sample_weight(
    estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray
) -> BaseEstimator:
    """Pass sample weights through, whether or not the estimator is a Pipeline.

    A Pipeline routes fit params by step name: `fit(X, y, sample_weight=w)`
    raises, `fit(X, y, clf__sample_weight=w)` works. Getting this wrong is not
    loud -- swap the two and you get a TypeError, but reach for a bare
    `**kwargs` and the weights can be dropped on the floor while training
    happily succeeds and reweighing appears to do nothing.
    """
    if isinstance(estimator, Pipeline):
        final_step = estimator.steps[-1][0]
        return estimator.fit(X, y, **{f"{final_step}__sample_weight": weights})
    return estimator.fit(X, y, sample_weight=weights)


def _default_estimator_factory() -> BaseEstimator:
    """A fast, boring classifier so the table can be produced without LightGBM.

    Callers that care about the numbers pass the tuned estimator instead; this
    default exists so the function is testable in isolation.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=schema.RANDOM_SEED,
                ),
            ),
        ]
    )


def tradeoff_curve(
    X_train: pd.DataFrame,
    y_train: ArrayLike,
    sensitive_train: ArrayLike,
    X_test: pd.DataFrame,
    y_test: ArrayLike,
    sensitive_test: ArrayLike,
    *,
    estimator_factory: EstimatorFactory | None = None,
    capacity_fraction: float | None = None,
    seed: int = schema.RANDOM_SEED,
) -> pd.DataFrame:
    """One row per strategy: the figure the fairness section is built on.

    Note what the table will show for `threshold_optimizer`: its PR-AUC equals
    the baseline's, exactly, because post-processing cannot reorder anything.
    That is not a copy-paste error -- it is the point. Post-processing buys
    parity without touching model quality, and the bill arrives somewhere else:
    its cutoffs are chosen to equalise odds, not to fit a budget, so it
    routinely selects more than `capacity_fraction` of the portfolio. That is a
    staffing problem, not a modelling one, and it is why `selection_rate` is a
    column rather than an assumption.
    """
    capacity = (
        settings.intervention_capacity_fraction if capacity_fraction is None else capacity_fraction
    )
    factory = _default_estimator_factory if estimator_factory is None else estimator_factory
    y_tr = np.asarray(y_train).astype(int).ravel()
    y_te = np.asarray(y_test).astype(int).ravel()
    s_tr = np.asarray(sensitive_train).ravel()
    s_te = np.asarray(sensitive_test).ravel()

    rows: list[dict[str, Any]] = []

    def record(strategy: str, proba: np.ndarray, decisions: np.ndarray) -> None:
        metrics = evaluate(y_te, proba, capacity_fraction=capacity)
        summary = fairness_summary(y_te, decisions, s_te)
        hits = int(((decisions == 1) & (y_te == 1)).sum())
        rows.append(
            {
                "strategy": strategy,
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "dp_diff": summary["demographic_parity_difference"],
                "eo_diff": summary["equalized_odds_difference"],
                # Recall at whatever this rule actually selects. For the three
                # score-based strategies that is the top-k list; for the
                # post-processor it is however many its cutoffs admit, so the
                # realised selection rate sits next to it.
                "recall_at_k": float(hits / max(1, int(y_te.sum()))),
                "selection_rate": float(decisions.mean()),
            }
        )

    # --- baseline: everything, protected attributes included
    # `clone` on top of the factory so a caller who hands back the same object
    # each call still gets four independently fitted models.
    baseline = clone(factory())
    baseline.fit(X_train, y_tr)
    proba_base = baseline.predict_proba(X_test)[:, 1]
    record("baseline", proba_base, top_k_mask(proba_base, capacity).astype(int))

    # --- unawareness: the same model with the protected columns deleted
    X_train_blind, X_test_blind = drop_protected(X_train), drop_protected(X_test)
    if X_train_blind.shape[1] == X_train.shape[1]:
        # Nothing to drop means the feature builder never exposed the
        # attribute; say so rather than reporting a duplicate baseline row.
        logger.warning("no protected-derived columns found; unawareness row duplicates baseline")
    blind = clone(factory())
    blind.fit(X_train_blind, y_tr)
    proba_blind = blind.predict_proba(X_test_blind)[:, 1]
    record("unawareness", proba_blind, top_k_mask(proba_blind, capacity).astype(int))

    # --- reweighing: same features, group-balanced sample weights
    weights = reweigh(y_tr, s_tr)
    reweighted = clone(factory())
    _fit_with_sample_weight(reweighted, X_train, y_tr, weights)
    proba_rw = reweighted.predict_proba(X_test)[:, 1]
    record("reweighing", proba_rw, top_k_mask(proba_rw, capacity).astype(int))

    # --- post-processing on the untouched baseline
    try:
        optimizer = fit_threshold_optimizer(baseline, X_train, y_tr, s_tr)
    except ValueError as exc:
        # fairlearn refuses to fit when any group arrives carrying a single
        # label -- MARRIAGE=3 is 323 rows in 30,000, so one thin batch is all
        # it takes. That is an operational fact about the sample, not a broken
        # pipeline, and the other three strategies are still worth reporting.
        logger.warning("threshold_optimizer could not be fitted (%s); dropping its row", exc)
        return pd.DataFrame(rows)
    decisions_to = np.asarray(
        optimizer.predict(X_test, sensitive_features=s_te, random_state=seed)
    ).astype(int)
    record("threshold_optimizer", proba_base, decisions_to)

    return pd.DataFrame(rows)
