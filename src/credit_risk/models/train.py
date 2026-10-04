"""Train the candidates, log everything, hand the best one to the gate.

Two model families are trained on every run, and neither is decoration.
LogisticRegression is the interpretable baseline: a compliance officer can read
its coefficients, and the adverse-action notice it supports needs no
approximation. LightGBM is the candidate that is expected to win on PR-AUC.
Reporting both is how the interpretability trade-off gets argued with numbers
instead of adjectives.

The hyperparameter grid is small on purpose. This is an operations course --
the deliverable is a reproducible, tracked, gated pipeline, not the last two
points of PR-AUC. Eight combinations across five folds is 40 fits and finishes
in well under a minute on four cores, which is short enough that nobody is
tempted to skip it or to comment it out of the DAG.

Run it directly with `python -m credit_risk.models.train`. If the MLflow server
is unreachable the run does not fail: it falls back to a local ./mlruns store
and says so loudly, because a training job that dies on a networking problem
is a training job that gets commented out of CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import schema
from credit_risk.config import PROJECT_ROOT, settings
from credit_risk.data.split import (
    BATCH_COL,
    MANIFEST_NAME,
    assign_batches,
    frame_sha256,
    load_split,
)
from credit_risk.fairness.metrics import (
    calibration_error_by_group,
    fairness_summary,
    flatten_summary,
    group_report,
    passes_gates,
    summaries_for_attributes,
)
from credit_risk.fairness.mitigation import (
    fit_threshold_optimizer,
    group_thresholds,
    save_group_thresholds,
    tradeoff_curve,
    unawareness_probe,
)
from credit_risk.features.build import FEATURE_NAMES, build_features
from credit_risk.models.evaluate import cost_matrix_report, evaluate, top_k_mask
from credit_risk.models.registry import evaluate_gate, record_refusal

logger = logging.getLogger(__name__)

# Fixed across every LightGBM run so the grid only varies what is being
# searched. `verbose=-1` because LightGBM otherwise prints a "No further
# splits with positive gain" wall that buries the MLflow output.
LIGHTGBM_FIXED: dict[str, Any] = {
    "objective": "binary",
    "n_jobs": 1,
    "verbose": -1,
    "random_state": schema.RANDOM_SEED,
}

# Deliberately eight combinations. See the module docstring.
LIGHTGBM_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.05, 0.1],
    "num_leaves": [15, 31],
    "min_child_samples": [20, 60],
    "n_estimators": [300],
}

CV_FOLDS = 5

# Run params naming the data a run saw. The two split hashes are computed from
# the frames the run actually trained and scored on, with the same function
# the split step used to write the manifest, so the two can be compared
# directly. `registry` reads TEST_SHA_PARAM back to refuse comparing PR-AUCs
# measured on different held-out rows.
TRAIN_SHA_PARAM: Final = "data_train_sha256"
TEST_SHA_PARAM: Final = "data_test_sha256"
# The raw download's digest, copied from the manifest -- only when the manifest
# describes these frames, since otherwise it is the origin of other rows.
SOURCE_SHA_PARAM: Final = "data_source_sha256"
# matched | mismatch | absent | unreadable: whether the manifest on disk
# describes the frames this run used.
MANIFEST_PARAM: Final = "data_manifest"
MANIFEST_GENERATED_PARAM: Final = "data_manifest_generated_at"
# The run keeps its own copy of the manifest here. The file in data/processed
# is rewritten by every clean_and_split; the artifact is not.
MANIFEST_ARTIFACT_DIR: Final = "data"


@dataclass(frozen=True)
class DataLineage:
    """The params that name a run's data, and the manifest to keep with it."""

    params: dict[str, str]
    # Set only when the manifest's split hashes equal this run's frames.
    manifest_path: Path | None = None


@dataclass(frozen=True)
class CandidateResult:
    """One trained model family, scored on the held-out batch."""

    name: str
    run_id: str | None
    estimator: BaseEstimator
    params: dict[str, Any]
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    metrics: dict[str, float]
    fairness: dict[str, float]
    gate_passed: bool
    gate_reasons: list[str] = field(default_factory=list)
    # SEX is the gate; the others are audited and reported so "it passed" can
    # never mean "it passed on the one attribute we chose to look at".
    fairness_by_attribute: dict[str, dict[str, float]] = field(default_factory=dict)
    group_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """What the DAG needs to decide whether to register anything."""

    best: CandidateResult
    candidates: list[CandidateResult]
    tracking_uri: str
    used_fallback_store: bool
    tradeoff: pd.DataFrame
    lineage: dict[str, str] = field(default_factory=dict)

    @property
    def run_id(self) -> str | None:
        return self.best.run_id

    @property
    def metrics(self) -> dict[str, float]:
        return self.best.metrics

    @property
    def estimator(self) -> BaseEstimator:
        return self.best.estimator


# ------------------------------------------------------------- estimators


def make_logistic_regression(seed: int = schema.RANDOM_SEED) -> Pipeline:
    """Scaled logistic regression -- the baseline a human can read.

    Scaling is not optional here: LIMIT_BAL runs to a million NT$ while the
    utilisation ratios sit in [0, 2], and without it the L2 penalty is applied
    almost entirely to the ratios.

    `class_weight="balanced"` because an unweighted logistic regression on a
    22% positive rate produces coefficients dominated by the majority class.
    It costs calibration, which is why the booster below does *not* use it.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def make_lightgbm(**overrides: Any) -> LGBMClassifier:
    """The candidate model.

    No class rebalancing. The decision is a capacity threshold, never p > 0.5,
    so the ranking is what matters -- and rebalancing shifts every predicted
    probability upward, which would wreck the Brier score that expected-loss
    reporting depends on.
    """
    params = {**LIGHTGBM_FIXED, **overrides}
    return LGBMClassifier(**params)


# ------------------------------------------------------------------ data


def load_training_splits(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The written train and test splits, read back through the data module.

    Batch 6 is never requested: it is the pool the traffic generator draws
    from, and evaluating on it would mean the live demo runs against rows the
    model was scored on.
    """
    return load_split("train", processed_dir), load_split("test", processed_dir)


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an in-memory clean frame exactly as the pipeline splits on disk.

    Used when a caller already holds the frame -- the DAG, and the tests that
    train on a synthetic sample rather than the real download.
    """
    labelled = frame if BATCH_COL in frame.columns else assign_batches(frame)
    train = labelled[labelled[BATCH_COL].isin(schema.TRAIN_BATCHES)].reset_index(drop=True)
    test = labelled[labelled[BATCH_COL] == schema.TEST_BATCH].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(
            f"empty split: {len(train)} train rows, {len(test)} test rows -- check the ID range"
        )
    return train, test


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Features for training, built by the exact function serving calls.

    Reindexing on FEATURE_NAMES pins the column order. The model is fitted on a
    DataFrame, so a reordered frame at serving time is a silent mismatch of
    features, not an error.
    """
    built = build_features(frame)
    return built[list(FEATURE_NAMES)]


def data_lineage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, processed_dir: Path | None = None
) -> DataLineage:
    """Name the data a run trains and scores on, from the frames themselves.

    The hashes come from the frames, not from the manifest, because a caller
    may hand train_all frames that never touched data/processed -- the DAG's
    in-memory path, the tests -- and the manifest on disk would then describe
    somebody else's rows. The manifest is adopted (its source digest copied,
    the file kept as an artifact) only when its train and test hashes equal the
    frames'. Never raises: lineage that cannot be established is recorded as
    such, and the training run goes ahead.
    """
    train_sha, test_sha = frame_sha256(train_df), frame_sha256(test_df)
    params = {
        TRAIN_SHA_PARAM: train_sha,
        TEST_SHA_PARAM: test_sha,
        SOURCE_SHA_PARAM: "unknown",
        MANIFEST_PARAM: "absent",
    }
    path = (settings.processed_dir if processed_dir is None else processed_dir) / MANIFEST_NAME
    if not path.is_file():
        return DataLineage(params)
    try:
        manifest = json.loads(path.read_text())
        splits = manifest["splits"]
        recorded = (splits["train"]["sha256"], splits["test"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("cannot read %s (%s); lineage records the frame hashes only", path, exc)
        return DataLineage({**params, MANIFEST_PARAM: "unreadable"})

    if recorded != (train_sha, test_sha):
        logger.warning(
            "%s describes other splits than the frames being trained on; not adopting it", path
        )
        return DataLineage({**params, MANIFEST_PARAM: "mismatch"})

    params[MANIFEST_PARAM] = "matched"
    params[SOURCE_SHA_PARAM] = str(manifest.get("source_sha256") or "unknown")
    params[MANIFEST_GENERATED_PARAM] = str(manifest.get("generated_at") or "unknown")
    return DataLineage(params, manifest_path=path)


# ---------------------------------------------------------------- mlflow


def resolve_tracking_uri(uri: str | None = None, *, timeout: float = 3.0) -> tuple[str, bool]:
    """Return (tracking_uri, used_fallback).

    A training run that has produced a model and then throws because a
    container is not up has wasted the expensive part of the work. Probe the
    server first and degrade to a local store, loudly, if it is not there.
    """
    target = settings.mlflow_tracking_uri if uri is None else uri
    if not target.startswith(("http://", "https://")):
        # file:, sqlite: and friends need no network and cannot be probed.
        return target, False
    try:
        response = requests.get(f"{target.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        return target, False
    except requests.RequestException as exc:
        fallback = (PROJECT_ROOT / "mlruns").as_uri()
        logger.warning(
            "=" * 72
            + "\nMLFLOW SERVER UNREACHABLE at %s (%s)\n"
            + "Falling back to the local store %s.\n"
            + "Runs will NOT appear in the tracking UI and the model registry is\n"
            + "unavailable, so nothing will be promoted from this run.\n"
            + "=" * 72,
            target,
            exc,
            fallback,
        )
        return fallback, True


def feature_importance_frame(
    estimator: BaseEstimator, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Importance per feature, for whichever family was fitted.

    A booster exposes split counts, a linear model exposes coefficients. They
    are not the same quantity and the column is named `importance` rather than
    pretending otherwise -- the plot is for spotting a feature that dominates,
    not for comparing the two families against each other.
    """
    final = estimator[-1] if isinstance(estimator, Pipeline) else estimator
    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        values = np.abs(np.asarray(final.coef_, dtype=float)).ravel()
    else:  # pragma: no cover - both supported families have one of the two
        raise TypeError(f"{type(final).__name__} exposes neither importances nor coefficients")
    frame = pd.DataFrame({"feature": list(feature_names), "importance": values})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(frame: pd.DataFrame, path: Path, *, top_n: int = 20) -> Path:
    """Write a horizontal bar chart, or a CSV if matplotlib is not installed.

    matplotlib arrives as an MLflow dependency rather than one we declare, so
    it is not guaranteed. Losing a picture is acceptable; losing a trained
    model because a plotting library moved is not.
    """
    top = frame.head(top_n).iloc[::-1]
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display in CI or in the Airflow container
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib ships with mlflow
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        logger.warning("matplotlib unavailable; wrote %s instead of a plot", csv_path)
        return csv_path

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.3 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#3b6ea5")
    ax.set_xlabel("importance")
    ax.set_title(f"Top {len(top)} features")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_tradeoff_plot(tradeoff: pd.DataFrame, path: Path, *, top_n: int = 0) -> Path:
    """Scatter the mitigation comparison: dp_diff on x, PR-AUC on y.

    The table says the same thing, but nobody argues with a table. Plotted, the
    shape of the trade-off is the argument: post-processing sits at the left
    edge on the baseline's own PR-AUC line -- parity bought for no ranking
    quality at all, with the bill arriving in the selection rate instead --
    while unawareness sits below the baseline for barely any movement left.

    Same matplotlib-optional contract as the importance plot: losing a picture
    is acceptable, losing a trained model to a plotting library is not.
    `top_n` is accepted and ignored so the two writers share a signature.
    """
    del top_n
    required = {"strategy", "dp_diff", "pr_auc"}
    missing = required - set(tradeoff.columns)
    if missing:
        raise ValueError(f"tradeoff table is missing {sorted(missing)}")

    try:
        import matplotlib

        matplotlib.use("Agg")  # no display in CI or in the Airflow container
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib ships with mlflow
        csv_path = path.with_suffix(".csv")
        tradeoff.to_csv(csv_path, index=False)
        logger.warning("matplotlib unavailable; wrote %s instead of a plot", csv_path)
        return csv_path

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tradeoff["dp_diff"], tradeoff["pr_auc"], s=90, color="#3b6ea5", zorder=3)
    for _, row in tradeoff.iterrows():
        ax.annotate(
            str(row["strategy"]),
            (row["dp_diff"], row["pr_auc"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )
    # The gate is the only vertical line that means anything here: points to
    # its right cannot be registered however good their PR-AUC is.
    ax.axvline(
        schema.MAX_DEMOGRAPHIC_PARITY_DIFF,
        color="#b3261e",
        linestyle="--",
        linewidth=1,
        label=f"gate dp_diff <= {schema.MAX_DEMOGRAPHIC_PARITY_DIFF}",
    )
    ax.set_xlabel("demographic parity difference (lower is fairer)")
    ax.set_ylabel("PR-AUC (higher is better)")
    ax.set_title("Fairness / performance trade-off by mitigation strategy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _log_dataframe(frame: pd.DataFrame, name: str, directory: Path) -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))
    return path


# --------------------------------------------------------------- training


def cross_validate_pr_auc(
    estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray, *, seed: int, folds: int = CV_FOLDS
) -> tuple[float, float]:
    """Stratified 5-fold average precision -- mean and spread.

    The spread is reported because a candidate that beats the baseline by less
    than its own fold-to-fold standard deviation has not beaten it.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


def search_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    grid: Mapping[str, Sequence[Any]] | None = None,
    seed: int = schema.RANDOM_SEED,
    folds: int = CV_FOLDS,
) -> GridSearchCV:
    """Grid search scored on average precision, not accuracy or ROC-AUC.

    Parallelism sits on the search, not on the booster: LightGBM's own n_jobs
    is pinned to 1 in LIGHTGBM_FIXED. Leaving both at -1 oversubscribes every
    core with 5 x n_threads workers and the search gets *slower*, which looks
    exactly like a slow model.
    """
    search = GridSearchCV(
        estimator=make_lightgbm(),
        param_grid=dict(LIGHTGBM_GRID if grid is None else grid),
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search


def fit_group_thresholds(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    sensitive: np.ndarray,
) -> dict[str, float]:
    """The cutoffs the group-aware policy would apply, as plain numbers.

    Fitted here rather than in serving because it needs the training labels,
    which the API container has no business holding.

    Fitted on the whole training split, which is also what the estimator was
    fitted on, and that is a deliberate choice rather than an oversight. The
    in-sample scores are more separated than the ones this model produces on
    batch 5 (measured: sigma 0.202 against 0.188), so the cutoffs carry a
    known bias. Re-estimating them on a 25% stratified slice was tried and is
    worse: across seeds 42/7/2024 it moved the SEX=1 cutoff over 0.423-0.506
    and the realised equalized-odds gap over 0.017-0.037, while the whole-split
    fit sits at 0.451 and 0.018. Trading a small bias for that much variance
    buys nothing, so the bias stays and is written down here instead.

    Returns an empty mapping when fairlearn refuses the sample. Callers get an
    operational fact as data; the base threshold is the documented fallback.
    """
    try:
        optimizer = fit_threshold_optimizer(estimator, X, y, sensitive)
    except ValueError as exc:
        # A group carrying a single label is a property of this batch, not a
        # defect, and it arrives after the grid search has already been paid
        # for. Losing the group-aware policy is survivable; losing the run is
        # not, so this comes back as {} rather than as an exception.
        logger.warning("no group-aware cutoffs from this sample (%s); the base cutoff applies", exc)
        return {}
    return group_thresholds(optimizer)


def score_candidate(
    name: str,
    estimator: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sensitive_test: np.ndarray,
    *,
    capacity_fraction: float,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    """Test metrics and fairness summary for one fitted model."""
    proba = estimator.predict_proba(X_test)[:, 1]
    decisions = top_k_mask(proba, capacity_fraction).astype(int)
    metrics = evaluate(y_test, proba, capacity_fraction=capacity_fraction)
    fairness = fairness_summary(y_test, decisions, sensitive_test)
    logger.info(
        "%s: pr_auc=%.4f roc_auc=%.4f recall@k=%.4f dp_diff=%.4f eo_diff=%.4f",
        name,
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["recall_at_k"],
        fairness["demographic_parity_difference"],
        fairness["equalized_odds_difference"],
    )
    return metrics, fairness, proba, decisions


def train_all(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    *,
    experiment: str | None = None,
    tracking_uri: str | None = None,
    capacity_fraction: float | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    folds: int = CV_FOLDS,
    seed: int = schema.RANDOM_SEED,
    log_to_mlflow: bool = True,
    protected: str = schema.PRIMARY_PROTECTED,
    thresholds_path: Path | None = None,
    processed_dir: Path | None = None,
) -> TrainingResult:
    """Train both families, log both runs, return the better one.

    "Better" is by PR-AUC alone. The fairness gate is evaluated and recorded
    here but is not part of the ranking, because a model can be the strongest
    candidate and still be unregisterable -- and hiding that behind a combined
    score would lose the finding.

    `processed_dir` is where the splits and their manifest are read from;
    the default is the configured data/processed.
    """
    capacity = (
        settings.intervention_capacity_fraction if capacity_fraction is None else capacity_fraction
    )
    if train_df is None or test_df is None:
        train_df, test_df = load_training_splits(processed_dir)
    lineage = data_lineage(train_df, test_df, processed_dir)
    logger.info(
        "data: train %s, test %s, manifest %s",
        lineage.params[TRAIN_SHA_PARAM][:12],
        lineage.params[TEST_SHA_PARAM][:12],
        lineage.params[MANIFEST_PARAM],
    )

    X_train, X_test = feature_frame(train_df), feature_frame(test_df)
    y_train = train_df[schema.TARGET].to_numpy(dtype=int)
    y_test = test_df[schema.TARGET].to_numpy(dtype=int)
    s_train = train_df[protected].to_numpy()
    s_test = test_df[protected].to_numpy()

    if tracking_uri:
        uri, used_fallback = tracking_uri, False
    elif log_to_mlflow:
        uri, used_fallback = resolve_tracking_uri()
    else:
        # Nothing is going to be written, so there is nothing worth probing.
        # Without this branch `--no-mlflow` still makes a live HTTP request to
        # the configured server, and a unit test run on a machine where this
        # project's own compose stack happens to be up takes a different code
        # path than the same test on CI.
        uri, used_fallback = settings.mlflow_tracking_uri, False
    if log_to_mlflow:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(settings.mlflow_experiment if experiment is None else experiment)

    # The leakage probe depends only on the features, so it is computed once
    # and attached to every run rather than per candidate.
    probe = unawareness_probe(X_train, s_train, seed=seed)
    logger.info(
        "unawareness probe: a linear model recovers %s from the remaining features "
        "with ROC-AUC %.3f -- dropping the column does not remove the attribute",
        protected,
        probe["roc_auc"],
    )

    logger.info("cross-validating the logistic baseline")
    logistic = make_logistic_regression(seed=seed)
    lr_cv_mean, lr_cv_std = cross_validate_pr_auc(
        logistic, X_train, y_train, seed=seed, folds=folds
    )
    logistic.fit(X_train, y_train)

    logger.info("grid-searching LightGBM")
    search = search_lightgbm(X_train, y_train, grid=grid, seed=seed, folds=folds)
    booster = search.best_estimator_
    cv_index = int(search.best_index_)
    lgbm_cv_mean = float(search.cv_results_["mean_test_score"][cv_index])
    lgbm_cv_std = float(search.cv_results_["std_test_score"][cv_index])

    # The comparison table uses the tuned configuration, so the price of each
    # mitigation is measured against the model that would actually ship.
    best_params = dict(search.best_params_)
    tradeoff = tradeoff_curve(
        X_train,
        y_train,
        s_train,
        X_test,
        y_test,
        s_test,
        estimator_factory=lambda: make_lightgbm(**best_params),
        capacity_fraction=capacity,
        seed=seed,
    )

    specs = [
        ("logistic_regression", logistic, {"model": "logistic_regression"}, lr_cv_mean, lr_cv_std),
        ("lightgbm", booster, {"model": "lightgbm", **best_params}, lgbm_cv_mean, lgbm_cv_std),
    ]

    candidates: list[CandidateResult] = []
    for name, estimator, params, cv_mean, cv_std in specs:
        metrics, fairness, proba, decisions = score_candidate(
            name, estimator, X_test, y_test, s_test, capacity_fraction=capacity
        )
        gate_ok, reasons = passes_gates(fairness)
        by_attribute = summaries_for_attributes(y_test, decisions, test_df)
        thresholds = fit_group_thresholds(estimator, X_train, y_train, s_train)
        run_id = None
        if log_to_mlflow:
            run_id = _log_run(
                name=name,
                estimator=estimator,
                params={
                    **params,
                    "cv_folds": folds,
                    "seed": seed,
                    "capacity_fraction": capacity,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "grid_size": len(search.cv_results_["params"]),
                    "protected_attribute": protected,
                },
                cv_mean=cv_mean,
                cv_std=cv_std,
                metrics=metrics,
                fairness=fairness,
                fairness_by_attribute=by_attribute,
                thresholds=thresholds,
                probe=probe,
                tradeoff=tradeoff,
                gate_ok=gate_ok,
                reasons=reasons,
                X_test=X_test,
                y_test=y_test,
                proba=proba,
                decisions=decisions,
                sensitive_test=s_test,
                lineage=lineage,
            )
        candidates.append(
            CandidateResult(
                name=name,
                run_id=run_id,
                estimator=estimator,
                params=params,
                cv_pr_auc_mean=cv_mean,
                cv_pr_auc_std=cv_std,
                metrics=metrics,
                fairness=fairness,
                gate_passed=gate_ok,
                gate_reasons=reasons,
                fairness_by_attribute=by_attribute,
                group_thresholds=thresholds,
            )
        )

    best = max(candidates, key=lambda candidate: candidate.metrics["pr_auc"])
    logger.info("best candidate: %s (pr_auc=%.4f)", best.name, best.metrics["pr_auc"])
    # Exported to the shared location serving reads at startup. Only the
    # winner's cutoffs go here; every candidate's are kept on its own run.
    #
    # Gated, because this file is a policy and not a metric. A candidate the
    # gate refuses never reaches the registry, but the API reads these cutoffs
    # off a mounted volume at startup with no idea which run produced them --
    # so an unconditional write lets a model nobody would register decide who
    # gets phoned. Leaving the previous file in place keeps the last policy
    # that did pass, which is the same thing the registry does with the model.
    if best.gate_passed:
        save_group_thresholds(best.group_thresholds, thresholds_path, attribute=protected)
    else:
        logger.warning(
            "%s failed the fairness gate (%s); leaving the exported group thresholds untouched",
            best.name,
            "; ".join(best.gate_reasons) or "no reason recorded",
        )
    return TrainingResult(
        best=best,
        candidates=candidates,
        tracking_uri=uri,
        used_fallback_store=used_fallback,
        tradeoff=tradeoff,
        lineage=dict(lineage.params),
    )


def _log_run(
    *,
    name: str,
    estimator: BaseEstimator,
    params: dict[str, Any],
    cv_mean: float,
    cv_std: float,
    metrics: dict[str, float],
    fairness: dict[str, float],
    fairness_by_attribute: dict[str, dict[str, float]],
    thresholds: dict[str, float],
    probe: dict[str, Any],
    tradeoff: pd.DataFrame,
    gate_ok: bool,
    reasons: Sequence[str],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    proba: np.ndarray,
    decisions: np.ndarray,
    sensitive_test: np.ndarray,
    lineage: DataLineage | None = None,
) -> str:
    """One MLflow run: params, metrics, artifacts, and the model itself."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        if lineage is not None:
            # Params, not tags: they describe the run's inputs and, like the
            # hyperparameters, can never change once logged.
            mlflow.log_params(lineage.params)
            if lineage.manifest_path is not None:
                mlflow.log_artifact(str(lineage.manifest_path), MANIFEST_ARTIFACT_DIR)
        mlflow.log_metrics(
            {
                "cv_pr_auc_mean": cv_mean,
                "cv_pr_auc_std": cv_std,
                **{f"test_{key}": value for key, value in metrics.items()},
                **{f"fair_{key}": value for key, value in fairness.items()},
                **{
                    key: value
                    for attribute, summary in fairness_by_attribute.items()
                    for key, value in flatten_summary(summary, f"by_{attribute}").items()
                },
                "unawareness_probe_roc_auc": float(probe["roc_auc"]),
                **{
                    f"calibration_ece_group_{group}": value
                    for group, value in calibration_error_by_group(
                        y_test, proba, sensitive_test
                    ).items()
                },
            }
        )
        # Tagged, not just logged: the gate outcome is what someone filters the
        # experiment list by when asked "which models were rejected and why".
        mlflow.set_tags(
            {
                "model_family": name,
                "fairness_gate": "pass" if gate_ok else "fail",
                "fairness_gate_reasons": "; ".join(reasons) or "none",
                "protected_attribute": params.get("protected_attribute", schema.PRIMARY_PROTECTED),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _log_dataframe(tradeoff, "fairness_tradeoff.csv", directory)
            mlflow.log_artifact(
                str(save_tradeoff_plot(tradeoff, directory / "fairness_tradeoff.png"))
            )
            _log_dataframe(cost_matrix_report(y_test, proba), "cost_matrix.csv", directory)
            _log_dataframe(
                group_report(y_test, decisions, sensitive_test).reset_index(),
                "group_report.csv",
                directory,
            )
            importances = feature_importance_frame(estimator, X_test.columns)
            _log_dataframe(importances, "feature_importance.csv", directory)
            mlflow.log_artifact(
                str(save_feature_importance_plot(importances, directory / "feature_importance.png"))
            )
            probe_path = directory / "unawareness_probe.json"
            probe_path.write_text(json.dumps(probe, indent=2))
            mlflow.log_artifact(str(probe_path))
            mlflow.log_artifact(
                str(
                    save_group_thresholds(
                        thresholds,
                        directory / "group_thresholds.json",
                        attribute=str(params.get("protected_attribute", schema.PRIMARY_PROTECTED)),
                    )
                )
            )

        # An input example plus a signature so the serving container fails at
        # load time on a schema mismatch instead of at 3am on a bad row.
        sample = X_test.head(5)
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            signature=infer_signature(sample, estimator.predict_proba(sample)),
            input_example=sample,
        )
        return str(run.info.run_id)


# ------------------------------------------------------- hand-off artefact

# The DAG runs train, evaluate and register as three separate subprocesses, so
# nothing survives in memory between them. This file is the hand-off: it is
# what `evaluate_and_gate` reads to decide, and what `register_model` reads to
# know which run to register. Writing it is what makes the pipeline steps
# genuinely independent rather than three functions pretending to be tasks.
TRAINING_RESULT_FILE: Final = "training_result.json"


def training_result_path(processed_dir: Path | None = None) -> Path:
    directory = settings.processed_dir if processed_dir is None else processed_dir
    return directory / TRAINING_RESULT_FILE


def save_training_result(result: TrainingResult, path: Path | None = None) -> Path:
    """Persist everything the downstream tasks need about the winning run."""
    target = training_result_path() if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.best.run_id,
        "model": result.best.name,
        "tracking_uri": result.tracking_uri,
        "used_fallback_store": result.used_fallback_store,
        "metrics": result.best.metrics,
        "fairness": result.best.fairness,
        "fairness_by_attribute": result.best.fairness_by_attribute,
        "gate_passed": result.best.gate_passed,
        "gate_reasons": result.best.gate_reasons,
        "group_thresholds": result.best.group_thresholds,
        "data_lineage": result.lineage,
        "candidates": [
            {
                "name": c.name,
                "run_id": c.run_id,
                "pr_auc": c.metrics.get("pr_auc"),
                "gate_passed": c.gate_passed,
                "gate_reasons": c.gate_reasons,
            }
            for c in result.candidates
        ],
        "tradeoff": result.tradeoff.to_dict(orient="records"),
    }
    target.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", target)
    return target


def load_training_result(path: Path | None = None) -> dict[str, Any]:
    """Read the hand-off artefact, with a message that says what to run first."""
    target = training_result_path() if path is None else path
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found -- run `python -m credit_risk.models.train` first"
        )
    return dict(json.loads(target.read_text()))


# ------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m credit_risk.models.train`."""
    parser = argparse.ArgumentParser(description="Train the credit-risk candidates")
    parser.add_argument("--experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default=None, help="override the MLflow tracking URI")
    parser.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="intervention capacity fraction (default: from settings)",
    )
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=schema.RANDOM_SEED)
    parser.add_argument(
        "--no-mlflow", action="store_true", help="train without touching any tracking store"
    )
    # Explicit, so a caller -- a test, or a DAG run pointed at a scratch
    # directory -- can keep the hand-off artefact out of data/processed instead
    # of overwriting the one the running pipeline is about to read.
    parser.add_argument(
        "--result", type=Path, default=None, help="where to write training_result.json"
    )
    parser.add_argument(
        "--thresholds", type=Path, default=None, help="where to export the group cutoffs"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    result = train_all(
        experiment=args.experiment,
        tracking_uri=args.tracking_uri,
        capacity_fraction=args.capacity,
        folds=args.folds,
        seed=args.seed,
        log_to_mlflow=not args.no_mlflow,
        thresholds_path=args.thresholds,
    )
    save_training_result(result, args.result)
    summary = {"run_id": result.run_id, "model": result.best.name, **result.metrics}
    print(json.dumps(summary, indent=2))
    print(result.tradeoff.to_string(index=False))
    # Non-zero exit when the winner cannot be registered, so the DAG task and a
    # human running this by hand learn the same thing at the same time.
    if result.best.gate_passed:
        return 0
    # The DAG stops here, so evaluate_and_gate never tags this run. Tag it with
    # the full gate's reasons, PR-AUC included; otherwise a combined refusal
    # reads as fairness-only.
    record_refusal(
        result.run_id,
        evaluate_gate(result.metrics, result.best.fairness),
        tracking_uri=result.tracking_uri,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
