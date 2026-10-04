"""The gate between a trained model and a model that serves traffic.

Registration is a decision, not a step. A run that produced a model is not
automatically a model anybody should be scored by, so this module answers one
question -- may this version be registered -- and returns the answer as data.

Nothing here raises when the answer is no. A candidate that fails the fairness
gate is a valid, informative experiment: the refusal and its reasons are
written back onto the run as tags so the rejection survives in MLflow next to
the metrics that caused it. A model card that only lists the models that
passed is a marketing document.

The registry itself may also be missing -- an empty registry on the first run,
a file-backed store that does not implement one, an unreachable server. Those
are operational facts, not programming errors, and they come back as a
RegistrationDecision with a reason too.

A registered version also carries, as tags, what serving needs to apply it:
the capacity threshold the evaluation computed, the capacity it was computed
for, when the run trained, from which commit, and the run id. MLflow keeps the
run id and a creation time on a version but not the cutoff, and until these
tags existed the cutoff lived only as a run metric nothing on the serving side
read. `python -m credit_risk.models.registry tag-threshold --version N` fills
the tags in on a version registered before they existed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.client import MlflowClient

from credit_risk.config import settings
from credit_risk.fairness.metrics import passes_gates

logger = logging.getLogger(__name__)

# MLflow's stores do not agree on an exception type. The SQL-backed registry
# raises MlflowException; the file-backed one raises a bare Exception when the
# model directory does not exist -- which is precisely the first-run case this
# module was written to survive. The contract here is that nothing raises, so
# the catch is deliberately wide and every branch logs what it swallowed.
StoreError = Exception

# The model-quality half of the gate. Fairness thresholds live in `schema`
# because compliance owns those; this one is the risk team's number.
MIN_PR_AUC: float = 0.54

# Metric dicts arrive either straight from `evaluate` (`pr_auc`) or read back
# out of an MLflow run, where training prefixed them (`test_pr_auc`). Accepting
# both beats making every caller remember which one it holds.
PR_AUC_KEYS: tuple[str, ...] = ("pr_auc", "test_pr_auc")

# Tag names on a registered model version. `serving.model_loader` reads them
# back; the names are the contract between the two, so they are defined once,
# here, where they are written.
THRESHOLD_TAG: Final = "threshold_at_k"
CAPACITY_TAG: Final = "capacity_fraction"
TRAINED_AT_TAG: Final = "trained_at"
GIT_SHA_TAG: Final = "git_sha"
RUN_ID_TAG: Final = "run_id"
# "registration" or "backfill". A tag filled in after the fact is a claim made
# later about an older run, and whoever reads it should be able to tell.
TAGGED_BY_TAG: Final = "tagged_by"

# Same prefix problem as PR_AUC_KEYS: unprefixed in the training hand-off,
# `test_`-prefixed on the run.
THRESHOLD_KEYS: tuple[str, ...] = ("threshold_at_k", "test_threshold_at_k")
CAPACITY_KEYS: tuple[str, ...] = ("capacity_fraction", "test_capacity_fraction")

# Set by MLflow itself when a run starts inside a git checkout. The only commit
# a backfill can honestly attribute to an old run; the environment of whoever
# runs the backfill describes their checkout, not the one that trained.
RUN_GIT_COMMIT_TAG: Final = "mlflow.source.git.commit"

TAG_THRESHOLD_COMMAND: Final = "tag-threshold"


@dataclass(frozen=True)
class RegistrationDecision:
    """Whether a version was registered, and why not if it was not."""

    registered: bool
    model_name: str
    version: str | None = None
    stage: str | None = None
    reasons: list[str] = field(default_factory=list)
    # The tags actually written onto the new version; empty when none were.
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return not self.registered


@dataclass(frozen=True)
class VersionMetadata:
    """One registered version as serving sees it: number, run, tags."""

    model_name: str
    version: str
    run_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def model_uri(self) -> str:
        """The URI of exactly this version, not of whatever a stage points at."""
        return f"models:/{self.model_name}/{self.version}"

    @property
    def trained_at(self) -> str | None:
        return self.tags.get(TRAINED_AT_TAG)


@dataclass(frozen=True)
class BackfillResult:
    """What `tag-threshold` found on a version and what it wrote."""

    model_name: str
    version: str
    run_id: str | None = None
    written: dict[str, str] = field(default_factory=dict)
    unchanged: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.reasons


def _client(client: MlflowClient | None = None) -> MlflowClient:
    """The caller's client, or one pointed at whatever URI is configured."""
    return client if client is not None else MlflowClient()


def _read_metric(metrics: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """The first of `keys` present in `metrics`, as a float."""
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def _iso_utc(epoch_ms: int | None) -> str:
    """MLflow's millisecond epoch as the UTC timestamp the API reports elsewhere."""
    if epoch_ms is None:
        return "unknown"
    moment = datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _tag_refusal(run_id: str, reasons: list[str], client: MlflowClient) -> None:
    """Record the refusal on the run that caused it.

    Best effort on purpose: if the tracking server has gone away we still want
    the caller to receive the decision. Losing a tag is survivable; swallowing
    the whole gate result is not.
    """
    try:
        client.set_tag(run_id, "registration_refused", "true")
        client.set_tag(run_id, "registration_refusal_reasons", "; ".join(reasons))
    except StoreError as exc:
        logger.warning("could not tag run %s with the refusal: %s", run_id, exc)


def evaluate_gate(metrics: dict[str, float], fairness_summary: dict[str, float]) -> list[str]:
    """Reasons this candidate must not be registered. Empty means it may be.

    Performance and fairness are checked independently and *both* sets of
    reasons are returned. Short-circuiting would hide the second problem until
    the first was fixed, which turns one retraining cycle into two.
    """
    reasons: list[str] = []
    pr_auc = _read_metric(metrics, PR_AUC_KEYS)
    if pr_auc is None:
        reasons.append(f"no PR-AUC in metrics (looked for {', '.join(PR_AUC_KEYS)})")
    elif pr_auc < MIN_PR_AUC:
        reasons.append(f"pr_auc {pr_auc:.4f} is below the minimum {MIN_PR_AUC:.2f}")

    _, fairness_reasons = passes_gates(fairness_summary)
    reasons.extend(fairness_reasons)
    return reasons


def version_tags(
    run_id: str,
    metrics: Mapping[str, Any] | None = None,
    *,
    tagged_by: str,
    git_sha: str | None = None,
    client: MlflowClient | None = None,
) -> dict[str, str]:
    """The tags a version registered from `run_id` should carry.

    Deterministic for a given run: `trained_at` is the run's own start time, not
    the moment this function ran, so tagging the same version twice writes the
    same values twice. The threshold and capacity come from `metrics` when the
    caller holds them (the training hand-off) and from the run's logged metrics
    otherwise.

    A value that cannot be found is left out, never invented. A version with no
    `threshold_at_k` tag is exactly what makes serving fall back to the
    configured cutoff -- and report that it did -- so a guessed tag would hide
    the one fact the fallback exists to surface.

    `git_sha=None` means "the commit MLflow recorded on the run, else unknown",
    which is the only honest choice for a backfill.
    """
    active = _client(client)
    run: Any = None
    try:
        run = active.get_run(run_id)
    except StoreError as exc:
        logger.warning("could not read run %s for its version tags: %s", run_id, exc)

    logged: dict[str, Any] = dict(run.data.metrics) if run is not None else {}
    params: dict[str, Any] = dict(run.data.params) if run is not None else {}
    run_tags: dict[str, str] = dict(run.data.tags) if run is not None else {}
    supplied: dict[str, Any] = dict(metrics or {})

    commit = run_tags.get(RUN_GIT_COMMIT_TAG, "unknown") if git_sha is None else git_sha
    tags = {
        RUN_ID_TAG: run_id,
        TRAINED_AT_TAG: _iso_utc(run.info.start_time if run is not None else None),
        GIT_SHA_TAG: commit,
        TAGGED_BY_TAG: tagged_by,
    }

    threshold = _read_metric(supplied, THRESHOLD_KEYS)
    if threshold is None:
        threshold = _read_metric(logged, THRESHOLD_KEYS)
    if threshold is not None:
        # repr round-trips a float exactly, so serving reads back the very
        # number the evaluation computed rather than a rounded neighbour of it.
        tags[THRESHOLD_TAG] = repr(threshold)

    # The realised fraction evaluate() reports first, the requested one logged
    # as a parameter second. They differ only when n * capacity is not whole.
    capacity = _read_metric(supplied, CAPACITY_KEYS)
    if capacity is None:
        capacity = _read_metric(logged, CAPACITY_KEYS)
    if capacity is None:
        capacity = _read_metric(params, ("capacity_fraction",))
    if capacity is not None:
        tags[CAPACITY_TAG] = repr(capacity)
    return tags


def tag_version(
    name: str,
    version: str | int,
    tags: Mapping[str, str],
    *,
    client: MlflowClient | None = None,
) -> list[str]:
    """Write `tags` onto a model version; return the keys that could not be written.

    Idempotent: MLflow overwrites a version tag in place, so writing the same
    tags twice leaves the version exactly as writing them once did. Best effort
    per key for the same reason `_tag_refusal` is -- a version that lost one tag
    is still a version, and the caller is told which tag it lost.
    """
    active = _client(client)
    failed: list[str] = []
    for key, value in tags.items():
        try:
            active.set_model_version_tag(name, str(version), key, value)
        except StoreError as exc:
            logger.warning("could not tag %s v%s with %s: %s", name, version, key, exc)
            failed.append(key)
    return failed


def _warn_untagged_threshold(name: str, version: str) -> None:
    logger.warning(
        "%s v%s carries no %s tag: serving will decide at the configured "
        "DECISION_THRESHOLD and report threshold_source=fallback. Fill it in with "
        "`python -m credit_risk.models.registry %s --version %s`",
        name,
        version,
        THRESHOLD_TAG,
        TAG_THRESHOLD_COMMAND,
        version,
    )


def register_if_passes(
    run_id: str,
    metrics: dict[str, float],
    fairness_summary: dict[str, float],
    *,
    model_name: str | None = None,
    artifact_path: str = "model",
    client: MlflowClient | None = None,
) -> RegistrationDecision:
    """Register the run's model only if it clears performance and fairness."""
    name = settings.model_name if model_name is None else model_name
    reasons = evaluate_gate(metrics, fairness_summary)
    active = _client(client)

    if reasons:
        logger.warning("refusing to register run %s: %s", run_id, "; ".join(reasons))
        _tag_refusal(run_id, reasons, active)
        return RegistrationDecision(registered=False, model_name=name, reasons=reasons)

    try:
        version = mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=name)
    except StoreError as exc:
        # An unreachable or unsupported registry is an operational fact, not a
        # verdict on the model. Report it as a refusal with a reason rather
        # than failing a pipeline whose model was fine.
        reason = f"registry unavailable: {exc}"
        logger.warning("%s", reason)
        return RegistrationDecision(registered=False, model_name=name, reasons=[reason])

    try:
        active.set_tag(run_id, "registered_version", str(version.version))
    except StoreError as exc:  # pragma: no cover - needs a half-dead server
        logger.warning("registered version %s but could not tag the run: %s", version.version, exc)

    # The version carries its own decision threshold from the moment it exists,
    # so promoting it moves the cutoff with the model instead of leaving the
    # previous one -- or a constant -- in charge.
    tags = version_tags(
        run_id, metrics, tagged_by="registration", git_sha=settings.git_sha, client=active
    )
    failed = tag_version(name, version.version, tags, client=active)
    written = {key: value for key, value in tags.items() if key not in failed}
    if THRESHOLD_TAG not in written:
        _warn_untagged_threshold(name, str(version.version))

    logger.info("registered %s version %s from run %s", name, version.version, run_id)
    return RegistrationDecision(
        registered=True,
        model_name=name,
        version=str(version.version),
        stage=str(getattr(version, "current_stage", None) or "None"),
        tags=written,
    )


def promote(
    version: str | int,
    stage: str = "Production",
    *,
    model_name: str | None = None,
    archive_existing: bool = True,
    client: MlflowClient | None = None,
) -> RegistrationDecision:
    """Move a registered version to a stage.

    `archive_existing=True` so exactly one version is ever in Production. The
    serving container resolves `models:/credit-risk/Production` at startup, and
    two versions sharing that stage means whichever one MLflow happens to
    return -- an ambiguity you only notice when the two disagree.
    """
    name = settings.model_name if model_name is None else model_name
    active = _client(client)
    try:
        active.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )
    except StoreError as exc:
        reason = f"could not promote {name} v{version} to {stage}: {exc}"
        logger.warning("%s", reason)
        return RegistrationDecision(
            registered=False, model_name=name, version=str(version), reasons=[reason]
        )
    logger.info("promoted %s version %s to %s", name, version, stage)
    return RegistrationDecision(registered=True, model_name=name, version=str(version), stage=stage)


def current_production_version(
    *,
    model_name: str | None = None,
    stage: str | None = None,
    client: MlflowClient | None = None,
) -> str | None:
    """The version currently serving, or None on an empty or absent registry.

    None is a legitimate answer on the first run of a fresh stack, so callers
    get it as a value. `/health` reports `degraded` from it instead of the API
    refusing to start.
    """
    name = settings.model_name if model_name is None else model_name
    target = settings.model_stage if stage is None else stage
    active = _client(client)
    try:
        versions = active.get_latest_versions(name, stages=[target])
    except StoreError as exc:
        logger.warning("no registry entry for %s at stage %s: %s", name, target, exc)
        return None
    if not versions:
        logger.info("model %s has no version in stage %s yet", name, target)
        return None
    return str(versions[0].version)


def version_metadata(
    version: str | int,
    *,
    model_name: str | None = None,
    client: MlflowClient | None = None,
) -> VersionMetadata | None:
    """Number, run id and tags of one version, or None if the registry cannot say."""
    name = settings.model_name if model_name is None else model_name
    active = _client(client)
    try:
        found = active.get_model_version(name, str(version))
    except StoreError as exc:
        logger.warning("could not read %s v%s from the registry: %s", name, version, exc)
        return None
    return VersionMetadata(
        model_name=name,
        version=str(found.version),
        run_id=str(found.run_id) if found.run_id else None,
        tags={str(key): str(value) for key, value in (found.tags or {}).items()},
    )


def production_version_metadata(
    *,
    model_name: str | None = None,
    stage: str | None = None,
    client: MlflowClient | None = None,
) -> VersionMetadata | None:
    """The version in `stage` with its tags, or None on an empty or absent registry.

    Serving loads the model by this version's own URI rather than by stage. Two
    lookups by stage -- one for the model, one for its tags -- would apply the
    tags of whichever version was promoted in between to the model loaded first.
    """
    name = settings.model_name if model_name is None else model_name
    active = _client(client)
    version = current_production_version(model_name=name, stage=stage, client=active)
    if version is None:
        return None
    return version_metadata(version, model_name=name, client=active)


def backfill_version_tags(
    version: str | int,
    *,
    model_name: str | None = None,
    dry_run: bool = False,
    client: MlflowClient | None = None,
) -> BackfillResult:
    """Fill in the tags a version registered before they existed is missing.

    Only missing tags are written; a tag already on the version is left alone,
    because the value written at registration describes the run better than one
    reconstructed later (`git_sha` especially). That also makes a second run of
    the backfill a no-op. The threshold comes from the run's logged
    `test_threshold_at_k`; a run that never logged one gets nothing written,
    since a version tagged with everything except the threshold would look
    finished and still decide at the fallback.
    """
    name = settings.model_name if model_name is None else model_name
    active = _client(client)
    found = version_metadata(version, model_name=name, client=active)
    if found is None:
        return BackfillResult(
            model_name=name,
            version=str(version),
            reasons=[f"{name} v{version} is not in the registry, or the registry is unreachable"],
            dry_run=dry_run,
        )
    if not found.run_id:
        return BackfillResult(
            model_name=name,
            version=found.version,
            reasons=[f"{name} v{found.version} records no run id to read the threshold from"],
            dry_run=dry_run,
        )

    wanted = version_tags(found.run_id, tagged_by="backfill", client=active)
    if THRESHOLD_TAG not in wanted and THRESHOLD_TAG not in found.tags:
        return BackfillResult(
            model_name=name,
            version=found.version,
            run_id=found.run_id,
            reasons=[
                f"run {found.run_id} logged no {' or '.join(THRESHOLD_KEYS)}; nothing was written"
            ],
            dry_run=dry_run,
        )

    missing = {key: value for key, value in wanted.items() if key not in found.tags}
    unchanged = {key: found.tags[key] for key in wanted if key in found.tags}
    current = found.tags.get(THRESHOLD_TAG)
    logged = wanted.get(THRESHOLD_TAG)
    if current is not None and logged is not None and current != logged:
        logger.warning(
            "%s v%s already carries %s=%s but its run logged %s; leaving the tag as it is",
            name,
            found.version,
            THRESHOLD_TAG,
            current,
            logged,
        )

    failed = [] if dry_run else tag_version(name, found.version, missing, client=active)
    return BackfillResult(
        model_name=name,
        version=found.version,
        run_id=found.run_id,
        written={key: value for key, value in missing.items() if key not in failed},
        unchanged=unchanged,
        reasons=[f"could not write tag {key}" for key in failed],
        dry_run=dry_run,
    )


def load_production_model(*, model_uri: str | None = None) -> Any | None:
    """Load the serving model, or return None with the reason logged.

    The sklearn flavour, not pyfunc, and the distinction is not cosmetic. Pyfunc
    hands back a `PyFuncModel` wrapper that exposes `predict` and nothing else:
    no `predict_proba`, so a ranking service cannot rank, and no underlying tree
    structure, so `shap.TreeExplainer` has nothing to read. Both failures appear
    only at the first request, well after start-up has reported success.

    Pyfunc remains the fallback, because a model logged under some other flavour
    is still better served degraded than not at all -- and the caller finds out
    from the attribute error rather than from silence.

    Returning None rather than raising is what lets the API come up degraded:
    the container is reachable, `/health` says model_loaded=0, Prometheus fires
    ModelNotLoaded, and an engineer sees a specific alert instead of a crash
    loop with no metrics at all.
    """
    uri = settings.model_uri if model_uri is None else model_uri
    try:
        return mlflow.sklearn.load_model(uri)
    except StoreError as exc:
        logger.warning("could not load %s: %s", uri, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - flavour mismatch, not a store failure
        logger.warning("sklearn flavour unavailable for %s (%s); falling back to pyfunc", uri, exc)
        try:
            return mlflow.pyfunc.load_model(uri)
        except Exception as fallback_exc:  # noqa: BLE001
            logger.warning("could not load %s: %s", uri, fallback_exc)
            return None


# ------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m credit_risk.models.registry`.

    With no command it registers the winning run and promotes it, both guarded
    by the gate -- the DAG's `register_model` task, called exactly as before.
    `tag-threshold --version N` is the one-off backfill for a version that was
    registered before versions carried their threshold.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == TAG_THRESHOLD_COMMAND:
        return _tag_threshold_main(args[1:])
    return _register_main(args)


def _tag_threshold_main(argv: Sequence[str]) -> int:
    """`tag-threshold`: fill in the version tags serving reads, from the run's metrics.

    Exit 0 when the version ends up carrying a threshold, 2 when it does not --
    the same convention as the registration command, so a script can tell a
    backfill that worked from one that found nothing to write.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m credit_risk.models.registry {TAG_THRESHOLD_COMMAND}",
        description=(
            "Tag an already-registered model version with the capacity threshold its "
            "run logged, so serving can decide at it without retraining."
        ),
    )
    parser.add_argument("--version", required=True, help="registered version number, e.g. 2")
    parser.add_argument("--model-name", default=settings.model_name, help="registered model name")
    parser.add_argument(
        "--tracking-uri",
        default=settings.mlflow_tracking_uri,
        help="MLflow tracking server (default: MLFLOW_TRACKING_URI, else http://localhost:15020)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the tags that would be written; write none"
    )
    args = parser.parse_args(list(argv))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    mlflow.set_tracking_uri(args.tracking_uri)

    result = backfill_version_tags(args.version, model_name=args.model_name, dry_run=args.dry_run)
    payload: dict[str, Any] = {
        "model_name": result.model_name,
        "version": result.version,
        "run_id": result.run_id,
        "dry_run": result.dry_run,
        "written": result.written,
        "unchanged": result.unchanged,
        "reasons": result.reasons,
    }
    print(json.dumps(payload, indent=2))
    if not result.ok:
        print(f"\nTAG-THRESHOLD FAILED: {'; '.join(result.reasons)}", file=sys.stderr)
        return 2
    if not result.dry_run:
        print(
            "\nThe API reads version tags when it loads the model: restart it "
            "(docker compose restart credit-api) and check /health for "
            '"threshold_source": "registry".',
            file=sys.stderr,
        )
    return 0


def _register_main(argv: Sequence[str]) -> int:
    """Register the winning run and promote it, both guarded by the gate.

    A separate DAG task from evaluation on purpose: the gate decides, this
    acts, and keeping them apart means the log tells you whether a missing
    Production model is a refused candidate or a broken registry.
    """
    # Imported inside the function: `models.train` imports `fairness.metrics`
    # and this module's gate, so pulling it in at module scope is a cycle.
    from credit_risk.models.train import load_training_result, training_result_path

    parser = argparse.ArgumentParser(
        description=f"Register and promote the gated candidate. See also: {TAG_THRESHOLD_COMMAND}"
    )
    parser.add_argument("--result", type=Path, default=None, help="path to training_result.json")
    parser.add_argument("--stage", default=settings.model_stage, help="stage to promote into")
    parser.add_argument(
        "--no-promote", action="store_true", help="register the version but leave it unstaged"
    )
    args = parser.parse_args(list(argv))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )

    result = load_training_result(args.result)
    run_id = result.get("run_id")
    if not run_id:
        print(
            "no run id in the training result -- was training run with --no-mlflow?",
            file=sys.stderr,
        )
        return 2

    # The tracking URI travels in the artefact so this process talks to the
    # same store training used, even when it fell back to a local ./mlruns.
    tracking_uri = result.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    decision = register_if_passes(run_id, result["metrics"], result["fairness"])
    payload: dict[str, Any] = {
        "registered": decision.registered,
        "model_name": decision.model_name,
        "version": decision.version,
        "reasons": decision.reasons,
        "version_tags": decision.tags,
    }

    if decision.registered and decision.version and not args.no_promote:
        promotion = promote(decision.version, args.stage)
        payload["promoted"] = promotion.registered
        payload["stage"] = promotion.stage
        payload["promotion_reasons"] = promotion.reasons

    out = (
        training_result_path(args.result.parent if args.result else None).parent
        / "registration.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    return 0 if decision.registered else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
