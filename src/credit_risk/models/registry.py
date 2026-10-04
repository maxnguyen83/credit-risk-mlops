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

Passing the gate makes a candidate registrable, not better than what serves.
Registration therefore compares the candidate's held-out PR-AUC with the
champion's -- the version serving resolves, `models:/<name>@champion` -- and
promotes only when there is no champion or the candidate is not worse beyond
`PROMOTION_PR_AUC_TOLERANCE`. Anything else is registered as the challenger
(alias `challenger`, stage Staging) with a tag saying why, and waits for a
person: `python -m credit_risk.models.registry set-champion --version N` is the
promotion after review, and also sets the alias on a version promoted before
aliases were used. The Production stage is still set on every promotion so
anything that reads stages keeps working; serving falls back to it only when
no version carries the alias.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.artifacts
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
SET_CHAMPION_COMMAND: Final = "set-champion"

# The champion alias is `settings.model_alias`. A candidate that passed the gate
# but lost the comparison is parked here, where nothing serves it.
CHALLENGER_ALIAS: Final = "challenger"
CHALLENGER_STAGE: Final = "Staging"

# Why a version holds the alias it holds, on the version itself: "champion" or
# "challenger", the comparison in words, and the version it was compared with.
PROMOTION_DECISION_TAG: Final = "promotion_decision"
PROMOTION_REASON_TAG: Final = "promotion_reason"
COMPARED_WITH_TAG: Final = "compared_with_version"

# The run param training writes with the content hash of the held-out split
# (`models.train.TEST_SHA_PARAM`; a test pins the two together). Named here
# rather than imported because `train` pulls in LightGBM and scikit-learn, and
# the API imports this module.
TEST_SPLIT_PARAM: Final = "data_test_sha256"

# PR-AUC differences this small are float noise, not a verdict.
_EPSILON: Final = 1e-12

# The error codes MLflow uses for "there is nothing there": a model, version
# or alias that does not exist. Measured on the SQL, file and REST stores of
# MLflow 2.19 -- a missing alias is INVALID_PARAMETER_VALUE on all three. Any
# other failure (a 5xx, a timeout, an exhausted DB pool) means the registry
# could not be read, which is not the same as the registry being empty.
NOT_FOUND_ERRORS: Final = frozenset({"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"})

# Exit code of the registration command when the candidate was registered but
# the promotion it earned -- champion or challenger -- could not be written.
# Distinct from 2 (refused by the gate, or nothing to register), and non-zero
# so the DAG's register_model task fails rather than going green over a
# half-written registry.
EXIT_PROMOTION_FAILED: Final = 3


class RegistryUnreadable(Exception):
    """A registry lookup failed for a reason other than "not found"."""


def _not_found(exc: BaseException) -> bool:
    return getattr(exc, "error_code", None) in NOT_FOUND_ERRORS


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
    alias: str | None = None
    # What a promotion changed ("alias champion", "stage Production", ...);
    # empty when the version already stood where it was asked to.
    changed: list[str] = field(default_factory=list)

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
    current_stage: str | None = None
    aliases: tuple[str, ...] = ()
    # How this version was found for serving: "alias" or "stage", and the URI
    # that names it that way. None when it was looked up by number.
    resolved_by: str | None = None
    resolved_uri: str | None = None
    # Lookups that failed for a reason other than "not found" on the way to
    # this version -- an alias that could not be read before the stage
    # answered. Empty when the resolution was clean.
    resolution_errors: tuple[str, ...] = ()

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
    """Move a registered version to a stage. Stages only; aliases are `set_champion`'s.

    `archive_existing=True` so exactly one version is ever in a stage. Serving
    falls back to `models:/credit-risk/Production` when no version carries the
    champion alias, and two versions sharing that stage means whichever one
    MLflow happens to return -- an ambiguity you only notice when the two
    disagree.
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
    try:
        return _stage_lookup(name, target, _client(client))
    except RegistryUnreadable as exc:
        logger.warning("%s", exc)
        return None


def _stage_lookup(name: str, stage: str, client: MlflowClient) -> str | None:
    """The version in `stage`; None if none is; RegistryUnreadable if the store cannot say."""
    try:
        versions = client.get_latest_versions(name, stages=[stage])
    except StoreError as exc:
        if not _not_found(exc):
            raise RegistryUnreadable(f"could not read {name} at stage {stage}: {exc}") from exc
        logger.info("no registry entry for %s at stage %s: %s", name, stage, exc)
        return None
    if not versions:
        logger.info("model %s has no version in stage %s yet", name, stage)
        return None
    return str(versions[0].version)


def _alias_lookup(name: str, alias: str, client: MlflowClient) -> str | None:
    """The version `alias` names; None if it names none; RegistryUnreadable if unknowable."""
    try:
        found = client.get_model_version_by_alias(name, alias)
    except StoreError as exc:
        if not _not_found(exc):
            raise RegistryUnreadable(f"could not read {name}@{alias}: {exc}") from exc
        logger.info("%s@%s resolves to nothing: %s", name, alias, exc)
        return None
    return str(found.version)


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
    return _as_metadata(name, found)


def _as_metadata(name: str, found: Any) -> VersionMetadata:
    """An MLflow ModelVersion as the plain record the rest of the code reads."""
    stage = getattr(found, "current_stage", None)
    return VersionMetadata(
        model_name=name,
        version=str(found.version),
        run_id=str(found.run_id) if found.run_id else None,
        tags={str(key): str(value) for key, value in (found.tags or {}).items()},
        current_stage=str(stage) if stage else None,
        aliases=tuple(str(alias) for alias in (getattr(found, "aliases", None) or ())),
    )


def alias_version(
    alias: str | None = None,
    *,
    model_name: str | None = None,
    client: MlflowClient | None = None,
) -> str | None:
    """The version number `alias` points at, or None if it points nowhere or cannot be read.

    Lenient, for callers that write next: `set_champion` sets the alias when
    this is not already its version, and a registry that cannot be read here
    refuses that write too. Decisions that read only -- the comparison with the
    champion -- use `resolve_champion`, which tells the two apart.
    """
    name = settings.model_name if model_name is None else model_name
    wanted = settings.model_alias if alias is None else alias
    try:
        return _alias_lookup(name, wanted, _client(client))
    except RegistryUnreadable as exc:
        logger.warning("%s", exc)
        return None


def resolve_champion(
    *,
    model_name: str | None = None,
    stage: str | None = None,
    alias: str | None = None,
    client: MlflowClient | None = None,
) -> VersionMetadata | None:
    """The champion -- alias first, then stage -- or None only if the registry has none.

    Strict: any lookup that fails for a reason other than "not found" raises
    RegistryUnreadable. An unreadable registry is not an empty one, and the
    comparison that decides whether to promote must not mistake one for the
    other.
    """
    name = settings.model_name if model_name is None else model_name
    wanted_alias = settings.model_alias if alias is None else alias
    wanted_stage = settings.model_stage if stage is None else stage
    active = _client(client)

    version = _alias_lookup(name, wanted_alias, active)
    resolved_by, resolved_uri = "alias", f"models:/{name}@{wanted_alias}"
    if version is None:
        version = _stage_lookup(name, wanted_stage, active)
        if version is None:
            return None
        resolved_by, resolved_uri = "stage", f"models:/{name}/{wanted_stage}"
    try:
        found = active.get_model_version(name, version)
    except StoreError as exc:
        raise RegistryUnreadable(f"could not read {name} v{version}: {exc}") from exc
    return replace(_as_metadata(name, found), resolved_by=resolved_by, resolved_uri=resolved_uri)


def production_version_metadata(
    *,
    model_name: str | None = None,
    stage: str | None = None,
    alias: str | None = None,
    client: MlflowClient | None = None,
) -> VersionMetadata | None:
    """The version serving should load, with its tags, or None if there is none.

    The champion alias first, the stage second. The stage is the fallback for a
    registry populated before aliases were used -- this project's own version 2
    is one -- and `resolved_by` says which of the two answered, so `/health`
    can report it rather than leave it to be inferred.

    Lenient, because serving would rather run on the stage than not run: an
    alias that cannot be read (as opposed to one that does not exist) also
    falls back to the stage, but the error is logged at WARNING and kept in
    `resolution_errors` instead of being swallowed.

    Serving loads the model by this version's own URI rather than by alias or
    stage. Two lookups by name -- one for the model, one for its tags -- would
    apply the tags of whichever version was promoted in between to the model
    loaded first.
    """
    name = settings.model_name if model_name is None else model_name
    wanted_alias = settings.model_alias if alias is None else alias
    wanted_stage = settings.model_stage if stage is None else stage
    active = _client(client)

    errors: list[str] = []
    try:
        version = _alias_lookup(name, wanted_alias, active)
    except RegistryUnreadable as exc:
        errors.append(str(exc))
        logger.warning("%s; falling back to the %s stage", exc, wanted_stage)
        version = None
    resolved_by, resolved_uri = "alias", f"models:/{name}@{wanted_alias}"
    if version is None:
        version = current_production_version(model_name=name, stage=wanted_stage, client=active)
        if version is None:
            return None
        resolved_by, resolved_uri = "stage", f"models:/{name}/{wanted_stage}"
        if not errors:
            logger.warning(
                "no version of %s carries the %r alias; using version %s from the %s stage. "
                "`python -m credit_risk.models.registry %s --version %s` sets the alias",
                name,
                wanted_alias,
                version,
                wanted_stage,
                SET_CHAMPION_COMMAND,
                version,
            )
    found = version_metadata(version, model_name=name, client=active)
    if found is None:
        return None
    return replace(
        found,
        resolved_by=resolved_by,
        resolved_uri=resolved_uri,
        resolution_errors=tuple(errors),
    )


# -------------------------------------------------------- champion/challenger


@dataclass(frozen=True)
class PromotionVerdict:
    """Whether a registered candidate should replace the champion, and why."""

    promote: bool
    reasons: list[str]
    champion_version: str | None = None
    candidate_pr_auc: float | None = None
    champion_pr_auc: float | None = None
    tolerance: float = 0.0

    @property
    def decision(self) -> str:
        return "champion" if self.promote else "challenger"

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": self.reasons,
            "champion_version": self.champion_version,
            "candidate_pr_auc": self.candidate_pr_auc,
            "champion_pr_auc": self.champion_pr_auc,
            "tolerance": self.tolerance,
        }


def _logged(run_id: str, client: MlflowClient) -> tuple[dict[str, Any], dict[str, str]] | None:
    """(metrics, params) a run logged, or None if the store cannot say."""
    try:
        run = client.get_run(run_id)
    except StoreError as exc:
        logger.warning("could not read run %s: %s", run_id, exc)
        return None
    return dict(run.data.metrics), {str(k): str(v) for k, v in run.data.params.items()}


def compare_with_champion(
    run_id: str,
    metrics: Mapping[str, Any],
    *,
    model_name: str | None = None,
    tolerance: float | None = None,
    client: MlflowClient | None = None,
) -> PromotionVerdict:
    """Should the candidate trained in `run_id` replace the champion?

    Yes when there is no champion -- the first model into an empty registry
    has to serve, or a fresh stack never does -- or when the candidate's
    PR-AUC is at most `tolerance` below the champion's. Both numbers are the
    ones training logged on the held-out split; if the two runs recorded
    different held-out split hashes the numbers do not compare and the answer
    is no. So is "cannot tell": a champion whose run logged no PR-AUC is not
    replaced on the strength of a missing number. Read-only.
    """
    name = settings.model_name if model_name is None else model_name
    allowed = settings.promotion_pr_auc_tolerance if tolerance is None else tolerance
    active = _client(client)

    candidate_metrics, candidate_params = _logged(run_id, active) or ({}, {})
    candidate = _read_metric(metrics, PR_AUC_KEYS)
    if candidate is None:
        candidate = _read_metric(candidate_metrics, PR_AUC_KEYS)

    try:
        champion = resolve_champion(model_name=name, client=active)
    except RegistryUnreadable as exc:
        return PromotionVerdict(
            False,
            [
                f"cannot compare with the champion: the registry could not be read ({exc}); "
                "left for review"
            ],
            candidate_pr_auc=candidate,
            tolerance=allowed,
        )
    if champion is None:
        return PromotionVerdict(
            True,
            ["no champion in the registry: the first model to pass the gate is promoted"],
            candidate_pr_auc=candidate,
            tolerance=allowed,
        )

    def verdict(promote: bool, reasons: list[str], best: float | None = None) -> PromotionVerdict:
        return PromotionVerdict(
            promote,
            reasons,
            champion_version=champion.version,
            candidate_pr_auc=candidate,
            champion_pr_auc=best,
            tolerance=allowed,
        )

    def cannot(why: str) -> PromotionVerdict:
        return verdict(
            False, [f"cannot compare with champion v{champion.version}: {why}; left for review"]
        )

    if champion.run_id == run_id:
        return verdict(True, [f"the same run as champion v{champion.version}"])
    if not champion.run_id:
        return cannot("it records no run")
    logged = _logged(champion.run_id, active)
    if logged is None:
        return cannot(f"its run {champion.run_id} cannot be read")
    champion_metrics, champion_params = logged
    best = _read_metric(champion_metrics, PR_AUC_KEYS)
    if best is None:
        return cannot(f"its run {champion.run_id} logged no PR-AUC")
    if candidate is None:
        return cannot("the candidate has no PR-AUC")

    ours, theirs = candidate_params.get(TEST_SPLIT_PARAM), champion_params.get(TEST_SPLIT_PARAM)
    if ours and theirs and ours != theirs:
        why = (
            f"scored on a different held-out split ({ours[:12]}) than champion "
            f"v{champion.version} ({theirs[:12]}): the PR-AUCs do not compare; left for review"
        )
        return verdict(False, [why], best)

    notes: list[str] = []
    if not (ours and theirs):
        missing = "the candidate" if not ours else f"champion v{champion.version}"
        notes.append(f"held-out split hash not recorded on {missing}; compared as logged")
    gap = best - candidate
    comparison = f"pr_auc {candidate:.4f} against champion v{champion.version}'s {best:.4f}"
    if gap <= allowed + _EPSILON:
        return verdict(
            True, [f"{comparison}: not worse beyond the {allowed:.4f} tolerance", *notes], best
        )
    return verdict(
        False,
        [f"{comparison} is {gap:.4f} below, beyond the {allowed:.4f} tolerance", *notes],
        best,
    )


def _promotion_tags(decision: str, reason: str | None, compared_with: str | None) -> dict[str, str]:
    tags = {PROMOTION_DECISION_TAG: decision}
    if reason:
        tags[PROMOTION_REASON_TAG] = reason
    if compared_with:
        tags[COMPARED_WITH_TAG] = compared_with
    return tags


def set_champion(
    version: str | int,
    *,
    model_name: str | None = None,
    stage: str | None = None,
    alias: str | None = None,
    reason: str | None = None,
    compared_with: str | None = None,
    dry_run: bool = False,
    client: MlflowClient | None = None,
) -> RegistrationDecision:
    """Make `version` the champion: the alias, the stage, and the tags that say why.

    Idempotent: only what is not already so is changed, and a second call
    changes nothing. The alias moves first because it is what serving
    resolves; if the stage transition then fails, serving already follows the
    version that was meant to be promoted, and the decision says what failed.
    A version holding the challenger alias gives it up.
    """
    name = settings.model_name if model_name is None else model_name
    target_stage = settings.model_stage if stage is None else stage
    target_alias = settings.model_alias if alias is None else alias
    active = _client(client)

    found = version_metadata(version, model_name=name, client=active)
    if found is None:
        return RegistrationDecision(
            registered=False,
            model_name=name,
            version=str(version),
            reasons=[f"{name} v{version} is not in the registry, or the registry is unreachable"],
        )

    previous = alias_version(target_alias, model_name=name, client=active)
    move_alias = previous != found.version
    move_stage = found.current_stage != target_stage
    drop_challenger = CHALLENGER_ALIAS in found.aliases
    changed = [
        *([f"alias {target_alias}"] if move_alias else []),
        *([f"stage {target_stage}"] if move_stage else []),
        *([f"dropped alias {CHALLENGER_ALIAS}"] if drop_challenger else []),
    ]
    done = RegistrationDecision(
        registered=True,
        model_name=name,
        version=found.version,
        stage=target_stage,
        alias=target_alias,
        changed=changed,
    )
    if dry_run or not changed:
        return done

    if move_alias:
        try:
            active.set_registered_model_alias(name, target_alias, found.version)
        except StoreError as exc:
            reason_text = f"could not point {name}@{target_alias} at v{found.version}: {exc}"
            logger.warning("%s", reason_text)
            unchanged = (
                f"nothing was changed: @{target_alias} still names "
                f"{f'v{previous}' if previous else 'no version'} and v{found.version} "
                f"is still in {found.current_stage or 'None'}"
            )
            return replace(
                done,
                registered=False,
                stage=None,
                alias=None,
                changed=[],
                reasons=[reason_text, unchanged],
            )
    if move_stage:
        moved = promote(found.version, target_stage, model_name=name, client=active)
        if not moved.registered:
            # Said in full, because this is the one state where the alias and
            # the stage name different versions on purpose of nobody's.
            where = (
                f"@{target_alias} now names v{found.version}, which the API serves after "
                f"its next restart"
                if move_alias
                else f"@{target_alias} already named v{found.version}"
            )
            split = (
                f"split state: {where}, but v{found.version} is still in "
                f"{found.current_stage or 'None'} and the {target_stage} stage was not "
                f"changed; re-run `python -m credit_risk.models.registry "
                f"{SET_CHAMPION_COMMAND} --version {found.version}` to finish"
            )
            logger.warning("%s", split)
            return replace(
                done,
                registered=False,
                stage=found.current_stage,
                changed=changed[:1] if move_alias else [],
                reasons=[*moved.reasons, split],
            )
    if drop_challenger:
        try:
            active.delete_registered_model_alias(name, CHALLENGER_ALIAS)
        except StoreError as exc:
            logger.warning("could not drop %s@%s: %s", name, CHALLENGER_ALIAS, exc)

    tag_version(
        name, found.version, _promotion_tags("champion", reason, compared_with), client=active
    )
    logger.info("%s v%s is the champion (%s)", name, found.version, ", ".join(changed))
    return done


def set_challenger(
    version: str | int,
    *,
    reason: str,
    model_name: str | None = None,
    compared_with: str | None = None,
    client: MlflowClient | None = None,
) -> RegistrationDecision:
    """Park a registered candidate as the challenger: alias, Staging, and why.

    The tags go first, because the explanation is the one thing a reviewer
    needs even if the alias or the stage cannot be written.
    """
    name = settings.model_name if model_name is None else model_name
    active = _client(client)
    tag_version(
        name, str(version), _promotion_tags("challenger", reason, compared_with), client=active
    )
    try:
        active.set_registered_model_alias(name, CHALLENGER_ALIAS, str(version))
    except StoreError as exc:
        reason_text = f"could not point {name}@{CHALLENGER_ALIAS} at v{version}: {exc}"
        logger.warning("%s", reason_text)
        state = (
            f"v{version} carries the promotion_reason tag but no alias and no stage; "
            "the champion is untouched"
        )
        return RegistrationDecision(
            registered=False, model_name=name, version=str(version), reasons=[reason_text, state]
        )
    moved = promote(str(version), CHALLENGER_STAGE, model_name=name, client=active)
    logger.warning("%s v%s registered as the challenger, not promoted: %s", name, version, reason)
    failed = (
        []
        if moved.registered
        else [
            f"@{CHALLENGER_ALIAS} names v{version} but it is not in {CHALLENGER_STAGE}; "
            "the champion is untouched"
        ]
    )
    return RegistrationDecision(
        registered=moved.registered,
        model_name=name,
        version=str(version),
        stage=CHALLENGER_STAGE if moved.registered else None,
        alias=CHALLENGER_ALIAS,
        reasons=[*moved.reasons, *failed],
        changed=[
            f"alias {CHALLENGER_ALIAS}",
            *([f"stage {CHALLENGER_STAGE}"] if moved.registered else []),
        ],
    )


def promote_or_challenge(
    run_id: str,
    version: str | int,
    metrics: Mapping[str, Any],
    *,
    model_name: str | None = None,
    stage: str | None = None,
    tolerance: float | None = None,
    client: MlflowClient | None = None,
) -> tuple[PromotionVerdict, RegistrationDecision]:
    """Compare a freshly registered version with the champion and act on the answer."""
    active = _client(client)
    verdict = compare_with_champion(
        run_id, metrics, model_name=model_name, tolerance=tolerance, client=active
    )
    reason = "; ".join(verdict.reasons)
    if verdict.promote:
        outcome = set_champion(
            version,
            model_name=model_name,
            stage=stage,
            reason=reason,
            compared_with=verdict.champion_version,
            client=active,
        )
    else:
        outcome = set_challenger(
            version,
            reason=reason,
            model_name=model_name,
            compared_with=verdict.champion_version,
            client=active,
        )
    return verdict, outcome


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

    The sklearn flavour, and only that flavour. Pyfunc hands back a
    `PyFuncModel` wrapper that exposes `predict` and nothing else: no
    `predict_proba`, so a ranking service cannot rank, and no underlying tree
    structure, so `shap.TreeExplainer` has nothing to read.

    There is no pyfunc fallback, on purpose. /predict, /predict/batch, /explain
    and LIME all score through `predict_proba`, so a PyFuncModel would be
    reported as loaded while every request failed, and ModelNotLoaded could
    never fire. A version without a loadable scikit-learn flavour is refused
    like a missing one. Training always logs through `mlflow.sklearn.log_model`,
    so only a version registered by hand can lack it.

    The artefacts are downloaded first and the flavour loaded from that copy,
    so the log can say which of the two went wrong. MLflow raises the same
    exception with the same code for a missing version as for a missing
    flavour; here a failure while downloading is the store's (unreachable, no
    such version, artifact store) and one after it is the version's own.

    Returning None rather than raising is what lets the API come up degraded:
    the container is reachable, `/health` says model_loaded=0, Prometheus fires
    ModelNotLoaded, and an engineer sees a specific alert instead of a crash
    loop with no metrics at all.
    """
    uri = settings.model_uri if model_uri is None else model_uri
    try:
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    except StoreError as exc:
        logger.warning("could not load %s: %s", uri, exc)
        return None
    try:
        return mlflow.sklearn.load_model(local)
    except Exception as exc:  # noqa: BLE001 - the version itself, not the store
        logger.warning(
            "%s cannot be served without a loadable scikit-learn flavour (serving needs predict_proba): %s",
            uri,
            exc,
        )
        return None


# ------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m credit_risk.models.registry`.

    With no command it registers the winning run if the gate passes and then
    promotes it if it is not worse than the champion -- the DAG's
    `register_model` task, called exactly as before. `tag-threshold --version N`
    is the one-off backfill for a version that was registered before versions
    carried their threshold. `set-champion --version N` promotes a version by
    hand: a reviewed challenger, a rollback, or a version promoted before the
    champion alias existed.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == TAG_THRESHOLD_COMMAND:
        return _tag_threshold_main(args[1:])
    if args and args[0] == SET_CHAMPION_COMMAND:
        return _set_champion_main(args[1:])
    return _register_main(args)


def _set_champion_main(argv: Sequence[str]) -> int:
    """`set-champion`: point the champion alias (and stage) at an existing version.

    Exit 0 when the version is the champion afterwards (including when it
    already was), 2 when it is not.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m credit_risk.models.registry {SET_CHAMPION_COMMAND}",
        description=(
            "Make an already-registered version the champion: set the alias serving "
            "resolves, move it to the serving stage, and drop its challenger alias. "
            "Only what is not already so is changed."
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
        "--dry-run", action="store_true", help="print what would change; change nothing"
    )
    args = parser.parse_args(list(argv))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    mlflow.set_tracking_uri(args.tracking_uri)

    outcome = set_champion(
        args.version,
        model_name=args.model_name,
        reason=f"promoted by hand with {SET_CHAMPION_COMMAND}",
        dry_run=args.dry_run,
    )
    payload: dict[str, Any] = {
        "model_name": outcome.model_name,
        "version": outcome.version,
        "alias": outcome.alias,
        "stage": outcome.stage,
        "dry_run": args.dry_run,
        "changed": outcome.changed,
        "reasons": outcome.reasons,
    }
    print(json.dumps(payload, indent=2))
    if not outcome.registered:
        print(f"\nSET-CHAMPION FAILED: {'; '.join(outcome.reasons)}", file=sys.stderr)
        return 2
    if outcome.changed and not args.dry_run:
        print(
            "\nThe API resolves the alias when it loads the model: restart it "
            "(docker compose restart credit-api) or run the deploy workflow, then "
            "`python3 scripts/verify_deploy.py` checks that /health reports "
            f'version {outcome.version} with "model_ref": "alias".',
            file=sys.stderr,
        )
    return 0


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
    """Register the winning run if it passes the gate; promote it if it beats the champion.

    A separate DAG task from evaluation on purpose: the gate decides, this
    acts, and keeping them apart means the log tells you whether a missing
    Production model is a refused candidate or a broken registry.

    Exit 0: registered, and promoted or parked as the challenger as decided.
    Exit 2: nothing registered (the gate refused it, or there is no run).
    Exit 3 (EXIT_PROMOTION_FAILED): registered, but the alias or stage the
    decision called for could not be written; stderr says what state the
    registry was left in. The DAG fails the task on any non-zero exit.
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
        "--no-promote",
        action="store_true",
        help="register the version but neither compare it nor give it an alias or a stage",
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
        verdict, outcome = promote_or_challenge(
            run_id, decision.version, result["metrics"], stage=args.stage
        )
        payload["promoted"] = verdict.promote and outcome.registered
        payload["stage"] = outcome.stage
        payload["alias"] = outcome.alias
        payload["promotion"] = verdict.as_dict()
        payload["promotion_reasons"] = outcome.reasons
        payload["promotion_failed"] = not outcome.registered
        if not outcome.registered:
            label = "PROMOTION FAILED" if verdict.promote else "CHALLENGER NOT RECORDED"
            print(f"\n{label}: {'; '.join(outcome.reasons)}", file=sys.stderr)
        elif not verdict.promote:
            print(
                f"\n{decision.model_name} v{decision.version} is registered as the "
                f"{CHALLENGER_ALIAS}, not promoted: {'; '.join(verdict.reasons)}.\n"
                "Promote it after review with `python -m credit_risk.models.registry "
                f"{SET_CHAMPION_COMMAND} --version {decision.version}`.",
                file=sys.stderr,
            )

    out = (
        training_result_path(args.result.parent if args.result else None).parent
        / "registration.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    if not decision.registered:
        return 2
    if payload.get("promotion_failed"):
        return EXIT_PROMOTION_FAILED
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
