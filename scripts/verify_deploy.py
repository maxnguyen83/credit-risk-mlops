#!/usr/bin/env python3
"""Post-deploy check: is the API serving the model the registry says it should?

Run by the deploy workflow after every deployment, and by hand whenever the
question is "what is actually live". Three answers have to agree:

1. MLflow: which version the champion alias names -- or, when no version
   carries the alias, which version holds the Production stage. The same
   order the API resolves them in.
2. The API's ``/health``: which version this process loaded, and whether it
   loaded one at all.
3. A real request: ``docs/examples/high_risk.json`` scored by
   ``/api/v1/predict``, naming the same version in its response.

A deploy that rebuilt every image and restarted every container can still get
this wrong. The API resolves ``models:/credit-risk@champion`` once, at startup
(ARCHITECTURE.md D3), so a promotion made after it booted leaves it serving the
previous version with every health check green. Comparing the two version
numbers is the only check that sees that.

Standard library only, and Python 3.9-compatible syntax, on purpose: it runs on
the deploy host with whatever ``python3`` is there -- 3.9 from Apple's command
line tools, on a Mac -- not inside the project virtualenv, which a deploy host
need not have.

    python3 scripts/verify_deploy.py                      # the full check
    python3 scripts/verify_deploy.py --registry-version   # print the version, or exit 3

Model name, alias and stage come from ``--model-name``/``--alias``/``--stage``,
else the ``MODEL_NAME``/``MODEL_ALIAS``/``MODEL_STAGE`` environment variables,
else the ``.env`` the stack runs with, else ``credit-risk``/``champion``/
``Production`` -- the same precedence the API's own settings use, so both sides
ask about the same model.

Exit codes: 0 every check passed; 1 a check failed or a service could not be
read; 3 (``--registry-version`` only) the registry answered and neither the
alias nor the stage names a version. 1 and 3 are kept apart because the deploy
workflow trains a model on 3 and must never do so merely because MLflow was
down -- which is also why only MLflow's "no such alias" answer falls back to
the stage, and any other error from the alias lookup is a failure.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE: Final = REPO_ROOT / "docs" / "examples" / "high_risk.json"
DEFAULT_ENV_FILE: Final = REPO_ROOT / ".env"

# The published ports from docker-compose.yml, and the variable names
# scripts/demo.sh already uses for them.
DEFAULT_API_URL: Final = "http://localhost:18000"
DEFAULT_MLFLOW_URL: Final = "http://localhost:15020"
DEFAULT_MODEL_NAME: Final = "credit-risk"
DEFAULT_ALIAS: Final = "champion"
DEFAULT_STAGE: Final = "Production"

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_NO_VERSION: Final = 3

LATEST_VERSIONS_PATH: Final = "/api/2.0/mlflow/registered-models/get-latest-versions"
ALIAS_PATH: Final = "/api/2.0/mlflow/registered-models/alias"
# What MLflow answers when no version carries the alias, or the model itself
# does not exist: 400 INVALID_PARAMETER_VALUE ("Registered model alias champion
# not found.") on 2.19, or RESOURCE_DOES_NOT_EXIST. Only these, carried in
# MLflow's own JSON error body, mean "no alias"; a 404 without them -- a proxy,
# a wrong URL, a server that predates aliases -- is a failure, not a fallback.
NO_ALIAS_ERRORS: Final = ("INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST")
POLL_SECONDS: Final = 2.0


class ServiceError(Exception):
    """A service did not answer, or answered with an error we cannot interpret."""


@dataclass(frozen=True)
class RegistryVersion:
    """One registered model version, as MLflow reports it."""

    name: str
    version: str
    stage: str
    run_id: str | None = None
    # Set when the version was found through the alias rather than the stage.
    alias: str | None = None

    @property
    def label(self) -> str:
        """How the version was found, as the problem messages name it."""
        return f"@{self.alias}" if self.alias else self.stage


@dataclass
class Observation:
    """Everything read from the live system.

    The verdict is a pure function of this object, which is what lets it be
    tested without a registry, an API or Docker. A ``*_error`` field set means
    the read itself failed; it is kept apart from "read fine, answer was bad"
    because the two call for different fixes.
    """

    registry: RegistryVersion | None = None
    registry_error: str | None = None
    health: Mapping[str, Any] | None = None
    health_error: str | None = None
    prediction: Mapping[str, Any] | None = None
    prediction_error: str | None = None
    served_git_sha: str | None = None


# ------------------------------------------------------------------ verdict


def _version_number(entry: Mapping[str, Any]) -> int:
    try:
        return int(str(entry.get("version")))
    except ValueError:
        return -1


def pick_stage_version(
    payload: Mapping[str, Any] | None, *, model_name: str, stage: str
) -> RegistryVersion | None:
    """The version holding ``stage`` in a get-latest-versions answer, or None.

    MLflow leaves ``model_versions`` out entirely when nothing holds the stage
    (proto3 drops an empty repeated field), so a missing key is the ordinary
    "none yet" answer rather than a malformed one. Should two versions ever
    share the stage, the highest number wins: it is the one
    ``models:/<name>/<stage>`` resolves to, so it is the one the API loads.
    """
    entries = [
        entry
        for entry in (payload or {}).get("model_versions") or []
        if entry.get("current_stage") == stage
    ]
    if not entries:
        return None
    best = max(entries, key=_version_number)
    run_id = best.get("run_id")
    return RegistryVersion(
        name=str(best.get("name") or model_name),
        version=str(best.get("version")),
        stage=stage,
        run_id=str(run_id) if run_id else None,
    )


def alias_query(model_name: str, alias: str) -> str:
    """The path and query of MLflow's get-model-version-by-alias call."""
    return ALIAS_PATH + "?" + urllib.parse.urlencode({"name": model_name, "alias": alias})


def pick_alias_version(
    payload: Mapping[str, Any] | None, *, model_name: str, alias: str
) -> RegistryVersion | None:
    """The version in a get-model-version-by-alias answer, or None if it names none."""
    entry = (payload or {}).get("model_version")
    if not isinstance(entry, Mapping) or not entry.get("version"):
        return None
    run_id = entry.get("run_id")
    return RegistryVersion(
        name=str(entry.get("name") or model_name),
        version=str(entry.get("version")),
        stage=str(entry.get("current_stage") or "None"),
        run_id=str(run_id) if run_id else None,
        alias=alias,
    )


def sha_matches(expected: str, served: str | None) -> bool:
    """True when two commit ids name the same commit, allowing an abbreviation.

    Seven characters minimum, git's own default: a shorter prefix could match
    a different commit, and "unknown" -- what the API reports when GIT_SHA was
    never passed in -- matches nothing.
    """
    if not served:
        return False
    a, b = expected.strip().lower(), served.strip().lower()
    if len(a) < 7 or len(b) < 7:
        return False
    return a.startswith(b) or b.startswith(a)


def _is_probability(value: Any) -> bool:
    # type() rather than isinstance(): a JSON true would pass isinstance(int).
    return type(value) in (int, float) and 0.0 <= value <= 1.0


def find_problems(
    obs: Observation,
    *,
    model_name: str,
    stage: str,
    expected_git_sha: str | None = None,
    alias: str | None = None,
) -> list[str]:
    """Every reason this deployment is not serving what it should. Empty means it is.

    All checks run and every failure is reported, the same contract as the
    registration gate: stopping at the first would hide the second until the
    first was fixed.
    """
    problems: list[str] = []
    expected = obs.registry

    if obs.registry_error:
        problems.append(f"could not read the MLflow registry: {obs.registry_error}")
    elif expected is None:
        nowhere = f" and no version carries @{alias}" if alias else ""
        problems.append(
            f"MLflow has no {stage} version of {model_name!r}{nowhere}, so nothing should serve"
        )

    if obs.health_error or obs.health is None:
        problems.append(f"API /health did not answer: {obs.health_error or 'no response'}")
    else:
        health = obs.health
        loaded = health.get("model_loaded") is True
        if health.get("status") != "ok" or not loaded:
            problems.append(
                f"the API is degraded (status={health.get('status')!r}, "
                f"model_loaded={health.get('model_loaded')!r}): "
                f"{health.get('detail') or 'no reason given'}"
            )
        if health.get("model_name") != model_name:
            problems.append(
                f"the API is configured for model {health.get('model_name')!r}, "
                f"not {model_name!r}"
            )
        served = str(health.get("model_version"))
        # Only once a model is loaded: degraded, the API reports "unknown", and
        # a version mismatch on top of "degraded" says the same thing twice.
        if loaded and expected is not None and served != expected.version:
            problems.append(
                f"the API serves version {served} but {expected.label} is version "
                f"{expected.version}: it has not reloaded since the promotion "
                f"(docker compose restart credit-api)"
            )

    if obs.prediction_error or obs.prediction is None:
        problems.append(f"POST /api/v1/predict failed: {obs.prediction_error or 'no response'}")
    else:
        prediction = obs.prediction
        if not _is_probability(prediction.get("default_probability")):
            problems.append(
                "the prediction has no default_probability in [0, 1]: "
                f"{prediction.get('default_probability')!r}"
            )
        if not prediction.get("decision"):
            problems.append("the prediction carries no decision")
        predicted_by = str(prediction.get("model_version"))
        if expected is not None and predicted_by != expected.version:
            problems.append(
                f"the prediction was made by version {predicted_by}, "
                f"not {expected.label} version {expected.version}"
            )

    if expected_git_sha and not sha_matches(expected_git_sha, obs.served_git_sha):
        problems.append(
            f"the API reports commit {obs.served_git_sha!r}, not the deployed "
            f"{expected_git_sha[:7]} (GIT_SHA did not reach the container)"
        )

    return problems


def render_markdown(
    obs: Observation,
    problems: list[str],
    *,
    model_name: str,
    stage: str,
    sample: Path,
    alias: str | None = None,
) -> str:
    """The model half of the deploy job summary, as GitHub-flavoured Markdown."""
    reg = obs.registry
    if reg is not None:
        run = f" (run `{reg.run_id[:8]}`)" if reg.run_id else ""
        if reg.alias:
            registry_cell = (
                f"`{reg.name}@{reg.alias}` → version **{reg.version}** "
                f"(stage `{reg.stage}`){run}"
            )
        else:
            registry_cell = f"`{reg.name}` version **{reg.version}** in `{reg.stage}`{run}"
            if alias:
                registry_cell += f" · no `@{alias}` alias, so the stage decides"
    elif obs.registry_error:
        registry_cell = f"unreadable: {obs.registry_error}"
    else:
        registry_cell = f"no `{stage}` version of `{model_name}`"

    if obs.health is not None:
        health = obs.health
        api_cell = (
            f"`{health.get('model_name')}` version **{health.get('model_version')}** · "
            f"status `{health.get('status')}` · algo `{health.get('algo')}`"
        )
        if health.get("model_ref"):
            api_cell += f" · via `{health.get('model_ref')}`"
    else:
        api_cell = f"no answer: {obs.health_error}"

    if obs.prediction is not None:
        p = obs.prediction
        probability = p.get("default_probability")
        shown = f"{probability:.3f}" if _is_probability(probability) else repr(probability)
        predict_cell = (
            f"`{sample.name}` → p={shown}, `{p.get('decision')}`, "
            f"version {p.get('model_version')}"
        )
    else:
        predict_cell = f"failed: {obs.prediction_error}"

    lines = [
        "### Model",
        "",
        "| | |",
        "|---|---|",
        f"| registry ({f'@{alias}, else ' if alias else ''}{stage}) | {registry_cell} |",
        f"| API serves | {api_cell} |",
        f"| API commit | `{(obs.served_git_sha or 'unknown')[:12]}` |",
        f"| sample prediction | {predict_cell} |",
        "",
    ]
    if problems:
        lines.append("**Verification failed:**")
        lines.append("")
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("**Verification passed:** the API serves the registry's model.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- I/O


def request_json(url: str, *, payload: Any = None, timeout: float = 15.0) -> tuple[int, Any]:
    """(status, parsed JSON body or None). Raises ServiceError only if nothing answered.

    An HTTP error status is returned rather than raised: a 404 from MLflow and a
    503 from a degraded API both carry a JSON body that says why, and that
    reason belongs in the report.
    """
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    method = "GET" if data is None else "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    # URLError is an OSError, as are refused connections and socket timeouts;
    # HTTPException covers a server that hangs up mid-response.
    except (OSError, http.client.HTTPException) as exc:
        raise ServiceError(f"{method} {url}: {exc}") from exc
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = None
    return status, body


def _describe_error(status: int, body: Any) -> str:
    if isinstance(body, Mapping):
        reason = body.get("message") or body.get("detail")
        code = body.get("error_code") or body.get("code")
        if reason or code:
            prefix = f"HTTP {status} {code}" if code else f"HTTP {status}"
            return f"{prefix}: {reason}" if reason else prefix
    return f"HTTP {status}"


def fetch_registry_version(
    mlflow_url: str, *, model_name: str, stage: str, timeout: float
) -> RegistryVersion | None:
    """The version holding ``stage``, None if nothing does, ServiceError if unknowable."""
    url = mlflow_url.rstrip("/") + LATEST_VERSIONS_PATH
    status, body = request_json(
        url, payload={"name": model_name, "stages": [stage]}, timeout=timeout
    )
    if status == 200:
        payload = body if isinstance(body, Mapping) else None
        return pick_stage_version(payload, model_name=model_name, stage=stage)
    if (
        status == 404
        and isinstance(body, Mapping)
        and body.get("error_code") == "RESOURCE_DOES_NOT_EXIST"
    ):
        # The registered model itself does not exist yet: a fresh registry.
        return None
    raise ServiceError(_describe_error(status, body))


def fetch_alias_version(
    mlflow_url: str, *, model_name: str, alias: str, timeout: float
) -> RegistryVersion | None:
    """The version ``alias`` names, None if none does, ServiceError if unknowable."""
    status, body = request_json(
        mlflow_url.rstrip("/") + alias_query(model_name, alias), timeout=timeout
    )
    if status == 200:
        payload = body if isinstance(body, Mapping) else None
        return pick_alias_version(payload, model_name=model_name, alias=alias)
    if (
        status in (400, 404)
        and isinstance(body, Mapping)
        and body.get("error_code") in NO_ALIAS_ERRORS
    ):
        return None
    raise ServiceError(_describe_error(status, body))


def fetch_serving_version(
    mlflow_url: str, *, model_name: str, alias: str, stage: str, timeout: float
) -> RegistryVersion | None:
    """The version the API should serve: the alias's, else the stage's -- the API's order."""
    found = fetch_alias_version(mlflow_url, model_name=model_name, alias=alias, timeout=timeout)
    if found is not None:
        return found
    return fetch_registry_version(mlflow_url, model_name=model_name, stage=stage, timeout=timeout)


def wait_for_health(
    api_url: str, *, wait_seconds: float, timeout: float
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Poll /health until it reports a loaded model or time runs out.

    Returns the last answer seen, whatever it said: a degraded answer after the
    deadline is evidence, and the verdict decides what it means.
    """
    deadline = time.monotonic() + max(wait_seconds, 0.0)
    url = api_url.rstrip("/") + "/health"
    last: Mapping[str, Any] | None = None
    error: str | None = None
    while True:
        try:
            status, body = request_json(url, timeout=timeout)
            if status == 200 and isinstance(body, Mapping):
                last, error = body, None
                if body.get("model_loaded") is True:
                    return last, None
            else:
                error = _describe_error(status, body)
        except ServiceError as exc:
            error = str(exc)
        if time.monotonic() >= deadline:
            return last, (None if last is not None else error)
        time.sleep(POLL_SECONDS)


def observe(
    *,
    api_url: str,
    mlflow_url: str,
    model_name: str,
    stage: str,
    sample: Mapping[str, Any],
    wait_seconds: float,
    timeout: float,
    alias: str = DEFAULT_ALIAS,
) -> Observation:
    """Read the registry, the API's health, one prediction and the deployed commit."""
    obs = Observation()
    try:
        obs.registry = fetch_serving_version(
            mlflow_url, model_name=model_name, alias=alias, stage=stage, timeout=timeout
        )
    except ServiceError as exc:
        obs.registry_error = str(exc)

    obs.health, obs.health_error = wait_for_health(
        api_url, wait_seconds=wait_seconds, timeout=timeout
    )
    if obs.health is None:
        # Nothing answered; a prediction and /version would only repeat that.
        obs.prediction_error = "skipped: the API did not answer /health"
        return obs

    api = api_url.rstrip("/")
    try:
        status, body = request_json(f"{api}/api/v1/predict", payload=sample, timeout=timeout)
        if status == 200 and isinstance(body, Mapping):
            obs.prediction = body
        else:
            obs.prediction_error = _describe_error(status, body)
    except ServiceError as exc:
        obs.prediction_error = str(exc)

    try:
        status, body = request_json(f"{api}/version", timeout=timeout)
        if status == 200 and isinstance(body, Mapping) and body.get("git_sha"):
            obs.served_git_sha = str(body["git_sha"])
    except ServiceError:
        pass  # reported through the git-sha check, when one was asked for
    return obs


# --------------------------------------------------------------------- cli


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE pairs from a dotenv file; comments, blanks and junk skipped.

    Only as much of the format as this script needs -- MODEL_NAME, MODEL_ALIAS
    and MODEL_STAGE -- including the optional quotes docker compose also strips.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def resolve_setting(
    cli_value: str | None,
    key: str,
    env_file: Mapping[str, str],
    default: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Flag, then environment, then .env, then default -- pydantic-settings' order."""
    source = os.environ if environ is None else environ
    for candidate in (cli_value, source.get(key), env_file.get(key)):
        if candidate:
            return candidate
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that the API serves the model version the MLflow registry holds.",
    )
    parser.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    parser.add_argument("--mlflow-url", default=os.environ.get("MLFLOW_URL", DEFAULT_MLFLOW_URL))
    parser.add_argument("--model-name", default=None, help="default: MODEL_NAME, then .env")
    parser.add_argument(
        "--alias", default=None, help="default: MODEL_ALIAS, then .env, then champion"
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="fallback when no version carries the alias; default: MODEL_STAGE, then .env",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="dotenv file to read MODEL_NAME/MODEL_ALIAS/MODEL_STAGE from (default: the repo's .env)",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="JSON account posted to /api/v1/predict (default: docs/examples/high_risk.json)",
    )
    parser.add_argument(
        "--expect-git-sha",
        default=None,
        help="also require /version to report this commit",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=60.0,
        help="seconds to wait for /health to report a loaded model (default 60)",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="per-request timeout")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="append a Markdown report to this file (e.g. $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--registry-version",
        action="store_true",
        help=(
            "print the version the alias names (else the one in the stage) and exit: "
            "0 found, 3 none, 1 unreadable"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = read_env_file(args.env_file)
    model_name = resolve_setting(args.model_name, "MODEL_NAME", env_file, DEFAULT_MODEL_NAME)
    alias = resolve_setting(args.alias, "MODEL_ALIAS", env_file, DEFAULT_ALIAS)
    stage = resolve_setting(args.stage, "MODEL_STAGE", env_file, DEFAULT_STAGE)

    if args.registry_version:
        try:
            found = fetch_serving_version(
                args.mlflow_url,
                model_name=model_name,
                alias=alias,
                stage=stage,
                timeout=args.timeout,
            )
        except ServiceError as exc:
            print(f"cannot read the registry at {args.mlflow_url}: {exc}", file=sys.stderr)
            return EXIT_FAILED
        if found is None:
            print(f"no version of {model_name!r} at @{alias} or in {stage}", file=sys.stderr)
            return EXIT_NO_VERSION
        if found.alias is None:
            print(f"no version carries @{alias}; {stage} holds {found.version}", file=sys.stderr)
        print(found.version)
        return EXIT_OK

    try:
        sample = json.loads(args.sample.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read the sample account {args.sample}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(
        f"registry {args.mlflow_url}  api {args.api_url}  "
        f"model {model_name!r} @{alias} (else {stage})"
    )
    obs = observe(
        api_url=args.api_url,
        mlflow_url=args.mlflow_url,
        model_name=model_name,
        stage=stage,
        sample=sample,
        wait_seconds=args.wait,
        timeout=args.timeout,
        alias=alias,
    )
    problems = find_problems(
        obs,
        model_name=model_name,
        stage=stage,
        expected_git_sha=args.expect_git_sha,
        alias=alias,
    )
    report = render_markdown(
        obs, problems, model_name=model_name, stage=stage, sample=args.sample, alias=alias
    )
    print(report)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    return EXIT_FAILED if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
