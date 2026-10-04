"""scripts/verify_deploy.py: the check that decides whether a deploy is live.

The script lives outside the package -- it runs on the deploy host with the
system ``python3``, not in the project virtualenv -- so it is loaded by path.
Nothing here needs Docker. The verdict is a pure function of an Observation,
and the CLI tests stand up a stub of the two HTTP APIs on localhost.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_deploy.py"
SAMPLE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "high_risk.json"
GIT_SHA = "4d5f478b0c1d2e3f4a5b6c7d8e9f00112233aabb"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_deploy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolve string annotations through
    # sys.modules, and an unregistered module fails at class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vd = _load()


def _registry_payload(*entries: tuple[str, str]) -> dict[str, Any]:
    return {
        "model_versions": [
            {
                "name": "credit-risk",
                "version": version,
                "current_stage": stage,
                "run_id": f"run{version}abcdef0123",
                "status": "READY",
            }
            for version, stage in entries
        ]
    }


def _health(version: str = "3", *, loaded: bool = True, **overrides: Any) -> dict[str, Any]:
    body = {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_name": "credit-risk",
        "model_version": version if loaded else "unknown",
        "algo": "LGBMClassifier" if loaded else "unknown",
        "detail": None if loaded else "RuntimeError: registry returned no model",
    }
    body.update(overrides)
    return body


def _prediction(version: str = "3", **overrides: Any) -> dict[str, Any]:
    body = {
        "default_probability": 0.7369,
        "decision": "intervene",
        "model_name": "credit-risk",
        "model_version": version,
    }
    body.update(overrides)
    return body


def _healthy_observation(version: str = "3") -> Any:
    return vd.Observation(
        registry=vd.RegistryVersion("credit-risk", version, "Production", "run3"),
        health=_health(version),
        prediction=_prediction(version),
        served_git_sha=GIT_SHA,
    )


def _problems(obs: Any, **kwargs: Any) -> list[str]:
    result: list[str] = vd.find_problems(
        obs, model_name="credit-risk", stage="Production", **kwargs
    )
    return result


# ------------------------------------------------------- registry parsing


def test_pick_stage_version_reads_the_production_entry() -> None:
    found = vd.pick_stage_version(
        _registry_payload(("3", "Production")), model_name="credit-risk", stage="Production"
    )
    assert found == vd.RegistryVersion("credit-risk", "3", "Production", "run3abcdef0123")


def test_a_registry_answer_without_versions_means_none_yet() -> None:
    # MLflow drops the empty repeated field, so `{}` is the normal empty answer.
    assert vd.pick_stage_version({}, model_name="credit-risk", stage="Production") is None
    assert vd.pick_stage_version(None, model_name="credit-risk", stage="Production") is None


def test_versions_in_other_stages_are_ignored() -> None:
    payload = _registry_payload(("4", "Staging"), ("1", "Archived"))
    assert vd.pick_stage_version(payload, model_name="credit-risk", stage="Production") is None


def test_the_highest_version_wins_if_two_share_the_stage() -> None:
    payload = _registry_payload(("2", "Production"), ("10", "Production"))
    found = vd.pick_stage_version(payload, model_name="credit-risk", stage="Production")
    assert found is not None and found.version == "10"  # numeric, not "2" > "10"


# --------------------------------------------------------------- verdict


def test_a_consistent_deployment_has_no_problems() -> None:
    assert _problems(_healthy_observation(), expected_git_sha=GIT_SHA) == []


def test_an_api_still_on_the_previous_version_fails_and_names_both() -> None:
    obs = _healthy_observation("3")
    obs.health = _health("2")
    obs.prediction = _prediction("2")

    problems = _problems(obs)

    assert any("serves version 2 but Production is version 3" in p for p in problems)
    assert any("made by version 2" in p for p in problems)


def test_a_degraded_api_fails_with_its_own_reason() -> None:
    obs = _healthy_observation()
    obs.health = _health(loaded=False)

    problems = _problems(obs)

    assert len([p for p in problems if "degraded" in p]) == 1
    assert "registry returned no model" in problems[0]
    # "unknown" vs 3 is the same fault; it is not reported a second time.
    assert not any("serves version" in p for p in problems)


def test_a_model_loaded_but_status_not_ok_still_fails() -> None:
    obs = _healthy_observation()
    obs.health = _health(status="degraded")
    assert any("degraded" in p for p in _problems(obs))


def test_an_empty_registry_fails() -> None:
    obs = _healthy_observation()
    obs.registry = None
    assert any("has no Production version" in p for p in _problems(obs))


def test_an_unreadable_registry_is_not_reported_as_an_empty_one() -> None:
    obs = _healthy_observation()
    obs.registry, obs.registry_error = None, "connection refused"

    problems = _problems(obs)

    assert any("could not read the MLflow registry: connection refused" in p for p in problems)
    assert not any("has no Production version" in p for p in problems)


def test_an_api_that_does_not_answer_fails() -> None:
    obs = vd.Observation(
        registry=vd.RegistryVersion("credit-risk", "3", "Production"),
        health_error="connection refused",
        prediction_error="skipped",
    )
    problems = _problems(obs)
    assert any("/health did not answer: connection refused" in p for p in problems)
    assert any("predict failed: skipped" in p for p in problems)


def test_the_api_configured_for_another_model_fails() -> None:
    obs = _healthy_observation()
    obs.health = _health(model_name="credit-risk-shadow")
    assert any("configured for model 'credit-risk-shadow'" in p for p in _problems(obs))


@pytest.mark.parametrize("probability", [1.5, -0.1, "0.7", True, None])
def test_a_prediction_without_a_real_probability_fails(probability: Any) -> None:
    obs = _healthy_observation()
    obs.prediction = _prediction(default_probability=probability)
    assert any("default_probability" in p for p in _problems(obs))


def test_a_prediction_without_a_decision_fails() -> None:
    obs = _healthy_observation()
    obs.prediction = _prediction(decision=None)
    assert any("no decision" in p for p in _problems(obs))


def test_a_failed_prediction_fails() -> None:
    obs = _healthy_observation()
    obs.prediction, obs.prediction_error = None, "HTTP 503 model_not_loaded: no model"
    assert any("HTTP 503 model_not_loaded" in p for p in _problems(obs))


@pytest.mark.parametrize(
    ("served", "ok"),
    [
        (GIT_SHA, True),
        (GIT_SHA[:7], True),
        (GIT_SHA.upper(), True),
        ("unknown", False),
        (None, False),
        ("0000000deadbeef", False),
        (GIT_SHA[:4], False),  # too short to name one commit
    ],
)
def test_the_deployed_commit_is_checked_only_when_asked(served: str | None, ok: bool) -> None:
    obs = _healthy_observation()
    obs.served_git_sha = served

    assert _problems(obs) == []  # not asked
    assert (_problems(obs, expected_git_sha=GIT_SHA) == []) is ok


# --------------------------------------------------------------- report


def test_the_summary_names_the_registry_version_the_served_version_and_the_verdict() -> None:
    obs = _healthy_observation()
    text = vd.render_markdown(obs, [], model_name="credit-risk", stage="Production", sample=SAMPLE)

    assert "`credit-risk` version **3** in `Production` (run `run3`)" in text
    assert "version **3** · status `ok`" in text
    assert "p=0.737, `intervene`" in text
    assert "Verification passed" in text


def test_the_summary_lists_every_problem() -> None:
    obs = vd.Observation(registry_error="refused", health_error="refused")
    problems = _problems(obs)
    text = vd.render_markdown(
        obs, problems, model_name="credit-risk", stage="Production", sample=SAMPLE
    )

    assert "unreadable: refused" in text
    assert "Verification failed" in text
    for problem in problems:
        assert f"- {problem}" in text


# ---------------------------------------------------------- configuration


def test_env_file_parsing_strips_quotes_comments_and_export(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nMODEL_NAME='credit-risk-v2'\nexport MODEL_STAGE=\"Staging\"\nnot a pair\n",
        encoding="utf-8",
    )
    assert vd.read_env_file(env) == {"MODEL_NAME": "credit-risk-v2", "MODEL_STAGE": "Staging"}
    assert vd.read_env_file(tmp_path / "missing") == {}


def test_settings_resolve_flag_then_environment_then_env_file_then_default() -> None:
    env_file = {"MODEL_NAME": "from-file"}
    environ = {"MODEL_NAME": "from-env"}
    resolve = vd.resolve_setting

    assert resolve("from-flag", "MODEL_NAME", env_file, "default", environ) == "from-flag"
    assert resolve(None, "MODEL_NAME", env_file, "default", environ) == "from-env"
    assert resolve(None, "MODEL_NAME", env_file, "default", {}) == "from-file"
    assert resolve(None, "MODEL_NAME", {}, "default", {}) == "default"


# ------------------------------------------------- CLI against a stub stack


def _alias_route(alias: str = "champion") -> tuple[str, str]:
    return ("GET", vd.alias_query("credit-risk", alias))


def _alias_payload(version: str, stage: str = "Production") -> dict[str, Any]:
    return {
        "model_version": {
            "name": "credit-risk",
            "version": version,
            "current_stage": stage,
            "run_id": f"run{version}abcdef0123",
            "aliases": ["champion"],
        }
    }


# What MLflow 2.19 answers when no version carries the alias -- measured
# against the live registry, whose version 2 predates aliases.
NO_ALIAS = (
    400,
    {
        "error_code": "INVALID_PARAMETER_VALUE",
        "message": "Registered model alias champion not found.",
    },
)


class _Stub:
    """MLflow and the API on one localhost port; each test sets the answers."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, Any]] = {}
        self.received: dict[str, Any] = {}
        self.url = ""


@pytest.fixture
def stub() -> Iterator[_Stub]:
    state = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                state.received[self.path] = json.loads(self.rfile.read(length))
            status, body = state.routes.get((method, self.path), (404, {"code": "not_found"}))
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._answer("GET")

        def do_POST(self) -> None:
            self._answer("POST")

        def log_message(self, *args: Any) -> None:
            pass

    # By default nothing carries the alias, answered the way MLflow 2.19
    # answers it -- not with the stub's generic 404, which is now an error.
    state.routes[_alias_route()] = NO_ALIAS
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


REGISTRY = ("POST", vd.LATEST_VERSIONS_PATH)


def _consistent(stub: _Stub, version: str = "3") -> None:
    stub.routes[REGISTRY] = (200, _registry_payload((version, "Production")))
    stub.routes[("GET", "/health")] = (200, _health(version))
    stub.routes[("POST", "/api/v1/predict")] = (200, _prediction(version))
    stub.routes[("GET", "/version")] = (200, {"git_sha": GIT_SHA})


def _run(stub: _Stub, *extra: str, tmp_path: Path) -> int:
    code: int = vd.main(
        [
            "--api-url",
            stub.url,
            "--mlflow-url",
            stub.url,
            "--model-name",
            "credit-risk",
            "--stage",
            "Production",
            "--env-file",
            str(tmp_path / "no.env"),
            "--wait",
            "0",
            "--timeout",
            "5",
            *extra,
        ]
    )
    return code


def test_cli_passes_against_a_consistent_stack_and_writes_the_summary(
    stub: _Stub, tmp_path: Path
) -> None:
    _consistent(stub)
    summary = tmp_path / "summary.md"

    code = _run(stub, "--expect-git-sha", GIT_SHA, "--summary", str(summary), tmp_path=tmp_path)

    assert code == vd.EXIT_OK
    assert "Verification passed" in summary.read_text(encoding="utf-8")
    # The documented sample is what was scored, and the registry was asked
    # about the right model and stage.
    assert stub.received["/api/v1/predict"] == json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert stub.received[vd.LATEST_VERSIONS_PATH] == {
        "name": "credit-risk",
        "stages": ["Production"],
    }


def test_cli_fails_when_the_api_serves_an_older_version(stub: _Stub, tmp_path: Path) -> None:
    _consistent(stub, "3")
    stub.routes[("GET", "/health")] = (200, _health("2"))
    assert _run(stub, tmp_path=tmp_path) == vd.EXIT_FAILED


def test_cli_reports_the_reason_a_degraded_api_gives(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _consistent(stub)
    stub.routes[("GET", "/health")] = (200, _health(loaded=False))
    stub.routes[("POST", "/api/v1/predict")] = (
        503,
        {"code": "model_not_loaded", "message": "No model is loaded."},
    )

    assert _run(stub, tmp_path=tmp_path) == vd.EXIT_FAILED
    out = capsys.readouterr().out
    assert "registry returned no model" in out
    assert "HTTP 503 model_not_loaded: No model is loaded." in out


def test_registry_version_prints_the_version(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _consistent(stub, "7")
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_OK
    assert capsys.readouterr().out.strip() == "7"


@pytest.mark.parametrize(
    "answer",
    [
        (200, {}),
        (
            404,
            {
                "error_code": "RESOURCE_DOES_NOT_EXIST",
                "message": "Registered Model with name=credit-risk not found",
            },
        ),
    ],
    ids=["nothing-in-the-stage", "model-never-registered"],
)
def test_registry_version_exits_3_when_nothing_is_registered(
    stub: _Stub, tmp_path: Path, answer: tuple[int, Any]
) -> None:
    stub.routes[REGISTRY] = answer
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_NO_VERSION


def test_registry_version_does_not_say_none_when_mlflow_is_down(tmp_path: Path) -> None:
    # The deploy workflow trains a model on exit 3. A registry that is merely
    # unreachable, or answering 500, must never look like an empty one.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
    code = vd.main(
        ["--mlflow-url", closed, "--registry-version", "--env-file", str(tmp_path / "x")]
    )
    assert code == vd.EXIT_FAILED


def test_registry_version_treats_a_server_error_as_unreadable(stub: _Stub, tmp_path: Path) -> None:
    stub.routes[REGISTRY] = (500, {"error_code": "INTERNAL_ERROR", "message": "db down"})
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_FAILED


def test_cli_fails_cleanly_on_an_unreadable_sample(stub: _Stub, tmp_path: Path) -> None:
    _consistent(stub)
    missing = tmp_path / "missing.json"
    assert _run(stub, "--sample", str(missing), tmp_path=tmp_path) == vd.EXIT_FAILED


# ---------------------------------------------------- the champion alias


def test_the_alias_decides_when_it_exists(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _consistent(stub, "3")
    # The stage still points at 3; the alias has moved on to 5, and the API
    # follows the alias, so the deploy check must too.
    stub.routes[_alias_route()] = (200, _alias_payload("5"))

    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_OK
    assert capsys.readouterr().out.strip() == "5"


def test_no_alias_falls_back_to_the_stage(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _consistent(stub, "2")
    stub.routes[_alias_route()] = NO_ALIAS

    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_OK
    assert capsys.readouterr().out.strip() == "2"


def test_no_alias_and_no_stage_is_still_exit_3(stub: _Stub, tmp_path: Path) -> None:
    # The deploy workflow bootstraps a model on exit 3 and only then.
    stub.routes[_alias_route()] = NO_ALIAS
    stub.routes[REGISTRY] = (200, {})
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_NO_VERSION


def test_an_alias_lookup_that_errors_is_not_a_missing_alias(stub: _Stub, tmp_path: Path) -> None:
    # Falling back to the stage on a 500 could approve an API that serves the
    # stage while the alias -- unreadable, not absent -- names another version.
    _consistent(stub, "3")
    stub.routes[_alias_route()] = (500, {"error_code": "INTERNAL_ERROR", "message": "db down"})
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_FAILED


def test_the_alias_can_be_chosen_with_a_flag(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _consistent(stub, "3")
    stub.routes[_alias_route("candidate")] = (200, _alias_payload("8"))

    assert _run(stub, "--registry-version", "--alias", "candidate", tmp_path=tmp_path) == 0
    assert capsys.readouterr().out.strip() == "8"


def test_cli_fails_when_the_api_still_serves_the_stage_version_after_an_alias_moved(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _consistent(stub, "2")
    stub.routes[_alias_route()] = (200, _alias_payload("3", stage="Production"))

    assert _run(stub, tmp_path=tmp_path) == vd.EXIT_FAILED
    out = capsys.readouterr().out
    assert "serves version 2 but @champion is version 3" in out


def test_a_bare_404_from_the_alias_lookup_is_not_a_missing_alias(
    stub: _Stub, tmp_path: Path
) -> None:
    # A proxy's 404 or a mistyped MLflow URL is not MLflow saying "no such
    # alias"; falling back to the stage on it could approve the wrong version.
    _consistent(stub, "3")
    stub.routes[_alias_route()] = (404, {"code": "not_found"})
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_FAILED


def test_mlflows_own_404_for_a_missing_model_means_no_alias(stub: _Stub, tmp_path: Path) -> None:
    stub.routes[_alias_route()] = (
        404,
        {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Registered Model not found"},
    )
    stub.routes[REGISTRY] = (200, {})
    assert _run(stub, "--registry-version", tmp_path=tmp_path) == vd.EXIT_NO_VERSION


def test_cli_passes_against_a_stack_serving_the_alias(stub: _Stub, tmp_path: Path) -> None:
    _consistent(stub, "3")
    stub.routes[_alias_route()] = (200, _alias_payload("3"))
    stub.routes[("GET", "/health")] = (200, _health("3", model_ref="alias"))
    summary = tmp_path / "summary.md"

    assert _run(stub, "--summary", str(summary), tmp_path=tmp_path) == vd.EXIT_OK

    text = summary.read_text(encoding="utf-8")
    assert "`credit-risk@champion` → version **3**" in text
    assert "via `alias`" in text


def test_the_summary_says_when_the_stage_answered_instead_of_the_alias() -> None:
    obs = _healthy_observation()
    text = vd.render_markdown(
        obs, [], model_name="credit-risk", stage="Production", sample=SAMPLE, alias="champion"
    )
    assert "no `@champion` alias" in text


def test_an_empty_registry_names_both_places_it_looked() -> None:
    obs = _healthy_observation()
    obs.registry = None
    problems = vd.find_problems(obs, model_name="credit-risk", stage="Production", alias="champion")
    assert any("has no Production version" in p and "@champion" in p for p in problems)
