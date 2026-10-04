"""The three DAG hand-off steps, exercised without Airflow.

The DAG runs train, evaluate, register and report as four separate subprocesses,
so the only thing joining them is a JSON file on disk. That seam is exactly
where a pipeline rots: a renamed key or a missing entry point produces a task
that fails at 3 a.m. rather than in review. These tests run each entry point the
DAG shells out to, against the artefact the previous one writes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.models import evaluate as evaluate_module
from credit_risk.models import registry as registry_module
from credit_risk.models import report as report_module
from credit_risk.models import train as train_module


def _result_payload(*, gate_passed: bool = True) -> dict:
    """A training result shaped exactly like the one train.py writes."""
    fairness = {
        "demographic_parity_difference": 0.0384 if gate_passed else 0.19,
        "demographic_parity_ratio": 0.695,
        "equalized_odds_difference": 0.0725 if gate_passed else 0.31,
        "selection_rate_gap": 0.0384,
    }
    return {
        "run_id": "abc123",
        "model": "lightgbm",
        "tracking_uri": "file:///tmp/mlruns",
        "used_fallback_store": True,
        "metrics": {
            "pr_auc": 0.5668 if gate_passed else 0.41,
            "roc_auc": 0.7930,
            "brier": 0.1243,
            "recall_at_k": 0.3490,
            "capacity_fraction": 0.10,
            "base_rate": 0.204,
        },
        "fairness": fairness,
        "fairness_by_attribute": {"SEX": fairness, "AGE_GROUP": fairness},
        "gate_passed": gate_passed,
        "gate_reasons": [] if gate_passed else ["dp too wide"],
        "group_thresholds": {"1": 0.51, "2": 0.49},
        "candidates": [
            {
                "name": "logistic_regression",
                "run_id": "def456",
                "pr_auc": 0.5062,
                "gate_passed": False,
                "gate_reasons": ["eo too wide"],
            },
            {
                "name": "lightgbm",
                "run_id": "abc123",
                "pr_auc": 0.5668,
                "gate_passed": gate_passed,
                "gate_reasons": [],
            },
        ],
        "tradeoff": [
            {"strategy": "baseline", "pr_auc": 0.5668, "dp_diff": 0.0384, "eo_diff": 0.0725},
            {"strategy": "reweighing", "pr_auc": 0.5689, "dp_diff": 0.0278, "eo_diff": 0.0332},
        ],
    }


@pytest.fixture
def artefact(tmp_path: Path) -> Path:
    path = tmp_path / "training_result.json"
    path.write_text(json.dumps(_result_payload()))
    return path


class RecordingClient:
    """Stands in for MlflowClient: keeps the run tags written, talks to no server."""

    def __init__(self) -> None:
        self.tracking_uris: list[str | None] = []
        self.tags: dict[str, dict[str, str]] = {}

    def __call__(self, tracking_uri: str | None = None, **_kwargs: object) -> RecordingClient:
        # Called where the code builds MlflowClient(...), so a test can check
        # the tag went to the store training logged the run to.
        self.tracking_uris.append(tracking_uri)
        return self

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.setdefault(run_id, {})[key] = value


@pytest.fixture
def mlflow_client(monkeypatch: pytest.MonkeyPatch) -> RecordingClient:
    """Every test here that refuses a candidate takes this, so none writes to a store."""
    client = RecordingClient()
    monkeypatch.setattr(registry_module, "MlflowClient", client)
    return client


# ------------------------------------------------------- hand-off artefact


def test_load_training_result_round_trips_what_train_writes(artefact: Path) -> None:
    loaded = train_module.load_training_result(artefact)
    assert loaded["run_id"] == "abc123"
    assert loaded["gate_passed"] is True
    assert loaded["metrics"]["pr_auc"] == pytest.approx(0.5668)


def test_missing_artefact_names_the_command_that_creates_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        train_module.load_training_result(tmp_path / "nope.json")
    # A path that does not exist is useless on its own; the message has to say
    # which step was skipped.
    assert "credit_risk.models.train" in str(excinfo.value)


def test_save_training_result_keeps_every_key_the_downstream_steps_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _result_payload()
    best = train_module.CandidateResult(
        name=payload["model"],
        run_id=payload["run_id"],
        estimator=None,  # type: ignore[arg-type]
        params={},
        cv_pr_auc_mean=0.55,
        cv_pr_auc_std=0.01,
        metrics=payload["metrics"],
        fairness=payload["fairness"],
        gate_passed=True,
        gate_reasons=[],
        fairness_by_attribute=payload["fairness_by_attribute"],
        group_thresholds=payload["group_thresholds"],
    )
    result = train_module.TrainingResult(
        best=best,
        candidates=[best],
        tracking_uri="file:///tmp/mlruns",
        used_fallback_store=True,
        tradeoff=pd.DataFrame(payload["tradeoff"]),
    )
    target = tmp_path / "training_result.json"
    train_module.save_training_result(result, target)

    written = json.loads(target.read_text())
    for key in ("run_id", "metrics", "fairness", "gate_passed", "tracking_uri", "tradeoff"):
        assert key in written, f"{key} is read downstream and must survive the round trip"


# ------------------------------------------------------------ evaluate cli


def test_evaluate_cli_passes_a_clean_candidate(
    artefact: Path, capsys: pytest.CaptureFixture
) -> None:
    code = evaluate_module.main(
        ["--result", str(artefact), "--out", str(artefact.parent / "g.json")]
    )
    assert code == 0
    decision = json.loads((artefact.parent / "g.json").read_text())
    assert decision["passed"] is True
    assert decision["reasons"] == []
    assert "GATE PASSED" in capsys.readouterr().out


def test_evaluate_cli_exits_non_zero_when_the_gate_refuses(
    tmp_path: Path, mlflow_client: RecordingClient
) -> None:
    path = tmp_path / "training_result.json"
    path.write_text(json.dumps(_result_payload(gate_passed=False)))
    out = tmp_path / "gate.json"

    code = evaluate_module.main(["--result", str(path), "--out", str(out)])

    # Non-zero is the whole point: the Airflow task must go red for the same
    # reason a human reading the log would.
    assert code == 2
    decision = json.loads(out.read_text())
    assert decision["passed"] is False
    assert decision["reasons"], "a refusal with no reason is not auditable"


def test_evaluate_cli_recomputes_the_gate_rather_than_trusting_the_flag(
    tmp_path: Path, mlflow_client: RecordingClient
) -> None:
    """A payload that claims it passed but breaches the thresholds is refused."""
    payload = _result_payload(gate_passed=False)
    payload["gate_passed"] = True  # a lie the artefact could carry
    payload["gate_reasons"] = []
    path = tmp_path / "training_result.json"
    path.write_text(json.dumps(payload))

    code = evaluate_module.main(["--result", str(path), "--out", str(tmp_path / "gate.json")])
    assert code == 2


def _artefact_with(
    tmp_path: Path, *, pr_auc: float, dp: float, eo: float, run_id: str | None = "abc123"
) -> Path:
    payload = _result_payload()
    payload["run_id"] = run_id
    payload["metrics"]["pr_auc"] = pr_auc
    payload["fairness"]["demographic_parity_difference"] = dp
    payload["fairness"]["equalized_odds_difference"] = eo
    path = tmp_path / "training_result.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize(
    ("pr_auc", "dp", "eo", "named"),
    [
        (0.50, 0.0384, 0.0725, ["pr_auc"]),
        (0.5668, 0.19, 0.0725, ["demographic_parity_difference"]),
        (0.5668, 0.0384, 0.31, ["equalized_odds_difference"]),
        (
            0.50,
            0.19,
            0.31,
            ["pr_auc", "demographic_parity_difference", "equalized_odds_difference"],
        ),
    ],
    ids=["pr_auc-alone", "dp-alone", "eo-alone", "all-three"],
)
def test_evaluate_cli_tags_the_run_with_the_refusal_and_its_reasons(
    tmp_path: Path,
    mlflow_client: RecordingClient,
    pr_auc: float,
    dp: float,
    eo: float,
    named: list[str],
) -> None:
    """FR-6 on the path the DAG takes.

    evaluate_and_gate fails the run, so register_model -- until now the only
    step that wrote these tags -- never starts. A model refused on PR-AUC alone
    passes training's fairness check too, so its run carried no refusal at all.
    """
    path = _artefact_with(tmp_path, pr_auc=pr_auc, dp=dp, eo=eo)
    out = tmp_path / "gate.json"

    assert evaluate_module.main(["--result", str(path), "--out", str(out)]) == 2

    tags = mlflow_client.tags["abc123"]
    assert tags["registration_refused"] == "true"
    for name in named:
        assert name in tags["registration_refusal_reasons"]
    assert tags["registration_refusal_reasons"] == "; ".join(json.loads(out.read_text())["reasons"])
    assert mlflow_client.tracking_uris == ["file:///tmp/mlruns"]


def test_evaluate_cli_writes_no_refusal_onto_a_run_that_passed(
    artefact: Path, mlflow_client: RecordingClient
) -> None:
    assert (
        evaluate_module.main(["--result", str(artefact), "--out", str(artefact.parent / "g.json")])
        == 0
    )
    assert mlflow_client.tags == {}


def _tracking_server_gone(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("tracking server gone")


class RefusesTags:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    set_tag = staticmethod(_tracking_server_gone)


@pytest.mark.parametrize(
    "client", [_tracking_server_gone, RefusesTags], ids=["no-client", "tag-refused"]
)
def test_evaluate_cli_still_refuses_when_the_refusal_cannot_be_tagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    client: object,
) -> None:
    # Losing the tag is survivable; a refusal that turns into a pass, or into a
    # crash that hides why the task failed, is not.
    monkeypatch.setattr(registry_module, "MlflowClient", client)
    path = _artefact_with(tmp_path, pr_auc=0.50, dp=0.0384, eo=0.0725)
    out = tmp_path / "gate.json"

    with caplog.at_level(logging.WARNING, logger=registry_module.__name__):
        assert evaluate_module.main(["--result", str(path), "--out", str(out)]) == 2

    assert json.loads(out.read_text())["passed"] is False
    assert "GATE REFUSED" in capsys.readouterr().err
    assert "abc123" in caplog.text and "tracking server gone" in caplog.text


def test_evaluate_cli_prints_the_verdict_before_it_reaches_for_the_tracking_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unreachable server is retried for minutes; by then the task log must
    # already say why the task failed.
    printed_by_then: list[str] = []

    def slow_store(*_args: object, **_kwargs: object) -> None:
        printed_by_then.append(capsys.readouterr().err)

    monkeypatch.setattr(registry_module, "record_refusal", slow_store)
    path = _artefact_with(tmp_path, pr_auc=0.50, dp=0.0384, eo=0.0725)

    assert evaluate_module.main(["--result", str(path), "--out", str(tmp_path / "gate.json")]) == 2
    assert len(printed_by_then) == 1
    assert "GATE REFUSED" in printed_by_then[0]


def test_evaluate_cli_says_a_refusal_without_a_run_could_not_be_tagged(
    tmp_path: Path, mlflow_client: RecordingClient, caplog: pytest.LogCaptureFixture
) -> None:
    path = _artefact_with(tmp_path, pr_auc=0.50, dp=0.0384, eo=0.0725, run_id=None)

    with caplog.at_level(logging.WARNING, logger=registry_module.__name__):
        code = evaluate_module.main(["--result", str(path), "--out", str(tmp_path / "gate.json")])

    # --no-mlflow leaves no run, so there is nothing to tag and no store to ask.
    assert code == 2
    assert mlflow_client.tracking_uris == []
    assert "no run" in caplog.text


def _trained(payload: dict) -> train_module.TrainingResult:
    """What train_all hands the CLI, for the candidate a hand-off payload describes."""
    best = train_module.CandidateResult(
        name=payload["model"],
        run_id=payload["run_id"],
        estimator=None,  # type: ignore[arg-type]
        params={},
        cv_pr_auc_mean=payload["metrics"]["pr_auc"],
        cv_pr_auc_std=0.01,
        metrics=payload["metrics"],
        fairness=payload["fairness"],
        gate_passed=payload["gate_passed"],
        gate_reasons=payload["gate_reasons"],
    )
    return train_module.TrainingResult(
        best=best,
        candidates=[best],
        tracking_uri=payload["tracking_uri"],
        used_fallback_store=False,
        tradeoff=pd.DataFrame(payload["tradeoff"]),
    )


def test_train_cli_tags_a_fairness_refusal_with_every_reason_the_gate_gives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mlflow_client: RecordingClient
) -> None:
    """A winner refused on fairness fails train_candidates, so evaluate_and_gate
    never runs for it. Its run is tagged here, with the full gate's reasons --
    PR-AUC included -- or a combined refusal reads as a fairness-only one."""
    result = _trained(_result_payload(gate_passed=False))
    monkeypatch.setattr(train_module, "train_all", lambda **_kwargs: result)

    assert train_module.main(["--result", str(tmp_path / "training_result.json")]) == 2

    tags = mlflow_client.tags["abc123"]
    assert tags["registration_refused"] == "true"
    for name in ("pr_auc", "demographic_parity_difference", "equalized_odds_difference"):
        assert name in tags["registration_refusal_reasons"]
    assert mlflow_client.tracking_uris == ["file:///tmp/mlruns"]


def test_train_cli_leaves_no_refusal_on_a_run_that_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mlflow_client: RecordingClient
) -> None:
    result = _trained(_result_payload())
    monkeypatch.setattr(train_module, "train_all", lambda **_kwargs: result)

    assert train_module.main(["--result", str(tmp_path / "training_result.json")]) == 0
    assert mlflow_client.tags == {}
    assert mlflow_client.tracking_uris == []


# ------------------------------------------------------------ registry cli


def test_registry_cli_registers_and_promotes_when_the_gate_passes(
    artefact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    def fake_register(run_id, metrics, fairness, **kwargs):
        calls["registered_run"] = run_id
        return registry_module.RegistrationDecision(
            registered=True, model_name="credit-risk", version="7"
        )

    def fake_promote(run_id, version, metrics, *, stage=None, **kwargs):
        calls["promoted"] = (run_id, version, stage, metrics["pr_auc"])
        verdict = registry_module.PromotionVerdict(True, ["no champion in the registry"])
        return verdict, registry_module.RegistrationDecision(
            registered=True,
            model_name="credit-risk",
            version=str(version),
            stage=stage,
            alias="champion",
        )

    monkeypatch.setattr(registry_module, "register_if_passes", fake_register)
    monkeypatch.setattr(registry_module, "promote_or_challenge", fake_promote)
    monkeypatch.setattr(registry_module.mlflow, "set_tracking_uri", lambda uri: None)

    code = registry_module.main(["--result", str(artefact)])

    assert code == 0
    assert calls["registered_run"] == "abc123"
    # Compared with the champion on the PR-AUC the hand-off carries, and
    # promoted into the configured stage.
    assert calls["promoted"] == ("abc123", "7", "Production", 0.5668)
    payload = json.loads((artefact.parent / "registration.json").read_text())
    assert payload["registered"] is True and payload["version"] == "7"
    assert payload["promoted"] is True and payload["alias"] == "champion"
    assert payload["promotion"]["decision"] == "champion"


def test_registry_cli_exits_zero_for_a_challenger_and_says_how_to_promote_it(
    artefact: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A candidate that lost the comparison is registered, not failed.

    The DAG task must stay green: the model passed the gate and is in the
    registry for review. What changes is that it does not serve.
    """
    monkeypatch.setattr(
        registry_module,
        "register_if_passes",
        lambda *a, **k: registry_module.RegistrationDecision(
            registered=True, model_name="credit-risk", version="8"
        ),
    )
    verdict = registry_module.PromotionVerdict(False, ["pr_auc 0.5400 is 0.0268 below"])
    outcome = registry_module.RegistrationDecision(
        registered=True, model_name="credit-risk", version="8", stage="Staging", alias="challenger"
    )
    monkeypatch.setattr(registry_module, "promote_or_challenge", lambda *a, **k: (verdict, outcome))
    monkeypatch.setattr(registry_module.mlflow, "set_tracking_uri", lambda uri: None)

    assert registry_module.main(["--result", str(artefact)]) == 0

    payload = json.loads((artefact.parent / "registration.json").read_text())
    assert payload["promoted"] is False
    assert payload["alias"] == "challenger"
    assert payload["promotion"]["reasons"] == ["pr_auc 0.5400 is 0.0268 below"]
    assert "set-champion --version 8" in capsys.readouterr().err


def test_registry_cli_does_not_promote_a_refused_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "training_result.json"
    path.write_text(json.dumps(_result_payload(gate_passed=False)))
    promoted: list[object] = []

    monkeypatch.setattr(
        registry_module,
        "register_if_passes",
        lambda *a, **k: registry_module.RegistrationDecision(
            registered=False, model_name="credit-risk", reasons=["dp too wide"]
        ),
    )
    monkeypatch.setattr(registry_module, "promote", lambda *a, **k: promoted.append(a))
    monkeypatch.setattr(registry_module, "promote_or_challenge", lambda *a, **k: promoted.append(a))
    monkeypatch.setattr(registry_module.mlflow, "set_tracking_uri", lambda uri: None)

    code = registry_module.main(["--result", str(path)])

    assert code == 2
    assert promoted == [], "a refused candidate must never reach a serving stage"


def test_registry_cli_reports_a_missing_run_id_instead_of_crashing(tmp_path: Path) -> None:
    payload = _result_payload()
    payload["run_id"] = None  # what --no-mlflow produces
    path = tmp_path / "training_result.json"
    path.write_text(json.dumps(payload))

    assert registry_module.main(["--result", str(path)]) == 2


# -------------------------------------------------------------- report cli


def test_report_renders_every_section_from_the_artefact(artefact: Path) -> None:
    html = report_module.build_report(json.loads(artefact.read_text()))

    assert "<!doctype html>" in html
    assert "GATE PASSED" in html
    # The candidate comparison is the point of training two families.
    assert "logistic_regression" in html and "lightgbm" in html
    # The trade-off table is what the presentation hangs on.
    assert "reweighing" in html
    # The gate thresholds must be visible, not implied.
    assert str(schema.MAX_DEMOGRAPHIC_PARITY_DIFF) in html
    assert str(schema.MAX_EQUALIZED_ODDS_DIFF) in html


def test_report_states_the_recall_ceiling(artefact: Path) -> None:
    """The ceiling note is the finding, not decoration: recall cannot exceed
    capacity / base_rate, so a bare recall number invites the wrong comparison."""
    html = report_module.build_report(json.loads(artefact.read_text()))
    assert "ceiling" in html.lower()
    assert "0.4902" in html  # 0.10 / 0.204


def test_report_shows_the_refusal_when_the_gate_blocks(tmp_path: Path) -> None:
    payload = _result_payload(gate_passed=False)
    html = report_module.build_report(payload)
    assert "GATE REFUSED" in html
    assert "dp too wide" in html


def test_report_escapes_values_rather_than_interpolating_markup(artefact: Path) -> None:
    payload = json.loads(artefact.read_text())
    payload["model"] = "<script>alert(1)</script>"
    html = report_module.build_report(payload)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_cli_writes_a_file_and_reports_the_gate(artefact: Path, tmp_path: Path) -> None:
    out = tmp_path / "run_report.html"
    code = report_module.main(["--result", str(artefact), "--out", str(out)])
    assert code == 0
    assert out.exists() and out.stat().st_size > 1000


def test_report_cli_fails_cleanly_without_a_training_result(tmp_path: Path) -> None:
    assert report_module.main(["--result", str(tmp_path / "missing.json")]) == 2
