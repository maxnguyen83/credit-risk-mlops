"""Does the model clear the bar, and does the gate stop it when it does not.

The targets asserted here are the ones in the design spec -- PR-AUC >= 0.54 and
ROC-AUC >= 0.78 -- and they are asserted twice: against the real held-out batch
when the dataset has been downloaded, and against a synthetic fixture that is
always available. The fixture exists so a reviewer who clones the repo at
midnight gets a meaningful signal instead of a skipped test, and it is built to
match the measured shape of the real data: 22% positives, a higher default rate
for SEX=1 than SEX=2, PAY_* in [-2, 8].

It is a fixture, not a claim about the world. No number produced from it goes
into the model card.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.data.split import BATCH_COL, write_splits
from credit_risk.fairness.mitigation import load_group_thresholds
from credit_risk.models import evaluate as evaluate_module
from credit_risk.models import registry as registry_module
from credit_risk.models import report as report_module
from credit_risk.models.evaluate import evaluate
from credit_risk.models.registry import (
    MIN_PR_AUC,
    current_production_version,
    evaluate_gate,
    load_production_model,
    promote,
    register_if_passes,
)
from credit_risk.models.train import (
    CandidateResult,
    TrainingResult,
    feature_frame,
    feature_importance_frame,
    fit_group_thresholds,
    load_training_splits,
    main,
    make_lightgbm,
    make_logistic_regression,
    resolve_tracking_uri,
    save_feature_importance_plot,
    save_tradeoff_plot,
    split_train_test,
    train_all,
)

MIN_ROC_AUC = 0.78

# A fairness summary that clears both gates, for the tests that are about the
# performance half of the decision and should not depend on a model's bias.
FAIR_ENOUGH = {"demographic_parity_difference": 0.01, "equalized_odds_difference": 0.02}


def synthetic_clean_frame(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """A frame shaped exactly like `data.split.clean` output.

    Delinquency is generated as an autocorrelated process rather than six
    independent draws, because the real repayment columns are strongly
    serially correlated and features like `max_consecutive_delinquent` are
    meaningless without that. A model trained on independent draws would look
    fine here and fall over on the real thing.
    """
    rng = np.random.default_rng(seed)
    sex = rng.choice([1, 2], size=n, p=[0.40, 0.60])
    education = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.47, 0.16, 0.02])
    marriage = rng.choice([1, 2, 3], size=n, p=[0.45, 0.54, 0.01])
    age = rng.integers(schema.AGE_MIN, schema.AGE_MAX + 1, size=n)
    limit = np.clip(
        rng.lognormal(mean=11.6, sigma=0.75, size=n) * (1.0 + 0.25 * (education == 1)),
        10_000,
        1_000_000,
    ).round(-3)

    # The latent trait that drives both the repayment history and the label --
    # which is why dropping SEX cannot hide it.
    propensity = (
        rng.normal(0.0, 1.0, n)
        + 0.30 * (sex == 1)
        + 0.25 * (education == 3)
        - 0.30 * np.log(limit / 150_000)
    )
    status = np.zeros((n, 6))
    carry = propensity.copy()
    for month in range(5, -1, -1):  # column 0 is the most recent month
        carry = 0.75 * carry + 0.25 * rng.normal(0, 1, n)
        status[:, month] = np.clip(
            np.round(1.8 * carry + rng.normal(0, 0.6, n)), schema.PAY_MIN, schema.PAY_MAX
        )

    utilization = np.clip(0.35 + 0.22 * propensity[:, None] + rng.normal(0, 0.18, (n, 6)), 0.0, 1.6)
    bills = (utilization * limit[:, None]).round(0)
    paid_share = np.clip(0.9 - 0.30 * np.maximum(status, 0) + rng.normal(0, 0.15, (n, 6)), 0.0, 1.2)
    payments = np.maximum(bills * paid_share, 0.0).round(0)

    logit = (
        -3.15
        + 1.05 * status.mean(axis=1)
        + 1.90 * utilization.mean(axis=1)
        + 0.80 * (status[:, 0] >= 1)
        + 0.22 * (sex == 1)
        - 0.25 * np.log(limit / 150_000)
    )
    target = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    frame = pd.DataFrame(
        {
            schema.ID_COL: np.arange(1, n + 1),
            schema.LIMIT_BAL: limit,
            schema.SEX: sex,
            schema.EDUCATION: education,
            schema.MARRIAGE: marriage,
            schema.AGE: age,
        }
    )
    for index, column in enumerate(schema.PAY_COLS):
        frame[column] = status[:, index].astype(int)
    for index, column in enumerate(schema.BILL_COLS):
        frame[column] = bills[:, index]
    for index, column in enumerate(schema.PAY_AMT_COLS):
        frame[column] = payments[:, index]
    frame[schema.TARGET] = target
    frame[schema.AGE_GROUP] = np.where(age <= schema.AGE_GROUP_CUTOFF, "young", "older")
    return frame


def synthetic_split(
    n: int = 8000, seed: int = 0, train_fraction: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = synthetic_clean_frame(n, seed)
    cut = int(n * train_fraction)
    return (
        frame.iloc[:cut].reset_index(drop=True),
        frame.iloc[cut:].reset_index(drop=True),
    )


@pytest.fixture
def local_mlflow(tmp_path):
    """A throwaway tracking store and registry, one per test.

    Pointed at tmp_path rather than the repo so a test run never leaves an
    `mlruns/` directory behind, and so `current_production_version` genuinely
    sees an empty registry instead of whatever the last run left there.
    """
    uri = (tmp_path / "mlruns").as_uri()
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        "credit-risk-test", artifact_location=str(tmp_path / "artifacts")
    )
    # Set it as well as create it: MLflow's fluent API caches the active
    # experiment id in a module global, and without this a test inherits the
    # id of a previous test's tmp_path store, which no longer exists.
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
    yield uri, experiment.name
    mlflow.set_tracking_uri(previous)


# --------------------------------------------------------------- fixture


def test_synthetic_fixture_matches_the_measured_shape_of_the_real_data():
    frame = synthetic_clean_frame()

    assert list(frame.columns) == [*schema.CLEAN_COLUMNS, schema.AGE_GROUP]
    assert frame.isna().sum().sum() == 0
    assert frame[schema.TARGET].mean() == pytest.approx(schema.BASE_POSITIVE_RATE, abs=0.02)

    by_sex = frame.groupby(schema.SEX)[schema.TARGET].mean()
    # The direction of the real base-rate gap: men default more often. If the
    # fixture lost this, the fairness tests would be measuring nothing.
    assert by_sex.loc[1] > by_sex.loc[2]

    for column in schema.PAY_COLS:
        assert frame[column].between(schema.PAY_MIN, schema.PAY_MAX).all()
    assert set(frame[schema.EDUCATION].unique()) <= set(schema.EDUCATION_CODES)
    assert set(frame[schema.MARRIAGE].unique()) <= set(schema.MARRIAGE_CODES)
    assert (frame[list(schema.PAY_AMT_COLS)] >= 0).all().all()
    assert (frame[schema.LIMIT_BAL] > 0).all()


# ----------------------------------------------------------- performance


@pytest.mark.slow
def test_lightgbm_clears_the_model_targets_on_the_fixture():
    train_df, test_df = synthetic_split()
    model = make_lightgbm(n_estimators=250, learning_rate=0.05, num_leaves=31)
    model.fit(feature_frame(train_df), train_df[schema.TARGET])

    proba = model.predict_proba(feature_frame(test_df))[:, 1]
    metrics = evaluate(test_df[schema.TARGET], proba)

    assert metrics["pr_auc"] >= MIN_PR_AUC
    assert metrics["roc_auc"] >= MIN_ROC_AUC
    # Calibration matters because expected loss is computed from these
    # probabilities, not from the ranking.
    assert metrics["brier"] <= 0.14


@pytest.mark.slow
def test_both_families_beat_a_random_ranking():
    train_df, test_df = synthetic_split(n=5000)
    X_train, X_test = feature_frame(train_df), feature_frame(test_df)
    y_train, y_test = train_df[schema.TARGET], test_df[schema.TARGET]
    base_rate = float(y_test.mean())

    scores = {}
    for name, model in (
        ("logistic_regression", make_logistic_regression()),
        ("lightgbm", make_lightgbm(n_estimators=150)),
    ):
        model.fit(X_train, y_train)
        scores[name] = evaluate(y_test, model.predict_proba(X_test)[:, 1])["pr_auc"]

    # A random ranking has PR-AUC equal to the base rate. Anything at or below
    # that is not a model, whatever its accuracy says.
    assert scores["logistic_regression"] > base_rate
    assert scores["lightgbm"] > base_rate


@pytest.mark.slow
@pytest.mark.needs_data
def test_lightgbm_clears_the_model_targets_on_the_real_held_out_batch():
    try:
        train_df, test_df = load_training_splits()
    except FileNotFoundError as exc:
        pytest.skip(f"processed splits are not on disk: {exc}")

    model = make_lightgbm(n_estimators=400, learning_rate=0.05, num_leaves=31)
    model.fit(feature_frame(train_df), train_df[schema.TARGET])

    proba = model.predict_proba(feature_frame(test_df))[:, 1]
    metrics = evaluate(test_df[schema.TARGET], proba)

    assert metrics["pr_auc"] >= MIN_PR_AUC
    assert metrics["roc_auc"] >= MIN_ROC_AUC


# ----------------------------------------------------------------- split


def test_split_train_test_honours_the_batch_column():
    frame = synthetic_clean_frame(n=600)
    frame[BATCH_COL] = np.repeat([1, 2, 3, 4, 5, 6], 100)

    train_df, test_df = split_train_test(frame)

    assert len(train_df) == 400
    assert len(test_df) == 100
    # Batch 6 is the traffic pool and must never reach either side.
    assert schema.SERVING_BATCH not in set(train_df[BATCH_COL]) | set(test_df[BATCH_COL])


def test_split_train_test_refuses_a_frame_that_yields_an_empty_side():
    # 600 accounts all land in batch 1, so there is no held-out batch 5 and
    # training would silently have no test set at all.
    with pytest.raises(ValueError, match="empty split"):
        split_train_test(synthetic_clean_frame(n=600))


# ---------------------------------------------------- tracking & plotting


def test_resolve_tracking_uri_falls_back_to_a_local_store():
    uri, used_fallback = resolve_tracking_uri("http://127.0.0.1:9", timeout=0.5)

    assert used_fallback is True
    assert uri.endswith("mlruns")


def test_resolve_tracking_uri_leaves_non_http_stores_alone():
    uri, used_fallback = resolve_tracking_uri("sqlite:///somewhere.db")

    assert (uri, used_fallback) == ("sqlite:///somewhere.db", False)


@pytest.mark.slow
@pytest.mark.parametrize("factory", [make_logistic_regression, make_lightgbm])
def test_feature_importance_works_for_both_families(tmp_path, factory):
    train_df, _ = synthetic_split(n=2000)
    X, y = feature_frame(train_df), train_df[schema.TARGET]
    model = factory()
    model.fit(X, y)

    importances = feature_importance_frame(model, X.columns)

    assert list(importances.columns) == ["feature", "importance"]
    assert set(importances["feature"]) == set(X.columns)
    assert importances["importance"].is_monotonic_decreasing
    assert (importances["importance"] >= 0).all()

    path = save_feature_importance_plot(importances, tmp_path / "importance.png")
    assert path.exists() and path.stat().st_size > 0


# ------------------------------------------------------------ the gate


def test_evaluate_gate_reports_every_problem_at_once():
    reasons = evaluate_gate(
        {"pr_auc": 0.41},
        {"demographic_parity_difference": 0.20, "equalized_odds_difference": 0.30},
    )
    # Three separate failures, all reported, so one retraining cycle fixes all
    # three instead of discovering them one at a time.
    assert len(reasons) == 3
    assert any("pr_auc" in reason for reason in reasons)
    assert any("demographic_parity" in reason for reason in reasons)
    assert any("equalized_odds" in reason for reason in reasons)


def test_evaluate_gate_accepts_the_prefixed_metric_names_training_logs():
    assert evaluate_gate({"test_pr_auc": 0.60}, FAIR_ENOUGH) == []
    assert "no PR-AUC" in evaluate_gate({"roc_auc": 0.9}, FAIR_ENOUGH)[0]


def test_register_if_passes_refuses_a_weak_model(local_mlflow):
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    decision = register_if_passes(run_id, {"pr_auc": MIN_PR_AUC - 0.01}, FAIR_ENOUGH)

    assert decision.registered is False
    assert decision.refused is True
    assert decision.version is None
    assert "pr_auc" in decision.reasons[0]
    # The refusal is written back onto the run: a rejected experiment is still
    # an experiment, and the reason has to outlive this process.
    tags = mlflow.get_run(run_id).data.tags
    assert tags["registration_refused"] == "true"
    assert "pr_auc" in tags["registration_refusal_reasons"]


def test_register_if_passes_refuses_an_unfair_model_however_accurate(local_mlflow):
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    decision = register_if_passes(
        run_id,
        {"pr_auc": 0.95},
        {"demographic_parity_difference": 0.20, "equalized_odds_difference": 0.01},
    )

    assert decision.registered is False
    assert any("demographic_parity" in reason for reason in decision.reasons)


def test_current_production_version_is_none_on_an_empty_registry(local_mlflow):
    assert current_production_version(model_name="nothing-registered-yet") is None


def test_load_production_model_returns_none_rather_than_raising(local_mlflow):
    assert load_production_model(model_uri="models:/does-not-exist/Production") is None


def test_promote_reports_failure_instead_of_raising(local_mlflow):
    decision = promote(1, "Production", model_name="does-not-exist")

    assert decision.registered is False
    assert "could not promote" in decision.reasons[0]


# ------------------------------------------------------------ end to end


@pytest.mark.slow
def test_train_all_logs_both_families_and_registers_through_the_gate(local_mlflow, tmp_path):
    _, experiment = local_mlflow
    train_df, test_df = synthetic_split(n=3000)

    result = train_all(
        train_df,
        test_df,
        experiment=experiment,
        tracking_uri=mlflow.get_tracking_uri(),
        # One configuration and three folds: this test is about the plumbing,
        # not about finding the best booster.
        grid={"n_estimators": [80], "learning_rate": [0.1], "num_leaves": [15]},
        folds=3,
        thresholds_path=tmp_path / "group_thresholds.json",
    )

    assert {candidate.name for candidate in result.candidates} == {
        "logistic_regression",
        "lightgbm",
    }
    assert result.used_fallback_store is False
    assert result.best.run_id is not None
    assert result.best.metrics["pr_auc"] == max(
        candidate.metrics["pr_auc"] for candidate in result.candidates
    )
    assert list(result.tradeoff["strategy"]) == [
        "baseline",
        "unawareness",
        "reweighing",
        "threshold_optimizer",
    ]
    # Every protected attribute in the frame is audited, not just the gated one,
    # and so is the SEX x AGE_GROUP joint.
    assert set(result.best.fairness_by_attribute) == {
        schema.SEX,
        schema.AGE_GROUP,
        schema.EDUCATION,
        schema.MARRIAGE,
        f"{schema.SEX}_x_{schema.AGE_GROUP}",
    }
    # The cutoffs serving will read are exported as plain JSON, not pickled --
    # and only when the winner cleared the gate, because these are the numbers
    # the API applies to real people. On a synthetic split either family can
    # win, so the assertion follows the gate rather than assuming which did.
    exported = load_group_thresholds(tmp_path / "group_thresholds.json")
    if result.best.gate_passed:
        assert exported == pytest.approx(result.best.group_thresholds)
    else:
        assert exported == {}, "a refused model's thresholds must never reach serving"

    # The convenience accessors the DAG reads.
    assert result.run_id == result.best.run_id
    assert result.metrics == result.best.metrics
    assert result.estimator is result.best.estimator

    run = mlflow.get_run(result.best.run_id)
    assert "test_pr_auc" in run.data.metrics
    assert "unawareness_probe_roc_auc" in run.data.metrics
    assert "by_SEX_x_AGE_GROUP_equalized_odds_difference" in run.data.metrics
    assert "by_SEX_x_AGE_GROUP_n_smallest_group" in run.data.metrics
    assert run.data.tags["fairness_gate"] in {"pass", "fail"}
    artifacts = {item.path for item in mlflow.MlflowClient().list_artifacts(result.best.run_id)}
    # fairness_tradeoff.png is the figure the spec asks the mitigation section
    # to be argued from -- dp_diff against PR-AUC, one point per strategy. A
    # CSV of the same numbers is not the same deliverable.
    assert {
        "fairness_tradeoff.csv",
        "fairness_tradeoff.png",
        "cost_matrix.csv",
        "feature_importance.png",
        "model",
    } <= artifacts

    decision = register_if_passes(result.best.run_id, result.best.metrics, result.best.fairness)
    # Whether this candidate passes depends on the fixture, so the assertion is
    # on the consistency of the decision, not on its direction.
    assert decision.registered == (evaluate_gate(result.best.metrics, result.best.fairness) == [])
    if not decision.registered:
        assert decision.reasons
        return

    assert decision.version is not None
    promotion = promote(decision.version, "Production")
    assert promotion.registered is True
    assert current_production_version() == decision.version


@pytest.mark.slow
def test_a_model_that_clears_the_gate_is_registered_and_promotable(local_mlflow):
    """The happy path, with the gate inputs supplied rather than trained.

    The fixture's own fairness numbers move with the seed, so the successful
    branch is exercised with metrics chosen to pass. The model logged is real;
    only the decision inputs are fixed.
    """
    train_df, test_df = synthetic_split(n=1500)
    X = feature_frame(train_df)
    model = make_lightgbm(n_estimators=40).fit(X, train_df[schema.TARGET])

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=X.head(3))
        run_id = run.info.run_id

    decision = register_if_passes(run_id, {"pr_auc": 0.61}, FAIR_ENOUGH)

    assert decision.registered is True
    assert decision.version is not None
    assert mlflow.get_run(run_id).data.tags["registered_version"] == decision.version
    # Registered is not the same as serving: nothing is in Production until
    # somebody promotes it, and /health must report that honestly.
    assert current_production_version() is None

    assert promote(decision.version, "Production").registered is True
    assert current_production_version() == decision.version

    scored = load_production_model()
    assert scored is not None
    predictions = scored.predict(feature_frame(test_df).head(5))
    assert len(predictions) == 5


def test_load_training_splits_reads_what_the_data_pipeline_wrote(tmp_path):
    frame = synthetic_clean_frame(n=600)
    frame[BATCH_COL] = np.repeat([1, 2, 3, 4, 5, 6], 100)
    write_splits(frame, tmp_path)

    train_df, test_df = load_training_splits(tmp_path)

    assert len(train_df) == 400
    assert len(test_df) == 100


def test_register_if_passes_reports_a_registry_it_cannot_reach(local_mlflow, monkeypatch):
    def explode(**_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("credit_risk.models.registry.mlflow.register_model", explode)
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    decision = register_if_passes(run_id, {"pr_auc": 0.9}, FAIR_ENOUGH)

    # The model was fine; the infrastructure was not. Those are different
    # findings and the reason string has to say which one happened.
    assert decision.registered is False
    assert "registry unavailable" in decision.reasons[0]
    assert "connection refused" in decision.reasons[0]


def test_a_refusal_survives_a_tracking_server_that_will_not_take_the_tag(local_mlflow):
    class DeadClient:
        def set_tag(self, *_args, **_kwargs):
            raise RuntimeError("tracking server gone")

    decision = register_if_passes(
        "run-that-does-not-matter", {"pr_auc": 0.1}, FAIR_ENOUGH, client=DeadClient()
    )

    # Losing the tag is survivable. Losing the decision is not.
    assert decision.registered is False
    assert "pr_auc" in decision.reasons[0]


def test_main_exits_non_zero_when_the_winner_fails_the_fairness_gate(monkeypatch, capsys, tmp_path):
    """The CLI contract, with training stubbed out.

    `python -m credit_risk.models.train` is a DAG task as well as something a
    human runs, and both read the exit code. A model that trained successfully
    but cannot be registered must not look like a success.

    `--result` is passed for a reason that is not cosmetic: without it the CLI
    writes its hand-off artefact to data/processed, so running the unit suite
    left a stub run id where the next DAG run would read a real one.
    """

    def fake_train_all(**kwargs):
        captured.update(kwargs)
        return TrainingResult(
            best=stub_candidate(gate_passed=kwargs["folds"] > 2),
            candidates=[stub_candidate(gate_passed=True)],
            tracking_uri="file:///tmp/mlruns",
            used_fallback_store=True,
            tradeoff=pd.DataFrame([{"strategy": "baseline", "pr_auc": 0.6}]),
        )

    def stub_candidate(*, gate_passed: bool) -> CandidateResult:
        return CandidateResult(
            name="lightgbm",
            run_id="abc123",
            estimator=make_lightgbm(),
            params={},
            cv_pr_auc_mean=0.6,
            cv_pr_auc_std=0.01,
            metrics={"pr_auc": 0.6},
            fairness={"demographic_parity_difference": 0.2},
            gate_passed=gate_passed,
            gate_reasons=[] if gate_passed else ["demographic_parity_difference 0.20 exceeds 0.05"],
        )

    captured: dict[str, object] = {}
    monkeypatch.setattr("credit_risk.models.train.train_all", fake_train_all)
    handoff = ["--result", str(tmp_path / "training_result.json")]

    assert main([*handoff, "--folds", "3", "--seed", "7", "--no-mlflow"]) == 0
    assert captured["folds"] == 3
    assert captured["seed"] == 7
    assert captured["log_to_mlflow"] is False
    assert "abc123" in capsys.readouterr().out
    # The artefact the next two DAG tasks read, written where it was asked for.
    assert json.loads((tmp_path / "training_result.json").read_text())["run_id"] == "abc123"

    assert main([*handoff, "--folds", "2", "--capacity", "0.05"]) == 2
    assert captured["capacity_fraction"] == 0.05


def test_resolve_tracking_uri_keeps_a_server_that_answers_its_health_check(monkeypatch):
    probed: dict[str, object] = {}

    class Healthy:
        def raise_for_status(self) -> None:
            return None

    def fake_get(url, timeout):
        probed["url"], probed["timeout"] = url, timeout
        return Healthy()

    monkeypatch.setattr("credit_risk.models.train.requests.get", fake_get)

    uri, used_fallback = resolve_tracking_uri("http://mlflow:15020/")

    assert (uri, used_fallback) == ("http://mlflow:15020/", False)
    # A TCP connect is not proof of a working tracking server; /health is.
    assert probed["url"] == "http://mlflow:15020/health"


def test_train_all_falls_back_to_the_written_splits_when_given_no_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "credit_risk.models.train.load_training_splits",
        lambda processed_dir=None: synthetic_split(n=1500),
    )

    result = train_all(
        grid={"n_estimators": [40], "learning_rate": [0.2], "num_leaves": [7]},
        folds=2,
        log_to_mlflow=False,
        thresholds_path=tmp_path / "group_thresholds.json",
    )

    # Nothing was logged, so there is no run to point the registry at -- which
    # is exactly the state the `--no-mlflow` CLI flag leaves behind.
    assert result.run_id is None
    assert len(result.candidates) == 2

    # The exported thresholds are what serving applies, so they are written only
    # when the winner cleared the gate. Asserting the file unconditionally would
    # have passed while quietly allowing a refused model's cutoffs to reach the
    # API -- which is the failure this branch exists to prevent. On a small
    # synthetic split either family can win, so the assertion follows the gate
    # rather than assuming which one did.
    exported = tmp_path / "group_thresholds.json"
    assert exported.exists() is result.best.gate_passed


def test_train_all_does_not_probe_the_tracking_server_when_it_will_not_log(monkeypatch, tmp_path):
    """`--no-mlflow` means no network, not "no network except one health check".

    The probe used to run regardless, so on a machine where this project's own
    compose stack happens to be up on 15020 the call succeeded and the test
    exercised a different branch than the same test on CI. A unit test that
    depends on what is listening on localhost is not a unit test.
    """

    def explode(*_args, **_kwargs):
        raise AssertionError("no HTTP request should be made with log_to_mlflow=False")

    monkeypatch.setattr("credit_risk.models.train.requests.get", explode)
    train_df, test_df = synthetic_split(n=700)

    result = train_all(
        train_df,
        test_df,
        grid={"n_estimators": [20], "learning_rate": [0.2], "num_leaves": [7]},
        folds=2,
        log_to_mlflow=False,
        thresholds_path=tmp_path / "group_thresholds.json",
    )

    assert result.used_fallback_store is False


def test_fit_group_thresholds_returns_nothing_when_fairlearn_refuses_the_sample():
    """A group carrying one label is a property of the batch, not a defect.

    fairlearn raises `Degenerate labels for sensitive feature value 3`, and
    MARRIAGE=3 is 323 rows in 30,000 -- so a thin batch hits this after the
    grid search has already been paid for. The run has to survive it; the
    group-aware policy is what gets dropped, and serving's documented fallback
    is the single configured cutoff.
    """
    rng = np.random.default_rng(3)
    n = 300
    sensitive = np.array([1] * 140 + [2] * 140 + [3] * 20)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (rng.random(n) < 0.3).astype(int)
    y[sensitive == 3] = 0
    model = make_logistic_regression().fit(X, y)

    assert fit_group_thresholds(model, X, y, sensitive) == {}

    # The same call on a sample fairlearn accepts still produces cutoffs, so
    # the guard cannot be hiding a permanently broken code path.
    healthy = sensitive[sensitive != 3]
    assert set(fit_group_thresholds(model, X[sensitive != 3], y[sensitive != 3], healthy)) == {
        "1",
        "2",
    }


def test_save_tradeoff_plot_writes_the_figure_the_fairness_section_argues_from(tmp_path):
    tradeoff = pd.DataFrame(
        [
            {"strategy": "baseline", "dp_diff": 0.038, "pr_auc": 0.567},
            {"strategy": "unawareness", "dp_diff": 0.036, "pr_auc": 0.560},
            {"strategy": "reweighing", "dp_diff": 0.028, "pr_auc": 0.569},
            {"strategy": "threshold_optimizer", "dp_diff": 0.024, "pr_auc": 0.567},
        ]
    )

    path = save_tradeoff_plot(tradeoff, tmp_path / "fairness_tradeoff.png")

    assert path.exists() and path.stat().st_size > 0

    # A table missing dp_diff would otherwise produce an empty axis rather than
    # an error, and an empty chart in a report is worse than no chart.
    with pytest.raises(ValueError, match="dp_diff"):
        save_tradeoff_plot(tradeoff.drop(columns=["dp_diff"]), tmp_path / "broken.png")


# ------------------------------------------------- the three DAG entry points


def _handoff(tmp_path, *, gate_passed: bool, run_id: str | None = "run-1") -> object:
    """The artefact training leaves for evaluate, register and report."""
    path = tmp_path / "training_result.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": "lightgbm",
                "tracking_uri": "file:///tmp/mlruns",
                "used_fallback_store": False,
                "metrics": {
                    "pr_auc": 0.61 if gate_passed else 0.10,
                    "base_rate": 0.204,
                    "recall_at_k": 0.349,
                    "capacity_fraction": 0.10,
                },
                "fairness": {
                    "demographic_parity_difference": 0.01 if gate_passed else 0.30,
                    "equalized_odds_difference": 0.02,
                },
                "fairness_by_attribute": {schema.SEX: {"demographic_parity_difference": 0.01}},
                "gate_passed": gate_passed,
                "gate_reasons": [] if gate_passed else ["demographic_parity_difference too high"],
                "group_thresholds": {"1": 0.45, "2": 0.40},
                "candidates": [
                    {
                        "name": "lightgbm",
                        "run_id": run_id,
                        "pr_auc": 0.61,
                        "gate_passed": gate_passed,
                        "gate_reasons": [],
                    }
                ],
                "tradeoff": [{"strategy": "baseline", "pr_auc": 0.61, "dp_diff": 0.01}],
            }
        )
    )
    return path


def test_evaluate_cli_exits_two_on_a_candidate_the_gate_refuses(tmp_path, capsys):
    """The DAG task that makes the fairness gate executable rather than stated.

    `evaluate_and_gate` runs this module as a subprocess and reads its exit
    code, so a refused candidate has to come back non-zero -- otherwise the
    task goes green, `register_model` runs next, and the gate exists only in
    the design document.
    """
    refused = _handoff(tmp_path, gate_passed=False)

    assert evaluate_module.main(["--result", str(refused)]) == 2
    decision = json.loads((tmp_path / "gate_decision.json").read_text())
    assert decision["passed"] is False
    assert any("demographic_parity" in reason for reason in decision["reasons"])
    assert "GATE REFUSED" in capsys.readouterr().err

    accepted = _handoff(tmp_path, gate_passed=True)
    assert evaluate_module.main(["--result", str(accepted), "--out", str(tmp_path / "g.json")]) == 0
    assert json.loads((tmp_path / "g.json").read_text())["passed"] is True


def test_registry_cli_registers_and_promotes_the_candidate_that_passed(tmp_path, local_mlflow):
    train_df, _ = synthetic_split(n=400)
    X = feature_frame(train_df)
    model = make_logistic_regression().fit(X, train_df[schema.TARGET])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=X.head(3))
        run_id = run.info.run_id

    path = _handoff(tmp_path, gate_passed=True, run_id=run_id)
    # The artefact carries the tracking URI so the subprocess talks to the same
    # store training used; the fixture has already pointed mlflow at it.
    json_path = tmp_path / "training_result.json"
    payload = json.loads(json_path.read_text())
    payload["tracking_uri"] = mlflow.get_tracking_uri()
    json_path.write_text(json.dumps(payload))

    assert registry_module.main(["--result", str(path)]) == 0

    written = json.loads((tmp_path / "registration.json").read_text())
    assert written["registered"] is True
    assert written["promoted"] is True
    assert current_production_version() == written["version"]


def test_registry_cli_refuses_and_says_which_of_the_two_failures_it_was(tmp_path, capsys):
    # No run id at all: training was run with --no-mlflow, so there is nothing
    # to register. That is a different finding from a model the gate refused,
    # and the DAG log has to distinguish them.
    assert (
        registry_module.main(["--result", str(_handoff(tmp_path, gate_passed=True, run_id=None))])
        == 2
    )
    assert "no run id" in capsys.readouterr().err

    # A refused candidate never reaches the registry either, but for a reason
    # that is about the model rather than about the plumbing.
    assert registry_module.main(["--result", str(_handoff(tmp_path, gate_passed=False))]) == 2
    assert json.loads((tmp_path / "registration.json").read_text())["registered"] is False


def test_report_cli_publishes_one_self_contained_page(tmp_path):
    out = tmp_path / "run_report.html"

    assert (
        report_module.main(
            ["--result", str(_handoff(tmp_path, gate_passed=True)), "--out", str(out)]
        )
        == 0
    )

    page = out.read_text()
    assert page.startswith("<!doctype html>")
    # No external stylesheet and no script tag: the file has to survive being
    # emailed or opened from a USB stick during the demo.
    assert "<link" not in page and "<script" not in page
    assert "GATE PASSED" in page

    refused = tmp_path / "refused.html"
    report_module.main(
        ["--result", str(_handoff(tmp_path, gate_passed=False)), "--out", str(refused)]
    )
    assert "GATE REFUSED" in refused.read_text()

    # A missing artefact is the ordinary "training has not run yet" state, and
    # the task has to say that rather than raising a traceback at the reader.
    assert report_module.main(["--result", str(tmp_path / "absent.json")]) == 2
