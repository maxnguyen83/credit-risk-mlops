"""Batch scoring: a file of accounts in, a ranked call list out.

The HTTP layer is a fake that speaks the API's own request and response models,
so a payload the real API would refuse is refused here too, and no test reaches
a network. Each test pins one promise the module makes to the risk desk or to
the DAG: which rows are scored, how many are called, what happens when the API
misbehaves, and what lands on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from credit_risk import schema
from credit_risk.config import Settings, settings
from credit_risk.scoring import batch
from credit_risk.serving.models import (
    EXAMPLE_APPLICATION,
    BatchPredictRequest,
    CreditApplication,
)

API = "http://api.test"
BATCH_PATH = "/api/v1/predict/batch"
EXPLAIN_PATH = "/api/v1/explain"

SCORES_COLUMNS = [
    "rank",
    "account_id",
    "default_probability",
    "risk_band",
    "decision",
    "in_call_list",
]
CALL_LIST_COLUMNS = [
    "rank",
    "account_id",
    "default_probability",
    "risk_band",
    "decision",
    "top_reasons",
]
SUMMARY_KEYS = {
    "generated_at",
    "run_dir",
    "input_path",
    "input_sha256",
    "n_input",
    "n_rejected",
    "rejected_fraction",
    "rejected_over_tolerance",
    "rejected_by_reason",
    "n_scored",
    "n_call_list",
    "capacity_fraction",
    "n_above_threshold",
    "model_name",
    "model_version",
    "threshold_used",
    "max_reasons",
    "n_reasons_missing",
    "dag_run_id",
}


# ------------------------------------------------------------------ fakes


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


Step = Exception | int | None


class FakeApi:
    """Stands in for ``requests.Session``: answers the two endpoints batch scoring calls.

    ``batch_script`` and ``explain_script`` say what the next calls do: an
    exception is raised, an int is returned as that HTTP status, ``None`` is a
    normal answer. Once a script runs out every call is answered normally.
    """

    def __init__(
        self,
        *,
        probabilities: Mapping[str, float] | None = None,
        versions: Iterable[str] = ("2",),
        thresholds: Iterable[float] = (0.5,),
        policy: str = "base",
        batch_script: Iterable[Step] = (),
        explain_script: Mapping[str, Step] | None = None,
        explain_version: str | None = None,
    ) -> None:
        self.probabilities = dict(probabilities or {})
        self.versions = list(versions)
        self.thresholds = list(thresholds)
        self.policy = policy
        self.batch_script = list(batch_script)
        self.explain_script = dict(explain_script or {})
        self.explain_version = explain_version
        self.urls: list[str] = []
        self.batch_calls: list[list[dict[str, Any]]] = []
        self.explain_calls: list[str] = []
        self.timeouts: list[Any] = []
        self._answered = 0

    def post(self, url: str, json: Any = None, timeout: Any = None) -> FakeResponse:
        self.urls.append(url)
        self.timeouts.append(timeout)
        if url == API + BATCH_PATH:
            return self._batch(json)
        if url == API + EXPLAIN_PATH:
            return self._explain(json)
        return FakeResponse(404, {"code": "not_found"})

    def probability(self, account_id: str) -> float:
        if account_id in self.probabilities:
            return self.probabilities[account_id]
        digest = hashlib.sha256(account_id.encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _batch(self, payload: Any) -> FakeResponse:
        self.batch_calls.append(list(payload["applications"]))
        step = self.batch_script.pop(0) if self.batch_script else None
        if isinstance(step, Exception):
            raise step
        if isinstance(step, int):
            return FakeResponse(step, {"code": "scripted", "message": "scripted failure"})
        # The real request model: a payload the API would refuse is refused here.
        request = BatchPredictRequest.model_validate(payload)
        index = min(self._answered, len(self.versions) - 1)
        version = self.versions[index]
        threshold = self.thresholds[min(self._answered, len(self.thresholds) - 1)]
        self._answered += 1
        predictions = []
        for application in request.applications:
            assert application.account_id is not None
            probability = self.probability(application.account_id)
            band = "high" if probability >= 0.5 else "medium" if probability >= 0.2212 else "low"
            predictions.append(
                {
                    "account_id": application.account_id,
                    "default_probability": probability,
                    "decision": "intervene" if probability >= threshold else "monitor",
                    "risk_band": band,
                    "threshold_used": threshold,
                    "threshold_policy": self.policy,
                    "model_name": "credit-risk",
                    "model_version": version,
                    "served_at": "2026-10-04T00:00:00Z",
                    "request_id": "r",
                }
            )
        return FakeResponse(
            200,
            {
                "count": len(predictions),
                "intervene_count": sum(p["decision"] == "intervene" for p in predictions),
                "high_risk_count": sum(p["risk_band"] == "high" for p in predictions),
                "predictions": predictions,
                "model_name": "credit-risk",
                "model_version": version,
                "served_at": "2026-10-04T00:00:00Z",
                "request_id": "r",
            },
        )

    def _explain(self, payload: Any) -> FakeResponse:
        application = CreditApplication.model_validate(payload)
        account_id = str(application.account_id)
        self.explain_calls.append(account_id)
        step = self.explain_script.get(account_id)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, int):
            return FakeResponse(step, {"code": "scripted", "message": "scripted failure"})
        return FakeResponse(
            200,
            {
                "account_id": account_id,
                "top_reasons": [f"first reason for {account_id}", "second reason"],
                "model_name": "credit-risk",
                "model_version": self.explain_version or self.versions[-1],
            },
        )


def _client(api: FakeApi, sleeps: list[float] | None = None) -> batch.ApiClient:
    record = sleeps if sleeps is not None else []
    return batch.ApiClient(API, session=api, sleep=record.append)


def _pool(n_rows: int = 20) -> pd.DataFrame:
    """Shaped like data/processed/serving_pool.parquet: ID, inputs, label, AGE_GROUP, batch.

    Every row is the documented example account; the fake API scores by
    account id, so identical features do not mean identical scores.
    """
    features = {key: value for key, value in EXAMPLE_APPLICATION.items() if key != "account_id"}
    frame = pd.DataFrame([features] * n_rows)
    frame.insert(0, schema.ID_COL, range(1, n_rows + 1))
    frame[schema.TARGET] = 0
    frame[schema.AGE_GROUP] = "older"
    frame["batch"] = schema.SERVING_BATCH
    return frame


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def _validated_run(data_dir: Path, frame: pd.DataFrame, name: str = "pool.parquet") -> Path:
    source = _write(frame, data_dir / "incoming" / name)
    run_dir = batch.new_run_dir(data_dir)
    batch.validate_input(source, run_dir, data_dir)
    return run_dir


def _scored_run(data_dir: Path, frame: pd.DataFrame, api: FakeApi) -> Path:
    run_dir = _validated_run(data_dir, frame)
    batch.score_run(run_dir, _client(api))
    return run_dir


# -------------------------------------------------------------- validation


def test_a_clean_pool_validates_and_the_label_never_reaches_the_api() -> None:
    frame = _pool(10)

    result = batch.validate_frame(frame)

    assert result.n_input == 10
    assert result.rejected.empty
    assert list(result.valid.columns) == ["input_row", "account_id", *batch.INPUT_COLUMNS]
    assert schema.TARGET not in result.valid.columns
    assert schema.AGE_GROUP not in result.valid.columns


def test_the_input_columns_are_the_api_fields_and_nothing_else() -> None:
    expected = set(EXAMPLE_APPLICATION) - {"account_id"}
    assert set(batch.INPUT_COLUMNS) == expected


def test_ID_becomes_a_string_account_id_and_account_id_wins_over_ID() -> None:
    frame = _pool(3)
    assert batch.validate_frame(frame).valid["account_id"].tolist() == ["1", "2", "3"]

    frame["account_id"] = ["A-1", "A-2", "A-3"]
    assert batch.validate_frame(frame).valid["account_id"].tolist() == ["A-1", "A-2", "A-3"]


def test_a_file_without_ids_gets_row_references_rather_than_anonymous_rows() -> None:
    frame = _pool(3).drop(columns=[schema.ID_COL])

    assert batch.validate_frame(frame).valid["account_id"].tolist() == ["row-1", "row-2", "row-3"]


def test_a_float_id_column_with_a_gap_still_reads_as_whole_ids() -> None:
    """One missing ID turns the parquet column into floats: 1.0 must still be "1"."""
    frame = _pool(3).astype({schema.ID_COL: "float64"})
    frame.loc[1, schema.ID_COL] = float("nan")

    assert batch.validate_frame(frame).valid["account_id"].tolist() == ["1", "row-2", "3"]


def test_bad_rows_are_rejected_with_the_api_reason_and_the_rest_go_through() -> None:
    frame = _pool(8).astype({"BILL_AMT1": "float64", "PAY_1": "float64"})
    frame.loc[1, schema.AGE] = 18
    frame.loc[3, schema.LIMIT_BAL] = 0
    frame.loc[4, "BILL_AMT1"] = float("nan")  # an empty cell is a missing value, not a number
    frame.loc[6, "PAY_1"] = 1.5

    result = batch.validate_frame(frame)

    assert result.valid["input_row"].tolist() == [1, 3, 6, 8]
    rejected = result.rejected.set_index("row")
    assert list(result.rejected.columns) == ["row", "account_id", "reason"]
    assert rejected.loc[2, "reason"].startswith("AGE: Input should be greater than or equal to 21")
    assert rejected.loc[4, "reason"].startswith("LIMIT_BAL: Input should be greater than 0")
    assert rejected.loc[5, "reason"].startswith("BILL_AMT1:")
    assert rejected.loc[7, "reason"].startswith("PAY_1:")
    assert rejected.loc[2, "account_id"] == "2"
    assert sum(result.rejected_by_reason.values()) == 4


def test_validation_applies_the_api_rules_not_the_data_dictionary() -> None:
    """EDUCATION 0 is undocumented but the API accepts it; 7 the API refuses."""
    frame = _pool(2)
    frame.loc[0, schema.EDUCATION] = 0
    frame.loc[1, schema.EDUCATION] = 7

    result = batch.validate_frame(frame)

    assert result.valid["input_row"].tolist() == [1]
    assert result.rejected["reason"].tolist() == [
        "EDUCATION: Input should be less than or equal to 6"
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.rename(columns={"PAY_1": "PAY_0"}), "PAY_1"),
        (lambda f: f.assign(customer_name="x"), "customer_name"),
    ],
)
def test_a_missing_or_unknown_column_fails_the_whole_file(
    change: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(batch.BadInputError, match=message):
        batch.validate_frame(change(_pool(3)))


def test_validate_writes_valid_and_rejected_files_and_reports_the_damage(data_dir: Path) -> None:
    frame = _pool(10)
    frame.loc[0, schema.AGE] = 99  # 1 of 10 rejected: over the 5% tolerance
    source = _write(frame, data_dir / "incoming" / "pool.csv")
    run_dir = batch.new_run_dir(data_dir)

    report = batch.validate_input(source, run_dir, data_dir)

    assert report["n_input"] == 10
    assert report["n_rejected"] == 1
    assert report["rejected_fraction"] == 0.1
    assert report["rejected_over_tolerance"] is True
    assert report["input_path"] == "incoming/pool.csv"
    assert report["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(pd.read_parquet(run_dir / "valid.parquet")) == 9
    rejected = pd.read_csv(run_dir / "rejected.csv")
    assert rejected["row"].tolist() == [1]
    assert json.loads((run_dir / "validation.json").read_text()) == report


def test_a_clean_file_leaves_no_rejected_csv_and_stays_under_tolerance(data_dir: Path) -> None:
    run_dir = _validated_run(data_dir, _pool(5))

    report = json.loads((run_dir / "validation.json").read_text())
    assert report["rejected_over_tolerance"] is False
    assert report["rejected_by_reason"] == {}
    assert not (run_dir / "rejected.csv").exists()


def test_a_file_with_no_valid_row_is_bad_input(data_dir: Path) -> None:
    frame = _pool(3)
    frame[schema.AGE] = 5
    source = _write(frame, data_dir / "incoming" / "bad.parquet")

    with pytest.raises(batch.BadInputError, match="no valid row"):
        batch.validate_input(source, batch.new_run_dir(data_dir), data_dir)


def test_two_runs_in_the_same_second_get_their_own_directories(data_dir: Path) -> None:
    moment = datetime(2026, 10, 4, 9, 30, tzinfo=UTC)

    first = batch.new_run_dir(data_dir, now=moment)
    second = batch.new_run_dir(data_dir, now=moment)

    assert first.name == "20261004T093000Z"
    assert second != first
    assert first.parent == second.parent == data_dir / "scored"


# ------------------------------------------------------------------ scoring


def test_scoring_posts_in_chunks_and_keeps_the_input_order(data_dir: Path) -> None:
    api = FakeApi()
    run_dir = _validated_run(data_dir, _pool(5))

    report = batch.score_run(run_dir, _client(api), chunk_size=2)

    assert [len(chunk) for chunk in api.batch_calls] == [2, 2, 1]
    posted = [item["account_id"] for chunk in api.batch_calls for item in chunk]
    assert posted == ["1", "2", "3", "4", "5"]
    scored = pd.read_parquet(run_dir / "scored.parquet")
    assert scored["account_id"].tolist() == posted
    assert scored["default_probability"].tolist() == [api.probability(a) for a in posted]
    assert report["n_scored"] == 5
    assert (report["model_name"], report["model_version"]) == ("credit-risk", "2")
    assert report["threshold_used"] == 0.5


def test_the_default_chunk_is_the_api_batch_cap(data_dir: Path) -> None:
    frame = _pool(10)
    big = pd.concat([frame] * (settings.max_batch_size // 10 + 1), ignore_index=True)
    big[schema.ID_COL] = range(1, len(big) + 1)
    api = FakeApi()
    run_dir = _validated_run(data_dir, big)

    batch.score_run(run_dir, _client(api))

    assert [len(chunk) for chunk in api.batch_calls] == [settings.max_batch_size, 10]


def test_a_dropped_connection_and_a_503_are_retried(data_dir: Path) -> None:
    api = FakeApi(batch_script=[requests.ConnectionError("reset"), 503])
    sleeps: list[float] = []
    run_dir = _validated_run(data_dir, _pool(3))

    batch.score_run(run_dir, _client(api, sleeps))

    assert len(api.batch_calls) == 3
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]


def test_an_api_that_stays_down_is_reported_as_unavailable(data_dir: Path) -> None:
    api = FakeApi(batch_script=[503] * 10)
    run_dir = _validated_run(data_dir, _pool(3))

    with pytest.raises(batch.ApiUnavailableError):
        batch.score_run(run_dir, _client(api))

    assert len(api.batch_calls) == batch.ATTEMPTS
    assert not (run_dir / "scored.parquet").exists()


def test_a_client_error_is_not_retried(data_dir: Path) -> None:
    api = FakeApi(batch_script=[422])
    run_dir = _validated_run(data_dir, _pool(3))

    with pytest.raises(batch.ApiResponseError, match="422"):
        batch.score_run(run_dir, _client(api))

    assert len(api.batch_calls) == 1


def test_predictions_that_do_not_line_up_with_the_accounts_sent_fail_the_run(
    data_dir: Path,
) -> None:
    """Scores are matched to accounts by position; a reordered answer would mislabel them."""

    class Reordering(FakeApi):
        def _batch(self, payload: Any) -> FakeResponse:
            response = super()._batch(payload)
            response._body["predictions"].reverse()
            return response

    run_dir = _validated_run(data_dir, _pool(3))

    with pytest.raises(batch.ApiResponseError, match="order"):
        batch.score_run(run_dir, _client(Reordering()))

    assert not (run_dir / "scored.parquet").exists()


def test_a_model_change_mid_run_fails_the_run(data_dir: Path) -> None:
    api = FakeApi(versions=["2", "3"])
    run_dir = _validated_run(data_dir, _pool(4))

    with pytest.raises(batch.ServedModelChangedError, match="3"):
        batch.score_run(run_dir, _client(api), chunk_size=2)

    assert not (run_dir / "scored.parquet").exists()


def test_a_threshold_change_mid_run_fails_the_run(data_dir: Path) -> None:
    api = FakeApi(thresholds=[0.5, 0.4])
    run_dir = _validated_run(data_dir, _pool(4))

    with pytest.raises(batch.ServedModelChangedError, match="threshold"):
        batch.score_run(run_dir, _client(api), chunk_size=2)


def test_per_group_cutoffs_are_recorded_as_no_single_threshold(data_dir: Path) -> None:
    api = FakeApi(thresholds=[0.5, 0.4], policy="group_aware_equalized_odds")
    run_dir = _validated_run(data_dir, _pool(4))

    report = batch.score_run(run_dir, _client(api), chunk_size=2)

    assert report["threshold_used"] is None


# ------------------------------------------------------- ranking and the list


@pytest.mark.parametrize(
    ("n", "fraction", "expected"),
    [
        (5000, 0.10, 500),
        (100, 0.07, 7),  # 0.07 * 100 is 7.000000000000001 in floating point
        (11, 0.10, 2),
        (10, 0.10, 1),
        (3, 0.10, 1),
        (1, 0.10, 1),
        (10, 1.0, 10),
    ],
)
def test_the_call_list_is_the_ceiling_of_capacity_times_n(
    n: int, fraction: float, expected: int
) -> None:
    assert batch.call_list_size(n, fraction) == expected


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, math.nan])
def test_a_capacity_outside_zero_to_one_is_refused(fraction: float) -> None:
    with pytest.raises(batch.BadInputError, match="capacity"):
        batch.call_list_size(10, fraction)


def test_ranking_is_by_probability_descending_with_ties_in_input_order() -> None:
    scored = pd.DataFrame(
        {
            "input_row": [1, 2, 3, 4, 5],
            "account_id": ["a", "b", "c", "d", "e"],
            "default_probability": [0.2, 0.9, 0.5, 0.9, 0.12345],
            "risk_band": ["low", "high", "high", "high", "low"],
            "decision": ["monitor", "intervene", "intervene", "intervene", "monitor"],
        }
    )

    ranked = batch.rank_scores(scored, capacity_fraction=0.4)

    assert ranked["account_id"].tolist() == ["b", "d", "c", "a", "e"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4, 5]
    assert ranked["in_call_list"].tolist() == [True, True, False, False, False]


def test_a_re_run_on_a_file_full_of_ties_calls_the_same_accounts() -> None:
    """Five rows sort stably by luck; sixty with two values do not, unless asked to."""
    n = 60
    scored = pd.DataFrame(
        {
            "input_row": range(1, n + 1),
            "account_id": [str(i) for i in range(1, n + 1)],
            "default_probability": [0.6 if i % 3 == 0 else 0.3 for i in range(n)],
            "risk_band": "high",
            "decision": "intervene",
        }
    )

    ranked = batch.rank_scores(scored, capacity_fraction=0.5)

    high = [str(i + 1) for i in range(n) if i % 3 == 0]
    low = [str(i + 1) for i in range(n) if i % 3]
    assert ranked["account_id"].tolist() == high + low


# ----------------------------------------------------------------- publish


def test_publish_writes_the_documented_files_and_points_latest_at_them(data_dir: Path) -> None:
    frame = _pool(20)
    frame.loc[0, schema.AGE] = 99
    api = FakeApi()
    run_dir = _scored_run(data_dir, frame, api)

    summary = batch.publish_run(
        run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.1, dag_run_id="manual__x"
    )

    scores = pd.read_csv(run_dir / "scores.csv", dtype={"account_id": str})
    call_list = pd.read_csv(run_dir / "call_list.csv", dtype={"account_id": str})
    assert list(scores.columns) == SCORES_COLUMNS
    assert list(call_list.columns) == CALL_LIST_COLUMNS
    assert len(scores) == 19
    assert scores["default_probability"].is_monotonic_decreasing
    assert all(round(p, 4) == p for p in scores["default_probability"])
    assert scores["in_call_list"].sum() == len(call_list) == 2  # ceil(0.1 * 19)
    assert call_list["account_id"].tolist() == scores["account_id"].head(2).tolist()
    top = call_list["account_id"].iloc[0]
    assert call_list["top_reasons"].iloc[0] == f"first reason for {top} | second reason"

    assert set(summary) == SUMMARY_KEYS
    assert summary["run_dir"] == f"scored/{run_dir.name}"
    assert summary["input_path"] == "incoming/pool.parquet"
    assert (summary["n_input"], summary["n_rejected"], summary["n_scored"]) == (20, 1, 19)
    assert summary["n_call_list"] == 2
    assert summary["capacity_fraction"] == 0.1
    assert summary["n_above_threshold"] == int((scores["decision"] == "intervene").sum())
    assert (summary["model_name"], summary["model_version"]) == ("credit-risk", "2")
    assert summary["threshold_used"] == 0.5
    assert summary["n_reasons_missing"] == 0
    assert summary["dag_run_id"] == "manual__x"
    assert summary["generated_at"].endswith("Z")
    assert json.loads((run_dir / "summary.json").read_text()) == summary
    assert json.loads((data_dir / "scored" / "latest.json").read_text()) == summary
    assert sorted(p.name for p in (data_dir / "scored").iterdir()) == [run_dir.name, "latest.json"]


def test_reasons_are_asked_for_the_call_list_only_and_in_rank_order(data_dir: Path) -> None:
    api = FakeApi()
    run_dir = _scored_run(data_dir, _pool(30), api)

    batch.publish_run(run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.1)

    call_list = pd.read_csv(run_dir / "call_list.csv", dtype={"account_id": str})
    assert api.explain_calls == call_list["account_id"].tolist()
    assert len(api.explain_calls) == 3


def test_a_failed_explanation_leaves_that_row_empty_and_the_run_goes_on(data_dir: Path) -> None:
    probabilities = {str(i): 1.0 - i / 100 for i in range(1, 21)}
    api = FakeApi(probabilities=probabilities, explain_script={"2": 500})
    run_dir = _scored_run(data_dir, _pool(20), api)

    summary = batch.publish_run(run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.15)

    call_list = pd.read_csv(run_dir / "call_list.csv", dtype={"account_id": str}).fillna("")
    assert call_list["account_id"].tolist() == ["1", "2", "3"]
    assert [bool(reason) for reason in call_list["top_reasons"]] == [True, False, True]
    assert summary["n_reasons_missing"] == 1


def test_an_unavailable_explainer_is_not_asked_again(data_dir: Path) -> None:
    probabilities = {str(i): 1.0 - i / 100 for i in range(1, 21)}
    api = FakeApi(probabilities=probabilities, explain_script={"1": 503})
    run_dir = _scored_run(data_dir, _pool(20), api)

    summary = batch.publish_run(run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.15)

    assert api.explain_calls == ["1"]
    assert summary["n_reasons_missing"] == 3
    assert summary["n_call_list"] == 3


def test_an_unreachable_api_still_publishes_the_call_list(data_dir: Path) -> None:
    api = FakeApi(explain_script={str(i): requests.ConnectionError("down") for i in range(1, 6)})
    run_dir = _scored_run(data_dir, _pool(5), api)

    summary = batch.publish_run(run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.4)

    assert len(api.explain_calls) == 1
    assert summary["n_reasons_missing"] == summary["n_call_list"] == 2
    assert (data_dir / "scored" / "latest.json").exists()


def test_reasons_from_another_model_version_are_not_used(data_dir: Path) -> None:
    api = FakeApi(explain_version="3")
    run_dir = _scored_run(data_dir, _pool(10), api)

    summary = batch.publish_run(run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.2)

    assert summary["n_reasons_missing"] == 2


def test_max_reasons_bounds_the_explain_calls(data_dir: Path) -> None:
    api = FakeApi()
    run_dir = _scored_run(data_dir, _pool(30), api)

    summary = batch.publish_run(
        run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.2, max_reasons=2
    )

    assert len(api.explain_calls) == 2
    assert summary["n_call_list"] == 6
    assert summary["max_reasons"] == 2
    assert summary["n_reasons_missing"] == 4


def test_a_failed_publish_leaves_the_previous_latest_untouched(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeApi()
    run_dir = _scored_run(data_dir, _pool(5), api)
    latest = data_dir / "scored" / "latest.json"
    latest.write_text('{"run_dir": "scored/previous"}')

    def crash(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(batch.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        batch.publish_run(run_dir, _client(api), data_dir=data_dir, capacity_fraction=0.1)

    assert json.loads(latest.read_text()) == {"run_dir": "scored/previous"}
    assert not [p for p in latest.parent.iterdir() if p.name.endswith(".tmp")]


def test_publish_before_score_is_bad_input(data_dir: Path) -> None:
    run_dir = _validated_run(data_dir, _pool(3))

    with pytest.raises(batch.BadInputError, match="scored.parquet"):
        batch.publish_run(run_dir, _client(FakeApi()), data_dir=data_dir)


# --------------------------------------------------------------------- CLI


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeApi], FakeApi]:
    """Route the CLI's own requests.Session to a fake and make its backoff instant."""

    def install(api: FakeApi) -> FakeApi:
        monkeypatch.setattr(batch.requests, "Session", lambda: api)
        monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
        return api

    return install


def test_run_scores_a_file_end_to_end_and_prints_the_summary(
    data_dir: Path,
    fake_session: Callable[[FakeApi], FakeApi],
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = fake_session(FakeApi())
    source = _write(_pool(20), data_dir / "processed" / "serving_pool.parquet")

    code = batch.main(
        ["run", "--input", str(source), "--data-dir", str(data_dir), "--api-url", API]
    )

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_scored"] == 20
    assert summary["n_call_list"] == 2
    assert json.loads((data_dir / "scored" / "latest.json").read_text()) == summary
    assert len(api.explain_calls) == 2


def test_the_subcommands_share_one_run_dir(
    data_dir: Path,
    fake_session: Callable[[FakeApi], FakeApi],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_session(FakeApi())
    source = _write(_pool(10), data_dir / "incoming" / "pool.parquet")
    common = ["--data-dir", str(data_dir), "--api-url", API]

    assert batch.main(["validate", "--input", str(source), *common]) == 0
    run_dir = capsys.readouterr().out.strip()
    assert batch.main(["score", "--run-dir", run_dir, *common]) == 0
    assert capsys.readouterr().out.strip() == run_dir
    assert batch.main(["publish", "--run-dir", run_dir, "--dag-run-id", "manual__1", *common]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["run_dir"] == f"scored/{Path(run_dir).name}"
    assert summary["dag_run_id"] == "manual__1"


@pytest.mark.parametrize(
    ("frame_change", "name"),
    [
        (lambda f: f.assign(AGE=5), "pool.parquet"),  # nothing valid
        (lambda f: f.drop(columns=["PAY_1"]), "pool.parquet"),  # missing column
        (lambda f: f, "pool.json"),  # unsupported format
    ],
)
def test_bad_input_exits_2(
    data_dir: Path,
    frame_change: Callable[[pd.DataFrame], pd.DataFrame],
    name: str,
) -> None:
    source = data_dir / "incoming" / name
    if source.suffix == ".json":
        source.parent.mkdir(parents=True)
        source.write_text("{}")
    else:
        _write(frame_change(_pool(3)), source)

    code = batch.main(["validate", "--input", str(source), "--data-dir", str(data_dir)])

    assert code == batch.EXIT_BAD_INPUT == 2


def test_a_missing_input_file_exits_2(data_dir: Path) -> None:
    code = batch.main(
        ["validate", "--input", str(data_dir / "nope.csv"), "--data-dir", str(data_dir)]
    )

    assert code == 2


def test_an_unavailable_api_exits_75_for_the_dag_to_retry(
    data_dir: Path, fake_session: Callable[[FakeApi], FakeApi]
) -> None:
    fake_session(FakeApi(batch_script=[requests.ConnectionError("refused")] * 10))
    run_dir = _validated_run(data_dir, _pool(3))

    code = batch.main(["score", "--run-dir", str(run_dir), "--api-url", API])

    assert code == batch.EXIT_TRANSIENT == 75


def test_a_model_change_mid_run_exits_1_without_a_retry(
    data_dir: Path, fake_session: Callable[[FakeApi], FakeApi]
) -> None:
    fake_session(FakeApi(versions=["2", "3"]))
    run_dir = _validated_run(data_dir, _pool(1001))

    code = batch.main(["score", "--run-dir", str(run_dir), "--api-url", API])

    assert code == 1


def test_an_invalid_capacity_on_the_command_line_exits_2(
    data_dir: Path, fake_session: Callable[[FakeApi], FakeApi]
) -> None:
    api = fake_session(FakeApi())
    run_dir = _scored_run(data_dir, _pool(3), api)

    code = batch.main(
        [
            "publish",
            "--run-dir",
            str(run_dir),
            "--capacity-fraction",
            "0",
            "--data-dir",
            str(data_dir),
            "--api-url",
            API,
        ]
    )

    assert code == 2


def test_the_api_url_defaults_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_API_URL", "http://credit-api:8000")
    assert Settings(_env_file=None).credit_api_url == "http://credit-api:8000"

    monkeypatch.delenv("CREDIT_API_URL")
    assert Settings(_env_file=None).credit_api_url == "http://localhost:18000"
