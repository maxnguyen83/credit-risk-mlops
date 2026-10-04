"""The console's view of Airflow: what it asks for, and what it makes of the answers.

Airflow is replaced by :class:`FakeAirflow`, an in-memory stand-in that keeps a
log of every call. It is a fake rather than a mock on purpose: the behaviours
under test -- "unpause before triggering", "refuse a second run" -- are about
the order and the state of calls, and a fake that holds state can fail those
tests where a mock that returns canned values cannot.

The HTTP client itself is tested against a stub session, so the REST paths,
the auth and the error mapping are pinned without a network.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import pytest
import requests

from credit_risk.console import pipelines
from credit_risk.console.airflow import AirflowClient, AirflowUnavailable, DagNotFound
from credit_risk.console.settings import ConsoleSettings

TRAINING = pipelines.TRAINING.dag_id
SCORING = pipelines.SCORING.dag_id


# ----------------------------------------------------------------- the fake


def chain_tasks(task_ids: list[str]) -> list[dict[str, Any]]:
    """Airflow's /tasks payload for a straight chain, in the order given."""
    return [
        {
            "task_id": task_id,
            "downstream_task_ids": [task_ids[index + 1]] if index + 1 < len(task_ids) else [],
        }
        for index, task_id in enumerate(task_ids)
    ]


@dataclass
class FakeDag:
    paused: bool = False
    tasks: list[dict[str, Any]] = field(default_factory=list)
    # Newest first, the order the console asks Airflow for.
    runs: list[dict[str, Any]] = field(default_factory=list)
    task_instances: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class FakeAirflow:
    """Enough of Airflow's stable REST API to drive the console, in memory."""

    def __init__(
        self,
        dags: dict[str, FakeDag] | None = None,
        *,
        down: AirflowUnavailable | None = None,
    ) -> None:
        self.dags = dags if dags is not None else default_dags()
        self.down = down
        self.calls: list[tuple[Any, ...]] = []
        self.paused_when_triggered: list[bool] = []

    def _dag(self, call: str, dag_id: str, *extra: Any) -> FakeDag:
        self.calls.append((call, dag_id, *extra))
        if self.down is not None:
            raise self.down
        if dag_id not in self.dags:
            raise DagNotFound(dag_id)
        return self.dags[dag_id]

    def get_dag(self, dag_id: str) -> dict[str, Any]:
        dag = self._dag("get_dag", dag_id)
        return {"dag_id": dag_id, "is_paused": dag.paused}

    def list_tasks(self, dag_id: str) -> list[dict[str, Any]]:
        return list(self._dag("list_tasks", dag_id).tasks)

    def recent_runs(self, dag_id: str, limit: int) -> list[dict[str, Any]]:
        return list(self._dag("recent_runs", dag_id).runs[:limit])

    def task_instances(self, dag_id: str, run_id: str) -> list[dict[str, Any]]:
        return list(self._dag("task_instances", dag_id, run_id).task_instances.get(run_id, []))

    def set_paused(self, dag_id: str, paused: bool) -> None:
        self._dag("set_paused", dag_id, paused).paused = paused

    def trigger(self, dag_id: str, conf: dict[str, Any]) -> dict[str, Any]:
        dag = self._dag("trigger", dag_id, conf)
        self.paused_when_triggered.append(dag.paused)
        run = {
            "dag_run_id": f"manual__{len(dag.runs) + 1}",
            "state": "queued",
            "conf": conf,
            "logical_date": "2026-10-04T10:00:00+00:00",
        }
        dag.runs.insert(0, run)
        return run


def default_dags(*, paused: bool = False) -> dict[str, FakeDag]:
    return {
        TRAINING: FakeDag(paused=paused, tasks=chain_tasks(list(pipelines.TRAINING.tasks))),
        SCORING: FakeDag(paused=paused, tasks=chain_tasks(list(pipelines.SCORING.tasks))),
    }


def run(run_id: str, state: str, **extra: Any) -> dict[str, Any]:
    return {
        "dag_run_id": run_id,
        "state": state,
        "logical_date": "2026-10-04T09:00:00+00:00",
        "start_date": "2026-10-04T09:00:01+00:00",
        "end_date": None if state in {"queued", "running"} else "2026-10-04T09:02:01+00:00",
        "conf": {},
        **extra,
    }


# ------------------------------------------------------ topological order


def test_topological_order_recovers_a_chain_whatever_order_airflow_lists_it() -> None:
    expected = list(pipelines.TRAINING.tasks)
    tasks = chain_tasks(expected)
    random.Random(7).shuffle(tasks)
    assert pipelines.topological_order(tasks) == expected


def test_topological_order_respects_a_skip_edge() -> None:
    # The real training DAG: download_raw feeds both validate_raw and
    # clean_and_split, and clean_and_split also waits for validate_raw.
    tasks = [
        {"task_id": "clean_and_split", "downstream_task_ids": []},
        {"task_id": "download_raw", "downstream_task_ids": ["clean_and_split", "validate_raw"]},
        {"task_id": "validate_raw", "downstream_task_ids": ["clean_and_split"]},
    ]
    assert pipelines.topological_order(tasks) == ["download_raw", "validate_raw", "clean_and_split"]


def test_topological_order_breaks_ties_in_the_order_airflow_listed_them() -> None:
    tasks = [
        {"task_id": "start", "downstream_task_ids": ["right", "left"]},
        {"task_id": "left", "downstream_task_ids": ["end"]},
        {"task_id": "right", "downstream_task_ids": ["end"]},
        {"task_id": "end", "downstream_task_ids": []},
    ]
    assert pipelines.topological_order(tasks) == ["start", "left", "right", "end"]


def test_topological_order_survives_a_cycle_and_an_unknown_downstream() -> None:
    tasks = [
        {"task_id": "a", "downstream_task_ids": ["b", "not_a_task"]},
        {"task_id": "b", "downstream_task_ids": ["a"]},
        {"task_id": "c", "downstream_task_ids": []},
    ]
    # Nothing is dropped and nothing is invented, even though no order exists.
    assert sorted(pipelines.topological_order(tasks)) == ["a", "b", "c"]


# ------------------------------------------------------------ status view


def test_status_reports_each_task_state_in_dag_order() -> None:
    fake = FakeAirflow()
    tasks = chain_tasks(list(pipelines.SCORING.tasks))
    tasks.reverse()
    fake.dags[SCORING].tasks = tasks
    fake.dags[SCORING].runs = [run("manual__2", "running", conf={"input": "incoming/x.csv"})]
    fake.dags[SCORING].task_instances["manual__2"] = [
        {"task_id": "score_batch", "state": "running", "duration": None},
        {"task_id": "validate_batch", "state": "success", "duration": 1.5},
    ]

    view = pipelines.collect(fake)
    scoring = next(p for p in view["pipelines"] if p["role"] == "scoring")

    assert view["available"] is True
    assert [t["task_id"] for t in scoring["tasks"]] == list(pipelines.SCORING.tasks)
    assert [t["state"] for t in scoring["tasks"]] == ["success", "running", None]
    assert scoring["tasks"][0]["duration"] == 1.5
    assert scoring["tasks"][0]["label"] == pipelines.TASK_LABELS["validate_batch"]
    assert scoring["latest_run"]["run_id"] == "manual__2"
    assert scoring["latest_run"]["input"] == "incoming/x.csv"
    assert scoring["active"] is True


def test_status_shows_the_running_run_rather_than_a_newer_queued_one() -> None:
    # Unpausing the @daily training DAG lets the scheduler start its own run;
    # the manual run queues behind it (max_active_runs=1). The diagram should
    # follow the run that is actually executing.
    fake = FakeAirflow()
    fake.dags[TRAINING].runs = [run("manual__9", "queued"), run("scheduled__8", "running")]

    training = pipelines.collect(fake)["pipelines"][0]

    assert training["latest_run"]["run_id"] == "scheduled__8"
    assert training["active"] is True
    assert training["n_active_runs"] == 2


def test_status_of_a_finished_run_is_not_active() -> None:
    fake = FakeAirflow()
    fake.dags[TRAINING].runs = [run("manual__1", "success")]
    training = pipelines.collect(fake)["pipelines"][0]
    assert training["active"] is False
    assert training["latest_run"]["state"] == "success"


def test_status_without_any_run_shows_the_tasks_unstarted() -> None:
    view = pipelines.collect(FakeAirflow(default_dags(paused=True)))
    training = view["pipelines"][0]
    assert training["latest_run"] is None
    assert training["is_paused"] is True
    assert [t["state"] for t in training["tasks"]] == [None] * len(pipelines.TRAINING.tasks)


def test_airflow_down_is_reported_not_raised() -> None:
    fake = FakeAirflow(down=AirflowUnavailable("Không kết nối được Airflow.", "ConnectionError"))
    view = pipelines.collect(fake)

    assert view["available"] is False
    assert "Airflow" in view["reason"]
    # Both diagrams still render, from the contract's task list.
    assert [p["dag_id"] for p in view["pipelines"]] == [TRAINING, SCORING]
    assert [t["task_id"] for t in view["pipelines"][1]["tasks"]] == list(pipelines.SCORING.tasks)
    assert all(p["latest_run"] is None for p in view["pipelines"])


def test_a_dag_airflow_has_not_loaded_is_flagged_but_the_other_still_reports() -> None:
    fake = FakeAirflow()
    del fake.dags[SCORING]
    fake.dags[TRAINING].runs = [run("manual__1", "success")]

    view = pipelines.collect(fake)
    training, scoring = view["pipelines"]

    assert view["available"] is True
    assert training["found"] is True
    assert training["latest_run"]["run_id"] == "manual__1"
    assert scoring["found"] is False
    assert scoring["reason"]
    assert [t["task_id"] for t in scoring["tasks"]] == list(pipelines.SCORING.tasks)


# ------------------------------------------------------------- triggering


def test_start_unpauses_a_paused_dag_before_triggering_it() -> None:
    fake = FakeAirflow(default_dags(paused=True))

    started = pipelines.start(fake, SCORING, {"input": "incoming/a.csv"})

    assert fake.paused_when_triggered == [False]
    names = [call[0] for call in fake.calls]
    assert names.index("set_paused") < names.index("trigger")
    assert ("set_paused", SCORING, False) in fake.calls
    assert ("trigger", SCORING, {"input": "incoming/a.csv"}) in fake.calls
    assert started["unpaused"] is True
    assert started["run_id"] == "manual__1"


def test_start_leaves_an_unpaused_dag_alone() -> None:
    fake = FakeAirflow()
    started = pipelines.start(fake, TRAINING, {})
    assert "set_paused" not in [call[0] for call in fake.calls]
    assert started["unpaused"] is False
    assert started["dag_id"] == TRAINING


def test_start_refuses_while_a_run_is_in_flight() -> None:
    fake = FakeAirflow()
    fake.dags[SCORING].runs = [run("manual__1", "running")]

    with pytest.raises(pipelines.PipelineBusy) as caught:
        pipelines.start(fake, SCORING, {"input": "processed/serving_pool.parquet"})

    assert caught.value.run_id == "manual__1"
    assert "trigger" not in [call[0] for call in fake.calls]


def test_start_rejects_a_dag_outside_the_allowlist_without_calling_airflow() -> None:
    fake = FakeAirflow()
    with pytest.raises(pipelines.UnknownPipeline):
        pipelines.start(fake, "some_other_dag", {})
    assert fake.calls == []


# ------------------------------------------------------------ HTTP client


class StubResponse:
    def __init__(self, status_code: int, body: Any = None, *, raw: str | None = None) -> None:
        self.status_code = status_code
        self._text = raw if raw is not None else json.dumps(body)

    def json(self) -> Any:
        return json.loads(self._text)


class StubSession:
    """Records each request and replays the queued answers in order."""

    def __init__(self, *answers: StubResponse | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> StubResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def client_with(*answers: StubResponse | Exception, username: str = "console") -> tuple:
    session = StubSession(*answers)
    client = AirflowClient("http://airflow:8080/", username, "s3cret", timeout=4.0, session=session)
    return client, session


def test_client_reads_a_dag_with_basic_auth_under_api_v1() -> None:
    client, session = client_with(StubResponse(200, {"dag_id": TRAINING, "is_paused": True}))
    assert client.get_dag(TRAINING)["is_paused"] is True
    sent = session.requests[0]
    assert sent["method"] == "GET"
    assert sent["url"] == f"http://airflow:8080/api/v1/dags/{TRAINING}"
    assert sent["auth"] == ("console", "s3cret")
    assert sent["timeout"] == 4.0


def test_client_sends_no_auth_when_no_username_is_configured() -> None:
    client, session = client_with(StubResponse(200, {"tasks": []}), username="")
    client.list_tasks(TRAINING)
    assert session.requests[0]["auth"] is None


def test_client_asks_for_the_newest_runs_first() -> None:
    client, session = client_with(StubResponse(200, {"dag_runs": [run("r1", "success")]}))
    runs = client.recent_runs(SCORING, limit=5)
    assert [r["dag_run_id"] for r in runs] == ["r1"]
    assert session.requests[0]["url"].endswith(f"/api/v1/dags/{SCORING}/dagRuns")
    assert session.requests[0]["params"] == {"order_by": "-execution_date", "limit": 5}


def test_client_url_encodes_the_run_id() -> None:
    client, session = client_with(StubResponse(200, {"task_instances": [{"task_id": "x"}]}))
    run_id = "manual__2026-10-04T10:15:30.123456+00:00"
    assert client.task_instances(SCORING, run_id) == [{"task_id": "x"}]
    url = session.requests[0]["url"]
    assert url.endswith("/dagRuns/manual__2026-10-04T10%3A15%3A30.123456%2B00%3A00/taskInstances")


def test_client_unpauses_through_patch_with_an_update_mask() -> None:
    client, session = client_with(StubResponse(200, {"dag_id": SCORING, "is_paused": False}))
    client.set_paused(SCORING, False)
    sent = session.requests[0]
    assert sent["method"] == "PATCH"
    assert sent["params"] == {"update_mask": "is_paused"}
    assert sent["json"] == {"is_paused": False}


def test_client_triggers_with_the_conf() -> None:
    client, session = client_with(StubResponse(200, {"dag_run_id": "manual__1", "state": "queued"}))
    created = client.trigger(SCORING, {"input": "incoming/a.csv"})
    assert created["dag_run_id"] == "manual__1"
    assert session.requests[0]["method"] == "POST"
    assert session.requests[0]["json"] == {"conf": {"input": "incoming/a.csv"}}


@pytest.mark.parametrize(
    ("answer", "fragment"),
    [
        (requests.ConnectionError("refused"), "kết nối"),
        (requests.Timeout("slow"), "kết nối"),
        (StubResponse(401, {"title": "Unauthorized"}), "đăng nhập"),
        (StubResponse(403, {"title": "Forbidden"}), "đăng nhập"),
        (StubResponse(500, {"title": "boom"}), "500"),
        (StubResponse(200, raw="<html>not json</html>"), "không đọc được"),
    ],
)
def test_client_turns_every_failure_into_airflow_unavailable(
    answer: StubResponse | Exception, fragment: str
) -> None:
    client, _ = client_with(answer)
    with pytest.raises(AirflowUnavailable) as caught:
        client.get_dag(TRAINING)
    assert fragment in caught.value.message


def test_client_maps_404_to_dag_not_found() -> None:
    client, _ = client_with(StubResponse(404, {"title": "DAG not found"}))
    with pytest.raises(DagNotFound):
        client.get_dag(SCORING)


def test_client_is_built_from_settings() -> None:
    settings = ConsoleSettings(
        airflow_url="http://example:1234",
        airflow_username="ops",
        airflow_password="pw",
        _env_file=None,
    )
    client = AirflowClient.from_settings(settings)
    assert client.base_url == "http://example:1234"
    assert client.auth == ("ops", "pw")
