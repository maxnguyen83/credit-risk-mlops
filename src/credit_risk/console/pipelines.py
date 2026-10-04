"""The two pipelines the console runs, and how their state is read from Airflow.

Only the DAGs named here can be started from the console. The allowlist is the
whole of the access control on triggering: the console's Airflow account can
start any DAG, so a path parameter must never reach Airflow unchecked.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from credit_risk.console.airflow import AirflowApi, AirflowUnavailable, DagNotFound

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSpec:
    """A DAG the console may start, and the task order to show when Airflow cannot say."""

    role: str
    dag_id: str
    title: str
    tasks: tuple[str, ...]


TRAINING: Final = PipelineSpec(
    role="training",
    dag_id="credit_risk_pipeline",
    title="Pipeline huấn luyện model",
    tasks=(
        "download_raw",
        "validate_raw",
        "clean_and_split",
        "build_features",
        "train_candidates",
        "evaluate_and_gate",
        "register_model",
        "publish_report",
    ),
)

SCORING: Final = PipelineSpec(
    role="scoring",
    dag_id="credit_risk_scoring",
    title="Pipeline chấm điểm khách hàng",
    tasks=("validate_batch", "score_batch", "publish_call_list"),
)

PIPELINES: Final[tuple[PipelineSpec, ...]] = (TRAINING, SCORING)
BY_DAG_ID: Final[dict[str, PipelineSpec]] = {spec.dag_id: spec for spec in PIPELINES}

# What a member of staff reads in each box; the task id is shown underneath.
TASK_LABELS: Final[dict[str, str]] = {
    "download_raw": "Tải dữ liệu",
    "validate_raw": "Kiểm tra dữ liệu",
    "clean_and_split": "Làm sạch & chia tập",
    "build_features": "Tạo đặc trưng",
    "train_candidates": "Huấn luyện model",
    "evaluate_and_gate": "Đánh giá & cổng kiểm định",
    "register_model": "Đăng ký model",
    "publish_report": "Xuất báo cáo",
    "validate_batch": "Kiểm tra file",
    "score_batch": "Chấm điểm",
    "publish_call_list": "Xuất danh sách gọi",
}

ACTIVE_STATES: Final = frozenset({"queued", "running"})

# Enough to see a running run behind a queued one (training has
# max_active_runs=1, so unpausing it can put the scheduler's run first).
RECENT_RUNS: Final = 5

DAG_NOT_LOADED: Final = (
    "Airflow chưa nạp DAG này (file DAG chưa có hoặc đang lỗi). Xem trang Airflow để biết lý do."
)


class UnknownPipeline(Exception):
    """A DAG id outside the allowlist."""


class PipelineBusy(Exception):
    """A run of this DAG is already queued or running."""

    def __init__(self, dag_id: str, run_id: str) -> None:
        super().__init__(f"{dag_id} is already running ({run_id})")
        self.dag_id = dag_id
        self.run_id = run_id


def topological_order(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    """Task ids with every task after all of its upstreams.

    Ties keep the order Airflow listed them in, so the result is stable from
    one poll to the next. Downstream ids that name no listed task are ignored,
    and a cycle -- which Airflow refuses to load -- appends whatever is left
    rather than losing it.
    """
    ids = [str(task["task_id"]) for task in tasks]
    known = set(ids)
    downstream = {
        str(task["task_id"]): [d for d in task.get("downstream_task_ids", []) if d in known]
        for task in tasks
    }
    indegree = dict.fromkeys(ids, 0)
    for children in downstream.values():
        for child in children:
            indegree[child] += 1

    ordered: list[str] = []
    ready = [task_id for task_id in ids if indegree[task_id] == 0]
    while ready:
        current = min(ready, key=ids.index)
        ready.remove(current)
        ordered.append(current)
        for child in downstream[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return ordered + [task_id for task_id in ids if task_id not in ordered]


def _run_view(run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    conf = run.get("conf")
    return {
        "run_id": run.get("dag_run_id"),
        "state": run.get("state"),
        "run_type": run.get("run_type"),
        "logical_date": run.get("logical_date") or run.get("execution_date"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "input": conf.get("input") if isinstance(conf, Mapping) else None,
    }


def _shown_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The run the diagram follows: the one executing, else the newest."""
    for run in runs:
        if run.get("state") == "running":
            return run
    return runs[0] if runs else None


def _task_views(
    order: Sequence[str], instances: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    views = []
    for task_id in order:
        instance = instances.get(task_id, {})
        views.append(
            {
                "task_id": task_id,
                "label": TASK_LABELS.get(task_id, task_id),
                "state": instance.get("state") or None,
                "start_date": instance.get("start_date"),
                "end_date": instance.get("end_date"),
                "duration": instance.get("duration"),
            }
        )
    return views


def _placeholder(spec: PipelineSpec, *, found: bool, reason: str | None) -> dict[str, Any]:
    return {
        "role": spec.role,
        "dag_id": spec.dag_id,
        "title": spec.title,
        "found": found,
        "reason": reason,
        "is_paused": None,
        "active": False,
        "n_active_runs": 0,
        "latest_run": None,
        "tasks": _task_views(spec.tasks, {}),
    }


def status(client: AirflowApi, spec: PipelineSpec) -> dict[str, Any]:
    """One pipeline's diagram: task order, the shown run, and each task's state in it."""
    dag = client.get_dag(spec.dag_id)
    order = topological_order(client.list_tasks(spec.dag_id)) or list(spec.tasks)
    runs = client.recent_runs(spec.dag_id, RECENT_RUNS)
    shown = _shown_run(runs)

    instances: dict[str, Mapping[str, Any]] = {}
    if shown is not None and shown.get("dag_run_id"):
        for instance in client.task_instances(spec.dag_id, str(shown["dag_run_id"])):
            instances.setdefault(str(instance.get("task_id")), instance)

    n_active = sum(1 for run in runs if run.get("state") in ACTIVE_STATES)
    view = _placeholder(spec, found=True, reason=None)
    view.update(
        is_paused=bool(dag.get("is_paused")),
        active=n_active > 0,
        n_active_runs=n_active,
        latest_run=_run_view(shown),
        tasks=_task_views(order, instances),
    )
    return view


def collect(client: AirflowApi) -> dict[str, Any]:
    """Both pipelines for the page; never raises, so the page always has something to draw."""
    try:
        views = []
        for spec in PIPELINES:
            try:
                views.append(status(client, spec))
            except DagNotFound:
                views.append(_placeholder(spec, found=False, reason=DAG_NOT_LOADED))
    except AirflowUnavailable as exc:
        return {
            "available": False,
            "reason": exc.message,
            "pipelines": [_placeholder(spec, found=False, reason=None) for spec in PIPELINES],
        }
    return {"available": True, "reason": None, "pipelines": views}


def start(client: AirflowApi, dag_id: str, conf: dict[str, Any]) -> dict[str, Any]:
    """Start one run of an allowlisted DAG, unpausing it first if it is paused.

    Refuses while a run is queued or running. Two scoring runs at once would
    race to write the same ``latest.json``, and a second training run only
    queues behind the first; either way a double click should not cause it.
    """
    if dag_id not in BY_DAG_ID:
        raise UnknownPipeline(dag_id)

    dag = client.get_dag(dag_id)
    for run in client.recent_runs(dag_id, RECENT_RUNS):
        if run.get("state") in ACTIVE_STATES:
            raise PipelineBusy(dag_id, str(run.get("dag_run_id")))

    unpaused = False
    if dag.get("is_paused"):
        client.set_paused(dag_id, False)
        unpaused = True

    created = client.trigger(dag_id, conf)
    log.info(
        "started dag_id=%s run_id=%s unpaused=%s conf=%s",
        dag_id,
        created.get("dag_run_id"),
        unpaused,
        conf,
    )
    return {
        "dag_id": dag_id,
        "run_id": created.get("dag_run_id"),
        "state": created.get("state"),
        "unpaused": unpaused,
    }
