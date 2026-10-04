"""A thin client for the parts of Airflow's stable REST API (v1) the console uses.

Every failure -- refused connection, timeout, bad credentials, a 5xx, a body
that is not JSON -- surfaces as :class:`AirflowUnavailable` with a message a
member of bank staff can read. The page shows that message instead of a stack
trace, and keeps rendering everything that does not need Airflow.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol
from urllib.parse import quote

import requests

from credit_risk.console.settings import ConsoleSettings

log = logging.getLogger(__name__)


class AirflowError(Exception):
    """Base for everything the console can be told by, or about, Airflow."""


class AirflowUnavailable(AirflowError):
    """Airflow could not be asked, or did not give a usable answer."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class DagNotFound(AirflowError):
    """Airflow is up but has not loaded this DAG (yet)."""

    def __init__(self, dag_id: str) -> None:
        super().__init__(dag_id)
        self.dag_id = dag_id


class AirflowApi(Protocol):
    """What the console needs from Airflow; the real client and the test fake both provide it."""

    def get_dag(self, dag_id: str) -> dict[str, Any]: ...

    def list_tasks(self, dag_id: str) -> list[dict[str, Any]]: ...

    def recent_runs(self, dag_id: str, limit: int) -> list[dict[str, Any]]: ...

    def task_instances(self, dag_id: str, run_id: str) -> list[dict[str, Any]]: ...

    def set_paused(self, dag_id: str, paused: bool) -> None: ...

    def trigger(self, dag_id: str, conf: dict[str, Any]) -> dict[str, Any]: ...


def _segment(value: str) -> str:
    # Run ids look like manual__2026-10-04T10:15:30+00:00; the ':' and '+' must
    # be escaped or the path no longer names the run.
    return quote(value, safe="")


class AirflowClient:
    """Basic-auth client for ``<airflow_url>/api/v1``."""

    def __init__(
        self,
        base_url: str,
        username: str = "",
        password: str = "",
        *,
        timeout: float = 8.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password) if username else None
        self.timeout = timeout
        self.session = session or requests.Session()

    @classmethod
    def from_settings(cls, settings: ConsoleSettings) -> AirflowClient:
        """Build the client the running console uses."""
        return cls(
            settings.airflow_url,
            settings.airflow_username,
            settings.airflow_password.get_secret_value(),
            timeout=settings.airflow_timeout_seconds,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}/api/v1{path}"
        try:
            response = self.session.request(
                method, url, params=params, json=json, auth=self.auth, timeout=self.timeout
            )
        except requests.RequestException as exc:
            log.warning("airflow %s %s failed: %s", method, path, type(exc).__name__)
            raise AirflowUnavailable(
                "Không kết nối được Airflow. Kiểm tra dịch vụ Airflow đã chạy chưa.",
                type(exc).__name__,
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise AirflowUnavailable(
                "Airflow từ chối đăng nhập của bảng điều khiển "
                "(kiểm tra AIRFLOW_USERNAME / AIRFLOW_PASSWORD và basic auth của API).",
                f"HTTP {status}",
            )
        if status == 404:
            raise DagNotFound(path)
        if status >= 400:
            log.warning("airflow %s %s answered %d", method, path, status)
            raise AirflowUnavailable(f"Airflow trả lỗi HTTP {status}.", f"HTTP {status}")
        try:
            return response.json()
        except ValueError as exc:
            raise AirflowUnavailable(
                "Airflow trả về dữ liệu không đọc được.", "response is not JSON"
            ) from exc

    def get_dag(self, dag_id: str) -> dict[str, Any]:
        """The DAG's metadata; ``is_paused`` is the field the console acts on."""
        body: dict[str, Any] = self._request("GET", f"/dags/{_segment(dag_id)}")
        return body

    def list_tasks(self, dag_id: str) -> list[dict[str, Any]]:
        """Every task with its ``downstream_task_ids``, which give the diagram its order."""
        body = self._request("GET", f"/dags/{_segment(dag_id)}/tasks")
        return list(body.get("tasks", []))

    def recent_runs(self, dag_id: str, limit: int) -> list[dict[str, Any]]:
        """The newest runs first. Airflow's default order is oldest first, by id."""
        body = self._request(
            "GET",
            f"/dags/{_segment(dag_id)}/dagRuns",
            params={"order_by": "-execution_date", "limit": limit},
        )
        return list(body.get("dag_runs", []))

    def task_instances(self, dag_id: str, run_id: str) -> list[dict[str, Any]]:
        """State and timing of each task in one run."""
        body = self._request(
            "GET", f"/dags/{_segment(dag_id)}/dagRuns/{_segment(run_id)}/taskInstances"
        )
        return list(body.get("task_instances", []))

    def set_paused(self, dag_id: str, paused: bool) -> None:
        """Pause or unpause. A run triggered on a paused DAG is created but never scheduled."""
        self._request(
            "PATCH",
            f"/dags/{_segment(dag_id)}",
            params={"update_mask": "is_paused"},
            json={"is_paused": paused},
        )

    def trigger(self, dag_id: str, conf: dict[str, Any]) -> dict[str, Any]:
        """Create a DAG run; Airflow chooses the run id."""
        body: dict[str, Any] = self._request(
            "POST", f"/dags/{_segment(dag_id)}/dagRuns", json={"conf": conf}
        )
        return body
