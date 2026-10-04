"""The console over HTTP: every endpoint the page calls, against a fake Airflow.

Each client gets its own app built by ``create_app`` with a temporary data
directory and an in-memory Airflow, so nothing here touches the live stack or
the repository's data folder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk.console import batches, pipelines
from credit_risk.console.airflow import AirflowUnavailable
from credit_risk.console.app import create_app
from credit_risk.console.settings import ConsoleSettings
from tests.unit.test_console_files import SUMMARY, csv_text, write_results
from tests.unit.test_console_pipelines import FakeAirflow, default_dags, run

TRAINING = pipelines.TRAINING.dag_id
SCORING = pipelines.SCORING.dag_id

MAX_UPLOAD = 4096


def make_settings(data_dir: Path) -> ConsoleSettings:
    return ConsoleSettings(
        airflow_url="http://airflow.invalid:8080",
        console_data_dir=data_dir,
        console_link_host="demo.local",
        console_repo_url="https://example.org/repo",
        console_max_upload_bytes=MAX_UPLOAD,
        _env_file=None,
    )


@pytest.fixture
def airflow() -> FakeAirflow:
    return FakeAirflow(default_dags(paused=True))


@pytest.fixture
def client(tmp_path: Path, airflow: FakeAirflow) -> TestClient:
    return TestClient(create_app(make_settings(tmp_path), airflow))


def upload(client: TestClient, body: bytes, content_type: str = "text/csv"):
    return client.post("/api/batches", content=body, headers={"Content-Type": content_type})


def valid_csv(n_rows: int = 3) -> bytes:
    return csv_text(["ID", *batches.REQUIRED_COLUMNS], n_rows=n_rows).encode()


# -------------------------------------------------------------- the page


def test_the_page_is_served_with_its_assets(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Cảnh báo sớm vỡ nợ — Bảng điều khiển" in page.text

    assets = re.findall(r'(?:src|href)="(/static/[^"]+)"', page.text)
    assert {"/static/console.css", "/static/console.js"} <= set(assets)
    for asset in assets:
        assert client.get(asset).status_code == 200, asset


def test_the_page_loads_nothing_from_another_origin(client: TestClient) -> None:
    page = client.get("/").text
    assert not re.search(r'(?:src|href)="(?:https?:)?//', page)
    script = client.get("/static/console.js").text
    assert "innerHTML" not in script  # every value from a file is set as text, never markup


def test_health_does_not_depend_on_airflow(tmp_path: Path) -> None:
    down = FakeAirflow(down=AirflowUnavailable("Không kết nối được Airflow.", "ConnectionError"))
    client = TestClient(create_app(make_settings(tmp_path), down))
    assert client.get("/health").json() == {"status": "ok"}
    assert down.calls == []


# ------------------------------------------------------------- pipelines


def test_pipelines_report_both_dags(client: TestClient, airflow: FakeAirflow) -> None:
    airflow.dags[SCORING].runs = [run("manual__4", "success", conf={"input": "incoming/a.csv"})]
    body = client.get("/api/pipelines").json()

    assert body["available"] is True
    assert [p["dag_id"] for p in body["pipelines"]] == [TRAINING, SCORING]
    scoring = body["pipelines"][1]
    assert scoring["latest_run"]["input"] == "incoming/a.csv"
    assert scoring["airflow_link"] == f"http://demo.local:18081/dags/{SCORING}/grid"
    assert len(body["pipelines"][0]["tasks"]) == 8


def test_pipelines_answer_200_when_airflow_is_down(tmp_path: Path) -> None:
    down = FakeAirflow(down=AirflowUnavailable("Không kết nối được Airflow.", "ConnectionError"))
    response = TestClient(create_app(make_settings(tmp_path), down)).get("/api/pipelines")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "Không kết nối được Airflow."
    assert len(body["pipelines"]) == 2


@pytest.mark.parametrize("dag_id", [TRAINING, SCORING])
def test_trigger_unpauses_then_starts_an_allowed_dag(
    client: TestClient, airflow: FakeAirflow, dag_id: str
) -> None:
    response = client.post(f"/api/pipelines/{dag_id}/runs")
    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] == "manual__1"
    assert body["unpaused"] is True
    assert airflow.paused_when_triggered == [False]


@pytest.mark.parametrize(
    "dag_id", ["example_bash_operator", "credit_risk", "..", "credit_risk_pipelinex"]
)
def test_trigger_refuses_any_other_dag(
    client: TestClient, airflow: FakeAirflow, dag_id: str
) -> None:
    response = client.post(f"/api/pipelines/{dag_id}/runs")
    assert response.status_code == 404
    assert airflow.calls == []


def test_trigger_while_running_is_a_conflict(client: TestClient, airflow: FakeAirflow) -> None:
    airflow.dags[TRAINING].runs = [run("scheduled__1", "running")]
    response = client.post(f"/api/pipelines/{TRAINING}/runs")
    assert response.status_code == 409
    assert response.json()["detail"] == "scheduled__1"


def test_trigger_with_airflow_down_is_503(tmp_path: Path) -> None:
    down = FakeAirflow(down=AirflowUnavailable("Không kết nối được Airflow.", "ConnectionError"))
    response = TestClient(create_app(make_settings(tmp_path), down)).post(
        f"/api/pipelines/{TRAINING}/runs"
    )
    assert response.status_code == 503
    assert response.json()["code"] == "airflow_unavailable"


def test_trigger_of_a_dag_airflow_has_not_loaded_is_503(
    client: TestClient, airflow: FakeAirflow
) -> None:
    del airflow.dags[SCORING]
    response = client.post(f"/api/pipelines/{SCORING}/runs")
    assert response.status_code == 503
    assert response.json()["code"] == "dag_not_loaded"


# --------------------------------------------------------------- batches


def test_upload_is_saved_and_the_scoring_dag_is_started_with_it(
    client: TestClient, airflow: FakeAirflow, tmp_path: Path
) -> None:
    response = upload(client, valid_csv(n_rows=3))

    assert response.status_code == 202
    body = response.json()
    assert body["n_rows"] == 3
    assert body["run_id"] == "manual__1"
    assert re.fullmatch(r"incoming/\d{8}T\d{6}Z(-\d+)?\.csv", body["input"])
    saved = tmp_path / body["input"]
    assert saved.read_bytes() == valid_csv(n_rows=3)
    assert ("trigger", SCORING, {"input": body["input"]}) in airflow.calls
    assert airflow.paused_when_triggered == [False]


def test_upload_name_never_comes_from_the_client(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/batches?name=../../evil.csv",
        content=valid_csv(),
        headers={
            "Content-Type": "text/csv",
            "Content-Disposition": 'attachment; filename="../x.csv"',
        },
    )
    assert response.status_code == 202
    saved = list(tmp_path.rglob("*.csv"))
    assert len(saved) == 1
    assert saved[0].parent == tmp_path / "incoming"


def test_upload_missing_columns_is_422_and_nothing_is_saved_or_started(
    client: TestClient, airflow: FakeAirflow, tmp_path: Path
) -> None:
    body = csv_text(["ID", "LIMIT_BAL", "AGE"]).encode()
    response = upload(client, body)
    assert response.status_code == 422
    assert response.json()["code"] == "missing_columns"
    assert "PAY_1" in response.json()["detail"]
    assert not (tmp_path / "incoming").exists()
    assert airflow.calls == []


def test_upload_too_big_is_413(client: TestClient, airflow: FakeAirflow, tmp_path: Path) -> None:
    body = valid_csv() + b"9" * MAX_UPLOAD
    response = upload(client, body)
    assert response.status_code == 413
    assert response.json()["code"] == "too_large"
    assert airflow.calls == []
    assert not (tmp_path / "incoming").exists()


def test_upload_too_big_is_413_even_without_a_content_length(client: TestClient) -> None:
    def chunks():
        yield valid_csv()
        yield b"9" * MAX_UPLOAD

    response = client.post("/api/batches", content=chunks(), headers={"Content-Type": "text/csv"})
    assert response.status_code == 413


def test_upload_empty_is_400(client: TestClient, airflow: FakeAirflow) -> None:
    response = upload(client, b"")
    assert response.status_code == 400
    assert response.json()["code"] == "empty_file"
    assert airflow.calls == []


@pytest.mark.parametrize("content_type", ["application/json", "multipart/form-data; boundary=x"])
def test_upload_with_the_wrong_content_type_is_415(
    client: TestClient, airflow: FakeAirflow, content_type: str
) -> None:
    response = upload(client, valid_csv(), content_type=content_type)
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"
    assert airflow.calls == []


def test_upload_when_airflow_is_down_is_503_and_leaves_no_file(tmp_path: Path) -> None:
    down = FakeAirflow(down=AirflowUnavailable("Không kết nối được Airflow.", "ConnectionError"))
    client = TestClient(create_app(make_settings(tmp_path), down))
    response = upload(client, valid_csv())
    assert response.status_code == 503
    assert list((tmp_path / "incoming").glob("*")) == []


def test_upload_while_scoring_runs_is_409_and_leaves_no_file(
    client: TestClient, airflow: FakeAirflow, tmp_path: Path
) -> None:
    airflow.dags[SCORING].runs = [run("manual__1", "queued")]
    response = upload(client, valid_csv())
    assert response.status_code == 409
    assert list((tmp_path / "incoming").glob("*")) == []


def test_sample_batch_scores_the_serving_pool(client: TestClient, airflow: FakeAirflow) -> None:
    response = client.post("/api/batches/sample")
    assert response.status_code == 202
    assert response.json()["input"] == "processed/serving_pool.parquet"
    assert ("trigger", SCORING, {"input": "processed/serving_pool.parquet"}) in airflow.calls


def test_upload_requirements_are_published(client: TestClient) -> None:
    body = client.get("/api/batches/requirements").json()
    assert body["required_columns"] == list(batches.REQUIRED_COLUMNS)
    assert body["max_bytes"] == MAX_UPLOAD


# --------------------------------------------------------------- results


def test_latest_results_before_any_scoring_is_an_empty_404(client: TestClient) -> None:
    response = client.get("/api/results/latest")
    assert response.status_code == 404
    body = response.json()
    assert body["available"] is False
    assert body["reason"]


def test_latest_results_returns_summary_and_rows(client: TestClient, tmp_path: Path) -> None:
    write_results(tmp_path)
    response = client.get("/api/results/latest", params={"limit": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["summary"]["n_call_list"] == SUMMARY["n_call_list"]
    assert [row["rank"] for row in body["rows"]] == [1, 2, 3]


def test_latest_results_default_to_fifty_rows(client: TestClient, tmp_path: Path) -> None:
    rows = "".join(f"{i},A-{i},0.5,high,intervene,x\n" for i in range(1, 81))
    write_results(
        tmp_path,
        call_list="rank,account_id,default_probability,risk_band,decision,top_reasons\n" + rows,
    )
    body = client.get("/api/results/latest").json()
    assert len(body["rows"]) == 50
    assert body["n_rows_total"] == 80


@pytest.mark.parametrize("name", ["call_list.csv", "scores.csv"])
def test_result_files_download_as_attachments(
    client: TestClient, tmp_path: Path, name: str
) -> None:
    run_dir = write_results(tmp_path)
    response = client.get(f"/api/results/latest/{name}")
    assert response.status_code == 200
    assert response.content == (run_dir / name).read_bytes()
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]


def test_downloads_cannot_leave_the_results_folder(client: TestClient, tmp_path: Path) -> None:
    write_results(tmp_path)
    (tmp_path / "secret.csv").write_text("secret\n", encoding="utf-8")
    for path in [
        "/api/results/latest/latest.json",
        "/api/results/latest/..%2F..%2Fsecret.csv",
        "/api/results/latest/%2E%2E%2Fsecret.csv",
    ]:
        response = client.get(path)
        assert response.status_code == 404, path
        assert b"secret" not in response.content


def test_downloads_follow_run_dir_only_inside_the_results_folder(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "scored").mkdir()
    (tmp_path / "call_list.csv").write_text("secret\n", encoding="utf-8")
    summary = {**SUMMARY, "run_dir": ".."}
    (tmp_path / "scored" / "latest.json").write_text(json.dumps(summary), encoding="utf-8")
    response = client.get("/api/results/latest/call_list.csv")
    assert response.status_code == 404
    assert b"secret" not in response.content


# ----------------------------------------------------------------- links


def test_links_point_at_the_published_host_ports(client: TestClient) -> None:
    links = {link["id"]: link for link in client.get("/api/links").json()["links"]}
    assert links["swagger"]["url"] == "http://demo.local:18000/docs"
    assert links["mlflow"]["url"] == "http://demo.local:15020"
    assert links["airflow"]["url"] == "http://demo.local:18081"
    assert links["prometheus"]["url"] == "http://demo.local:19090"
    assert links["alertmanager"]["url"] == "http://demo.local:19093"
    assert links["objectstore"]["url"] == "http://demo.local:19011"
    assert links["github"]["url"] == "https://example.org/repo"
    grafana = sorted(link["url"] for key, link in links.items() if key.startswith("grafana"))
    assert grafana == [
        "http://demo.local:13000/d/credit-fairness-monitor",
        "http://demo.local:13000/d/credit-model-behaviour",
        "http://demo.local:13000/d/credit-service-health",
    ]
    assert all(link["title"] and link["description"] for link in links.values())


def test_settings_come_from_the_documented_environment_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AIRFLOW_URL", "http://af:9999")
    monkeypatch.setenv("AIRFLOW_USERNAME", "console")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "pw")
    monkeypatch.setenv("CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_LINK_HOST", "10.0.0.5")
    monkeypatch.setenv("CONSOLE_REPO_URL", "https://example.org/r")
    settings = ConsoleSettings(_env_file=None)
    assert settings.airflow_url == "http://af:9999"
    assert settings.airflow_username == "console"
    assert settings.airflow_password.get_secret_value() == "pw"
    assert settings.console_data_dir == tmp_path
    assert settings.console_link_host == "10.0.0.5"
    assert settings.console_repo_url == "https://example.org/r"
    # The password must never be printed by a stray log of the settings object.
    assert "pw" not in repr(settings)


def test_the_static_assets_are_declared_as_package_data() -> None:
    # The images install the package non-editable; a file that is not matched
    # here is missing from the wheel and 404s in the container, while every
    # test in a source checkout still passes.
    import fnmatch
    import tomllib

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["credit_risk.console"]
    static = root / "src" / "credit_risk" / "console" / "static"
    shipped = [p.relative_to(static.parent).as_posix() for p in static.rglob("*") if p.is_file()]
    assert shipped
    for path in shipped:
        assert any(fnmatch.fnmatch(path, pattern) for pattern in patterns), path


def test_upload_that_cannot_be_written_is_500_and_starts_nothing(
    tmp_path: Path, airflow: FakeAirflow
) -> None:
    # The likeliest failure in the container: a data mount the console cannot
    # write to. Simulated here by a file where the incoming folder should be.
    (tmp_path / "incoming").write_text("not a folder", encoding="utf-8")
    client = TestClient(create_app(make_settings(tmp_path), airflow))
    response = upload(client, valid_csv())
    assert response.status_code == 500
    assert response.json()["code"] == "cannot_save"
    assert airflow.calls == []
