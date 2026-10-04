"""Where the other services are, as the browser sees them.

The ports are the host ports docker-compose publishes, not the ports inside
the compose network: these links are followed by someone's browser.
"""

from __future__ import annotations

from typing import Final

from credit_risk.console.settings import ConsoleSettings

API_PORT: Final = 18000
MLFLOW_PORT: Final = 15020
AIRFLOW_PORT: Final = 18081
GRAFANA_PORT: Final = 13000
PROMETHEUS_PORT: Final = 19090
ALERTMANAGER_PORT: Final = 19093
OBJECT_STORE_PORT: Final = 19011


def _url(settings: ConsoleSettings, port: int, path: str = "") -> str:
    return f"http://{settings.console_link_host}:{port}{path}"


def airflow_dag_link(settings: ConsoleSettings, dag_id: str) -> str:
    """The DAG's grid view, where each run's logs are one click away."""
    return _url(settings, AIRFLOW_PORT, f"/dags/{dag_id}/grid")


def service_links(settings: ConsoleSettings) -> list[dict[str, str]]:
    """One card per place a reviewer or an analyst may want to go next."""

    def card(key: str, title: str, url: str, description: str) -> dict[str, str]:
        return {"id": key, "title": title, "url": url, "description": description}

    return [
        card(
            "swagger",
            "API chấm điểm (Swagger)",
            _url(settings, API_PORT, "/docs"),
            "Thử chấm điểm và giải thích từng khách hàng trực tiếp trên trình duyệt.",
        ),
        card(
            "mlflow",
            "MLflow",
            _url(settings, MLFLOW_PORT),
            "Lịch sử huấn luyện, chỉ số đánh giá và các phiên bản model đã đăng ký.",
        ),
        card(
            "airflow",
            "Airflow",
            _url(settings, AIRFLOW_PORT),
            "Lịch chạy, nhật ký và lịch sử từng bước của hai pipeline.",
        ),
        card(
            "grafana-service",
            "Grafana — Sức khỏe dịch vụ",
            _url(settings, GRAFANA_PORT, "/d/credit-service-health"),
            "Lưu lượng, độ trễ và tỉ lệ lỗi của API chấm điểm.",
        ),
        card(
            "grafana-model",
            "Grafana — Hành vi model",
            _url(settings, GRAFANA_PORT, "/d/credit-model-behaviour"),
            "Tỉ lệ khách rủi ro cao, phân phối điểm và độ lệch dữ liệu (PSI).",
        ),
        card(
            "grafana-fairness",
            "Grafana — Giám sát công bằng",
            _url(settings, GRAFANA_PORT, "/d/credit-fairness-monitor"),
            "Tỉ lệ được chọn theo nhóm giới tính và cảnh báo khi chênh lệch vượt ngưỡng.",
        ),
        card(
            "prometheus",
            "Prometheus",
            _url(settings, PROMETHEUS_PORT),
            "Chỉ số thô của hệ thống và trạng thái các luật cảnh báo.",
        ),
        card(
            "alertmanager",
            "Alertmanager",
            _url(settings, ALERTMANAGER_PORT),
            "Các cảnh báo đang kích hoạt và nơi chúng được gửi đi.",
        ),
        card(
            "objectstore",
            "Kho lưu trữ artefact",
            _url(settings, OBJECT_STORE_PORT),
            "File model và báo cáo mà MLflow lưu trong SeaweedFS.",
        ),
        card(
            "github",
            "Mã nguồn (GitHub)",
            settings.console_repo_url,
            "Mã nguồn, CI/CD, model card và tài liệu kiến trúc của dự án.",
        ),
    ]
