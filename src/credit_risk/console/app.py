"""The console's HTTP surface: one page, and the small JSON API that page calls.

``create_app`` takes its settings and its Airflow client as arguments so tests
can hand it a temporary data directory and an in-memory Airflow; the module
level ``app`` is the one uvicorn serves, built from the environment.

Every error leaves in one shape, ``{"code", "message", "detail"}``, where
``message`` is Vietnamese and written for the member of staff who will read it
on the page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from credit_risk.console import batches, links, pipelines, results
from credit_risk.console.airflow import AirflowApi, AirflowClient, AirflowUnavailable, DagNotFound
from credit_risk.console.settings import ConsoleSettings

log = logging.getLogger(__name__)

STATIC_DIR: Final = Path(__file__).resolve().parent / "static"

DEFAULT_RESULT_ROWS: Final = 50

# The page loads nothing from anywhere else, so it can say so. A value from a
# results file that somehow became markup would still have nowhere to load from.
SECURITY_HEADERS: Final[dict[str, str]] = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class ConsoleError(Exception):
    """A failure the console has classified, with a status and a message for the page."""

    def __init__(self, status_code: int, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class Console:
    """What every handler needs: the configuration and a way to reach Airflow."""

    settings: ConsoleSettings
    airflow: AirflowApi


router = APIRouter()


def _console(request: Request) -> Console:
    console: Console = request.app.state.console
    return console


def _render_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ConsoleError | batches.UploadRejected)
    body = {"code": exc.code, "message": exc.message, "detail": exc.detail}
    return JSONResponse(body, status_code=exc.status_code)


def _start(console: Console, dag_id: str, conf: dict[str, Any]) -> dict[str, Any]:
    try:
        return pipelines.start(console.airflow, dag_id, conf)
    except pipelines.UnknownPipeline as exc:
        raise ConsoleError(
            404, "unknown_pipeline", "Bảng điều khiển không chạy pipeline này.", dag_id
        ) from exc
    except pipelines.PipelineBusy as exc:
        raise ConsoleError(
            409,
            "pipeline_busy",
            "Pipeline đang có một lần chạy chưa xong. Đợi lần đó kết thúc rồi thử lại.",
            exc.run_id,
        ) from exc
    except DagNotFound as exc:
        raise ConsoleError(503, "dag_not_loaded", pipelines.DAG_NOT_LOADED, dag_id) from exc
    except AirflowUnavailable as exc:
        raise ConsoleError(503, "airflow_unavailable", exc.message, exc.detail) from exc


async def _read_body(request: Request, max_bytes: int) -> bytes:
    """The request body, refused as soon as it passes ``max_bytes``.

    Content-Length is checked first, but not trusted: a chunked upload has
    none, so the stream is counted as it arrives and never held past the limit.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise batches.too_large(max_bytes)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise batches.too_large(max_bytes)
    return bytes(body)


def _save_and_start(console: Console, parsed: batches.ParsedUpload) -> dict[str, Any]:
    data_dir = console.settings.console_data_dir
    try:
        relative = batches.save_upload(data_dir, parsed.text, now=datetime.now(UTC))
    except OSError as exc:
        log.error("cannot save an upload under %s: %s", data_dir, exc)
        raise ConsoleError(
            500,
            "cannot_save",
            "Không lưu được file vào thư mục dữ liệu chung. Báo quản trị hệ thống kiểm tra "
            "quyền ghi của thư mục incoming.",
            type(exc).__name__,
        ) from exc
    try:
        started = _start(console, pipelines.SCORING.dag_id, {"input": relative})
    except ConsoleError:
        # No run will ever read it; leaving it would look like a batch in flight.
        batches.discard(data_dir, relative)
        raise
    log.info("upload %s (%d rows) -> run %s", relative, parsed.n_rows, started["run_id"])
    return {**started, "input": relative, "n_rows": parsed.n_rows}


# ------------------------------------------------------------------ page


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The single-page console."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness only. Airflow being down is shown on the page, not a reason to restart this."""
    return {"status": "ok"}


# ------------------------------------------------------------- pipelines


@router.get("/api/pipelines")
def get_pipelines(request: Request) -> dict[str, Any]:
    """Both DAGs' diagrams. Answers 200 with ``available: false`` when Airflow cannot be asked."""
    console = _console(request)
    view = pipelines.collect(console.airflow)
    for pipeline in view["pipelines"]:
        pipeline["airflow_link"] = links.airflow_dag_link(console.settings, pipeline["dag_id"])
    view["checked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return view


@router.post("/api/pipelines/{dag_id}/runs", status_code=202)
def trigger_pipeline(dag_id: str, request: Request) -> dict[str, Any]:
    """Start an allowlisted DAG with an empty conf, unpausing it first if needed."""
    return _start(_console(request), dag_id, {})


# --------------------------------------------------------------- batches


@router.get("/api/batches/requirements")
def batch_requirements(request: Request) -> dict[str, Any]:
    """What an upload must look like, so the page can say so before anyone tries."""
    return {
        "required_columns": list(batches.REQUIRED_COLUMNS),
        "max_bytes": _console(request).settings.console_max_upload_bytes,
        "content_type": "text/csv",
    }


@router.post("/api/batches", status_code=202)
async def upload_batch(request: Request) -> dict[str, Any]:
    """Accept a CSV body, save it under ``incoming/`` and start the scoring DAG on it.

    The body is the raw file with ``Content-Type: text/csv`` -- not a multipart
    form, which would need a parser this image does not ship.
    """
    console = _console(request)
    batches.check_content_type(request.headers.get("content-type"))
    payload = await _read_body(request, console.settings.console_max_upload_bytes)
    parsed = batches.parse_upload(payload)
    return await run_in_threadpool(_save_and_start, console, parsed)


@router.post("/api/batches/sample", status_code=202)
def score_sample(request: Request) -> dict[str, Any]:
    """Score the demo portfolio: the 5,000 held-back customers in the serving pool."""
    conf = {"input": batches.SAMPLE_INPUT}
    return {**_start(_console(request), pipelines.SCORING.dag_id, conf), **conf}


# --------------------------------------------------------------- results


@router.get("/api/results/latest", response_model=None)
def latest_results(
    request: Request, limit: int = Query(DEFAULT_RESULT_ROWS, ge=1, le=1000)
) -> dict[str, Any] | JSONResponse:
    """The latest summary and the top of its call list; 404 with a reason before any scoring."""
    try:
        return results.latest(_console(request).settings.console_data_dir, limit)
    except results.NoResults as exc:
        return JSONResponse({"available": False, "reason": exc.reason}, status_code=404)


@router.get("/api/results/latest/{name}")
def download_result(name: str, request: Request) -> FileResponse:
    """One of the latest run's files, by its published name only."""
    try:
        path = results.result_file(_console(request).settings.console_data_dir, name)
    except results.NoResults as exc:
        raise ConsoleError(404, "not_found", exc.reason, name) from exc
    return FileResponse(path, media_type="text/csv", filename=f"{path.stem}_{path.parent.name}.csv")


# ----------------------------------------------------------------- links


@router.get("/api/links")
def get_links(request: Request) -> dict[str, Any]:
    """The other services, as cards for section 4 of the page."""
    return {"links": links.service_links(_console(request).settings)}


# ------------------------------------------------------------------- app


def create_app(
    settings: ConsoleSettings | None = None, airflow: AirflowApi | None = None
) -> FastAPI:
    """Build the console around a configuration and an Airflow client."""
    settings = settings or ConsoleSettings()
    app = FastAPI(
        title="Credit-risk operations console",
        # The console's own API is an implementation detail of its page; the
        # Swagger staff should find is the scoring API's, linked from the page.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.console = Console(settings, airflow or AirflowClient.from_settings(settings))
    app.add_exception_handler(ConsoleError, _render_error)
    app.add_exception_handler(batches.UploadRejected, _render_error)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    log.info(
        "console ready: airflow=%s data_dir=%s", settings.airflow_url, settings.console_data_dir
    )
    return app


app = create_app()
