"""The console's file handling: what an upload must look like, and how results are read.

Both sides touch the shared data directory that Airflow also writes to, so the
two failure modes that matter are a bad file getting through to the scoring
DAG, and a value read from a file steering a path outside the results folder.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from credit_risk.console import batches, results
from credit_risk.features.build import REQUIRED_INPUT_COLUMNS

NOW = datetime(2026, 10, 4, 10, 15, 30, tzinfo=UTC)


def csv_text(columns: tuple[str, ...] | list[str], n_rows: int = 2) -> str:
    rows = [",".join(columns)]
    rows += [",".join("1" for _ in columns) for _ in range(n_rows)]
    return "\n".join(rows) + "\n"


# ----------------------------------------------------------------- uploads


def test_required_columns_are_the_model_inputs() -> None:
    # The console checks the same 23 columns the feature builder needs; a
    # drift between the two would let through a file the DAG then rejects.
    assert batches.REQUIRED_COLUMNS == REQUIRED_INPUT_COLUMNS


def test_a_well_formed_upload_is_accepted_and_counted() -> None:
    text = csv_text(["ID", *batches.REQUIRED_COLUMNS], n_rows=3)
    parsed = batches.parse_upload(text.encode())
    assert parsed.n_rows == 3
    assert parsed.columns[0] == "ID"


def test_an_excel_bom_and_padded_header_are_tolerated() -> None:
    text = csv_text([f" {c} " for c in batches.REQUIRED_COLUMNS])
    parsed = batches.parse_upload(b"\xef\xbb\xbf" + text.encode())
    assert parsed.n_rows == 2
    # Saved without the BOM, so the scoring DAG reads the first column by name.
    assert not parsed.text.startswith("﻿")


def test_missing_columns_are_named() -> None:
    columns = [c for c in batches.REQUIRED_COLUMNS if c not in {"PAY_1", "BILL_AMT3"}]
    with pytest.raises(batches.UploadRejected) as caught:
        batches.parse_upload(csv_text(columns).encode())
    assert caught.value.status_code == 422
    assert caught.value.code == "missing_columns"
    assert caught.value.detail == ["PAY_1", "BILL_AMT3"]


@pytest.mark.parametrize("payload", [b"", b"   \n\n"])
def test_an_empty_upload_is_rejected(payload: bytes) -> None:
    with pytest.raises(batches.UploadRejected) as caught:
        batches.parse_upload(payload)
    assert caught.value.status_code == 400
    assert caught.value.code == "empty_file"


def test_a_header_without_rows_is_rejected() -> None:
    with pytest.raises(batches.UploadRejected) as caught:
        batches.parse_upload(csv_text(batches.REQUIRED_COLUMNS, n_rows=0).encode())
    assert caught.value.status_code == 422
    assert caught.value.code == "no_rows"


def test_blank_lines_are_not_counted_as_customers() -> None:
    text = csv_text(batches.REQUIRED_COLUMNS, n_rows=1) + "\n,,,\n"
    assert batches.parse_upload(text.encode()).n_rows == 1


def test_a_binary_file_is_rejected() -> None:
    with pytest.raises(batches.UploadRejected) as caught:
        batches.parse_upload(b"PK\x03\x04\xff\xfe\x00 an xlsx is a zip")
    assert caught.value.status_code == 400
    assert caught.value.code == "not_utf8"


def test_a_malformed_csv_is_rejected() -> None:
    header = ",".join(batches.REQUIRED_COLUMNS)
    with pytest.raises(batches.UploadRejected) as caught:
        batches.parse_upload(f'{header}\n"unterminated,1\n'.encode())
    assert caught.value.status_code == 400
    assert caught.value.code == "malformed_csv"


@pytest.mark.parametrize("content_type", ["text/csv", "text/csv; charset=utf-8", "TEXT/CSV"])
def test_csv_content_types_are_accepted(content_type: str) -> None:
    batches.check_content_type(content_type)


@pytest.mark.parametrize("content_type", [None, "", "application/json", "multipart/form-data"])
def test_other_content_types_are_refused(content_type: str | None) -> None:
    with pytest.raises(batches.UploadRejected) as caught:
        batches.check_content_type(content_type)
    assert caught.value.status_code == 415


def test_an_upload_is_saved_under_incoming_with_a_server_chosen_name(tmp_path: Path) -> None:
    relative = batches.save_upload(tmp_path, "a,b\n1,2\n", now=NOW)
    assert relative == "incoming/20261004T101530Z.csv"
    assert (tmp_path / relative).read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_two_uploads_in_the_same_second_do_not_overwrite_each_other(tmp_path: Path) -> None:
    first = batches.save_upload(tmp_path, "first\n", now=NOW)
    second = batches.save_upload(tmp_path, "second\n", now=NOW)
    assert first != second
    assert (tmp_path / first).read_text(encoding="utf-8") == "first\n"
    assert (tmp_path / second).read_text(encoding="utf-8") == "second\n"


# ----------------------------------------------------------------- results


SUMMARY: dict[str, Any] = {
    "generated_at": "2026-10-04T10:20:00Z",
    "run_dir": "20261004T102000Z",
    "input_path": "processed/serving_pool.parquet",
    "n_input": 5000,
    "n_rejected": 0,
    "n_scored": 5000,
    "n_call_list": 500,
    "capacity_fraction": 0.1,
    "n_above_threshold": 612,
    "model_name": "credit-risk",
    "model_version": "3",
    "threshold_used": 0.4123,
    "dag_run_id": "manual__1",
}

CALL_LIST_HEADER = "rank,account_id,default_probability,risk_band,decision,top_reasons\n"


def write_results(
    data_dir: Path,
    summary: dict[str, Any] | None = None,
    call_list: str | None = None,
    scores: str | None = "account_id,default_probability\n1,0.9\n",
) -> Path:
    summary = SUMMARY if summary is None else summary
    scored = data_dir / "scored"
    run_dir = scored / str(summary.get("run_dir", "missing"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (scored / "latest.json").write_text(json.dumps(summary), encoding="utf-8")
    if call_list is None:
        call_list = CALL_LIST_HEADER + "".join(
            f"{i},A-{i},{1 - i / 100:.4f},high,intervene,PAY_1 late | utilisation high\n"
            for i in range(1, 8)
        )
    (run_dir / "call_list.csv").write_text(call_list, encoding="utf-8")
    if scores is not None:
        (run_dir / "scores.csv").write_text(scores, encoding="utf-8")
    return run_dir


def test_latest_result_returns_the_summary_and_the_first_rows(tmp_path: Path) -> None:
    write_results(tmp_path)
    latest = results.latest(tmp_path, limit=5)

    assert latest["summary"]["n_scored"] == 5000
    assert latest["summary"]["model_version"] == "3"
    assert latest["n_rows_total"] == 7
    assert len(latest["rows"]) == 5
    first = latest["rows"][0]
    assert first == {
        "rank": 1,
        "account_id": "A-1",
        "default_probability": pytest.approx(0.99),
        "risk_band": "high",
        "decision": "intervene",
        "top_reasons": ["PAY_1 late", "utilisation high"],
    }
    assert latest["downloads"] == {
        "call_list.csv": "/api/results/latest/call_list.csv",
        "scores.csv": "/api/results/latest/scores.csv",
        "rejected.csv": None,
    }


def test_nothing_scored_yet_is_an_empty_state(tmp_path: Path) -> None:
    with pytest.raises(results.NoResults) as caught:
        results.latest(tmp_path, limit=50)
    assert caught.value.reason


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]", ""])
def test_an_unreadable_summary_is_an_empty_state_not_a_crash(tmp_path: Path, content: str) -> None:
    (tmp_path / "scored").mkdir()
    (tmp_path / "scored" / "latest.json").write_text(content, encoding="utf-8")
    with pytest.raises(results.NoResults):
        results.latest(tmp_path, limit=50)


def test_missing_summary_keys_and_a_missing_call_list_still_render(tmp_path: Path) -> None:
    run_dir = write_results(tmp_path, summary={"run_dir": "r1", "n_scored": 10}, scores=None)
    (run_dir / "call_list.csv").unlink()

    latest = results.latest(tmp_path, limit=50)

    assert latest["summary"]["n_scored"] == 10
    assert latest["summary"]["model_version"] is None
    assert latest["rows"] == []
    assert latest["n_rows_total"] is None
    assert latest["downloads"]["call_list.csv"] is None
    assert latest["downloads"]["scores.csv"] is None


def test_odd_cells_in_the_call_list_become_nulls(tmp_path: Path) -> None:
    call_list = "rank,account_id,default_probability\nfirst,A-1,not-a-number\n"
    write_results(tmp_path, call_list=call_list)
    row = results.latest(tmp_path, limit=50)["rows"][0]
    assert row["rank"] is None
    assert row["default_probability"] is None
    assert row["account_id"] == "A-1"
    assert row["top_reasons"] == []


def test_an_absolute_run_dir_from_the_airflow_container_is_found_by_name(tmp_path: Path) -> None:
    write_results(tmp_path)
    summary = {**SUMMARY, "run_dir": "/opt/airflow/data/scored/20261004T102000Z"}
    (tmp_path / "scored" / "latest.json").write_text(json.dumps(summary), encoding="utf-8")
    assert results.latest(tmp_path, limit=50)["n_rows_total"] == 7


def test_the_run_dir_the_scorer_writes_is_found(tmp_path: Path) -> None:
    # credit_risk.scoring.batch records run_dir relative to the data dir
    # ("scored/<stamp>"), so it reads the same in every container.
    write_results(tmp_path)
    summary = {**SUMMARY, "run_dir": "scored/20261004T102000Z"}
    (tmp_path / "scored" / "latest.json").write_text(json.dumps(summary), encoding="utf-8")
    assert results.latest(tmp_path, limit=50)["n_rows_total"] == 7


@pytest.mark.parametrize("run_dir", ["..", "../..", "../outside", "", ".", None, 7])
def test_a_run_dir_that_escapes_the_results_folder_is_ignored(tmp_path: Path, run_dir: Any) -> None:
    data_dir = tmp_path / "data"
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "call_list.csv").write_text("secret\n", encoding="utf-8")
    (data_dir / "scored").mkdir(parents=True)
    (data_dir / "scored" / "call_list.csv").write_text("secret\n", encoding="utf-8")
    summary = {**SUMMARY, "run_dir": run_dir}
    (data_dir / "scored" / "latest.json").write_text(json.dumps(summary), encoding="utf-8")

    latest = results.latest(data_dir, limit=50)
    assert latest["rows"] == []
    with pytest.raises(results.NoResults):
        results.result_file(data_dir, "call_list.csv")


def test_a_symlink_out_of_the_results_folder_is_not_followed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "scores.csv").write_text("secret\n", encoding="utf-8")
    (data_dir / "scored").mkdir(parents=True)
    (data_dir / "scored" / "r1").symlink_to(elsewhere, target_is_directory=True)
    (data_dir / "scored" / "latest.json").write_text(json.dumps({"run_dir": "r1"}), "utf-8")

    with pytest.raises(results.NoResults):
        results.result_file(data_dir, "scores.csv")


def test_only_the_published_file_names_can_be_downloaded(tmp_path: Path) -> None:
    write_results(tmp_path)
    assert results.result_file(tmp_path, "scores.csv").name == "scores.csv"
    for name in ["latest.json", "../scored/latest.json", "call_list.csv/../scores.csv"]:
        with pytest.raises(results.NoResults):
            results.result_file(tmp_path, name)


def test_a_result_file_linked_out_of_the_results_folder_is_not_served(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = write_results(data_dir)
    (tmp_path / "secret.csv").write_text("secret\n", encoding="utf-8")
    (run_dir / "scores.csv").unlink()
    (run_dir / "scores.csv").symlink_to(tmp_path / "secret.csv")

    assert results.latest(data_dir, limit=50)["downloads"]["scores.csv"] is None
    with pytest.raises(results.NoResults):
        results.result_file(data_dir, "scores.csv")


def test_non_finite_numbers_become_nulls_so_the_response_can_be_json(tmp_path: Path) -> None:
    call_list = CALL_LIST_HEADER + "1,A-1,nan,high,intervene,x\n2,A-2,inf,high,intervene,x\n"
    write_results(tmp_path, call_list=call_list)
    summary_path = tmp_path / "scored" / "latest.json"
    summary_path.write_text(
        '{"run_dir": "20261004T102000Z", "threshold_used": NaN, '
        '"rejected_by_reason": {"bad_age": Infinity}}',
        encoding="utf-8",
    )

    latest = results.latest(tmp_path, limit=50)

    assert latest["summary"]["threshold_used"] is None
    assert latest["summary"]["rejected_by_reason"] == {"bad_age": None}
    assert [row["default_probability"] for row in latest["rows"]] == [None, None]
    json.dumps(latest, allow_nan=False)
