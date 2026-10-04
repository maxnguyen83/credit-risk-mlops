"""Data-quality tests for the raw UCI file, and for how it gets here.

The counts asserted below were measured on the published file, not copied
from its documentation. If UCI re-exports the dataset these tests go red,
which is the point: every downstream number in this repo -- base rates,
fairness gaps, the model card -- describes the file these assertions
describe.

The tests that need the 30,000-row parquet are marked ``needs_data``; the
ingestion tests below them stub the network and run everywhere.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd
import pytest
import requests

from credit_risk import schema
from credit_risk.data import download as download_module
from credit_risk.data.download import (
    DownloadError,
    TransientDownloadError,
    download_raw,
    load_raw,
    load_raw_metadata,
    raw_meta_path,
    raw_parquet_path,
)
from credit_risk.data.validate import validate_raw

# Measured 2026-09-30 on the published file.
UNDOCUMENTED_EDUCATION_COUNTS: Final[dict[int, int]] = {0: 14, 5: 280, 6: 51}
UNDOCUMENTED_MARRIAGE_COUNTS: Final[dict[int, int]] = {0: 54}
GROUP_SIZES_BY_SEX: Final[dict[int, int]] = {1: 11_888, 2: 18_112}


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    path = raw_parquet_path()
    if not path.exists():
        pytest.skip(f"raw dataset not at {path}; run `make data` first")
    return pd.read_parquet(path)


@pytest.mark.needs_data
class TestPublishedFile:
    """Assertions about the file itself."""

    def test_shape(self, raw_df: pd.DataFrame) -> None:
        assert raw_df.shape == (schema.RAW_N_ROWS, schema.RAW_N_COLS)

    def test_columns_match_the_contract(self, raw_df: pd.DataFrame) -> None:
        assert tuple(raw_df.columns) == schema.RAW_COLUMNS

    def test_first_repayment_column_is_still_misnamed(self, raw_df: pd.DataFrame) -> None:
        """PAY_0 in the export, PAY_1 everywhere else. Renaming is our job."""
        assert schema.PAY_0_RAW in raw_df.columns
        assert schema.PAY_1 not in raw_df.columns

    def test_no_nulls_anywhere(self, raw_df: pd.DataFrame) -> None:
        assert int(raw_df.isna().sum().sum()) == 0

    def test_target_positive_rate(self, raw_df: pd.DataFrame) -> None:
        rate = float(raw_df[schema.TARGET_RAW].mean())
        assert rate == pytest.approx(schema.BASE_POSITIVE_RATE, abs=0.001)

    def test_undocumented_education_codes(self, raw_df: pd.DataFrame) -> None:
        counts = raw_df[schema.EDUCATION].value_counts()
        observed = {
            int(code): int(counts[code])
            for code in schema.UNDOCUMENTED_EDUCATION
            if code in counts.index
        }
        assert observed == UNDOCUMENTED_EDUCATION_COUNTS

    def test_undocumented_marriage_codes(self, raw_df: pd.DataFrame) -> None:
        counts = raw_df[schema.MARRIAGE].value_counts()
        observed = {
            int(code): int(counts[code])
            for code in schema.UNDOCUMENTED_MARRIAGE
            if code in counts.index
        }
        assert observed == UNDOCUMENTED_MARRIAGE_COUNTS

    def test_base_rates_by_sex_are_what_the_fairness_analysis_assumes(
        self, raw_df: pd.DataFrame
    ) -> None:
        grouped = raw_df.groupby(schema.SEX)[schema.TARGET_RAW]
        for code, expected in schema.BASE_RATE_BY_SEX.items():
            assert int(grouped.size()[code]) == GROUP_SIZES_BY_SEX[code]
            assert float(grouped.mean()[code]) == pytest.approx(expected, abs=0.001)

    def test_validator_reports_only_the_known_warnings(self, raw_df: pd.DataFrame) -> None:
        report = validate_raw(raw_df)
        assert report.ok, [c.to_dict() for c in report.failures]
        assert report.bad_row_fraction == 0.0
        assert {c.name for c in report.warnings} == {"education_codes", "marriage_codes"}
        assert report.check("education_codes").n_bad_rows == sum(
            UNDOCUMENTED_EDUCATION_COUNTS.values()
        )
        assert report.check("marriage_codes").n_bad_rows == sum(
            UNDOCUMENTED_MARRIAGE_COUNTS.values()
        )


class TestIngestion:
    """How the file lands on disk. The network is stubbed, never called."""

    def test_raw_path_lives_under_the_configured_raw_dir(self) -> None:
        assert raw_parquet_path().name == download_module.RAW_PARQUET_NAME

    def test_an_html_error_page_is_not_mistaken_for_an_archive(self) -> None:
        """A 200 response full of HTML is the classic way this breaks."""
        with pytest.raises(DownloadError, match="not a zip"):
            download_module._read_archive(b"<html><body>Service unavailable</body></html>")

    def test_archive_without_a_spreadsheet_is_rejected(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "nothing useful here")
        with pytest.raises(DownloadError, match="no .xls member"):
            download_module._read_archive(buffer.getvalue())

    def test_existing_parquet_short_circuits_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        dest = tmp_path / "credit_default_raw.parquet"
        monkeypatch.setattr(download_module, "_fetch", lambda url: b"pretend-zip")
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)
        download_raw(dest=dest)

        def explode(url: str) -> bytes:
            raise AssertionError(f"re-downloaded {url} when the parquet already existed")

        monkeypatch.setattr(download_module, "_fetch", explode)
        assert download_raw(dest=dest) == dest

    def test_a_parquet_without_its_sidecar_is_downloaded_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        """A crash between the two writes used to leave exactly this, and the
        next run skipped the download and wrote source_sha256: null forever."""
        dest = tmp_path / "raw.parquet"
        synthetic_raw_df.to_parquet(dest, index=False)
        calls: list[str] = []

        def record(url: str) -> bytes:
            calls.append(url)
            return b"pretend-zip"

        monkeypatch.setattr(download_module, "_fetch", record)
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)

        download_raw(dest=dest)

        assert calls == [schema.DATASET_URL]
        assert len(load_raw_metadata(dest)["sha256"]) == 64

    def test_a_parquet_that_is_not_the_one_its_sidecar_describes_is_downloaded_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        dest = tmp_path / "raw.parquet"
        monkeypatch.setattr(download_module, "_fetch", lambda url: b"pretend-zip")
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)
        download_raw(dest=dest)
        synthetic_raw_df.head(10).to_parquet(dest, index=False)  # replaced behind its back

        calls: list[str] = []
        monkeypatch.setattr(download_module, "_fetch", lambda url: calls.append(url) or b"zip")
        download_raw(dest=dest)

        assert calls == [schema.DATASET_URL]
        assert len(pd.read_parquet(dest)) == len(synthetic_raw_df)

    @staticmethod
    def _second_rename_dies(monkeypatch: pytest.MonkeyPatch) -> None:
        """Let the first os.replace through and kill the process on the second."""
        real = os.replace
        calls: list[str] = []

        def replace(src: object, dst: object) -> None:
            calls.append(str(dst))
            if len(calls) == 2:
                raise KeyboardInterrupt("killed between the two renames")
            real(src, dst)  # type: ignore[arg-type]

        monkeypatch.setattr(download_module.os, "replace", replace)

    @staticmethod
    def _counting_fetch(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> list[str]:
        calls: list[str] = []

        def fetch(url: str) -> bytes:
            calls.append(url)
            return f"zip-{len(calls)}".encode()

        monkeypatch.setattr(download_module, "_fetch", fetch)
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: frame)
        return calls

    def test_a_crash_between_the_two_renames_is_downloaded_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        dest = tmp_path / "raw.parquet"
        calls = self._counting_fetch(monkeypatch, synthetic_raw_df)
        with monkeypatch.context() as crash:
            self._second_rename_dies(crash)
            with pytest.raises(KeyboardInterrupt):
                download_raw(dest=dest)

        download_raw(dest=dest)

        assert len(calls) == 2, "the half-finished download was trusted"
        assert load_raw_metadata(dest)["parquet_sha256"] == _sha256(dest)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_a_crash_while_replacing_an_old_download_is_downloaded_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        """The case the rename order exists for. The sidecar on disk predates the
        parquet digest, so it is trusted on its archive hash alone. Rename the
        new parquet first, die, and that old sidecar vouches for a file it never
        described. Rename the new sidecar first, die, and its digest disowns the
        old parquet left in place."""
        dest = tmp_path / "raw.parquet"
        synthetic_raw_df.head(50).to_parquet(dest, index=False)
        raw_meta_path(dest).write_text(json.dumps({"sha256": "0" * 64, "n_rows": 50}) + "\n")
        calls = self._counting_fetch(monkeypatch, synthetic_raw_df)
        with monkeypatch.context() as crash:
            self._second_rename_dies(crash)
            with pytest.raises(KeyboardInterrupt):
                download_raw(dest=dest, force=True)

        download_raw(dest=dest)

        assert len(calls) == 2, "a sidecar vouched for a parquet it does not describe"
        assert len(pd.read_parquet(dest)) == len(synthetic_raw_df)
        assert load_raw_metadata(dest)["n_rows"] == len(synthetic_raw_df)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_another_downloads_temporary_files_are_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        """Two downloads into one directory (a manual run beside the DAG) must
        not write through, rename or delete each other's half-written files."""
        dest = tmp_path / "raw.parquet"
        theirs = {
            tmp_path / "raw.parquet.tmp": b"another run's parquet, half written",
            tmp_path / "raw.meta.json.tmp": b"another run's sidecar",
        }
        for path, content in theirs.items():
            path.write_bytes(content)
        self._counting_fetch(monkeypatch, synthetic_raw_df)

        download_raw(dest=dest)

        for path, content in theirs.items():
            assert path.read_bytes() == content, f"{path.name} was touched"
        assert len(pd.read_parquet(dest)) == len(synthetic_raw_df)
        assert sorted(p.name for p in tmp_path.glob("*.tmp")) == sorted(p.name for p in theirs)
        # Unique temporaries are created 0600; the files they become must not be.
        plain = tmp_path / "plain"
        plain.write_bytes(b"")
        for written in (dest, raw_meta_path(dest)):
            assert written.stat().st_mode & 0o777 == plain.stat().st_mode & 0o777

    def test_download_writes_parquet_and_records_the_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        payload = b"pretend-this-is-the-uci-zip"
        monkeypatch.setattr(download_module, "_fetch", lambda url: payload)
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)

        dest = tmp_path / "raw.parquet"
        assert download_raw(dest=dest) == dest
        assert dest.exists()

        metadata = json.loads((tmp_path / "raw.meta.json").read_text())
        assert metadata["url"] == schema.DATASET_URL
        assert len(metadata["sha256"]) == 64
        assert metadata["n_bytes"] == len(payload)
        assert metadata["n_rows"] == len(synthetic_raw_df)
        assert metadata["n_cols"] == schema.RAW_N_COLS
        # The sidecar names the exact parquet it describes, so a parquet
        # swapped in later is recognised as a different file.
        assert metadata["parquet_sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()

    def test_force_refetches_even_when_the_parquet_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        calls: list[str] = []

        def record(url: str) -> bytes:
            calls.append(url)
            return b"pretend-this-is-the-uci-zip"

        monkeypatch.setattr(download_module, "_fetch", record)
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)

        dest = tmp_path / "raw.parquet"
        dest.write_bytes(b"stale")
        download_raw(dest=dest, force=True)

        assert calls == [schema.DATASET_URL]
        assert len(pd.read_parquet(dest)) == len(synthetic_raw_df)

    def test_load_raw_says_what_to_run_when_the_file_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="credit_risk.data.download"):
            load_raw(tmp_path / "absent.parquet")

    def test_cli_prints_the_parquet_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        synthetic_raw_df: pd.DataFrame,
    ) -> None:
        dest = tmp_path / "raw.parquet"
        monkeypatch.setattr(download_module, "_fetch", lambda url: b"pretend-zip")
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)

        assert download_module.main(["--dest", str(dest)]) == 0
        assert capsys.readouterr().out.strip() == str(dest)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Response:
    """The two attributes of requests.Response the fetch reads."""

    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


def _scripted_get(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[Exception | _Response]
) -> list[str]:
    """Replace requests.get with one that plays ``outcomes`` in order."""
    calls: list[str] = []

    def get(url: str, **kwargs: object) -> _Response:
        calls.append(url)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(download_module.requests, "get", get)
    return calls


class TestTransientFailures:
    """UCI has bad days. A blip must be retried; a wrong URL must not be."""

    def test_a_dropped_connection_is_retried_with_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _scripted_get(
            monkeypatch,
            [
                requests.ConnectionError("reset by peer"),
                requests.Timeout("read timed out"),
                _Response(200, b"the-zip"),
            ],
        )
        slept: list[float] = []

        assert download_module._fetch(schema.DATASET_URL, sleep=slept.append) == b"the-zip"

        assert len(calls) == 3
        assert slept == [download_module.BACKOFF_SECONDS, 2 * download_module.BACKOFF_SECONDS]

    def test_a_server_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _scripted_get(monkeypatch, [_Response(503), _Response(200, b"zip")])

        assert download_module._fetch(schema.DATASET_URL, sleep=lambda s: None) == b"zip"
        assert len(calls) == 2

    def test_a_client_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 404 means the URL is wrong. Retrying it only delays the red task."""
        calls = _scripted_get(monkeypatch, [_Response(404), _Response(200, b"zip")])

        with pytest.raises(DownloadError, match="404") as excinfo:
            download_module._fetch(schema.DATASET_URL, sleep=lambda s: None)

        assert not isinstance(excinfo.value, TransientDownloadError)
        assert len(calls) == 1

    def test_a_source_that_stays_down_is_reported_as_transient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = download_module.FETCH_ATTEMPTS
        calls = _scripted_get(monkeypatch, [_Response(502) for _ in range(attempts)])

        with pytest.raises(TransientDownloadError, match="502"):
            download_module._fetch(schema.DATASET_URL, sleep=lambda s: None)
        assert len(calls) == attempts

    def test_the_cli_exits_with_the_transient_code_so_the_dag_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(url: str) -> bytes:
            raise TransientDownloadError("still 503 after 4 attempts")

        monkeypatch.setattr(download_module, "_fetch", down)

        code = download_module.main(["--dest", str(tmp_path / "raw.parquet")])

        assert code == download_module.EXIT_TRANSIENT == 75

    def test_the_dag_retries_on_the_exit_code_the_download_uses(self) -> None:
        """The DAG cannot import this package (ADR 0006), so it keeps its own
        copy of the number. This is what stops the two drifting apart."""
        dag_file = Path(__file__).resolve().parents[2] / "dags" / "credit_risk_pipeline.py"
        tree = ast.parse(dag_file.read_text())
        assigned = {
            node.target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        value = assigned.get("TRANSIENT_EXIT_CODE")
        assert isinstance(value, ast.Constant), "dags/ must define TRANSIENT_EXIT_CODE"
        assert value.value == download_module.EXIT_TRANSIENT


class TestProvenance:
    """The sidecar that lets the splits manifest name its source."""

    def test_metadata_round_trips_through_the_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        monkeypatch.setattr(download_module, "_fetch", lambda url: b"pretend-zip")
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)

        dest = tmp_path / "raw.parquet"
        download_raw(dest=dest)

        metadata = load_raw_metadata(dest)
        assert len(metadata["sha256"]) == 64
        assert metadata["n_rows"] == len(synthetic_raw_df)

    def test_a_sidecar_that_does_not_describe_its_parquet_is_not_believed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_raw_df: pd.DataFrame
    ) -> None:
        """split.py reads the sidecar without going through download_raw, so a
        split run after a crash would otherwise copy another file's digest
        into the manifest."""
        monkeypatch.setattr(download_module, "_fetch", lambda url: b"pretend-zip")
        monkeypatch.setattr(download_module, "_read_archive", lambda blob: synthetic_raw_df)
        dest = tmp_path / "raw.parquet"
        download_raw(dest=dest)
        assert load_raw_metadata(dest)["sha256"]

        synthetic_raw_df.head(10).to_parquet(dest, index=False)

        assert load_raw_metadata(dest) == {}
        dest.unlink()
        assert load_raw_metadata(dest) == {}, "a digest cannot vouch for a missing file"

    def test_a_parquet_placed_by_hand_has_no_provenance_and_that_is_not_fatal(
        self, tmp_path: Path
    ) -> None:
        """Fixtures and samples arrive without a sidecar; the pipeline still runs."""
        assert load_raw_metadata(tmp_path / "raw.parquet") == {}

    def test_an_unreadable_sidecar_is_ignored_rather_than_crashing_ingestion(
        self, tmp_path: Path
    ) -> None:
        """A half-written JSON file must not take the whole DAG down with it."""
        dest = tmp_path / "raw.parquet"
        raw_meta_path(dest).write_text('{"sha256": "trunca')
        assert load_raw_metadata(dest) == {}

    def test_a_sidecar_holding_the_wrong_shape_is_ignored(self, tmp_path: Path) -> None:
        dest = tmp_path / "raw.parquet"
        raw_meta_path(dest).write_text("[1, 2, 3]")
        assert load_raw_metadata(dest) == {}
