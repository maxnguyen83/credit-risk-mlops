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

import io
import json
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from credit_risk import schema
from credit_risk.data import download as download_module
from credit_risk.data.download import (
    DownloadError,
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
        synthetic_raw_df.to_parquet(dest, index=False)

        def explode(url: str) -> bytes:
            raise AssertionError(f"re-downloaded {url} when the parquet already existed")

        monkeypatch.setattr(download_module, "_fetch", explode)
        assert download_raw(dest=dest) == dest

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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], synthetic_raw_df: pd.DataFrame
    ) -> None:
        dest = tmp_path / "raw.parquet"
        synthetic_raw_df.to_parquet(dest, index=False)

        assert download_module.main(["--dest", str(dest)]) == 0
        assert capsys.readouterr().out.strip() == str(dest)


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
