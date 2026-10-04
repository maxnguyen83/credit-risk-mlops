"""scripts/traffic.py: what the banner promises for each failure mode.

A presenter reads the `modes` line aloud, so it has to name the alert a mode
really drives -- and name none for --bias, which changes the arrival mix and not
a within-group rate. The script lives outside the package, so it is loaded by
path. Nothing here sends a request.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "traffic.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("traffic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolve string annotations through
    # sys.modules, and an unregistered module fails at class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


traffic = _load()


def _modes_line(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    traffic.describe(traffic.parse_args(list(argv)), "pool", "http://api")
    lines = [line for line in capsys.readouterr().out.splitlines() if "modes" in line]
    assert len(lines) == 1
    return lines[0]


def test_bias_promises_no_alert(capsys: pytest.CaptureFixture[str]) -> None:
    line = _modes_line(capsys, "--bias", "0.9")
    assert "FairnessGapExceeded" not in line
    assert "no alert" in line


@pytest.mark.parametrize(
    ("argv", "alert"),
    [
        (("--broken", "0.2"), "HighErrorRate"),
        (("--drift", "3.0"), "FeatureDriftHigh"),
        (("--slow",), "SlowBatchPredictions"),
    ],
)
def test_other_modes_name_their_alert(
    capsys: pytest.CaptureFixture[str], argv: tuple[str, ...], alert: str
) -> None:
    assert alert in _modes_line(capsys, *argv)


def test_normal_traffic_names_no_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert "none (normal traffic)" in _modes_line(capsys)
