#!/usr/bin/env python3
"""Traffic generator for the credit-risk API.

Drives the demo. Normal mode replays the batch-6 serving pool at a chosen rate;
three of the four failure modes each aim at one alert, so a presenter can say
"watch FeatureDriftHigh" and then be right -- given a run longer than that
rule's `for:` clause. The fourth fires nothing, by design.

    --broken 0.2   -> HighErrorRate
    --drift 3.0    -> FeatureDriftHigh, then HighRiskShareShift
    --bias 0.9     -> no alert: it changes the arrival mix, not a within-group
                      rate, so FairnessGapExceeded stays quiet (see below)
    --slow         -> SlowBatchPredictions

`--slow` drives SlowBatchPredictions and not SlowPredictions, because it posts
to /predict/batch and the SlowPredictions regex (`endpoint=~".*predict"`, fully
anchored by PromQL) deliberately excludes that endpoint -- the batch path has a
budget of its own. Pointing --slow at /predict instead would not work: one
record is fast however large you make the payload, so the only way to move the
single-record p95 from a client is to saturate the service, which is what
scripts/load_test.py is for.

Size alone is not what makes it slow, and it is worth knowing which lever is
doing the work: one 1000-record batch costs about 55ms, and even a dozen a
second keep p95 under 100ms. It is the default 20 calls/s of them that
oversubscribes the worker -- measured, 235 of 240 batches came back above the
500ms budget. Drop --rps far enough and this mode stops demonstrating anything.

The --bias run is the one worth watching. Latency does not move, the error
count stays at zero, every response is a 200 -- and who the system acts on
changes. No conventional operational metric notices.

What --bias does NOT do is make the model discriminate: credit_selection_rate is
a within-group rate, and reweighting the request mix leaves each group's rate
unchanged in expectation. It changes the precision with which each rate is
measured. Push it far enough and the minority group falls under the serving-side
minimum-sample floor and stops being published at all -- which is the monitor
saying it can no longer measure parity, and is worth showing as exactly that
rather than narrating it as bias.

It runs with no dataset on disk: if the serving pool parquet is missing it
falls back to synthetic records drawn from the published marginals. A demo
script that needs a 6-step data pipeline to have run first is a demo script
that fails live.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Importable without the package being installed: the demo runs this script
# straight out of a fresh clone, before anyone has run `make install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import settings  # noqa: E402
from credit_risk.schema import (  # noqa: E402
    AGE,
    BILL_COLS,
    EDUCATION,
    ID_COL,
    LIMIT_BAL,
    MARRIAGE,
    PAY_AMT_COLS,
    PAY_COLS,
    SERVING_BATCH,
    SEX,
)

DEFAULT_URL = "http://localhost:18000/api/v1/predict"

# The 23 model inputs, in the order the schema declares them. Derived rather
# than typed out: a column renamed in schema.py must not leave this script
# quietly sending a field the API stopped accepting.
FEATURE_COLS: tuple[str, ...] = (
    LIMIT_BAL,
    SEX,
    EDUCATION,
    MARRIAGE,
    AGE,
    *PAY_COLS,
    *BILL_COLS,
    *PAY_AMT_COLS,
)

INT_COLS: tuple[str, ...] = (SEX, EDUCATION, MARRIAGE, AGE, *PAY_COLS)
MONEY_COLS: tuple[str, ...] = (LIMIT_BAL, *BILL_COLS, *PAY_AMT_COLS)

# Columns --drift shifts. Repayment status codes (PAY_*) are deliberately left
# alone: they are a small ordinal scale, and adding 3 sigma to them produces
# values outside the documented -2..8 range, which the API would reject as a
# validation error. That would fire HighErrorRate instead of FeatureDriftHigh
# and make the demo prove the wrong point.
DRIFT_COLS: tuple[str, ...] = (LIMIT_BAL, *BILL_COLS, *PAY_AMT_COLS)

# Filenames the split job might write for batch 6. Checked in order; a miss is
# not fatal.
POOL_CANDIDATES: tuple[str, ...] = (
    f"batch_{SERVING_BATCH}.parquet",
    f"batch_0{SERVING_BATCH}.parquet",
    "serving_pool.parquet",
    "batch_6.parquet",
)


@dataclass
class Tally:
    """Running totals for the live summary."""

    sent: int = 0
    ok: int = 0
    errors: int = 0
    by_status: dict[int, int] = field(default_factory=dict)
    # Bounded on purpose: a 20-minute run at 200 rps is 240,000 samples, and
    # nobody needs a p95 over the whole run when the point is the last minute.
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=5_000))

    def record(self, status: int, elapsed: float) -> None:
        self.sent += 1
        self.latencies.append(elapsed)
        self.by_status[status] = self.by_status.get(status, 0) + 1
        if 200 <= status < 300:
            self.ok += 1
        else:
            self.errors += 1

    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        # Nearest-rank, which is what you want on a handful of samples; the
        # interpolating variants report a latency nobody actually observed.
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


# ------------------------------------------------------------------- pool


def synthetic_pool(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Records drawn from the published marginals of the UCI Taiwan dataset.

    Not a substitute for the real pool -- the correlations are gone. It exists
    so `--help` to first request never depends on the pipeline having run.
    """
    limit = np.clip(rng.lognormal(mean=11.7, sigma=0.65, size=n), 10_000, 1_000_000)
    # 39.6% male / 60.4% female, the measured split of the 30,000 accounts.
    sex = rng.choice([1, 2], size=n, p=[0.396, 0.604])
    education = rng.choice([1, 2, 3, 4], size=n, p=[0.353, 0.468, 0.164, 0.015])
    marriage = rng.choice([1, 2, 3], size=n, p=[0.455, 0.532, 0.013])
    age = np.clip(rng.normal(35.5, 9.2, size=n).round(), 21, 79)

    data: dict[str, np.ndarray] = {
        LIMIT_BAL: limit.round(-3),
        SEX: sex,
        EDUCATION: education,
        MARRIAGE: marriage,
        AGE: age,
    }
    # PAY_* is dominated by -1 (paid in full), 0 (revolving) and 1..2 (late).
    for col in PAY_COLS:
        data[col] = rng.choice([-2, -1, 0, 1, 2], size=n, p=[0.09, 0.19, 0.53, 0.13, 0.06])
    for col in BILL_COLS:
        # Bills can be negative: a customer who overpaid is in credit.
        data[col] = (limit * rng.beta(1.3, 2.4, size=n) - rng.exponential(2_000, size=n)).round()
    for col in PAY_AMT_COLS:
        data[col] = np.clip(rng.exponential(5_500, size=n).round(), 0, None)

    frame = pd.DataFrame(data)
    frame[ID_COL] = np.arange(1, n + 1)
    return frame


def load_pool(
    explicit: Path | None, size: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, str]:
    """The batch-6 serving pool if it is on disk, synthetic records otherwise."""
    candidates = [explicit] if explicit else [settings.processed_dir / c for c in POOL_CANDIDATES]
    for path in candidates:
        if path is not None and path.exists():
            frame = pd.read_parquet(path)
            missing = [c for c in FEATURE_COLS if c not in frame.columns]
            if missing:
                print(f"! {path.name} is missing {missing}; using synthetic records", flush=True)
                break
            return frame, str(path)
    return synthetic_pool(size, rng), "synthetic (serving pool not on disk)"


# ---------------------------------------------------------------- payloads


def build_record(row: pd.Series, index: int) -> dict[str, object]:
    """One clean request body."""
    record: dict[str, object] = {"account_id": f"A-{int(row.get(ID_COL, index)):06d}"}
    # int where the schema says int. The demographic codes and PAY_* are
    # categorical, and 2.0 is a different thing to 2 for a strict validator --
    # parquet round-trips integers as floats often enough to matter.
    for col in INT_COLS:
        record[col] = int(row[col])
    for col in MONEY_COLS:
        record[col] = float(row[col])
    return record


def apply_drift(record: dict[str, object], sigma: float, stds: dict[str, float]) -> None:
    """Shift the monetary columns by `sigma` training standard deviations."""
    for col in DRIFT_COLS:
        record[col] = float(record[col]) + sigma * stds[col]


def break_record(record: dict[str, object], rng: random.Random) -> dict[str, object]:
    """Two ways a caller breaks a contract: omission and wrong type."""
    broken = dict(record)
    if rng.random() < 0.5:
        broken.pop(rng.choice(list(FEATURE_COLS)), None)
    else:
        broken[rng.choice(list(FEATURE_COLS))] = "not-a-number"
    return broken


# ------------------------------------------------------------------ sending


def send(session: requests.Session, url: str, payload: object, timeout: float) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        response = session.post(url, json=payload, timeout=timeout)
        return response.status_code, time.perf_counter() - started
    except requests.RequestException:
        # 0 means "never reached a server". Counting it as a 5xx would blame the
        # API for a problem that is DNS, the port mapping, or nothing listening.
        return 0, time.perf_counter() - started


def _std(column: pd.Series) -> float:
    """Population std, floored at 1.0 so a constant column cannot make drift a no-op."""
    value = float(column.astype(float).std(ddof=0))
    return value if value == value and value > 0.0 else 1.0


def batch_url(url: str) -> str:
    return url.rstrip("/") + "/batch" if not url.rstrip("/").endswith("/batch") else url


# --------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate traffic against the credit-risk API, including the demo failure modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="prediction endpoint")
    parser.add_argument("--rps", type=float, default=20.0, help="requests per second")
    parser.add_argument("--seconds", type=int, default=60, help="how long to run")
    parser.add_argument("--pool", type=Path, default=None, help="parquet of records to replay")
    parser.add_argument(
        "--broken",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="fraction of requests sent with a missing field or a wrong type (-> HighErrorRate)",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="shift LIMIT_BAL, BILL_AMT* and PAY_AMT* by SIGMA std devs (-> FeatureDriftHigh)",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="force this fraction of records to SEX=1 (-> FairnessGapExceeded)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="send oversized batch payloads to /predict/batch (-> SlowBatchPredictions)",
    )
    parser.add_argument(
        # Defaults to the API's own cap. Server-side cost is roughly linear in
        # records per call, and a batch well short of the cap leaves the p95
        # under the 500ms budget -- which makes --slow a mode that demonstrates
        # nothing. The cap is the largest payload the endpoint will accept, so
        # this is the honest worst case rather than an arbitrary large number.
        "--batch-size",
        type=int,
        default=settings.max_batch_size,
        help="records per payload when --slow is set",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout")
    parser.add_argument("--seed", type=int, default=7, help="seed for record selection and modes")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary")
    return parser.parse_args(argv)


def describe(args: argparse.Namespace, source: str, url: str) -> None:
    modes = []
    if args.broken:
        modes.append(f"broken={args.broken:.0%} -> HighErrorRate")
    if args.drift:
        modes.append(f"drift={args.drift}sigma -> FeatureDriftHigh")
    if args.bias:
        # The arrival mix moves, not a within-group rate: see the docstring.
        modes.append(f"bias={args.bias:.0%} SEX=1 -> no alert, watch the per-group rates")
    if args.slow:
        modes.append(f"slow batches of {args.batch_size} -> SlowBatchPredictions")
    print(f"  target : {url}")
    print(f"  pool   : {source}")
    print(f"  rate   : {args.rps:g} rps for {args.seconds}s")
    print(f"  modes  : {', '.join(modes) if modes else 'none (normal traffic)'}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)

    pool, source = load_pool(args.pool, size=5_000, rng=nprng)
    stds = {col: _std(pool[col]) for col in DRIFT_COLS}

    url = batch_url(args.url) if args.slow else args.url
    describe(args, source, url)

    tally = Tally()
    interval = 1.0 / args.rps if args.rps > 0 else 0.0
    deadline = time.perf_counter() + args.seconds
    next_send = time.perf_counter()
    next_report = time.perf_counter() + 1.0

    session = requests.Session()
    # Pool size tracks the rate so a slow API creates back-pressure rather than
    # an unbounded pile of in-flight threads.
    workers = max(4, min(64, int(args.rps) + 4))
    pending: list = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_send:
                # Sleep rather than spin. A busy-wait here pegs a core, which at
                # 20 rps competes with the API for the CPU we are measuring.
                time.sleep(min(0.002, next_send - now))
                continue
            next_send += interval

            index = rng.randrange(len(pool))
            record = build_record(pool.iloc[index], index)

            if args.bias and rng.random() < args.bias:
                record[SEX] = 1
            if args.drift:
                apply_drift(record, args.drift, stds)

            if args.slow:
                rows = [
                    build_record(pool.iloc[rng.randrange(len(pool))], i)
                    for i in range(args.batch_size)
                ]
                # The field is `applications`. BatchPredictRequest forbids extras,
                # so the wrong key here is not a partial success -- it is a 422
                # on every request, HighErrorRate firing for the wrong reason,
                # and main() returning 1 to whatever invoked the script.
                payload: object = {"applications": rows}
            elif args.broken and rng.random() < args.broken:
                payload = break_record(record, rng)
            else:
                payload = record

            pending.append(executor.submit(send, session, url, payload, args.timeout))

            # Drain whatever has completed. Keeps `pending` from growing without
            # blocking the send loop on the slowest request in flight.
            still_running = []
            for future in pending:
                if future.done():
                    status, elapsed = future.result()
                    tally.record(status, elapsed)
                else:
                    still_running.append(future)
            pending = still_running

            if not args.quiet and time.perf_counter() >= next_report:
                next_report += 1.0
                print(
                    f"  sent={tally.sent:6d}  ok={tally.ok:6d}  err={tally.errors:5d}"
                    f"  p95={tally.p95() * 1000:7.1f}ms",
                    flush=True,
                )

        for future in pending:
            status, elapsed = future.result()
            tally.record(status, elapsed)

    print()
    print(f"  sent          {tally.sent}")
    print(f"  ok (2xx)      {tally.ok}")
    print(f"  errors        {tally.errors}")
    print(f"  p95 latency   {tally.p95() * 1000:.1f} ms")
    if tally.latencies:
        print(f"  mean latency  {statistics.fmean(tally.latencies) * 1000:.1f} ms")
    for status in sorted(tally.by_status):
        label = "unreachable" if status == 0 else str(status)
        print(f"  status {label:<12} {tally.by_status[status]}")

    # A run that reached nothing is a failure even though every "request" was
    # accounted for -- otherwise `make traffic` exits 0 against a stopped stack.
    if tally.sent and tally.ok == 0 and args.broken < 1.0:
        print("\n! no successful responses -- is the stack up? (make up && make smoke)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
