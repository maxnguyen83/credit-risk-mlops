#!/usr/bin/env python3
"""Saturation test for the credit-risk API: what rate can it actually sustain?

`traffic.py` is an open-loop generator -- it sends at whatever `--rps` you name
and reports what came back. That answers "does the system behave under this
load". It cannot answer "what load can this system take", because the rate is
an input rather than a result, and its drain loop and `min(64, rps + 4)` worker
cap mean asking it for more than the API can serve just piles up futures.

This script closes the loop. It holds a fixed number of connections, each
sending the next request the moment the previous one returns, and steps that
number up a ladder until p95 latency breaches the SLO. The highest rate reached
while still inside the budget is the throughput number in the spec's system
metrics table -- previously a target with no instrument behind it.

    scripts/load_test.py                       full ladder, 10s per step
    scripts/load_test.py --seconds 30          longer steps, tighter numbers
    scripts/load_test.py --strict              exit 1 if the target is missed

Two honesty notes that belong on any number this produces:

  * Client and server share this laptop's CPU. The generator is deliberately
    cheap -- payloads are built and serialised to bytes once, up front, so the
    measured loop does nothing but socket work -- but a result here is still a
    floor on what the API could do with a load generator of its own.
  * The figure is for the deployed worker count. The spec's target is written
    for one worker; check `docker compose` before quoting it against anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Same trick, same reason as traffic.py: this runs out of a fresh clone.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import traffic  # noqa: E402

# Powers of two, not a linear ramp. Throughput climbs, plateaus, and then
# latency knees; a linear ramp spends most of its steps on the plateau
# re-measuring the same number, and the knee is the only interesting part.
DEFAULT_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

# The p95 the spec holds /predict to. A step that breaches it has found the
# knee, and every step above it is measuring a queue rather than a service.
DEFAULT_P95_BUDGET = 0.100

DEFAULT_TARGET_RPS = 200.0

_local = threading.local()


def _session() -> requests.Session:
    """One Session per thread. Sharing one across threads is not safe."""
    existing = getattr(_local, "session", None)
    if existing is None:
        existing = requests.Session()
        # Without this the pool defaults to 10 and the 32-connection step
        # measures urllib3 discarding and reopening sockets, not the API.
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=256)
        existing.mount("http://", adapter)
        existing.mount("https://", adapter)
        _local.session = existing
    return existing


@dataclass
class Step:
    """What one rung of the ladder measured."""

    concurrency: int
    seconds: float = 0.0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return len(self.latencies) + self.errors

    @property
    def rps(self) -> float:
        return self.completed / self.seconds if self.seconds > 0 else 0.0

    def quantile(self, q: float) -> float:
        """Nearest-rank, matching traffic.py: never reports a latency nobody saw."""
        if not self.latencies:
            return float("inf")
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    def healthy(self, budget: float) -> bool:
        """A step counts only if it was fast AND correct."""
        return self.errors == 0 and self.quantile(0.95) <= budget


def build_payloads(count: int, pool: pd.DataFrame, rng: np.random.Generator) -> list[bytes]:
    """Pre-serialise the request bodies so the measured loop only does I/O.

    json.dumps inside the send loop is client CPU charged to the server's
    latency. At 200 rps across 64 threads that is not a rounding error, and it
    is the classic way a load test understates the system it is measuring.
    """
    indices = rng.integers(0, len(pool), size=count)
    return [
        json.dumps(traffic.build_record(pool.iloc[int(index)], int(index))).encode()
        for index in indices
    ]


def _drive(
    url: str,
    payloads: list[bytes],
    deadline: float,
    timeout: float,
    sink: list[tuple[list[float], int]],
    slot: int,
) -> None:
    """Send back-to-back until the deadline, then publish this thread's totals.

    Each worker accumulates into locals and writes one tuple into its own slot
    of `sink`. Incrementing a shared counter instead would be a read-modify-
    write with no lock, which under load drops errors -- and a load test that
    undercounts failures reports a throughput the service cannot actually serve.
    """
    session = _session()
    headers = {"Content-Type": "application/json"}
    latencies: list[float] = []
    errors = 0
    index = 0
    while time.perf_counter() < deadline:
        body = payloads[index % len(payloads)]
        index += 1
        started = time.perf_counter()
        try:
            response = session.post(url, data=body, headers=headers, timeout=timeout)
        except requests.RequestException:
            errors += 1
            continue
        elapsed = time.perf_counter() - started
        if 200 <= response.status_code < 300:
            latencies.append(elapsed)
        else:
            errors += 1
    sink[slot] = (latencies, errors)


def run_step(
    url: str, concurrency: int, seconds: float, payloads: list[bytes], timeout: float
) -> Step:
    """Hold `concurrency` connections open for `seconds` and report what happened."""
    sink: list[tuple[list[float], int]] = [([], 0)] * concurrency
    deadline = time.perf_counter() + seconds
    started = time.perf_counter()

    threads = [
        threading.Thread(
            target=_drive, args=(url, payloads, deadline, timeout, sink, slot), daemon=True
        )
        for slot in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        # Generous slack over the deadline: a thread parked in a socket read
        # has to come back before its samples are counted, and joining with no
        # timeout at all is how a wedged API hangs the whole test.
        thread.join(timeout=seconds + timeout + 5.0)

    step = Step(concurrency=concurrency, seconds=time.perf_counter() - started)
    for latencies, errors in sink:
        step.latencies.extend(latencies)
        step.errors += errors
    return step


def warm_up(url: str, payloads: list[bytes], timeout: float) -> None:
    """Score a few records before measuring anything.

    The first request through a fresh process pays for lazy imports, the first
    pandas frame and the first predict call on a cold estimator. Measured, that
    lands entirely in the 1-connection step and makes the baseline look worse
    than every step above it.
    """
    session = _session()
    headers = {"Content-Type": "application/json"}
    for body in payloads[:20]:
        try:
            session.post(url, data=body, headers=headers, timeout=timeout)
        except requests.RequestException:
            return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the sustained request rate the credit-risk API holds inside its p95 SLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=traffic.DEFAULT_URL, help="prediction endpoint")
    parser.add_argument("--seconds", type=float, default=10.0, help="measured seconds per step")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(DEFAULT_LADDER),
        metavar="N",
        help="the ladder of open connections to step through",
    )
    parser.add_argument(
        "--p95-budget",
        type=float,
        default=DEFAULT_P95_BUDGET,
        metavar="SECONDS",
        help="p95 above this ends the ramp; the SLO for /predict",
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        default=DEFAULT_TARGET_RPS,
        help="the throughput target the verdict is read against",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when the target is not met (for CI; off by default so a laptop run still reports)",
    )
    parser.add_argument("--pool", type=Path, default=None, help="parquet of records to replay")
    parser.add_argument(
        "--payloads", type=int, default=512, help="distinct request bodies to cycle through"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout")
    parser.add_argument("--seed", type=int, default=7, help="seed for record selection")
    return parser.parse_args(argv)


def report(step: Step) -> str:
    return (
        f"  conn={step.concurrency:>4}  rps={step.rps:8.1f}  "
        f"p50={step.quantile(0.50) * 1000:7.1f}ms  p95={step.quantile(0.95) * 1000:7.1f}ms  "
        f"p99={step.quantile(0.99) * 1000:7.1f}ms  errors={step.errors}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    pool, source = traffic.load_pool(args.pool, size=5_000, rng=rng)
    payloads = build_payloads(args.payloads, pool, rng)

    print(f"  target    : {args.url}")
    print(f"  pool      : {source}")
    print(f"  ladder    : {args.concurrency}")
    print(f"  p95 budget: {args.p95_budget * 1000:.0f} ms over {args.seconds:g}s steps")
    print()

    warm_up(args.url, payloads, args.timeout)

    steps: list[Step] = []
    for concurrency in args.concurrency:
        step = run_step(args.url, concurrency, args.seconds, payloads, args.timeout)
        steps.append(step)
        print(report(step), flush=True)

        if step.completed == 0:
            print("\n! nothing completed -- is the stack up? (make up && make smoke)")
            return 1
        if step.errors:
            print(f"\n  stopping: {step.errors} request(s) failed at {concurrency} connections")
            break
        if step.quantile(0.95) > args.p95_budget:
            print(f"\n  stopping: p95 breached the {args.p95_budget * 1000:.0f}ms budget")
            break

    healthy = [step for step in steps if step.healthy(args.p95_budget)]
    print()
    if not healthy:
        # Reported rather than crashed: "not even one connection fits the SLO"
        # is a result, and it is the one worth putting in front of people.
        print(f"  no step stayed inside the {args.p95_budget * 1000:.0f}ms p95 budget")
        best_effort = max(steps, key=lambda s: s.rps)
        print(
            f"  best observed rate: {best_effort.rps:.1f} rps at p95 "
            f"{best_effort.quantile(0.95) * 1000:.1f}ms"
        )
        return 1 if args.strict else 0

    best = max(healthy, key=lambda step: step.rps)
    mean = statistics.fmean(best.latencies)
    print(f"  sustained     {best.rps:.1f} rps")
    print(f"  at            {best.concurrency} concurrent connections")
    print(
        f"  p95           {best.quantile(0.95) * 1000:.1f} ms (budget {args.p95_budget * 1000:.0f} ms)"
    )
    print(f"  mean          {mean * 1000:.1f} ms")
    print(f"  samples       {len(best.latencies)}")
    print()

    met = best.rps >= args.target_rps
    verdict = "MET" if met else "NOT MET"
    print(f"  target >= {args.target_rps:g} rps: {verdict}")
    if not met:
        print("  (client and server share this machine; re-run against a remote API before")
        print("   concluding the service is the bottleneck)")
    return 1 if args.strict and not met else 0


if __name__ == "__main__":
    raise SystemExit(main())
