# ADR 0005 — Airflow standalone in one container, not CeleryExecutor

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

The pipeline needs scheduling, retries, backfill and a run history UI.

## Decision

`apache/airflow:2.8.4` in `standalone` mode with `SequentialExecutor` and a SQLite metadata DB.

## Alternatives rejected

CeleryExecutor (webserver + scheduler + worker + redis + metadata DB); or no orchestrator at all.

## Rationale

Standalone gives all four capabilities for one container. CeleryExecutor gives distributed workers, which a handful of sequential tasks over 30,000 rows will never need. Dropping the orchestrator entirely would mean hand-rolling retry, backoff, partial-rerun and run history in a shell script, which is how a pipeline ends up with no error handling at all.

## Consequences

No task parallelism and no horizontal scaling. Both irrelevant at this size.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D5](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
