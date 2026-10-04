# ADR 0002 — Batch scoring, not streaming

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

The business action is a monthly intervention list of ~3,000 accounts.

## Decision

Nightly batch scoring plus on-demand single lookups over HTTP.

## Alternatives rejected

Kafka or Flink stream processing.

## Rationale

Nothing in the use case has a sub-second requirement. Streaming would add two or three containers plus a partitioning and retention policy, consumer-lag monitoring and an at-least-once delivery story -- ongoing operational cost bought to solve a latency problem this system does not have.

## Consequences

A limit change today is not reflected until tonight's run. For this use case that is correct behaviour.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D2](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
