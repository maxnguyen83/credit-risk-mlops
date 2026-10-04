# ADR 0001 — Modular monolith for the API, not microservices

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

One model, one team of four, four weeks. We need clear component responsibilities.

## Decision

One FastAPI service with enforced module boundaries: `serving/`, `explain/`, `features/`, `fairness/`.

## Alternatives rejected

Separate predict / explain / fairness services.

## Rationale

No network hops, no service discovery, no distributed tracing, no extra health checks. Clear component responsibilities come from module boundaries enforced in the package layout and the import graph, not from putting a network between them.

## Consequences

`/explain` and `/predict` share a process, so a burst of SHAP traffic degrades scoring latency. Accepted because explanation is human-in-the-loop and scoring is a nightly batch, so the peaks do not coincide. The boundary is drawn so `explain/` can be extracted without touching `routes.py`.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D1](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
