# ADR 0003 — MLflow Registry is the source of truth for the served model

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

The API must serve a specific, identifiable model version, and promoting a new one must not require a code deploy.

## Decision

The API resolves `models:/credit-risk/Production` at startup and pulls the artifact from MinIO via boto3.

## Alternatives rejected

Baking `model.pkl` into the serving image.

## Rationale

Promotion becomes a registry operation rather than a rebuild. `/health` can report which model is serving, which is the first question asked during an incident. This is the difference between having Docker and having MLOps.

## Consequences

The API depends on MLflow at startup. Mitigated with `depends_on: service_healthy` and an explicit degraded state (`/health` reports `model_loaded: false`, `/predict` returns 503) instead of a crash loop.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D3](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
