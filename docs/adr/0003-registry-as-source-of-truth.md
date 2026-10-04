# ADR 0003 — MLflow Registry is the source of truth for the served model

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

The API must serve a specific, identifiable model version, and promoting a new one must not require a code deploy.

## Decision

The API resolves `models:/credit-risk@champion` at startup and pulls the artifact from MinIO via boto3. When no version carries the alias it falls back to `models:/credit-risk/Production`, and `/health` reports which of the two answered (`model_ref`).

Promotion moves the `champion` alias and also sets the Production stage, for anything that still reads stages. Registration promotes a gated candidate only when the registry has no champion or the candidate's held-out PR-AUC is no more than `PROMOTION_PR_AUC_TOLERANCE` below the champion's on the same split; otherwise it becomes the `challenger` (Staging) and a person promotes it with `python -m credit_risk.models.registry set-champion --version N`.

## Alternatives rejected

Baking `model.pkl` into the serving image.

## Rationale

Promotion becomes a registry operation rather than a rebuild. `/health` can report which model is serving, which is the first question asked during an incident. This is the difference between having Docker and having MLOps.

## Consequences

The API depends on MLflow at startup. Mitigated with `depends_on: service_healthy` and an explicit degraded state (`/health` reports `model_loaded: false`, `/predict` returns 503) instead of a crash loop.

Aliases rather than stages because stages are deprecated in MLflow and an alias says what a version is for (`champion`, `challenger`) instead of a fixed lifecycle position. The stage fallback exists for the registry this project already has: version 2 was promoted before the alias was used.

A promotion reaches traffic only after the API restarts; `deploy.yml` restarts and verifies. There is deliberately no reload endpoint.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D3](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
