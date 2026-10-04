# ADR 0004 — Postgres and an S3 object store behind MLflow, not SQLite and a folder

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

MLflow needs a backend store for runs and a store for artifacts, reachable from more than one container.

## Decision

`postgres:16-alpine` for the backend store and SeaweedFS for artifacts over the S3 API.

SeaweedFS rather than MinIO: MinIO withdrew its community images, so
`minio/minio` denies anonymous pulls on Docker Hub and every quay.io tag now
requires authentication. CI found it on a clean runner; a developer machine
with the image cached never would. The S3 API is identical, so nothing above
the service definition changed.

## Alternatives rejected

`sqlite:///mlflow.db` plus a local artifact directory.

## Rationale

SQLite serialises writers and does not survive concurrent pipeline runs; a local folder cannot be read by another container without a shared mount. Both are right for a tutorial and wrong for a system.

## Consequences

Two extra containers and roughly 300 MB of RAM. It also forces the `service_completed_successfully` dependency on bucket creation, which is the correct lesson.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D4](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
