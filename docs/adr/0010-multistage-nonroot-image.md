# ADR 0010 — Multi-stage Dockerfile with a non-root runtime

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

The serving image needs a compiler toolchain to install its dependencies but not to answer requests.

## Decision

Builder stage installs into `/opt/venv`; runtime stage copies the venv, installs `libgomp1`, and runs as uid 10001.

## Alternatives rejected

A single-stage image running as root.

## Rationale

Shipping `build-essential` to production means a larger image and a larger attack surface, neither of which serves an HTTP request. `libgomp1` is required because LightGBM links against OpenMP; without it the import fails with a linker error that reads like a missing Python package.

## Consequences

A more complex Dockerfile and a longer cold build.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D10](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
