# ADR 0006 — Two Python interpreters inside the Airflow image

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

Airflow 2.8.4 pins pandas and numpy through its constraints file. The project pins different versions and additionally needs LightGBM, SHAP and LIME.

## Decision

Airflow keeps its own constrained environment. The project is installed into a second virtualenv at `/opt/credit-risk/venv`, and the DAG shells out to it through `CREDIT_RISK_PYTHON`.

## Alternatives rejected

Installing everything into one interpreter and letting pip resolve it.

## Rationale

One interpreter means one dependency set silently loses, and the symptom appears later as a wrong number rather than an install error. Two interpreters make the isolation visible.

## Consequences

A larger image, and a subprocess boundary instead of a Python import — the DAG sees exit codes and stdout rather than exceptions. Child output is piped into the task log to keep failures debuggable.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D6](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
