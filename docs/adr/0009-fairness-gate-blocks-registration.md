# ADR 0009 — Fairness thresholds block model registration

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

A fairness result that lives in a notebook protects only the model that was examined. Every model trained afterwards is unguarded, including the ones retrained by somebody who never read the notebook.

## Decision

`demographic_parity_difference <= 0.05` and `equalized_odds_difference <= 0.08` are evaluated in the pipeline. A candidate that breaches either is not registered; the refusal is logged to MLflow as a run tag with its reason.

## Alternatives rejected

Measuring fairness in a notebook and writing it up.

## Rationale

A gate protects every future model, including one retrained by somebody who never read the report. That is the difference between an audit and a control.

## Consequences

A genuinely better-performing model can be refused. That is the intended behaviour, and refusals are recorded so the trade-off is visible rather than silent.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D9](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
