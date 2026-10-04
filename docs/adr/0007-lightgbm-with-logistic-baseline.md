# ADR 0007 — LightGBM as the candidate, logistic regression as the baseline

**Status:** Accepted · **Date:** 2026-10-01 · **Deciders:** DDM501 team

## Context

Tabular data, 20,000 training rows, and a hard requirement for per-prediction explanation.

## Decision

Train both families on every run and compare them in MLflow.

## Alternatives rejected

LightGBM alone; or a neural network.

## Rationale

The baseline quantifies what gradient boosting actually buys and turns the interpretability trade-off into a number. LightGBM is directly supported by `shap.TreeExplainer`, which computes exact Shapley values for trees instead of the sampling approximation `KernelExplainer` would need. Deep learning on 30k tabular rows loses to gradient boosting and makes explanation an order of magnitude slower.

## Consequences

Two model families to maintain, evaluate and test.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D7](../../ARCHITECTURE.md#4-decisions-and-trade-offs)
