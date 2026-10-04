# ADR 0008 — CI on GitHub-hosted runners, CD on a self-hosted runner

**Status:** Accepted · **Date:** 2026-10-01 · **Revised:** 2026-10-03 (continuous deployment added) · **Deciders:** DDM501 team

## Context

A reviewer must be able to see a current, trustworthy build status at any time. Deployment, unlike CI, targets one specific machine: the one running the compose stack, the MLflow registry and the model artifacts.

## Decision

`ci.yml` runs on `ubuntu-latest`, with a matrix over Python 3.11 and 3.12. `deploy.yml` runs on a self-hosted runner labelled `credit-risk` on the stack host, after CI succeeds on a push to `main`, and fails unless the API serves the registry's Production version.

## Alternatives rejected

Everything on the self-hosted macOS runner used in the labs. Everything on hosted runners, deploying over SSH or a tunnel. Deploying by hand.

## Rationale

A self-hosted runner is green only while somebody's laptop is awake, so CI stays on hosted runners. The version matrix exists because the serving image runs 3.12 while the Airflow-side virtualenv runs 3.11, so that divergence is tested rather than assumed away. A hosted runner cannot reach a laptop behind NAT; a self-hosted runner on the host needs no inbound access, only outbound HTTPS to GitHub.

## Consequences

Deploys happen only while the host is up (a queued job fails after 24 hours). The runner can run code on the host, so it is used only with a private repository, and the deploy job runs only from `main` after CI, never on `pull_request`. No ARM64 coverage in CI, and the dataset must be downloaded and cached inside the CI workflow.

---

Full trade-off discussion: [`ARCHITECTURE.md` section 4, D8](../../ARCHITECTURE.md#4-decisions-and-trade-offs) · runner setup: [`docs/RUNNER.md`](../RUNNER.md)
