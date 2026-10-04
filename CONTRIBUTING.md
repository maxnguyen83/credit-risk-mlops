# Contributing

Team roles, working agreements, and definition of done for the DDM501 final
project.

---

## 1. Team and ownership

The system is split into **four vertical slices**. Each person owns a slice end
to end — its code, its tests, and its section of the documentation.

One rule shapes the split more than any other: **every member owns a piece of
Responsible AI.** Fairness, explainability and ethics are not a module somebody
bolts on at the end — they are properties of the data, the model, the API and
the monitoring, and each one fails differently. A team where one person "did the
fairness part" has three people who cannot say whether their own component is
fair, which is the same as not knowing whether it works.

| | Member | Branch | Primary ownership | Responsible-AI ownership |
|---|---|---|---|---|
| **P1** | `maxnguyen83` | `p1/core`, `p1/docs` | Data and pipeline — download, validation, cleaning, splits, feature engineering, the Airflow DAG | **Bias in the data**: base rates by group, representation, `DATASHEET.md` |
| **P2** | `Ducmanh2212` | `p2/core`, `p2/docs` | Model and tracking — training, hyperparameter search, cross-validation, evaluation, MLflow, the registry gate | **Fairness measurement and mitigation**: fairlearn metrics, three mitigation strategies, the trade-off curve, `MODEL_CARD.md` |
| **P3** | `hieunt-fsb-ai` | `p3/core`, `p3/docs` | Serving and deployment — FastAPI, OpenAPI, Dockerfiles, docker-compose | **Explainability**: SHAP, LIME, the `/explain` endpoint, the adverse-action notice draft |
| **P4** | `thientd2609` | `p4/core`, `p4/docs` | Monitoring and CI/CD — Prometheus, Grafana, Alertmanager, GitHub Actions, the traffic generator | **Fairness in production**: `credit_selection_rate`, the `FairnessGapExceeded` alert, the fairness dashboard |

### Files each member owns

Code and tests live on the member's `core` branch, documentation on `docs`.

```
P1  core  src/credit_risk/data/**          src/credit_risk/features/**
          dags/**                          Dockerfile.airflow
          tests/data_quality/**            tests/conftest.py
          tests/unit/test_dag.py  test_features.py  test_validate.py
    docs  ARCHITECTURE.md  REQUIREMENTS.md  DATASHEET.md
          docs/BACKLOG.md  docs/adr/README.md  docs/adr/0002, 0005, 0006

P2  core  src/credit_risk/models/**        src/credit_risk/fairness/**
          Dockerfile.mlflow                tests/model/**
          tests/unit/test_evaluate.py  test_fairness.py  test_pipeline_cli.py
          tests/unit/test_registry_*.py
    docs  MODEL_CARD.md  ETHICS.md  docs/adr/0003, 0007, 0009

P3  core  src/credit_risk/serving/main.py  routes.py  models.py  model_loader.py
          src/credit_risk/explain/**       tests/integration/**
          tests/unit/test_explain.py       Dockerfile  .dockerignore
          docker-compose.yml               docs/openapi.json  docs/examples/**
    docs  README.md  docs/adr/0001, 0004, 0010

P4  core  src/credit_risk/serving/metrics.py
          monitoring/**                    .github/**
          scripts/** except gen_requirements.py
          tests/unit/test_metrics.py  test_verify_deploy.py
          docs/ALERTING.md                 .gitattributes
    docs  CONTRIBUTING.md  docs/RUNNER.md  docs/adr/0008
```

Shared, on `main` before any member branch, edited by whoever needs to and
always reviewed by another member: `src/credit_risk/schema.py`,
`src/credit_risk/config.py`, `pyproject.toml`, `requirements*.txt` with
`scripts/gen_requirements.py`, `Makefile`, `.env.example`, `.gitignore` and
the package skeleton. So far `Makefile` has changed on `p1/core`, `config.py`
on `p2/core`, and `.env.example` on both `p2/core` and `p3/core`.

`ETHICS.md` is written by **all four** — one section each — and committed on
`p2/docs`. Everyone will be asked about it.

---

## 2. Cross-module contracts

Two interfaces are shared between slices. Changing either one breaks somebody
else's code, so changes go through a PR that the other owner reviews.

**`features/build.py` (P1) is the single source of feature construction.** Both
`models/train.py` and `serving/routes.py` import it. This is the architectural
defence against train/serve skew: there is no second code path that could drift.
`tests/unit/test_features.py` asserts the two entry points produce byte-identical
output for the same record.

**`serving/metrics.py` (P4) exposes a fixed set of names** that
`serving/routes.py` (P3) calls. Renaming a metric silently breaks the alert rules
that query it, so the metric names are also asserted in
`tests/integration/test_api.py` and in `scripts/smoke.sh`.

---

## 3. Git workflow

```
main             protected; always green; never pushed to directly
  └── p1/core        ingestion, validation, splits, features, the DAG
  └── p1/docs        P1's documentation
  └── p2/core        training, evaluation, registry gate, fairness
  └── p2/docs        P2's documentation
  └── p3/core        API, SHAP/LIME, Dockerfiles, compose
  └── p3/docs        P3's documentation
  └── p4/core        metrics, alerts, dashboards, CI
  └── p4/docs        P4's documentation
```

**Branches.** Two per member: `p{1..4}/core` for code and tests,
`p{1..4}/docs` for documentation. They merge into `main` in member order:
p1, then p2, p3 and p4.

**Commits.** Conventional Commits, scoped to the area:

```
feat(data): fold undocumented EDUCATION codes into the "other" bucket
fix(serving): guard utilisation ratio when LIMIT_BAL is zero
test(model): assert protected-attribute invariance under group thresholds
docs(arch): record the two-interpreter decision for the Airflow image
chore(ci): cache the UCI download between runs
```

Write the body when the *why* is not obvious from the subject. A commit message
that only repeats the diff is a wasted opportunity.

**Pull requests.**

- Every change reaches `main` through a PR. No direct pushes.
- **At least one approving review from a different member.** This is not
  bureaucracy: it is how four people end up able to answer questions about the
  whole system, and it leaves a trail showing the change was looked at by
  somebody other than its author.
- CI must be green. No merging red.
- The PR description states what changed and why, and names anything the
  reviewer should look at sceptically.

**Everybody commits under their own GitHub account.** The history is the record
of who understands what, and it is the first thing anyone reads when something
breaks six months from now. Committing on somebody else's behalf destroys that
record. If you pair, use `Co-authored-by:` trailers.

---

## 4. Definition of done

A task is done when **all** of these hold. "It works on my machine" is not on the
list.

- [ ] The code runs, and you have seen it run — not merely compiled.
- [ ] Tests cover the branches you added, including the failure paths.
- [ ] `make lint` clean (`ruff check` and `ruff format --check`).
- [ ] `make typecheck` clean (`mypy`).
- [ ] `make test` passes and total coverage stays at or above 80%.
- [ ] Public functions have type hints and a docstring saying what they are for.
- [ ] Comments explain **why**, not what. If a comment restates the code, delete it.
- [ ] No secrets, tokens, passwords or real personal data in code, tests,
      fixtures, logs or commit history.
- [ ] Documentation updated: your section of `README.md` or `ARCHITECTURE.md`,
      plus an ADR under `docs/adr/` if you made a decision somebody could
      reasonably have made differently.
- [ ] A reviewer from another slice has approved it.

---

## 5. Local setup

```bash
git clone <repo-url> && cd credit-risk-mlops
cp .env.example .env          # then edit; .env is gitignored and stays that way

make install                  # virtualenv + package + dev tools
make data                     # download, clean and split the dataset
make test                     # the full suite with the coverage gate

make up                       # the whole stack
make ps                       # wait for healthy
make smoke                    # prove the running stack answers correctly
```

On macOS, LightGBM needs OpenMP, which is not installed by default:

```bash
brew install libomp
```

Without it the import fails with a `Library not loaded: @rpath/libomp.dylib`
error that looks like a missing Python package but is not one. Linux and the
Docker images are unaffected — `libgomp1` is installed in the runtime stage.

---

## 6. Habits that wreck a project like this one

| Habit | What it actually costs |
|---|---|
| Letting coverage drift and fixing it in the last week | Tests written to raise a number assert nothing. You end up with a green gate and no safety net. |
| One person owning all of Responsible AI | Three people who cannot say whether their own component is fair, and one bottleneck |
| Merging with red CI | The signal stops meaning anything. Once a red build is normal, nobody reads it. |
| Decisions made in chat and never written down | The alternative you rejected comes back in three weeks and nobody remembers why it was rejected |
| Committing `.env`, a token, or real personal data | Deleting the file does not remove it from the history. The credential has to be rotated. |
