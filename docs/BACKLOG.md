# Backlog

Open work, grouped by owner. Every item names the file it touches and what
breaks today if it is left alone. Ordered within each section by value, not by
effort.

Owners are the four slices in [`CONTRIBUTING.md`](../CONTRIBUTING.md):
**P1** data and pipeline · **P2** model and fairness ·
**P3** serving and explainability · **P4** monitoring and CI.

---

## P1 — data and pipeline

| # | Task | Why it matters | Effort |
|---|---|---|---|
| 4 | **Document what folding the undocumented category codes loses.** `EDUCATION` 0/5/6 and `MARRIAGE` 0 are merged into "other" — 345 and 54 accounts whose original category is now unrecoverable. Add a paragraph to `DATASHEET.md` on whether that was the right call versus keeping a separate level. | Right now the cleaning step looks like a formatting fix. It is a modelling decision with a cost, made silently. | 20 min |

---

## P2 — model and fairness

| # | Task | Why it matters | Effort |
|---|---|---|---|
| 1 | **Compute `SEX × AGE_GROUP` in `summaries_for_attributes`** (about four lines). `MODEL_CARD.md` and `ETHICS.md` both state that intersectional fairness is reported; nothing currently crosses two attributes. | A documented claim the code does not implement. Either deliver it or delete the sentence — delivering is worth more, because a model can clear every single-attribute gate and still be badly skewed on an intersection. | 45 min |
| 2 | **Explain why the interpretable baseline is the less fair model.** Logistic regression fails the fairness gate (`eo_diff` 0.1125) while LightGBM passes at 0.0725. Write the hypothesis into `MODEL_CARD.md`. | This is counter-intuitive and currently unexplained. It also undercuts the usual "simpler models are safer" argument, which somebody will raise. | 30 min |
| 3 | **Fix the artefact name in `MODEL_CARD.md`.** It says the trade-off table is logged as `tradeoff_curve.json`; the file written is `fairness_tradeoff.csv`/`.png`. | A document that names a file which does not exist. | 10 min |
| 4 | **Fix the invariance claim in `README.md`.** It says the test bounds the probability change at 0.02 under the group-aware policy. `tests/model/test_invariance.py` asserts exactly `0.0` on the unawareness path. | The code is stronger than the sentence describing it. | 5 min |
| 5 | **Re-export `data/processed/group_thresholds.json`** through `save_group_thresholds` so the attribute name travels with the numbers. | The mounted file is the old flat `{"1":…,"2":…}` shape, which slips past the attribute-mismatch guard in `mitigation.py`. A marital-status policy could be applied to sex codes without anything complaining. | 5 min |

---

## P3 — serving and explainability

| # | Task | Why it matters | Effort |
|---|---|---|---|
| 2 | **Add `.dockerignore`** — `.venv/`, `data/`, `mlruns/`, `.git/`, `.env`, `htmlcov/`, `__pycache__/`. Measure the build context before and after. | The context is currently about 1.1 GB. Every build ships it to the daemon. | 5 min |
| 3 | **Add `json_schema_extra` examples** to `PredictResponse`, `ExplainResponse` and `FairnessReportResponse`; dump `app.openapi()` to a committed `docs/openapi.json`. | Only the request model has an example, so the generated docs show request shapes and empty response shapes. | 30 min |
| 4 | **Draw the three missing flows in `ARCHITECTURE.md`**: the `/predict/batch` sequence, the degraded-start 503 branch, and the `./data:/app/data:ro` edge that `docker-compose.yml` already calls the weakest seam in the design. | The document names a weakness and then does not show it. | 90 min |
| 5 | **Correct the services table in `ARCHITECTURE.md`**: the object store is `chrislusf/seaweedfs`, and `createbucket` builds from `Dockerfile.mlflow`. | Stale after the MinIO removal. | 10 min |

---

## P4 — monitoring and CI

| # | Task | Why it matters | Effort |
|---|---|---|---|
| 1 | **Add a CI job running `pytest tests/model -m "not needs_data"`.** | Every model-quality assertion — the PR-AUC floor, ROC-AUC, Brier, monotonicity, protected-attribute invariance — is currently deselected by the `not slow` filter. The exclusion saves 6.8 seconds and means a model regression passes CI. | 20 min |
| 2 | **Repoint the Grafana *High-risk share* panel thresholds.** They are hard-coded at `0.2212 ± 0.10` — the label prevalence — while the panel description says it reads against the baseline gauge. The measured baseline is 0.124, which sits 0.003 inside green. | The panel is one small shift away from showing green during a real drift event. | 10 min |
| 3 | **Fix the demo table in `README.md`.** It says `FeatureDriftHigh` fires. The rule has `for: 20m` and `make drift` runs 300 s, so the reachable state is `pending` — which is what was measured. Same for `FairnessGapExceeded` at `for: 15m`. | Claims an alert fires when the scenario cannot make it fire. | 10 min |
| 4 | **Update `ARCHITECTURE.md` §5.** The `HighRiskShareShift` row describes a rule that no longer exists in that form; `SlowBatchPredictions` is missing (8 listed, 9 exist); `credit_baseline_high_risk_share` is missing (9 metrics listed, 10 exist). | The observability section no longer matches the observability. | 15 min |
| 5 | **Add `pip-audit` as a CI job and a `.github/dependabot.yml`** for `pip` and `github-actions`. | 18 pinned runtime dependencies and nothing watching them for advisories. | 20 min |
| 6 | **Fix the ADR citation in the `ci.yml` header** — it cites ADR 0007 (LightGBM) for a decision recorded in ADR 0008 (GitHub-hosted runners). | One word, but it reads as a comment nobody checked. | 2 min |

---

## Done

Derived `requirements.txt` from `pyproject.toml`; CI alert-rule count derived
from the rule files rather than hard-coded; Python 3.11/3.12 matrix in the test
job; worked request examples under `docs/examples/`; every documented metric
reconciled against the registered model; MinIO replaced with SeaweedFS after its
community images stopped being pullable. `REQUIREMENTS.md`; a `DagBag` test for the
DAG plus `make dag-test`; `disallow_untyped_defs = true` in mypy. The capacity
threshold is tagged onto each model version at registration and the API decides
at it, with a `tag-threshold` backfill for versions registered earlier (was P3 #1;
version 2 still needs the backfill run against the live registry).
