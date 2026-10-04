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
| 2 | **Explain why the interpretable baseline is the less fair model.** Logistic regression fails the fairness gate (`eo_diff` 0.1125) while LightGBM passes at 0.0725. Write the hypothesis into `MODEL_CARD.md`. | This is counter-intuitive and currently unexplained. It also undercuts the usual "simpler models are safer" argument, which somebody will raise. | 30 min |

---

## P3 — serving and explainability

| # | Task | Why it matters | Effort |
|---|---|---|---|
| 4 | **Draw the three missing flows in `ARCHITECTURE.md`**: the `/predict/batch` sequence, the degraded-start 503 branch, and the `./data:/app/data:ro` edge that `docker-compose.yml` already calls the weakest seam in the design. | The document names a weakness and then does not show it. | 90 min |

---

## P4 — monitoring and CI

Nothing open: all six items are under *Done*.

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
version 2 still needs the backfill run against the live registry). Pipeline
hardening (P1): rows failing an error-level check are quarantined instead of
trained on; the download retries transient failures and exits 75 so Airflow
retries the task; the raw parquet is trusted only with a sidecar that describes
it; `validate_raw` keeps its report in XCom when it fails; a failed run posts
`PipelineTaskFailed` to Alertmanager.

Model and fairness (P2): `summaries_for_attributes` reports `SEX × AGE_GROUP` as
one joint attribute, `SEX_x_AGE_GROUP` (`INTERSECTIONS` in `fairness/metrics.py`;
`tests/unit/test_fairness.py::test_sex_by_age_group_is_reported_as_one_joint_attribute`,
and the run's `by_SEX_x_AGE_GROUP_*` metrics in
`tests/model/test_performance.py::test_train_all_logs_both_families_and_registers_through_the_gate`)
(was P2 #1). `MODEL_CARD.md` names the trade-off artefacts `models/train.py`
logs, `fairness_tradeoff.csv` and `fairness_tradeoff.png` (was P2 #3).
`README.md` states the invariance bound as exactly 0.0 on the mitigated path,
which is what
`tests/model/test_invariance.py::test_flipping_sex_cannot_move_the_mitigated_probability_at_all`
asserts (was P2 #4). `save_group_thresholds` (`fairness/mitigation.py`) writes
`{"attribute": …, "thresholds": …}`, and every training run whose winner passes
the gate exports `group_thresholds.json` through it (`models/train.py`);
`load_group_thresholds` returns no cutoffs when the recorded attribute is not
the one asked for
(`tests/unit/test_fairness.py::test_group_thresholds_round_trip_through_the_exported_file`,
`::test_thresholds_fitted_on_another_attribute_are_refused_not_misapplied`). A
flat file written before the attribute was recorded still loads, without that
check (`::test_a_flat_thresholds_file_still_loads`) (was P2 #5).

Serving (P3): `.dockerignore` excludes every path the item listed, plus the
Alertmanager secrets; the context size after it was not recorded (was P3 #2).
`PredictResponse`, `ExplainResponse`, `FairnessReportResponse` and
`HealthResponse` carry `json_schema_extra` examples (`serving/models.py`), and
`make openapi` writes the committed `docs/openapi.json`
(`tests/integration/test_api.py::test_openapi_publishes_a_response_example_for_each_documented_answer`,
`::test_the_committed_openapi_document_matches_the_app`) (was P3 #3). The
services table in `ARCHITECTURE.md` lists `chrislusf/seaweedfs` and
`createbucket` built from `Dockerfile.mlflow` (was P3 #5).

Monitoring and CI (P4): the `model` job in `ci.yml` runs
`pytest tests/model -m "not needs_data"` (was P4 #1). In
`monitoring/grafana/dashboards/model-behaviour.json` the *High-risk share* stat
is uncoloured, and *Distance from this model's own baseline* colours
`abs(credit_high_risk_share - credit_baseline_high_risk_share)` at 0.05 and
0.10, as `HighRiskShareShift` does; no threshold sits at 0.2212 any more (was
P4 #2). The demo table in `README.md` shows `FeatureDriftHigh` as **pending**
and `FairnessGapExceeded` as not firing, each with its `for:` (was P4 #3).
`ARCHITECTURE.md` §5 lists the 10 metrics in `serving/metrics.py` and the 9
rules in `monitoring/prometheus/alerts/`, `HighRiskShareShift` measured against
`credit_baseline_high_risk_share` (was P4 #4). The `dependency audit` job runs
`pip-audit -r requirements.txt`, report-only, and `.github/dependabot.yml`
opens weekly PRs for `pip` and `github-actions` (was P4 #5). The `ci.yml`
header cites `docs/adr/0008` (was P4 #6).
