# Requirements

Credit Default Early-Warning System — DDM501 final project.

This document states what the system must do (functional requirements), how
well it must do it (non-functional requirements), how much each one matters,
and **what checks it**. Every threshold below already exists in code, an alert
rule or a test; this file puts them in one place. Where a requirement has no
automated check, the "Verified by" column says so.

---

## 1. Who uses the system, and through what

The business context is in [`ARCHITECTURE.md`](ARCHITECTURE.md) §1: a risk team
that can act on about 10% of a ~30,000-account portfolio per month needs to know
which accounts to contact.

| Persona | What they need | Interface |
|---|---|---|
| **Core-banking batch job** | Score the portfolio overnight to build the intervention list | `POST /api/v1/predict/batch` (up to 1,000 accounts per call) |
| **Risk officer** (collections team) | Look up one account, see its risk and why it was flagged, before contacting the holder | `POST /api/v1/predict`, `POST /api/v1/explain` |
| **Compliance reviewer** | Check that the serving model treats protected groups within the policy limits; see why a candidate was refused | `GET /api/v1/fairness/report`; Grafana *Fairness Monitor*; `FairnessGapExceeded` alert; MLflow fairness metrics and `fairness_gate` tags per run |
| **On-call engineer** | Know whether the service is up, which model is serving, and what is failing | `GET /health`, `GET /version`, `GET /metrics` through Prometheus, Alertmanager and Grafana |
| **ML engineer** | Retrain, compare candidates, promote a model | Airflow DAG `credit_risk_pipeline` (`:18081`, `make dag-test`), MLflow tracking and registry (`:15020`) |
| **Account holder** | Not a user of the system, but affected by it. A contact is explained with the `top_reasons` from `/explain`, relayed by the risk officer | none directly |

---

## 2. Priority

| Priority | Meaning |
|---|---|
| **P0** | The system is not fit for purpose without it. |
| **P1** | Required to operate, monitor or review the system safely. |
| **P2** | A target or a convenience. Tracked, not enforced. |

**Where checks run.** "CI `test`" is the `test` job in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), which runs every test not
marked `slow` or `needs_data` on Python 3.11 and 3.12. Tests marked **(slow)** run
in CI's `model` job, or in `data quality` when they need the dataset, and in
`make test`. **(Airflow image)** means the test skips anywhere Airflow is
not installed, CI included. Alert names refer to `monitoring/prometheus/alerts/`.

---

## 3. Functional requirements

### 3.1 Training pipeline — one DAG task each

| ID | Requirement | Pri | DAG task / code | Verified by |
|---|---|---|---|---|
| FR-1 | Download the UCI archive, land it as parquet and record its SHA-256 in a sidecar written before the parquet; reuse an existing parquet only when its sidecar describes it. Retry connection errors, timeouts, truncated bodies and HTTP 408, 425, 429, 500, 502, 503 and 504 with backoff; when the archive stays down, exit 75 so Airflow retries the task, and fail every other error (other HTTP errors, TLS and proxy errors) without a retry. | P0 | `download_raw` · `data/download.py` | `tests/data_quality/test_raw.py::TestIngestion` (incl. `::test_a_parquet_without_its_sidecar_is_downloaded_again`, `::test_a_crash_while_replacing_an_old_download_is_downloaded_again`), `::TestTransientFailures` (CI `test`); `tests/unit/test_dag.py::test_a_transient_exit_raises_an_exception_airflow_retries` (Airflow image) |
| FR-2 | Validate schema, ranges and nulls. Fail the run when more than 5% of rows break an error-severity check; undocumented category codes warn and do not fail. Keep the report in XCom (`validation_report`) whether the task passes or fails. | P0 | `validate_raw` · `data/validate.py` | `tests/unit/test_validate.py::test_assert_ok_raises_above_the_threshold`, `::test_undocumented_codes_warn_but_do_not_fail`, `::test_cli_exits_non_zero_but_still_prints_the_evidence` (CI `test`); `tests/unit/test_dag.py::test_a_failed_validation_still_leaves_its_report_in_xcom` (Airflow image) |
| FR-3 | Fold `EDUCATION` 0/5/6 and `MARRIAGE` 0 into "other", add `AGE_GROUP`, and cut six 5,000-row batches in ID order (1–4 train, 5 test, 6 serving pool), with a manifest of file hashes. Quarantine every row that failed an error-severity check of the raw or the cleaned frame (5% of the file at most, both stages together) to `quarantine.parquet`, count it in the manifest, and keep it out of every split. | P0 | `clean_and_split` · `data/split.py` | `tests/data_quality/test_processed.py::test_the_portfolio_splits_into_six_equal_batches`, `::test_undocumented_rows_are_folded_not_dropped`, `::test_write_splits_emits_files_and_a_manifest`, `::test_rows_failing_an_error_check_are_quarantined_not_trained_on`, `::test_an_undocumented_sex_code_does_not_become_a_third_group`, `::test_a_code_cleaning_cannot_fold_is_quarantined_too` (CI `test`, `data quality`) |
| FR-4 | Build model features with one function shared by training and serving; fail fast if the train or test split cannot be turned into a feature matrix. | P0 | `build_features` · `features/build.py` | `tests/unit/test_features.py::test_train_and_serve_paths_agree_exactly`, `::test_cli_fails_on_a_frame_it_cannot_build` (CI `test`) |
| FR-5 | Train a logistic-regression baseline and a LightGBM candidate with cross-validated hyperparameter search; log parameters, metrics and artifacts to MLflow. | P0 | `train_candidates` · `models/train.py` | `tests/model/test_performance.py::test_train_all_logs_both_families_and_registers_through_the_gate` (slow) |
| FR-6 | Refuse registration when, on held-out batch 5, PR-AUC < 0.54, demographic parity difference > 0.05 or equalized odds difference > 0.08. Record the refusal and its reason as run tags. | P0 | `evaluate_and_gate` · `models/evaluate.py`, `models/registry.py` | `tests/unit/test_pipeline_cli.py::test_evaluate_cli_exits_non_zero_when_the_gate_refuses`; `tests/model/test_performance.py::test_register_if_passes_refuses_a_weak_model`, `::test_register_if_passes_refuses_an_unfair_model_however_accurate` (CI `test`) |
| FR-7 | Register the candidate that passed as a new version of `credit-risk`. Make it the champion (the `champion` alias and the Production stage) only when the registry has no champion or its held-out PR-AUC is no more than `PROMOTION_PR_AUC_TOLERANCE` (0.005) below the champion's; otherwise register it as the `challenger` (stage Staging) with the reason as a tag. Tag the version with its capacity threshold (`threshold_at_k`), `capacity_fraction`, `trained_at`, `git_sha` and `run_id`. | P0 | `register_model` · `models/registry.py` | `tests/unit/test_pipeline_cli.py::test_registry_cli_registers_and_promotes_when_the_gate_passes`, `::test_registry_cli_exits_zero_for_a_challenger_and_says_how_to_promote_it`, `::test_registry_cli_does_not_promote_a_refused_candidate`; `tests/unit/test_registry_promotion.py::test_a_worse_candidate_is_registered_as_challenger_and_not_promoted`, `::test_a_candidate_within_the_tolerance_is_promoted`; `tests/unit/test_registry_tags.py::test_registration_tags_the_version_with_everything_serving_needs` (CI `test`) |
| FR-8 | Publish a self-contained HTML report of the run: candidate comparison, recall ceiling, gate outcome. | P2 | `publish_report` · `models/report.py` | `tests/unit/test_pipeline_cli.py::test_report_renders_every_section_from_the_artefact`, `::test_report_states_the_recall_ceiling` (CI `test`) |
| FR-9 | Run FR-1..FR-8 in order as one Airflow DAG, with validation finishing before cleaning starts. Post a `PipelineTaskFailed` alert (severity `critical`, one per failed task, or one with `task_id=dagrun` when the run failed without a failed task) to Alertmanager when a run fails, and resolve it when a run succeeds. | P1 | `dags/credit_risk_pipeline.py` | `tests/unit/test_dag.py` (Airflow image): no import errors, the eight task ids, every upstream edge, `::test_a_failed_run_posts_one_alert_per_failed_task`, `::test_an_unreachable_alertmanager_never_breaks_the_callback` |

### 3.2 Serving API

| ID | Requirement | Pri | Endpoint | Verified by |
|---|---|---|---|---|
| FR-10 | Score one account: default probability, decision (`intervene` / `monitor`), risk band, the threshold and policy applied, model name and version, request id. | P0 | `POST /api/v1/predict` | `tests/integration/test_api.py::test_predict_returns_the_documented_schema`, `::test_predict_decision_follows_the_threshold` (CI `test`) |
| FR-11 | Score up to 1,000 accounts in one call; reject a larger or an empty batch with 422. | P0 | `POST /api/v1/predict/batch` | `tests/integration/test_api.py::test_batch_accepts_the_documented_maximum`, `::test_batch_rejects_one_over_the_maximum`; `tests/integration/test_errors.py::test_empty_batch_is_rejected` (CI `test`) |
| FR-12 | Explain one decision with SHAP and LIME, three plain-language top reasons, and how far the two methods agree. Reasons never cite a protected attribute. | P0 | `POST /api/v1/explain` | `tests/integration/test_api.py::test_explain_returns_shap_lime_and_agreement`, `::test_explain_is_reproducible_for_the_same_record`; `tests/unit/test_explain.py::test_top_reasons_never_cites_a_protected_attribute` (CI `test`) |
| FR-13 | Report, for the serving model, the live selection rate and applied threshold per `SEX` group against the registration limits. | P1 | `GET /api/v1/fairness/report` | `tests/integration/test_api.py::test_fairness_report_covers_every_protected_group` (CI `test`) |
| FR-14 | Report which model version, run and algorithm are serving, the decision threshold and where it came from, and the API version; answer 200 with `status: degraded` when no model is loaded. | P0 | `GET /health`, `GET /version` | `tests/integration/test_api.py::test_health_reports_which_model_is_serving`, `::test_health_is_200_but_degraded_without_a_model`, `::test_version_reports_api_and_model_build`; `tests/integration/test_serving_threshold.py::test_health_reports_the_run_the_threshold_and_where_it_came_from` (CI `test`) |
| FR-15 | Expose every series the alert rules query, in Prometheus text format. | P1 | `GET /metrics` | `tests/integration/test_api.py::test_metrics_is_plain_text_and_exports_every_alerted_series` (CI `test`); CI `smoke (compose)` checks Prometheus loaded every rule |
| FR-16 | Reject a missing, mistyped, out-of-range or unknown field with 422 and a structured error that does not echo the submitted values. Answer 503 on every inference route while no model is loaded. | P0 | all `/api/v1/*` | `tests/integration/test_errors.py::test_missing_field_is_422_and_counted`, `::test_unknown_field_is_rejected_rather_than_ignored`, `::test_validation_detail_never_echoes_the_submitted_values`, `::test_every_inference_route_is_503_while_degraded` (CI `test`) |
| FR-17 | Load the model from `models:/credit-risk@champion` in MLflow at startup, falling back to the Production stage when no version carries the alias, and report which one answered as `model_ref` in `/health`; no model is baked into the image. | P0 | `serving/model_loader.py` | `tests/integration/test_serving_alias.py::test_the_api_serves_the_alias_and_falls_back_to_the_stage`, `::test_the_loader_records_how_the_version_was_found`; `tests/unit/test_registry_promotion.py::test_serving_resolution_prefers_the_alias_and_falls_back_to_the_stage`; `tests/integration/test_api.py::test_model_holder_degrades_when_the_registry_is_empty` (CI `test`); `make smoke` fails without a loaded model (CI runs it with `SMOKE_ALLOW_DEGRADED=1`) |
| FR-18 | Apply one threshold to everyone by default (`THRESHOLD_POLICY=base`); per-group cut-offs (`group_aware_equalized_odds`) are opt-in. | P2 | `serving/routes.py` | `tests/integration/test_api.py::test_base_policy_applies_one_threshold_to_everyone`, `::test_group_aware_policy_applies_the_fitted_cutoffs` (CI `test`) |
| FR-22 | Decide at the capacity threshold of the serving version, read from its `threshold_at_k` tag (`THRESHOLD_SOURCE=registry`, the default). When the version has no usable tag, decide at `DECISION_THRESHOLD` and report `threshold_source: "fallback"` in `/health`; `THRESHOLD_SOURCE=env` always uses `DECISION_THRESHOLD`. The share flagged on accounts the threshold was not computed on stays within 10% ± 1 point. | P0 | `serving/model_loader.py`, `serving/routes.py`; backfill: `python -m credit_risk.models.registry tag-threshold` | `tests/integration/test_serving_threshold.py::test_the_served_selection_rate_holds_capacity_on_accounts_never_scored`, `::test_the_loader_reads_the_threshold_tag_of_the_version_it_loads`, `::test_an_untagged_version_falls_back_loudly`, `::test_every_decision_is_made_at_the_loaded_threshold`; `tests/unit/test_registry_tags.py` (CI `test`); `tests/model/test_served_capacity.py` on `serving_pool` (CI `data_quality`) |

### 3.3 Monitoring

| ID | Requirement | Pri | Where | Verified by |
|---|---|---|---|---|
| FR-19 | Alert when the live selection-rate gap between protected groups exceeds 0.05 for 15 minutes. Groups below the sample floor are not published. | P1 | `FairnessGapExceeded` · `fairness.yml` | `tests/unit/test_metrics.py::test_a_group_below_the_sample_floor_is_not_published` (CI `test`); rule loading in CI `smoke (compose)`; firing exercised by hand with `make bias` |
| FR-20 | Publish PSI per feature against the training baseline and alert above 0.25 for 20 minutes; alert when the high-risk share moves more than 10 points from the registration baseline for 15 minutes. | P1 | `FeatureDriftHigh`, `HighRiskShareShift` · `model.yml` | `tests/unit/test_metrics.py::test_psi_flags_a_clearly_shifted_distribution`, `::test_observe_features_publishes_psi_against_the_baseline`, `::test_baseline_high_risk_share_is_published_and_replaced` (CI `test`); firing exercised by hand with `make drift` |
| FR-21 | Provision the Prometheus datasource and three dashboards from files, with no manual setup. | P2 | `monitoring/grafana/provisioning/` | No automated check |

---

## 4. Non-functional requirements

Measured values are from 2026-10-03 on the local compose stack: latency with
`curl` from one client (5 × `/predict`, 3 × `/explain`, 1 × 1,000-record batch),
model metrics from MLflow runs `8aec69fc` and `3914189d`, the pipeline from one
`airflow dags test` run.

| ID | Requirement | Target | Measured | Pri | Verified by |
|---|---|---|---|---|---|
| NFR-1 | Single-account latency | p95 `/predict` ≤ 100 ms | 5–23 ms | P1 | `SlowPredictions` (p95 > 100 ms for 10 min); `scripts/load_test.py` p95 budget |
| NFR-2 | Batch latency | p95 `/predict/batch` ≤ 500 ms per call of ≤ 1,000 records | 69 ms for 1,000 records | P1 | `SlowBatchPredictions` (p95 > 500 ms for 10 min) |
| NFR-3 | Explanation latency | p95 `/explain` ≤ 500 ms | 28–40 ms | P2 | `SlowExplanations` (severity info, for 10 min) |
| NFR-4 | Throughput | ≥ 200 requests/s on `/predict` with p95 inside 100 ms | not yet recorded | P2 | `scripts/load_test.py --strict` (exits 1 when missed); manual |
| NFR-5 | Availability | `credit-api` answers Prometheus scrapes | — | P0 | `ApiDown` (critical, no scrape for 1 min) |
| NFR-6 | A running API is serving a model | no more than 2 min without one; start degraded rather than crash-loop | — | P0 | `ModelNotLoaded` (critical, for 2 min); FR-14 and FR-16 tests |
| NFR-7 | Error ratio | errors / (errors + predictions) ≤ 5% over 5 min | — | P1 | `HighErrorRate` (for 5 min); `tests/integration/test_errors.py` asserts each error is counted |
| NFR-8 | Model quality on held-out batch 5 | PR-AUC ≥ 0.54 (blocks registration); ROC-AUC ≥ 0.78, Brier ≤ 0.14 | PR-AUC 0.5668, ROC-AUC 0.7930, Brier 0.1243 | P0 | FR-6 gate; `tests/model/test_performance.py::test_lightgbm_clears_the_model_targets_on_the_fixture` (slow) |
| NFR-9 | Fairness at registration, on `SEX` | dp ≤ 0.05, eo ≤ 0.08 (blocks registration) | dp 0.0384, eo 0.0725 | P0 | FR-6 gate; `tests/unit/test_fairness.py::test_passes_gates_rejects_the_known_gap_and_names_both_reasons` (CI `test`) |
| NFR-10 | Protected-attribute invariance of the mitigated model | flipping `SEX` moves the probability by exactly 0.0 | — | P1 | `tests/model/test_invariance.py::test_flipping_sex_cannot_move_the_mitigated_probability_at_all` (slow) |
| NFR-11 | Input data quality | bad-row fraction ≤ 5% across the raw and cleaned checks, and no row failing an error check of either in any split | 0.0 on the UCI file, 0 rows quarantined | P0 | FR-2: `validate_raw` fails the DAG; FR-3: `clean_and_split` quarantines the rest |
| NFR-12 | Reproducibility | same input gives identical split hashes and, with seed 42, identical metrics | a DAG run reproduced v1's metrics to four decimals | P1 | `tests/data_quality/test_processed.py::test_rerunning_write_splits_produces_identical_hashes`, `::test_batch_assignment_is_deterministic` (CI `test`) |
| NFR-13 | Metric cardinality | < 60 exported series after 1,000 predictions; no label carries an account id | — | P1 | `tests/unit/test_metrics.py::test_series_count_stays_bounded_under_traffic`; `tests/integration/test_api.py::test_group_label_never_leaks_an_account` (CI `test`) |
| NFR-14 | Container hardening | API image is multi-stage and runs as a non-root user; secrets live only in the git-ignored `.env` | uid 10001 | P1 | `Dockerfile` (`USER appuser`); CI `build image` |
| NFR-15 | Code quality | ruff clean; mypy clean with `disallow_untyped_defs`; coverage of `src/credit_risk` ≥ 80%; Python 3.11 and 3.12 | 96.6% in CI (not slow, not needs_data); 98.78% for the full suite | P1 | CI `lint`, `typecheck`, `test` (3.11 and 3.12 matrix) |
| NFR-16 | Pipeline run time | each step ≤ 1 h, a whole run ≤ 2 h | 9 min 14 s end to end | P2 | `SUBPROCESS_TIMEOUT` and `dagrun_timeout` in `dags/credit_risk_pipeline.py` |

---

## 5. Known gaps

Stated here so that nobody reads a requirement above as met when it is not.

- **Version 2 decides at the 0.5 fallback until its tag is backfilled.** It was
  registered before versions carried `threshold_at_k`, so `/health` reports
  `threshold_source: "fallback"` and FR-22's capacity is not met by the running
  stack until `python -m credit_risk.models.registry tag-threshold --version 2`
  is run against the registry and the API is restarted. Measured offline, 0.5
  flags 10.56% of `serving_pool` and the capacity threshold 0.5167 flags 10.14%.
- **The capacity threshold is computed once, on batch 5.** If the population
  moves, the share flagged moves away from 10% with it. `HighRiskShareShift`
  alerts on that; nothing recomputes the threshold until the next training run.
- **NFR-4 has an instrument but no recorded result.** `scripts/load_test.py`
  exists; nobody has yet published a run against the compose stack.
- **Alert rules are checked for loading, not for firing.** CI confirms
  Prometheus loaded every rule. That each one fires on its condition has only
  been exercised by hand with `make drift`, `make bias` and `make broken`.
- **`tests/unit/test_dag.py` skips wherever Airflow is not installed**, CI
  included. The DAG's coverage is therefore not part of the 80% gate.
- **The splits are ID ranges, not periods.** The file has no dates, so batch 5
  is not "later" than batches 1–4, and the default rate differs by split (22.8%
  train, 20.4% test, 21.2% serving pool). Nothing here measures behaviour over
  time; DATASHEET.md has the table.
- **`PipelineTaskFailed` reaches a person only if Alertmanager routes it
  somewhere.** The DAG posts the alert; delivery depends on the receiver
  configured in `monitoring/alertmanager/`.
- **The latency figures come from one client** sending a handful of requests.
  They show headroom against the targets; they are not a load test.
