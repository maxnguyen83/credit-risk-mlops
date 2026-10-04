# Model Card: Credit Default Early-Warning

Following Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

All numbers below describe the version now serving: MLflow run
`3914189dcbc645808759fd94c2900e5f`, registered as `credit-risk` **version 2**
and promoted to Production by the Airflow DAG on 2026-10-03. They are measured
on the held-out test split (batch 5, 5,000 accounts never seen during training).
Version 1 (run `8aec69fc6fb3434f8f0e09ae7c14dc60`, trained with `make train`)
logged identical test metrics; version 2 replaced it in Production.

The per-group tables were regenerated from the same configuration re-run
offline on the DAG's splits (`python -m credit_risk.models.train --no-mlflow`).
That re-run reproduces version 2's logged PR-AUC, ROC-AUC, Brier, threshold and
fairness metrics to every digit, so the tables describe the served model.
Reproduce with:

```bash
make dag-test                 # the whole DAG, in the running stack
make data && make train       # or the training step on its own
```

---

## Model details

| | |
|---|---|
| **Developed by** | `maxnguyen83`, `Ducmanh2212`, `hieunt-fsb-ai`, `thientd2609` |
| **Model date** | October 2026 |
| **Model type** | Gradient-boosted decision trees (LightGBM 4.5.0), binary classification with probability output |
| **Baseline compared against** | Logistic regression (standardised features, `class_weight="balanced"`) |
| **Input** | 23 account attributes; 38 engineered features |
| **Output** | Probability of default next month, plus a decision at the version's 10%-capacity threshold, read from its `threshold_at_k` registry tag (0.5167 for version 2's run). A version without the tag decides at the configured `DECISION_THRESHOLD` (0.5) and `/health` reports `threshold_source: "fallback"`; version 2 is in that state until its tag is backfilled |
| **Training data** | UCI *Default of Credit Card Clients*, batches 1–4 (20,000 accounts) |
| **Evaluation data** | Batch 5 (5,000 accounts), held out and never trained on |
| **Versioning** | MLflow Model Registry, `models:/credit-risk/Production` (version 2 at the time of writing) |
| **License** | Coursework. Not licensed for deployment. |
| **Contact** | See `CONTRIBUTING.md` |

---

## Intended use

**Primary use.** Rank existing credit-card accounts by probability of missing
next month's minimum payment, so a risk team with capacity for ~10% of the
portfolio can choose which accounts to contact.

**Primary users.** Risk analysts at the issuing bank; a nightly batch job in the
core-banking system.

**Out of scope — explicitly.**

- **Scoring new credit applications.** The features contain six months of
  repayment history on an existing account. A new applicant has none, so using
  this model for origination is target leakage and it would fail on exactly the
  population it was aimed at.
- **Automated adverse action.** The model ranks; a human decides. There is no
  configuration in which it declines anybody by itself.
- **Any population other than Taiwanese credit-card holders circa 2005.**
- **Real lending decisions.** It has never been validated against a current
  population, a hold-out period, or by a credit-risk professional.

---

## Factors

**Groups reported.** `SEX` (primary), `AGE_GROUP` (≤35 / >35), `EDUCATION`,
`MARRIAGE`, and the `SEX × AGE_GROUP` intersection. These are the protected
attributes present in the data. Only `SEX` is gated; the other three and the
intersection are summarised below with group sizes, and the reason they are not
gated is stated with them.

**Instrumentation.** All features derive from bank records — billing statements
and payment history. Nothing is self-reported or inferred from a third party.

**Environment.** A static 2005 snapshot. No seasonality, no macroeconomic
variation, no policy change is represented.

---

## Metrics

### Why PR-AUC is the primary metric

The positive class is 22.12% of the data. ROC-AUC weights performance on the
majority class heavily and flatters an imbalanced classifier; average precision
(PR-AUC) tracks the quality of the ranking where the decision is actually made —
at the top. Both are reported; only PR-AUC gates registration.

### Why recall is reported against a ceiling

The risk team can act on 10% of the portfolio. With a base rate of 20.4% on the
test split, **even a perfect ranker** can only reach

```
max recall@10%  =  capacity / base_rate  =  0.10 / 0.204  =  0.490
```

An absolute recall target above that is unachievable, not ambitious. The model is
therefore assessed on **how much of the achievable ceiling it captures**.

### Overall (version 2, test split)

| Metric | Value | Target | Enforced by |
|---|---|---|---|
| PR-AUC (average precision) | **0.5668** | ≥ 0.54 | **Blocks registration** (`MIN_PR_AUC`, `models/registry.py`) |
| ROC-AUC | **0.7930** | ≥ 0.78 | Asserted in `tests/model/test_performance.py` |
| Brier score | **0.1243** | ≤ 0.14 | Asserted in `tests/model/test_performance.py`, on the synthetic fixture |
| Recall@10% | **0.3490** | ≥ 0.33 | Reported only |
| — as a share of the 0.490 ceiling | **71.2%** | ≥ 67% | Reported only |
| Precision@10% | **0.7120** | — | Reported only |
| Lift@10% | **3.49×** | ≥ 2.2× | Reported only |
| Threshold at 10% capacity | **0.5167** | — | Logged as `test_threshold_at_k`; tagged onto the version; the cutoff the API decides at |
| Accounts flagged at that threshold | 500 of 5,000 | — | Exactly the capacity, by construction |

**The API decides at the capacity threshold the serving version carries.**
Registration tags each model version with `threshold_at_k`, and `/predict`
compares the probability with that tag. A version without the tag decides at
`DECISION_THRESHOLD`, a configured constant of 0.5, and `/health` reports
`threshold_source: "fallback"` while that is the case. Version 2 was registered
before versions carried the tag, so it serves at 0.5 until
`python -m credit_risk.models.registry tag-threshold --version 2` copies its
run's logged 0.5167 onto it (no retraining) and the API is restarted.

The difference is small but not zero. Measured offline with version 2's
configuration re-run on the DAG's splits:

| Accounts flagged | Test split (batch 5) | `serving_pool` (batch 6) |
|---|---|---|
| At the 0.5 fallback | 528 of 5,000 (10.56%) | 528 of 5,000 (10.56%) |
| At the capacity threshold, 0.5167 | 500 of 5,000 (10.00%) | 507 of 5,000 (10.14%) |

Batch 6 was never trained or evaluated on, so its 10.14% is how close the
threshold computed on batch 5 lands on new accounts. A test trains a LightGBM
on batches 1–4, registers it with its batch-5 threshold, serves it through the
API and asserts that the share it flags on batch 6 stays within a point of 10%
(`tests/model/test_served_capacity.py`; 10.56% at its own threshold of 0.5144
when written). It checks the mechanism on real accounts, not version 2 itself,
which lives in the running registry.

### Business impact at the capacity threshold (offline)

Using the stated cost assumptions — NT$30,000 lost per unflagged default,
NT$500 per intervention:

| | NT$ |
|---|---|
| Expected loss, no model | 30,600,000 |
| Expected loss, with the model at 10% capacity | 19,992,000 |
| **Avoided loss** | **10,608,000** |
| Cost of false alarms (144 × 500) | 72,000 |

The cost matrix is an assumption, stated so it can be argued with. Change the
two constants in `config.py` and the conclusion changes; that is the point of
putting them in configuration rather than in prose.

---

## Quantitative analysis — disaggregated by `SEX`

At the 10%-capacity threshold on the held-out split:

| Group | n | Defaults | Selection rate | TPR | FPR | Precision | Accuracy |
|---|---|---|---|---|---|---|---|
| female (`SEX = 2`) | 2,803 | 520 | **0.0831** | 0.3135 | 0.0307 | 0.6996 | 0.8477 |
| male (`SEX = 1`) | 2,197 | 500 | **0.1215** | 0.3860 | 0.0436 | 0.7228 | 0.8266 |

| Fairness metric | Value | Gate | Result |
|---|---|---|---|
| Demographic parity difference | **0.0384** | ≤ 0.05 | pass |
| Demographic parity ratio | 0.6840 | — | |
| Equalized odds difference | **0.0725** | ≤ 0.08 | pass |
| Selection-rate gap | 0.0384 | — | |

The per-group rows are the table every run logs as its `group_report.csv`
artifact. Each summary row can be recomputed from them: the parity difference is
`0.1215 − 0.0831 = 0.0384`, and the equalized-odds difference is the larger of
the TPR gap (`0.3860 − 0.3135 = 0.0725`) and the FPR gap
(`0.0436 − 0.0307 = 0.0129`).

### Reading these numbers honestly

The model passes both gates. It is **not** group-neutral.

Men are flagged at 12.15% and women at 8.31% — **1.46 times as often**. The
observed default rates differ in the same direction (22.76% against 18.55% on
this split; 24.17% against 20.78% across all 30,000 accounts), so the model is
tracking a real difference in the recorded data rather than inventing one. Two
things follow, and both matter:

1. **Passing a threshold is not the same as being fair.** `dp_diff = 0.0384`
   clears a gate set at 0.05. It does not mean the 1.46× ratio is acceptable —
   that is a policy question, and the gate is where somebody wrote their policy
   down, not a proof of innocence.
2. **The base rates are not neutral facts.** They record the outcome of a lending
   process that was itself shaped by the same categories. See `ETHICS.md` §1.

---

## Quantitative analysis — attributes reported but not gated

Same model, same decisions (the top 10% by score), same test split. The SEX
limits are shown for scale only; nothing below blocks registration.

| Attribute | Groups | Smallest group on the test split | DP difference | DP ratio | EO difference | Against the SEX limits (0.05 / 0.08) |
|---|---|---|---|---|---|---|
| `AGE_GROUP` (≤35 / >35) | 2 | older, n = 2,200 | 0.0114 | 0.893 | 0.0095 | inside both |
| `EDUCATION` | 4 | other, n = 130 (9 defaults) | **0.0974** | 0.191 | **0.3802** | dp 1.9×, eo 4.8× |
| `MARRIAGE` | 3 | other, n = 55 (14 defaults) | 0.0472 | 0.536 | **0.2967** | eo 3.7× |
| `SEX × AGE_GROUP` | 4 | male, older, n = 1,011 | 0.0468 | 0.621 | **0.1061** | eo 1.3× |

Where the gaps come from:

| Group | n | Defaults | Selection rate | TPR | FPR |
|---|---|---|---|---|---|
| `EDUCATION` graduate school | 1,720 | 275 | 0.0686 | 0.3127 | 0.0221 |
| `EDUCATION` university | 2,365 | 555 | 0.1205 | 0.3802 | 0.0409 |
| `EDUCATION` high school | 785 | 181 | 0.1197 | 0.3260 | 0.0579 |
| `EDUCATION` other (codes 0, 4, 5, 6) | 130 | 9 | 0.0231 | 0.0000 | 0.0248 |
| `MARRIAGE` married | 2,231 | 479 | 0.1017 | 0.3361 | 0.0377 |
| `MARRIAGE` single | 2,714 | 527 | 0.0995 | 0.3681 | 0.0348 |
| `MARRIAGE` other (codes 0, 3) | 55 | 14 | 0.0545 | 0.0714 | 0.0488 |
| male, ≤35 | 1,186 | 252 | 0.1197 | 0.4008 | 0.0439 |
| male, >35 | 1,011 | 248 | 0.1236 | 0.3710 | 0.0433 |
| female, ≤35 | 1,614 | 285 | 0.0768 | 0.2947 | 0.0301 |
| female, >35 | 1,189 | 235 | 0.0917 | 0.3362 | 0.0314 |

**Why `EDUCATION` and `MARRIAGE` are not gated.** Both large EO values come from
the "other" group, and both rest on a handful of defaults. The model catches
none of the 9 defaults in `EDUCATION = other` and 1 of the 14 in
`MARRIAGE = other`; one more catch would move those groups' TPR by 0.11 and
0.07. A gate on these attributes would accept or refuse a model on the outcome
of one or two accounts. That is a reason not to gate on them as they stand. It
is not evidence that the gaps are noise: in both cases the smallest group is the
one whose defaults the model almost never flags. Gating them properly needs a
minimum group size and an interval on the gap, neither of which exists yet.

**Why `SEX × AGE_GROUP` is not gated, although it breaches 0.08.** Its cells are
large (1,011 to 1,614 accounts), so this is not a small-sample effect: young
women's defaults are caught at a TPR of 0.295 against 0.401 for young men. A
normal-approximation 95% interval on that difference runs from about 0.03 to
0.19. The registration policy was written for `SEX` alone, and extending it to an
intersection is a policy decision we are flagging rather than taking.
`SEX_x_AGE_GROUP` is computed on every training run from now on (logged as
`by_SEX_x_AGE_GROUP_*`); run `3914189d` predates that metric, so the figures
here come from the offline re-run described at the top.

---

## Mitigation strategies evaluated

Three approaches were implemented and compared. The full trade-off table is
logged to MLflow on every run as `fairness_tradeoff.csv`, with the figure as
`fairness_tradeoff.png`.

| Strategy | Intervention point | What it achieves | What it costs |
|---|---|---|---|
| **Baseline** (no mitigation) | — | pr 0.5668, dp 0.0384, eo 0.0725 | — |
| **Unawareness** (drop `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) | Pre-processing | pr 0.5597, dp 0.0360, eo 0.0486 | Lowers eo by a third (0.0725 → 0.0486) but barely moves dp (0.0384 → 0.0360), and costs ranking quality (PR-AUC −0.007). The attribute survives through proxies: a linear probe recovers `SEX` from the remaining features at **ROC-AUC 0.566** — a floor rather than a ceiling, since a linear model is the weakest probe one could use |
| **Reweighing** (Kamiran–Calders) | Pre-processing | pr 0.5689, dp 0.0278, eo 0.0332 | **Free, not better**: +0.002 PR-AUC on one seed and one split, inside the cross-validation standard deviation of 0.0096. Lower dp and eo than the baseline, and group-blind at serving time. Measured here but not a registration candidate: only the unmitigated logistic and LightGBM models compete for the registry |
| **ThresholdOptimizer** (equalized odds) | Post-processing | pr 0.5668, dp 0.0241, eo 0.0202 | Tightest fairness — but it selects **13.26%** of accounts against a 10% intervention capacity, a third more calls than the team can staff, and it needs **different thresholds per group**, which may be unlawful in a credit decision regardless of intent |

**Shipped default: the group-blind policy.** The group-aware policy is
implemented and measured but is not the default, because in a real deployment the
legal constraint binds before the fairness metric does. `THRESHOLD_POLICY` in
`.env` selects between them, and the active policy is returned in every
`/predict` response so it is auditable rather than implicit.

The unawareness probe is kept in the repository specifically because
"we removed the sensitive column" is the most common unverified fairness claim,
and this measures it.

---

## Explainability

Two methods are served, because a single attribution method is a claim no one can
check.

| | SHAP | LIME |
|---|---|---|
| Method | `TreeExplainer` — exact Shapley values for tree ensembles | Local linear surrogate |
| Scope | Local. `explain_global` in `shap_explainer.py` can draw a global summary plot, but neither the pipeline nor the API calls it; the global view logged to MLflow is LightGBM's gain importance (`feature_importance.png`) | Local |
| Determinism | Deterministic | Stochastic; re-seeded on every request so the same record gets the same explanation |
| Cost | The cheaper of the two: one pass over the trees per record | The slower of the two: 1,000 perturbed predictions per record (`NUM_SAMPLES`) |

Both costs are measured on live traffic as
`credit_explain_duration_seconds{method}`.

`/api/v1/explain` returns both plus an `agreement` field reporting the overlap of
their top-3 reasons. When the two methods disagree about why a decision was made,
the response says so.

The top three positive SHAP contributions are rendered into plain-language
sentences as a draft adverse-action notice. **An attribution is not a
justification**: SHAP reports what moved the prediction relative to the average,
not that the decision was correct, fair, or causally grounded.

---

## Ethical considerations

Covered in full in `ETHICS.md`. The three that most affect how this model should
be read:

1. **Feedback loop.** Cutting the limit of a flagged account may itself cause the
   default that the next training round records as a correct prediction. No metric
   in this repository can detect that. A held-out control group is the only
   remedy, and this dataset has none.
2. **Asymmetric harm.** A false negative costs the bank recoverable money. A false
   positive costs a specific person credit access at the moment a model believes
   they are struggling. The cost matrix treats both as currency; they are not.
3. **Historical data, not ground truth.** The model learns the association
   recorded in 2005 Taiwanese lending data. It does not learn cause.

---

## Caveats and recommendations

- Re-validate before any use on current data. A 2005 snapshot has no claim on
  present behaviour.
- Keep the fairness gates in the pipeline. They protect models nobody has
  reviewed yet, which is most of them.
- Monitor `credit_selection_rate` per group in production. Offline fairness
  measured at training time says nothing about the population actually arriving.
- Watch `credit_feature_psi`. A PSI above 0.25 on any feature means the inputs
  have moved and the reported metrics no longer describe what the model is doing.
- `SEX × AGE_GROUP` is reported on every run but **not gated**, and it currently
  exceeds the 0.08 equalized-odds level applied to `SEX` (0.1061). Three-way
  intersections, such as young women with high-school education, are not
  computed. Extending the gate to intersections, with a minimum group size, is
  the first improvement we would make.
