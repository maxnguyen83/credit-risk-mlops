# Datasheet: Default of Credit Card Clients

Following the structure proposed in Gebru et al., *Datasheets for Datasets*
(Communications of the ACM, 2021) — reference [7] on the DDM501 reading list.

Every count in this document was **measured from the file**, not copied from the
source page. Where the data disagrees with its own published dictionary, the
disagreement is recorded rather than quietly patched.

---

## Motivation

**Why was the dataset created?** To compare the predictive accuracy of six data
mining methods on the probability of customer default, for a study by Yeh and Lien
(2009), *The comparisons of data mining techniques for the predictive accuracy of
probability of default of credit card clients*, Expert Systems with Applications.

**Who created it and who funded it?** I-Cheng Yeh, Department of Information
Management, Chung Hua University, Taiwan. Donated to the UCI Machine Learning
Repository in 2016.

**Why are we using it?** Three requirements had to be met at once: it must carry
**protected attributes** so fairness analysis is real rather than decorative, it
must be **downloadable without authentication** so CI can reproduce the pipeline,
and it must be **small enough** that a four-person team spends its four weeks on
operations rather than on data wrangling. Alternatives considered and rejected
are recorded in the design spec: German Credit (1,000 rows — too small to justify
a versioned pipeline), Home Credit Default Risk (2.7 GB and a Kaggle token — CI
cannot fetch it), Credit Card Fraud (features PCA-anonymised, no protected
attributes — fairness and SHAP both become meaningless), and the ACS/folktables
suite (`www2.census.gov` returns HTTP 403 to automated requests, and the package
has had no release since February 2023).

---

## Composition

**What does an instance represent?** One credit-card account holder at a Taiwanese
bank, observed over six months.

**How many instances are there?** **30,000 rows, 25 columns.** No sampling; this
is the complete released dataset.

**Is any data missing?** **No.** Zero nulls in any column — verified, not assumed.
This is unusual and worth stating plainly: the dataset's quality problems are
*encoding* problems, not *missingness* problems.

**What does each instance consist of?**

| Group | Columns | Notes |
|---|---|---|
| Identifier | `ID` | Sequential. Dropped before training. |
| Credit limit | `LIMIT_BAL` | NT dollars, includes family supplementary credit |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | **Protected attributes** — see below |
| Repayment history | `PAY_0`, `PAY_2`..`PAY_6` | April–September 2005. −2 no consumption, −1 paid in full, 0 revolving, 1–8 months overdue |
| Bill statements | `BILL_AMT1`..`BILL_AMT6` | NT dollars; may be negative when the account is overpaid |
| Previous payments | `PAY_AMT1`..`PAY_AMT6` | NT dollars |
| Target | `default payment next month` | 1 = defaulted, 0 = did not |

**Class balance.** Positive rate **0.2212** (6,636 of 30,000). Imbalanced enough
that ROC-AUC flatters a model; PR-AUC is used as the primary metric for that
reason.

**Known errors, sources of noise, redundancies.** Four, all measured:

| Issue | Observed | Our handling |
|---|---|---|
| `EDUCATION` contains codes outside its dictionary (1–4) | `0` in 14 rows, `5` in 280, `6` in 51 — 345 rows, 1.15% | Folded into `4 = other`. A data-quality test asserts nothing outside {1,2,3,4} survives cleaning. |
| `MARRIAGE` contains a code outside its dictionary (1–3) | `0` in 54 rows, 0.18% | Folded into `3 = other`, asserted the same way. |
| The first repayment column is exported as `PAY_0` while the rest are `PAY_2`..`PAY_6` | All rows | Renamed to `PAY_1`. A test asserts no column named `PAY_0` remains. |
| `BILL_AMT*` can be negative | Present | **Valid** — an overpaid account. Validation permits negative bills but rejects negative `PAY_AMT` and non-positive `LIMIT_BAL`. |

The header also sits on the **second row** of the spreadsheet, behind a banner
row. Reading the file naively yields a frame whose column names are the banner.

**Does the dataset contain confidential or sensitive data?** It contains
demographic attributes and financial behaviour. There are no direct identifiers —
no names, addresses, account numbers or national ID. See *Privacy* below for why
that is not the same as anonymous.

---

## Protected attributes and base rates

Measured directly on the raw file. These are **base rates in the world as
recorded**, before any model exists:

| Attribute | Group | n | Default rate |
|---|---|---|---|
| `SEX` | 1 (male) | 11,888 | **24.17%** |
| | 2 (female) | 18,112 | **20.78%** |
| | | | *gap 3.39 points* |
| `EDUCATION` | 1 graduate school | 10,585 | 19.23% |
| | 2 university | 14,030 | 23.73% |
| | 3 high school | 4,917 | 25.16% |
| | | | *gap 5.93 points* |
| `MARRIAGE` | 1 married | 13,659 | 23.47% |
| | 2 single | 15,964 | 20.93% |
| | 3 other | 323 | 26.01% |

A model fitted to this data will reproduce these differences. **That is not a
bug in the model; it is the data.** The question the project has to answer is not
*is there a gap* — there is — but **whether a gap in observed outcomes may be
turned into a gap in treatment**, and if not, what it costs to remove it. That
question is worked through in `ETHICS.md` and measured in
`src/credit_risk/fairness/`.

Note also that the group sizes are uneven: women make up 60% of the sample. An
accuracy metric averaged over the whole population is therefore dominated by the
majority group, which is one reason per-group metrics are reported separately.

---

## Collection process

**How was the data acquired?** Directly observed from bank records — billing
statements, payment records and account attributes. Not self-reported, not
inferred.

**Over what timeframe?** April to September 2005, with the target observed in
October 2005.

**Was consent obtained? Were subjects notified?** Not documented by the
depositors, and we cannot verify it. Recorded here as an unknown rather than
assumed to be fine.

**Ethical review?** No ethical review process is documented in the source
publication or the UCI entry.

---

## Preprocessing, cleaning and labelling

All steps live in `src/credit_risk/data/` and `src/credit_risk/features/`, and
each is covered by a test:

1. **Download and verify** — fetch the ZIP from UCI (retrying dropped
   connections and 5xx answers), record its SHA-256, extract the single `.xls`
   member, convert to Parquet. The archive digest and the Parquet's own digest
   go into a sidecar written before the Parquet; a Parquet without a matching
   sidecar is downloaded again. The Excel reader is needed exactly once;
   everything downstream reads Parquet.
2. **Validate** — shape, column set, dtypes, ranges, nulls, category codes. If
   more than 5% of rows fail, the pipeline **stops** rather than training.
   Rows that fail below that tolerance are **quarantined**: written as received
   to `data/processed/quarantine.parquet` with the checks they failed, counted
   in the manifest, and left out of every split. The published file has none.
3. **Clean** — rename `PAY_0` to `PAY_1` and the target to `default_next_month`;
   fold undocumented category codes; derive `AGE_GROUP` (35 and under / over 35).
4. **Split** — six deterministic batches of 5,000 by sorted `ID`, with a
   manifest recording a hash per split so reruns are verifiably identical.
   Batches are assigned before quarantine, so a set-aside row leaves a gap in its
   own batch rather than moving other accounts between batches.
5. **Feature engineering** — utilisation ratios, payment ratios, delinquency
   counts and runs, trends. Every division is guarded; a NaN reaching the model
   is a silent wrong answer rather than a crash.

**The raw data is kept.** Cleaning is never done in place.

### A simulation we are declaring, not hiding

The file is a static snapshot. Splitting it into six "arrival batches" so the
pipeline ingests something rather than copying a file is a **simulation**, and so
is the drift injected by `scripts/traffic.py`. Neither represents real temporal
change. We do it because an orchestration layer that only ever copies one file
demonstrates nothing — and we state it here, in the README, and in the
presentation, because a simulated arrival presented as a real one would be a
fabricated result.

**`ID` is not time, and the splits do not share a base rate.** The file has no
date column; ordering by `ID` is the only deterministic order available, and
nothing documents how IDs were assigned. The batches therefore differ in risk,
measured on the published file:

| Split | Batches | Rows | Default rate |
|---|---|---|---|
| train | 1–4 | 20,000 | 22.8% |
| test | 5 | 5,000 | 20.4% |
| serving pool | 6 | 5,000 | 21.2% |
| whole file | — | 30,000 | 22.1% |

Two consequences. Held-out metrics are measured on a population with a lower
base rate than the training data (the recall ceiling in the model card uses the
test split's 20.4% for that reason). And no result here says anything about how
the model behaves as time passes: a random or stratified split would equalise
the rates, and a real temporal hold-out is impossible without dates. We keep the
ID split because it is reproducible and states its own limits, and we record
this as a property of the simulation rather than of the population.

---

## Uses

**What has the dataset been used for?** The original comparative study of six
classification methods, and since then, very widely as a teaching and
benchmarking set for imbalanced tabular classification and for algorithmic
fairness.

**What is it used for here?** Training and evaluating a default early-warning
model, and as the substrate for the fairness, explainability and drift-monitoring
work.

**What should it *not* be used for?**

- **Scoring new credit applications.** The features include six months of
  repayment history on an existing account. An applicant does not have that
  history, so using this model for origination is target leakage, and the model
  would fail on exactly the population it was pointed at.
- **Any population other than Taiwanese credit-card holders in 2005.** Credit
  behaviour, the regulatory environment, and the meaning of the demographic
  categories have all moved.
- **Real lending decisions of any kind.** This is coursework. The model has never
  been validated against a current population, against a hold-out period, or by
  a credit-risk professional.

**Is there anything about the composition that could result in unfair treatment?**
Yes, and it is the point of the project. The demographic attributes correlate
with the target, group sizes are unbalanced, and the categorical encodings for
education and marital status are coarse and culturally specific. A model that
optimises accuracy alone will encode all of it.

---

## Distribution and maintenance

**License and terms.** The UCI Machine Learning Repository distributes the
dataset under a Creative Commons Attribution 4.0 International licence. Cite Yeh
and Lien (2009).

**Where does the pipeline get it?**
`https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip`
— verified reachable, HTTP 200, 5,539,494 bytes, **no authentication required**.
That last property is why this dataset is usable in CI at all.

**Will it be updated?** No. It is a 2005 snapshot and will not change. Our
pipeline records the SHA-256 of the downloaded archive so a substitution at the
source would be detected rather than silently trained on.

**Who maintains this datasheet?** P1 (`maxnguyen83`), the data and pipeline
owner. Corrections belong in a pull request against this file.

---

## Privacy

There are no direct identifiers. That is **not** the same as anonymous.

`AGE` + `SEX` + `EDUCATION` + `MARRIAGE` together form a quasi-identifier. In a
sufficiently small population, or when joined against another dataset, that
combination can single out an individual. Small cells make it worse: the
`MARRIAGE = 3` group has 323 members, and combinations within it will be smaller
still.

The controls that follow from this:

- Request payloads are **never logged at INFO**. Logs carry a `request_id` and
  derived aggregates only.
- **No Prometheus metric is labelled with per-account data.** This is
  simultaneously a privacy control and a cardinality control.
- `.env` is gitignored and contains no real credentials in the committed example.
- Any real deployment would additionally need retention limits, access control
  and audit logging over the prediction store. This project has none of those,
  and that limitation is stated rather than left implied.
