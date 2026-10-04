# Ethics

Written jointly by all four team members — one section each — because every one
of us will be asked about it.

This is not a compliance checklist. It is the set of arguments we had while
building the system, written down with the conclusions we actually reached,
including the ones where we concluded that the honest answer is uncomfortable.

---

## 1. What the system does to people (P1 — data)

The model ranks credit-card holders by their probability of missing next month's
minimum payment. The top 10% go on an intervention list. "Intervention" means a
phone call, an offer to restructure the debt, or a reduction of the credit limit.

Two of those three are help. The third is a withdrawal of access to money,
applied to somebody a statistical model believes is in trouble, before they have
actually missed anything.

That asymmetry is the ethical centre of the project. A false negative — a
default we failed to flag — costs the bank money, and money is recoverable. A
false positive costs a specific person their credit headroom at the exact moment
a model thinks they are struggling. Those two errors are not interchangeable even
when a cost matrix says they are, and no confusion matrix contains that fact.

### The data carries history, not truth

The base rates in `DATASHEET.md` are not neutral facts about men and women:

```
SEX = 1 (male)    n = 11,888   default rate 24.17%
SEX = 2 (female)  n = 18,112   default rate 20.78%
```

A 3.4-point gap in observed defaults among Taiwanese credit-card holders in 2005
reflects who was granted credit, at what limit, under what social and economic
conditions. The dataset records the outcome of a lending process that was itself
shaped by the same categories. Training on it and calling the result "what the
data says" launders a historical process into an apparently objective score.

We do not have the counterfactual — what would have happened under a different
lending policy — and we cannot recover it. The honest position is that the model
learns the *recorded* relationship between demographics and default, and that
this is not the same thing as a *causal* one.

### Education and marital status are worse, not better

`EDUCATION` spans a 5.9-point gap, wider than sex. `MARRIAGE` spans 5.1 points.
Both are more socially loaded than they look: education encodes class and
opportunity; marital status in a 2005 Taiwanese dataset encodes age, gender role
and household structure at once. The `MARRIAGE = 3` ("other") bucket has 323
people in it, and it has the highest default rate of the three. A model is
perfectly capable of learning "unusual household structure implies risk" from 323
examples, and nobody would defend that sentence if it were written in a policy
document.

---

## 2. Fairness: what we measured, what we changed, and what it cost (P2 — model)

### The measurement

We report, per protected group: selection rate, true positive rate, false
positive rate, precision, and calibration. From those:

- **Demographic parity difference** — is the same fraction of each group flagged?
- **Equalized odds difference** — among people who actually default, is each
  group caught at the same rate, and among those who do not, is each group
  falsely flagged at the same rate?

The gates are `dp_diff <= 0.05` and `eo_diff <= 0.08`, and they **block model
registration** rather than appearing in a report. A model that outperforms the
baseline and breaches a gate is refused, and the refusal is logged to MLflow with
its reason.

### Fairness through unawareness is not enough, and we measured why

The intuitive fix is to delete the protected columns from the feature set. We
implemented it (it drops `SEX`, `EDUCATION`, `MARRIAGE` and `AGE`), and we
implemented the probe that tests it: a linear model trained to predict `SEX`
from the *remaining* features. On the reference split that probe reaches
ROC-AUC 0.566. That is modestly above chance, and it is a floor, because a
linear model is the weakest probe one could use: the attribute is still present
in the data, encoded in utilisation and repayment patterns.

What deleting the columns did and did not change, measured on the same split:
the equalized-odds gap fell by a third (0.0725 → 0.0486), the parity gap barely
moved (0.0384 → 0.0360), and PR-AUC dropped from 0.5668 to 0.5597. It helps one
definition of fairness a little and leaves the other where it was.

The probe is in the repository on purpose. "We removed the sensitive column" is
the most common fairness claim made by teams who have not tested it.

### Three strategies, and the price of each

| Strategy | Where it intervenes | What it costs |
|---|---|---|
| Unawareness (drop the protected columns) | Pre-processing | PR-AUC 0.5668 → 0.5597; lowers the equalized-odds gap by a third, barely moves parity — see above |
| Reweighing (Kamiran–Calders) | Pre-processing | No measurable PR-AUC cost on this split (0.5689 against 0.5668, one seed); group-blind at serving time |
| `ThresholdOptimizer` (equalized odds) | Post-processing | Tightest control of the gap — and requires **different thresholds for different groups** |

The trade-off curve — PR-AUC against `dp_diff` for all three plus the unmitigated
baseline — is the figure the presentation is built on, because it converts
"we tried some fairness things" into a choice with a stated price.

### The part that does not resolve cleanly

`ThresholdOptimizer` works by applying a different decision threshold depending on
the person's sex. It produces the fairest outcome by our metrics, and it may be
**illegal**: many jurisdictions prohibit using a protected attribute in a credit
decision at all, regardless of the direction or the intent.

So the technically fairest option may be the legally forbidden one. We do not
have a resolution for that, and we are not going to manufacture one. What we do
instead:

- The threshold policy is a **configuration value**, not a hard-coded choice
  (`THRESHOLD_POLICY` in `.env`), so which policy is in force is explicit,
  visible in `/predict` responses, and auditable.
- The default shipped configuration is the **group-blind** policy, because in a
  real deployment the legal constraint binds first.
- The group-aware policy is implemented, measured, and reported, so the cost of
  the legal constraint is a number rather than an assumption.

Making the conflict visible is the contribution. Hiding it behind whichever
option scores better would not be.

---

## 3. Explanation, and what it is actually for (P3 — serving)

`/explain` returns SHAP and LIME attributions plus three plain-language reasons.
Its purpose is not to make the model look transparent. It exists because a person
who is refused credit, or has their limit cut, is entitled to know why — and in
many jurisdictions the lender is legally required to tell them.

### An explanation is not a justification

SHAP tells you which features moved this prediction relative to the average
prediction. It does not tell you the decision was correct, or fair, or that the
feature is causally connected to default. A large SHAP value on `PAY_1` means the
model leaned on recent delinquency. It does not mean recent delinquency *caused*
the risk, and it certainly does not mean the decision was right.

Presenting an attribution as a justification is the most common misuse of
explainability, and it is worse than no explanation at all, because it
manufactures confidence.

### Why two methods, and why we report their disagreement

SHAP is theoretically grounded and, for tree ensembles, exact. LIME fits a local
surrogate and is stochastic — the same record explained twice can yield different
orderings.

We serve both and return an `agreement` field reporting how much the top-3
reasons overlap. When two explanation methods disagree about why a decision was
made, the honest response is to show that, not to pick whichever one reads better.
A reviewer asking "do your explainers agree with each other?" gets a number
instead of a shrug.

### Explanations can leak

A sufficiently detailed attribution, returned often enough, lets a caller
reconstruct the decision boundary and — with a little work — infer things about
the training population. `/explain` is therefore a separate endpoint from
`/predict`, so that in a real deployment it can carry its own authorisation and
its own rate limit. In this coursework build it is open, and that is a limitation
we are stating rather than a property we are claiming.

---

## 4. Operating it responsibly (P4 — monitoring)

### The failure mode that no ordinary dashboard shows

`scripts/traffic.py --drift 3.0` shifts the incoming feature distribution. During
that run, measured on the running stack: latency flat at 15 ms p95, error rate
zero, uptime 100%, every panel an SRE would check is green. What moves is
`credit_feature_psi`, from 0.03 to **12.1**, and the share of accounts flagged,
from 12.4% down to **4.5%**. Nothing is broken and the answers changed.

That is the argument for treating ML systems as their own operational category.
A system can be **completely healthy and answering differently at the same
time**, and you will not find out from infrastructure monitoring.

### What we found when we tested the fairness alert, and did not hide

`FairnessGapExceeded` **did not fire in any scenario we ran.** The measured
selection-rate gap is 0.046 under the normal arrival mix (male 0.128, female
0.082), 0.025 under a 95%-male arrival mix, and 0.040 with group-aware
thresholds enabled — all inside the 0.05 limit.

We are reporting the silence rather than lowering the threshold until we got a
screenshot. Four consequences we think are worth stating:

- **The alert is armed and honest.** Its expression evaluates against live data;
  it is quiet because the system is within the policy somebody wrote down. An
  alert that has only ever been seen firing is an alert nobody has tested in its
  normal state.
- **The margin is thin.** Ordinary operation sits at 0.046 against a limit of
  0.05, a margin of 0.004; the 0.025 figure comes from the skewed `--bias` run,
  not from normal traffic. A slightly different population would make it flap.
  We kept the limit equal to the registration gate so the monitor and the gate
  enforce one policy, and we accept that this costs headroom.
- **When it flaps, the threshold is not the thing to change.** Firing, resolving
  and firing again is what a 0.004 margin predicts, so expect it. The response,
  which the alert's own description repeats:
  1. read the per-group rates (`credit_selection_rate`) and compare the traffic
     mix, the serving model version and `THRESHOLD_POLICY` with the last quiet
     period;
  2. hand what you found to compliance, who own the policy — not to the on-call
     engineer, there is nothing to restart;
  3. if only the mix moved, a time-boxed Alertmanager silence that names the
     reason stops the noise, leaves the alert visible, and expires on its own;
  4. if the model or the policy changed, or the gap keeps widening, treat it as a
     regression.

  Raising the alert's limit on its own would leave the monitor and the gate
  enforcing two different policies, the second one decided by whoever was tired
  of the notifications. Moving the limit is a compliance decision, and it moves
  the registration gate (`evaluate_and_gate`, ADR 0009) and the alert together.
- **Mitigation moves this metric the wrong way, on purpose.** Group-aware
  thresholds cut equalized-odds difference from 0.0725 to 0.0202 and widen the
  selection-rate gap from 0.025 to 0.040. Demographic parity and equalized odds
  are different definitions of fair; improving one degrades the other, and the
  monitor is what makes that visible instead of theoretical.

### Monitoring people is itself an ethical act

`credit_selection_rate{group}` requires knowing each requester's group. That is a
sensitive attribute flowing into a metrics pipeline, so:

- The label is the **group**, never the individual. Cardinality stays bounded and
  no metric can be traced to a person.
- Only aggregates are exported. No request payload, no identifier, no raw feature
  value is ever a label.
- Grafana runs anonymous read-only **in this coursework build**. In a real
  deployment, group-disaggregated statistics would need access control of their
  own — "who is allowed to see default rates broken down by sex" is a real
  question with a real answer, and that answer is not "anyone on the network".

We note the tension plainly: you cannot detect discrimination without measuring
the protected attribute, and measuring it creates its own risk. The resolution is
aggregation and access control, not avoidance — a system that refuses to look
cannot report that it is fair.

### Alerts are addressed to humans, on purpose

`FairnessGapExceeded` is meant for a person. Whether it reaches one depends on
configuration, and the default is no. With a Telegram bot token and chat id
mounted as files (`docs/ALERTING.md`), it is delivered to that chat with the other
warnings — silently, in a group of its own, with a description that says what to
check and that it goes to compliance. Without them, which is how a fresh clone
starts, it is recorded and shown in the Alertmanager UI and in Grafana and
reaches someone only if they look. Either way it does not trigger an automatic
retrain, a threshold adjustment, or a rollback.
That is a design decision, not a missing feature: a system that automatically
adjusts its own fairness behaviour in response to its own metrics is one where
nobody can say afterwards who decided what. Keeping a human in the loop keeps
accountability locatable.

---

## 5. Feedback loops, and the limit of what we can verify (all)

The most serious problem with this system is one we cannot measure from inside
it.

If the model flags an account and the bank reduces that person's credit limit,
the reduction itself may push them into default. The next training round then
observes a default at that account and records the model as having been correct.
The model gets more confident about a pattern it helped create.

This is a **self-fulfilling prophecy**, and it is invisible to every metric in
the repository — PR-AUC, calibration, drift, fairness, all of them look fine
while it happens.

The standard mitigation is a **held-out control group**: a random fraction of
flagged accounts receives no intervention, so the counterfactual stays
observable. That costs money, it is a deliberate decision not to act on
information you have, and it is the only way to find out whether the system helps.

We cannot implement it — our data is a 2005 snapshot with no intervention arm and
no follow-up. So we record it as the first thing a real deployment would need,
and as the clearest example in this project of a problem that **more engineering
does not solve**.

---

## 6. Limits of this work

Stated plainly so nobody has to infer them.

| Limit | Consequence |
|---|---|
| Data is from Taiwan, 2005 | Nothing here transfers to another population or another decade without revalidation |
| No causal identification | The model captures association. Every statement about "drivers of risk" is a statement about correlation |
| Simulated drift and simulated batch arrival | Demonstrates the monitoring machinery; proves nothing about real temporal behaviour |
| No control group | Feedback loops are undetectable |
| Single protected attribute in the gate | `SEX × AGE_GROUP` is reported on every run but not gated, and its equalized-odds gap (0.1061) is above the level that gates `SEX`. Three-way intersections — say, young women with high-school education — are not computed |
| No human-subjects review, no consent record | Unknown provenance for the original collection; recorded as unknown rather than assumed acceptable |
| Coursework build | Open Grafana, no authentication on the API, no audit log, no retention policy |

**This model must not be used to make real credit decisions about real people.**
Not because the code is bad, but because it has never been validated on a current
population, never reviewed by a credit-risk professional, and never tested for
the feedback loop described in section 5.
