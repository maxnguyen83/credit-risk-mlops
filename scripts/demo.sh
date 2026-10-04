#!/usr/bin/env bash
#
# The 15-minute presentation, scripted end to end.
#
# Nobody types during a demo. Typing is how you discover that the port was
# 18000 and not 8000, live, with an audience. Every command below is the command
# that will actually be run, in the order it will be run, with the narration
# printed above it so the presenter reads rather than remembers.
#
#   ./scripts/demo.sh              interactive: pauses for Enter between beats
#   DEMO_AUTO=1 ./scripts/demo.sh  unattended: fixed pauses, for rehearsal or
#                                  for recording the backup video
#   DEMO_SKIP_UP=1 ./scripts/demo.sh   stack is already running, skip the build
#
# Runtime is about 15 minutes with the default traffic durations. Shorten with
# DEMO_TRAFFIC_SECONDS if the slot is tighter.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

API_URL="${API_URL:-http://localhost:18000}"
PROM_URL="${PROM_URL:-http://localhost:19090}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:19093}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:13000}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:15020}"

DEMO_AUTO="${DEMO_AUTO:-0}"
DEMO_SKIP_UP="${DEMO_SKIP_UP:-0}"
DEMO_TRAFFIC_SECONDS="${DEMO_TRAFFIC_SECONDS:-90}"
PAUSE_SECONDS="${PAUSE_SECONDS:-6}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

if [[ -t 1 ]]; then
  B=$'\033[1m'; CYAN=$'\033[36m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
  B=''; CYAN=''; GREEN=''; YELLOW=''; DIM=''; OFF=''
fi

chapter() {
  printf '\n\n%s%s══ %s %s%s\n' "$B" "$CYAN" "$1" "${2:-}" "$OFF"
  printf '%s%s%s\n' "$DIM" "$(printf '─%.0s' {1..72})" "$OFF"
}
say()   { printf '\n%s\n' "$*"; }
point() { printf '  %s→%s %s\n' "$YELLOW" "$OFF" "$*"; }
run() {
  printf '\n%s$ %s%s\n' "$GREEN" "$*" "$OFF"
  "$@" || printf '  %s(command returned %d — continuing)%s\n' "$DIM" "$?" "$OFF"
}
pause() {
  if [[ "$DEMO_AUTO" == "1" ]]; then
    sleep "$PAUSE_SECONDS"
  else
    printf '\n%s[Enter to continue]%s ' "$DIM" "$OFF"
    read -r _ || true
  fi
}

# Alert state straight from Prometheus, formatted so it is readable from the
# back of a room. Prometheus is the source of truth here rather than Grafana:
# the point is that the rule evaluated, not that a panel turned red.
alerts() {
  curl -fsS "${PROM_URL}/api/v1/alerts" 2>/dev/null | "$PYTHON" -c '
import json, sys
try:
    data = json.load(sys.stdin)["data"]["alerts"]
except Exception:
    print("  (could not read alerts from Prometheus)"); raise SystemExit(0)
if not data:
    print("  no alerts pending or firing"); raise SystemExit(0)
order = {"firing": 0, "pending": 1}
for a in sorted(data, key=lambda a: (order.get(a["state"], 9), a["labels"]["alertname"])):
    name = a["labels"]["alertname"]
    sev = a["labels"].get("severity", "-")
    extra = a["labels"].get("feature", "")
    mark = "FIRING " if a["state"] == "firing" else "pending"
    print(f"  [{mark}] {name:<22} severity={sev:<8} {extra}")
' || echo "  (Prometheus not reachable at ${PROM_URL})"
}

# Alert rules as Prometheus actually loaded them, not as the YAML claims.
rules() {
  curl -fsS "${PROM_URL}/api/v1/rules" 2>/dev/null | "$PYTHON" -c '
import json, sys
try:
    groups = json.load(sys.stdin)["data"]["groups"]
except Exception:
    print("  (could not read rules from Prometheus)"); raise SystemExit(0)
loaded = [r for g in groups for r in g["rules"] if r["type"] == "alerting"]
print(f"  {len(loaded)} alert rules loaded, every one with a for: duration")
for rule in sorted(loaded, key=lambda r: r["name"]):
    name = rule["name"]
    seconds = int(rule.get("duration", 0))
    severity = rule["labels"].get("severity", "-")
    print(f"    {name:<24} for={seconds:>5}s  severity={severity}")
' || echo "  (Prometheus not reachable at ${PROM_URL})"
}

traffic() {
  printf '\n%s$ %s scripts/traffic.py %s%s\n' "$GREEN" "$PYTHON" "$*" "$OFF"
  "$PYTHON" scripts/traffic.py "$@" || point "traffic generator exited non-zero (is the API up?)"
}

# ============================================================ 0. preflight

chapter "0" "Preflight"

docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Colima or Docker Desktop first."; exit 1; }
[[ -f docker-compose.yml ]] || { echo "run this from the repository root"; exit 1; }
if [[ ! -f .env ]]; then
  say "No .env yet — copying .env.example. It contains placeholders only; no secret is committed."
  cp .env.example .env
fi
point "API          ${API_URL}"
point "Prometheus   ${PROM_URL}"
point "Alertmanager ${ALERTMANAGER_URL}"
point "Grafana      ${GRAFANA_URL}  (admin / admin)"
point "MLflow       ${MLFLOW_URL}"
say "Ports are deliberately in the 1xxxx range: four other lab stacks live on this machine and 8000/9090/3000 are all taken."
pause

# ====================================================== 1. bring the stack up

chapter "1" "The stack — nine services, one command"

say "Everything is declared in docker-compose.yml: Postgres and an S3 object store behind MLflow, Airflow for the pipeline, the API, and Prometheus + Alertmanager + Grafana for observability. No cloud, no Kubernetes: one model, one team, one machine. A control plane costs real operational work and buys nothing at this scale."

if [[ "$DEMO_SKIP_UP" == "1" ]]; then
  point "DEMO_SKIP_UP=1 — assuming the stack is already running"
else
  run docker compose up -d --build
fi
run docker compose ps
say "Health checks, not just 'the container started'. depends_on: service_healthy is what stops the API racing MLflow to the registry."
pause

# ================================================== 2. health and which model

chapter "2" "Is it healthy — and which model is answering?"

say "There are two different questions here and most systems only answer the first."
run bash -c "curl -fsS ${API_URL}/health | ${PYTHON} -m json.tool"
point "model_loaded is the field that matters. The API is designed to start and answer /health even when it could not reach MLflow — and then return 503 from every prediction. A port check calls that healthy."
run bash -c "curl -fsS ${API_URL}/metrics | grep -E '^credit_model_(loaded|info)' || true"
say "credit_model_info carries the version, the algorithm, the training timestamp and the git sha as labels. During an incident the first question is always 'which model is running', and this is the only place that answers it."
pause

# ============================================================ 3. a prediction

chapter "3" "One account, one decision"

say "A risk officer looking at a single cardholder. Two months late, using 95% of the limit, payments shrinking."
run bash -c "curl -fsS -X POST ${API_URL}/api/v1/predict -H 'Content-Type: application/json' -d @- <<'JSON' | ${PYTHON} -m json.tool
{\"account_id\":\"A-000042\",\"LIMIT_BAL\":200000.0,\"SEX\":2,\"EDUCATION\":2,\"MARRIAGE\":1,\"AGE\":37,
 \"PAY_1\":2,\"PAY_2\":2,\"PAY_3\":1,\"PAY_4\":0,\"PAY_5\":0,\"PAY_6\":0,
 \"BILL_AMT1\":189000.0,\"BILL_AMT2\":182000.0,\"BILL_AMT3\":175000.0,
 \"BILL_AMT4\":168000.0,\"BILL_AMT5\":160000.0,\"BILL_AMT6\":151000.0,
 \"PAY_AMT1\":6000.0,\"PAY_AMT2\":6500.0,\"PAY_AMT3\":7000.0,
 \"PAY_AMT4\":7200.0,\"PAY_AMT5\":7500.0,\"PAY_AMT6\":8000.0}
JSON"
point "The response carries threshold_used and threshold_policy, not just a probability. A decision you cannot reconstruct later is a decision you cannot defend to a regulator."
pause

# =========================================================== 4. an explanation

chapter "4" "Why — SHAP and LIME, on the same account"

say "Consumer credit law in most jurisdictions requires a reason when someone is declined or has their limit cut. That makes explainability a business requirement here, not a garnish."
run bash -c "curl -fsS -X POST ${API_URL}/api/v1/explain -H 'Content-Type: application/json' -d @- <<'JSON' | ${PYTHON} -m json.tool | head -60
{\"account_id\":\"A-000042\",\"LIMIT_BAL\":200000.0,\"SEX\":2,\"EDUCATION\":2,\"MARRIAGE\":1,\"AGE\":37,
 \"PAY_1\":2,\"PAY_2\":2,\"PAY_3\":1,\"PAY_4\":0,\"PAY_5\":0,\"PAY_6\":0,
 \"BILL_AMT1\":189000.0,\"BILL_AMT2\":182000.0,\"BILL_AMT3\":175000.0,
 \"BILL_AMT4\":168000.0,\"BILL_AMT5\":160000.0,\"BILL_AMT6\":151000.0,
 \"PAY_AMT1\":6000.0,\"PAY_AMT2\":6500.0,\"PAY_AMT3\":7000.0,
 \"PAY_AMT4\":7200.0,\"PAY_AMT5\":7500.0,\"PAY_AMT6\":8000.0}
JSON"
point "top_reasons is a draft adverse-action notice in plain language."
point "agreement reports how far SHAP and LIME overlap — which answers the obvious challenge before anyone has to ask it."
pause

# =========================================================== 5. the dashboards

chapter "5" "Three dashboards, provisioned from the repo"

point "Service Health    ${GRAFANA_URL}/d/credit-service-health"
point "Model Behaviour   ${GRAFANA_URL}/d/credit-model-behaviour"
point "Fairness Monitor  ${GRAFANA_URL}/d/credit-fairness-monitor"
say "Nobody clicked these into existence. The JSON is in monitoring/grafana/dashboards/ and the datasource uid is pinned, so they come up populated on a fresh clone."
if [[ "$DEMO_AUTO" != "1" ]] && command -v open >/dev/null 2>&1; then
  open "${GRAFANA_URL}/d/credit-service-health" >/dev/null 2>&1 || true
fi
say "Open Service Health now and leave it on screen for the next three chapters."
pause

# ========================================================= 6. normal traffic

chapter "6" "Baseline — what healthy looks like"

say "Replaying batch 6, the 5,000 accounts held back as the production traffic pool. Nothing is wrong with any of this."
traffic --rps 20 --seconds "${DEMO_TRAFFIC_SECONDS}"
say "On Service Health: throughput split by decision, p50/p95/p99 from the histogram buckets, error ratio flat at zero."
say "On Model Behaviour: the high-risk share settles on credit_baseline_high_risk_share — the rate this exact model flagged on its evaluation set, published as a gauge at model load. Note it is NOT the 22.12% default rate: 22% of these accounts do default, but the model puts far fewer than that on the intervention list at its cutoff. Comparing a model's output rate against the label prevalence is how you get an alert that fires on day one and is muted by day two."
say "Alert state right now:"
alerts
pause

# =============================================================== 7. drift

chapter "7" "Scenario A — the data moves"

say "Now the population changes. Every monetary feature — limit, six months of bills, six months of payments — shifts by three standard deviations. In the real world this is a portfolio expansion, a change upstream, or a new segment being onboarded."
point "Expect: credit_feature_psi climbs past 0.25 → FeatureDriftHigh (20m)"
point "Expect: the high-risk share pulls away from credit_baseline_high_risk_share → HighRiskShareShift (15m)"
traffic --drift 3.0 --rps 20 --seconds "${DEMO_TRAFFIC_SECONDS}"
say "Model Behaviour → 'Feature PSI (current)': the bars go amber and then red. Below 0.10 is stable, 0.10–0.25 moderate, above 0.25 the training distribution is gone."
run bash -c "curl -fsS '${PROM_URL}/api/v1/query?query=max(credit_feature_psi)' | ${PYTHON} -m json.tool | head -20"
say "Alert state:"
alerts
point "The 'for:' durations are real — 20 minutes for drift. In a live slot you will usually see these Pending rather than Firing, and Pending is the honest thing to show. An alert that fires in ten seconds is an alert that will page someone at 3am for a blip."
pause

# =============================================================== 8. bias

chapter "8" "Scenario B — nothing is broken, and no operational metric can see it"

say "This is the one to watch. The request mix skews hard towards male cardholders. No malformed payloads, no slow queries, no errors."
point "Watch Service Health while this runs. Latency will not move. The error ratio will stay at zero. Every response will be a 200."
# 0.7 rather than 0.9 on purpose. At 90% male the women left in the 200-slot
# window fall under the serving-side minimum-sample floor and their series stops
# being published — correct behaviour, but it leaves the gap expression with one
# series and nothing to show. 0.7 keeps both groups measurable.
traffic --bias 0.7 --rps 20 --seconds "${DEMO_TRAFFIC_SECONDS}"
say "Now switch to Fairness Monitor."
run bash -c "curl -fsS '${PROM_URL}/api/v1/query?query=credit_selection_rate' | ${PYTHON} -m json.tool | head -30"
run bash -c "curl -fsS '${PROM_URL}/api/v1/query?query=max(credit_selection_rate)-min(credit_selection_rate)' | ${PYTHON} -m json.tool | head -20"
say "Two live selection rates, one per protected group, measured on production traffic — not on a test set in a notebook nobody re-runs. The gap between them is evaluated every 15 seconds against 0.05, the same demographic-parity threshold the training pipeline gates on."
point "Every operational dashboard is green. Uptime 100%, errors zero, p95 inside budget. And the only number in the building that describes WHO the system is acting on is this one."
point "Be precise about what the skew does and does not do: it does not make the model discriminate. A selection rate is a within-group rate, so reweighting the mix leaves each group's rate unchanged in expectation — it changes how precisely each is measured. That is why the serving code refuses to publish a group holding fewer than 30 of the 200 window slots, and why the alert carries a 15-minute for:. A fairness alert that fires on a small sample is a fairness alert that gets muted, and a muted fairness alert is worse than none."
point "Push the skew further and the minority series disappears rather than turning red. That is the monitor declining to measure what it cannot measure, which is the honest behaviour and the one worth understanding."
say "Alert state:"
alerts
pause

# ============================================================== 9. wrap up

chapter "9" "What is left running"

say "Every rule Prometheus is evaluating, read back from Prometheus itself rather than from the YAML — the count below is whatever actually loaded:"
rules
point "Alertmanager  ${ALERTMANAGER_URL}  — routed by severity: critical and pipeline failures within seconds, warnings grouped per component, info never sent"
point "Delivery is a switch, not an edit: with a Telegram bot token and chat id mounted as files, the same routes deliver to that chat; without them every receiver is a no-op and alerts are recorded and displayed only. Anything incomplete falls back to no-op with a warning — an unconfigured notifier must never stop the stack from starting, and one that retries against a bot that does not exist is not 'off'. docs/ALERTING.md has the steps."
say
say "To reset: docker compose down -v"
say "To re-run any single scenario:  make traffic | make drift | make bias"
printf '\n%s%sDemo complete.%s\n\n' "$B" "$GREEN" "$OFF"
