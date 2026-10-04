#!/usr/bin/env bash
#
# Smoke test for a running stack. Answers one question: is this deployment
# actually able to do its job, or does it merely have an open port?
#
# Run by `make smoke`, by the CI smoke job, and by hand before a demo.
#
#   BASE_URL              default http://localhost:18000
#   SMOKE_WAIT_SECONDS    how long to wait for the API to become ready (120)
#   SMOKE_ALLOW_DEGRADED  1 = tolerate an API with no model loaded (see below)
#
# Every curl here uses -f. Without it curl exits 0 on an HTTP 500 and prints the
# error body to stdout, so a smoke test written without -f passes against a
# completely broken service. That is not a hypothetical; it is the single most
# common way these scripts lie.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:18000}"
SMOKE_WAIT_SECONDS="${SMOKE_WAIT_SECONDS:-120}"
SMOKE_ALLOW_DEGRADED="${SMOKE_ALLOW_DEGRADED:-0}"
CURL=(curl -fsS --max-time 15)

if [[ -t 1 ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; DIM=''; OFF=''
fi

failures=0
step()  { printf '\n%s==>%s %s\n' "$DIM" "$OFF" "$*"; }
ok()    { printf '    %sok%s   %s\n' "$GREEN" "$OFF" "$*"; }
warn()  { printf '    %swarn%s %s\n' "$YELLOW" "$OFF" "$*"; }
bad()   { printf '    %sFAIL%s %s\n' "$RED" "$OFF" "$*"; failures=$((failures + 1)); }
die()   { printf '\n%sFAIL%s %s\n' "$RED" "$OFF" "$*"; exit 1; }

# The metric families every alert rule and dashboard panel queries. prometheus
# client emits the HELP and TYPE lines for a declared metric even when it has no
# samples yet, so a missing name here means the serving code stopped declaring
# it -- at which point the alert never fires and the system looks healthy.
ALERTED_METRICS=(
  credit_predictions_total
  credit_errors_total
  credit_prediction_latency_seconds
  credit_explain_duration_seconds
  credit_model_loaded
  credit_model_info
  credit_high_risk_share
  credit_baseline_high_risk_share
  credit_selection_rate
  credit_feature_psi
)

SAMPLE_ACCOUNT='{
  "account_id": "A-000042",
  "LIMIT_BAL": 200000.0,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 37,
  "PAY_1": 2, "PAY_2": 2, "PAY_3": 1, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
  "BILL_AMT1": 189000.0, "BILL_AMT2": 182000.0, "BILL_AMT3": 175000.0,
  "BILL_AMT4": 168000.0, "BILL_AMT5": 160000.0, "BILL_AMT6": 151000.0,
  "PAY_AMT1": 6000.0, "PAY_AMT2": 6500.0, "PAY_AMT3": 7000.0,
  "PAY_AMT4": 7200.0, "PAY_AMT5": 7500.0, "PAY_AMT6": 8000.0
}'

printf '%s\n' "smoke test against ${BASE_URL}"

# ---------------------------------------------------------------- readiness

step "waiting for ${BASE_URL}/health to answer"
health=''
deadline=$(( $(date +%s) + SMOKE_WAIT_SECONDS ))
while [[ $(date +%s) -lt $deadline ]]; do
  if health=$("${CURL[@]}" "${BASE_URL}/health" 2>/dev/null); then
    break
  fi
  health=''
  sleep 2
done
[[ -n "$health" ]] || die "no 200 from /health within ${SMOKE_WAIT_SECONDS}s -- is the stack up? (make up)"
ok "/health responded"

# A 200 from /health is not the same as a working service. The API is designed
# to start, bind the port and answer /health even when MLflow was unreachable
# and no model could be pulled -- and then return 503 from every prediction.
# This is the distinction ModelNotLoaded exists to alert on, so the smoke test
# has to make it too.
step "waiting for a model to be loaded"
model_loaded=0
while [[ $(date +%s) -lt $deadline ]]; do
  if health=$("${CURL[@]}" "${BASE_URL}/health" 2>/dev/null) &&
     grep -qE '"model_loaded"[[:space:]]*:[[:space:]]*(true|1)' <<<"$health"; then
    model_loaded=1
    break
  fi
  sleep 2
done

if [[ $model_loaded -eq 1 ]]; then
  ok "model_loaded is true"
  printf '    %s%s%s\n' "$DIM" "$health" "$OFF"
elif [[ "$SMOKE_ALLOW_DEGRADED" == "1" ]]; then
  # CI has no registered model: nothing in the pipeline trains one, and training
  # LightGBM inside the smoke job would turn a 90-second check into a 6-minute
  # one. With this flag the job still proves the stack composes, the API serves,
  # and every metric the alert rules query is declared. The demo runs the same
  # script with the flag off, where a missing model is a hard failure.
  warn "no model loaded; continuing because SMOKE_ALLOW_DEGRADED=1"
  warn "this is the exact state ModelNotLoaded alerts on"
else
  die "API is up but no model is loaded after ${SMOKE_WAIT_SECONDS}s (check MLflow and the Production stage)"
fi

# ---------------------------------------------------------------- endpoints

if [[ $model_loaded -eq 1 ]]; then
  step "the decision threshold is the serving version's own"
  # "fallback" is a working service deciding at DECISION_THRESHOLD because the
  # serving version carries no threshold_at_k tag. A warning, not a failure: the
  # model serves, but not at the cutoff its capacity was computed for.
  if grep -qE '"threshold_source"[[:space:]]*:[[:space:]]*"fallback"' <<<"$health"; then
    warn "threshold_source is fallback -- run: python -m credit_risk.models.registry tag-threshold --version <N>, then docker compose restart credit-api"
  elif grep -qE '"threshold_source"' <<<"$health"; then
    ok "$(grep -oE '"threshold_source"[[:space:]]*:[[:space:]]*"[a-z]+"' <<<"$health")"
  else
    warn "/health reports no threshold_source -- the API image predates it"
  fi

  step "POST /api/v1/predict"
  prediction=$("${CURL[@]}" -X POST "${BASE_URL}/api/v1/predict" \
    -H 'Content-Type: application/json' -d "$SAMPLE_ACCOUNT") ||
    die "/api/v1/predict did not return 2xx"
  for expected in default_probability decision model_version; do
    if grep -q "\"${expected}\"" <<<"$prediction"; then
      ok "response carries ${expected}"
    else
      bad "response is missing ${expected}: ${prediction}"
    fi
  done

  step "POST /api/v1/explain"
  if explanation=$("${CURL[@]}" -X POST "${BASE_URL}/api/v1/explain" \
      -H 'Content-Type: application/json' -d "$SAMPLE_ACCOUNT" 2>/dev/null); then
    for expected in shap lime; do
      grep -q "\"${expected}\"" <<<"$explanation" &&
        ok "explanation carries ${expected}" ||
        bad "explanation is missing ${expected}"
    done
  else
    bad "/api/v1/explain did not return 2xx"
  fi

  step "a malformed payload is rejected, not accepted"
  # 422 is the correct answer. A 200 here means the API is scoring rubbish, and
  # a 500 means it is crashing on input a caller can send by accident.
  status=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 \
    -X POST "${BASE_URL}/api/v1/predict" \
    -H 'Content-Type: application/json' -d '{"account_id":"A-1","LIMIT_BAL":"abc"}')
  [[ "$status" == "422" ]] && ok "malformed payload -> 422" || bad "malformed payload -> ${status}, expected 422"
fi

# ------------------------------------------------------------------ metrics

step "GET /metrics"
metrics=$("${CURL[@]}" "${BASE_URL}/metrics") || die "/metrics did not return 2xx"
grep -q '^# TYPE ' <<<"$metrics" || die "/metrics returned no Prometheus exposition"
ok "metrics endpoint served $(wc -l <<<"$metrics" | tr -d ' ') lines"

step "every metric the alert rules query is declared"
for metric in "${ALERTED_METRICS[@]}"; do
  if grep -qE "^# (HELP|TYPE) ${metric}([[:space:]]|$)" <<<"$metrics"; then
    ok "${metric}"
  else
    bad "${metric} is not exported -- every alert rule reading it can never fire"
  fi
done

if [[ $model_loaded -eq 1 ]]; then
  step "the drift baseline was installed at model load"
  # The gap the HELP/TYPE check above cannot see: prometheus_client emits the
  # HELP and TYPE lines for a declared metric family whether or not anything
  # ever calls set_baseline(), so credit_feature_psi passes that check with zero
  # series and FeatureDriftHigh can never fire. The label set is what proves the
  # baseline was installed -- the VALUES only appear after PSI_MIN_SAMPLES
  # requests, which a smoke run does not send, so this is a warn and not a fail.
  if grep -q '^credit_feature_psi{' <<<"$metrics"; then
    ok "credit_feature_psi has $(grep -c '^credit_feature_psi{' <<<"$metrics") feature series"
  else
    warn "credit_feature_psi is declared but has no series yet -- expected on a cold API (PSI needs ~50 scored requests). If it is still empty after sustained traffic, the reference sample never loaded and FeatureDriftHigh cannot fire."
  fi

  step "the high-risk baseline gauge is populated"
  # HighRiskShareShift is gated on this being non-zero, so an unset gauge is a
  # rule that silently never evaluates rather than a rule that reads wrong.
  baseline=$(grep -E '^credit_baseline_high_risk_share ' <<<"$metrics" | awk '{print $2}')
  if [[ -n "$baseline" ]] && awk "BEGIN{exit !($baseline > 0)}"; then
    ok "credit_baseline_high_risk_share = ${baseline}"
  else
    warn "credit_baseline_high_risk_share is 0 -- HighRiskShareShift is gated on it being set from registry metadata and will never fire until it is"
  fi

  step "histogram buckets exist for the latency SLO"
  # SlowPredictions runs histogram_quantile over this exact series. The family
  # name alone is not enough: with no observations there are no _bucket samples
  # and the rule evaluates to nothing, forever.
  if grep -q '^credit_prediction_latency_seconds_bucket{' <<<"$metrics"; then
    ok "credit_prediction_latency_seconds_bucket"
  else
    bad "no _bucket samples after a successful prediction"
  fi
fi

step "no per-request label has crept into the metrics"
# The Lab 04 trap, asserted rather than remembered: one label carrying an
# account id turned 23 time series into 639. If this ever matches, a deploy is
# about to make Prometheus the slowest process in the compose file.
if offenders=$(grep -oE '^credit_[a-z_]+\{([^}]*,)?(account|account_id|request_id|user|id)=' <<<"$metrics" | sort -u); then
  bad "per-request labels found:"
  printf '         %s\n' "$offenders"
else
  ok "labels are bounded (no account/request identifiers)"
fi

# -------------------------------------------------------------------- verdict

printf '\n'
if [[ $failures -eq 0 ]]; then
  printf '%sSMOKE PASSED%s\n' "$GREEN" "$OFF"
  exit 0
fi
printf '%sSMOKE FAILED%s — %d check(s)\n' "$RED" "$OFF" "$failures"
exit 1
