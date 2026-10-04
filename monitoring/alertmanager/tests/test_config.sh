#!/bin/sh
# Tests for the Alertmanager configs and the switch between them. Runs inside
# the prom/alertmanager image, which has amtool and a busybox shell, so the
# amtool here parses YAML exactly as the deployed Alertmanager does:
#
#   docker run --rm --entrypoint /bin/sh \
#     -v "$PWD/monitoring/alertmanager:/etc/alertmanager:ro" \
#     prom/alertmanager:v0.27.0 /etc/alertmanager/tests/test_config.sh
#
# (CI runs it in the `alert rules` job.) Checks, in order:
#   1. both configs load;
#   2. they share one routing tree, byte for byte;
#   3. each kind of alert reaches the receiver meant for it, in both;
#   4. entrypoint.sh starts Telegram only on a complete, valid pair of secrets
#      and falls back to the no-op config on anything else.

set -eu

AM=/etc/alertmanager
NOOP=$AM/alertmanager.yml
TELEGRAM=$AM/alertmanager.telegram.yml
failures=0

ok() { echo "ok    $*"; }
fail() {
  echo "FAIL  $*"
  failures=$((failures + 1))
}

# --- 1. both configs load --------------------------------------------------
for f in "$NOOP" "$TELEGRAM"; do
  if amtool check-config "$f" > /dev/null; then ok "check-config $f"; else fail "check-config $f"; fi
done

# --- 2. one routing tree -----------------------------------------------------
sed -n '/^route:/,/^receivers:/p' "$NOOP" > /tmp/noop.route
sed -n '/^route:/,/^receivers:/p' "$TELEGRAM" > /tmp/telegram.route
if diff -u /tmp/noop.route /tmp/telegram.route; then
  ok "route: and inhibit_rules: identical in both configs"
else
  fail "route: or inhibit_rules: differ between the configs (diff above)"
fi

# --- 3. where each kind of alert lands -------------------------------------
# expect RECEIVER LABEL=VALUE...
expect() {
  want=$1
  shift
  for f in "$NOOP" "$TELEGRAM"; do
    if got=$(amtool config routes test --config.file="$f" --verify.receivers="$want" "$@" 2>&1); then
      ok "$* -> $want ($(basename "$f"))"
    else
      fail "$* -> expected $want, got: $got ($(basename "$f"))"
    fi
  done
}

expect pipeline alertname=PipelineTaskFailed severity=critical dag_id=credit_risk_pipeline task_id=train_candidates
expect critical alertname=ModelNotLoaded severity=critical component=model
expect critical alertname=ApiDown severity=critical component=serving
expect warning alertname=FairnessGapExceeded severity=warning component=fairness
expect warning alertname=FeatureDriftHigh severity=warning component=model feature=PAY_0
expect null alertname=SlowExplanations severity=info component=serving
expect null alertname=FromSomewhereElse

# --- 4. the start-up switch ------------------------------------------------
# AM_BIN=echo makes the entrypoint print the flags it would start Alertmanager
# with instead of starting it.
secrets=$(mktemp -d)
starts() {
  SECRETS_DIR=$1 AM_BIN=echo sh "$AM/entrypoint.sh" 2> /dev/null
}
expect_start() {
  case_name=$1
  want=$2
  if starts "$3" | grep -q -- "--config.file=$want "; then
    ok "start, $case_name -> $want"
  else
    fail "start, $case_name -> expected $want, got: $(starts "$3")"
  fi
}

expect_start "no secrets directory" "$NOOP" /nonexistent
expect_start "empty secrets directory" "$NOOP" "$secrets"

printf '%s' '123456:not-a-real-token' > "$secrets/telegram_bot_token"
expect_start "token without chat id" "$NOOP" "$secrets"

# Leading zeros are octal to YAML (0123 would become chat 83); two lines would
# be glued into one number; anything but digits and a leading minus is not an id.
for bad in abc 12-3 '12 34' 0 00 0123 -0123 '-' '' '123\n456' '  \n\n'; do
  printf '%b' "$bad" > "$secrets/telegram_chat_id"
  expect_start "chat id '$bad'" "$NOOP" "$secrets"
done

# filled_in ID: the generated config carries exactly this chat id and loads.
filled_in() {
  if grep -q "^ *chat_id: $1\$" /tmp/alertmanager.yml \
    && amtool check-config /tmp/alertmanager.yml > /dev/null; then
    ok "filled-in config carries chat id $1 and loads"
  else
    fail "filled-in config: chat id $1 not substituted, or it does not load"
  fi
}

printf '%s\n' '-1001234567890' > "$secrets/telegram_chat_id"
expect_start "token and group chat id" /tmp/alertmanager.yml "$secrets"
filled_in -1001234567890

# Blank lines, surrounding spaces and a Windows line ending are tolerated.
printf '\n  4242  \r\n\n' > "$secrets/telegram_chat_id"
expect_start "token and private chat id, padded" /tmp/alertmanager.yml "$secrets"
filled_in 4242

chmod 000 "$secrets/telegram_bot_token"
expect_start "unreadable token file" "$NOOP" "$secrets"

rm -f "$secrets/telegram_bot_token"
expect_start "chat id without token" "$NOOP" "$secrets"
rm -rf "$secrets"

echo
if [ "$failures" -ne 0 ]; then
  echo "$failures check(s) failed"
  exit 1
fi
echo "all alertmanager checks passed"
