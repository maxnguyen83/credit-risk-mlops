#!/bin/sh
# Picks the Alertmanager config, then becomes Alertmanager.
#
#   telegram_bot_token AND telegram_chat_id readable in the secrets mount
#       -> alertmanager.telegram.yml, chat id filled in, checked with amtool
#   anything else
#       -> alertmanager.yml, where every receiver is a no-op
#
# Anything short of a complete, valid pair falls back to the no-op config with
# a WARNING in `docker compose logs alertmanager`, rather than exiting: a typo
# in a notifier must never be the reason the alert UI is down.
#
# Runs as `nobody` under the busybox /bin/sh of prom/alertmanager. The secrets
# are only read here; the filled-in config is written to /tmp inside the
# container and never to the read-only mount. It runs again on every
# `docker compose restart alertmanager`, which is how a new chat id is picked
# up (a new token needs no restart: Alertmanager reads that file on every
# notification).
#
# SECRETS_DIR and AM_BIN exist for tests/test_config.sh, which runs this with
# AM_BIN=echo to see which config would be started.

set -u

CONFIG_DIR=/etc/alertmanager
SECRETS_DIR=${SECRETS_DIR:-/run/secrets/alertmanager}
AM_BIN=${AM_BIN:-/bin/alertmanager}
TOKEN_FILE=$SECRETS_DIR/telegram_bot_token
CHAT_FILE=$SECRETS_DIR/telegram_chat_id
GENERATED=/tmp/alertmanager.yml

log() { echo "entrypoint: $*" >&2; }
off() { log "WARNING: $* -- Telegram delivery is OFF, starting with the no-op config"; }

telegram_config() {
  # Nothing configured is the normal case, not a warning.
  if [ ! -e "$TOKEN_FILE" ] && [ ! -e "$CHAT_FILE" ]; then
    log "alert delivery: off (no Telegram secrets in $SECRETS_DIR)"
    return 1
  fi
  for f in "$TOKEN_FILE" "$CHAT_FILE"; do
    if [ ! -s "$f" ] || [ ! -r "$f" ]; then
      off "$f is missing, empty or not readable by uid $(id -u)"
      return 1
    fi
  done

  chat_id=$(tr -d ' \t\r\n' < "$CHAT_FILE")
  case $chat_id in
    '' | - | *[!0-9-]* | ?*-* | 0 | -0*)
      off "telegram_chat_id must be a non-zero integer (a group's id is negative)"
      return 1
      ;;
  esac

  if ! sed "s/^\( *chat_id:\) .*@TELEGRAM_CHAT_ID@.*$/\1 $chat_id/" \
    "$CONFIG_DIR/alertmanager.telegram.yml" > "$GENERATED"; then
    off "could not write $GENERATED"
    return 1
  fi
  if ! grep -q "^ *chat_id: $chat_id\$" "$GENERATED" || grep -q '@TELEGRAM_CHAT_ID@' "$GENERATED"; then
    off "could not fill the chat id in: the '# @TELEGRAM_CHAT_ID@' line in alertmanager.telegram.yml changed"
    return 1
  fi
  if ! check=$(amtool check-config "$GENERATED" 2>&1); then
    log "$check"
    off "the filled-in Telegram config does not pass amtool check-config"
    return 1
  fi
  log "alert delivery: Telegram (bot token from $TOKEN_FILE)"
  return 0
}

if telegram_config; then
  config=$GENERATED
else
  config=$CONFIG_DIR/alertmanager.yml
fi

exec "$AM_BIN" --config.file="$config" --storage.path=/alertmanager "$@"
