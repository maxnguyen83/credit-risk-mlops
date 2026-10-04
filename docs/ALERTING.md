# Alert delivery

Prometheus evaluates the rules in `monitoring/prometheus/alerts/` and sends what
fires to Alertmanager. Alertmanager groups, deduplicates and inhibits, then hands
each group to a receiver. Whether a receiver sends anything depends on one thing:
whether a Telegram bot token and chat id are present in the secrets directory
when the container starts.

- **No secrets (the default).** Every receiver is a no-op. Alerts are visible in
  the Alertmanager UI (<http://127.0.0.1:19093>) and in Grafana, and go no
  further. The stack needs no secret to start.
- **Both secrets present.** The same routes deliver to one Telegram chat.

`monitoring/alertmanager/entrypoint.sh` makes that choice on every container
start. Anything short of a complete, valid pair — one file missing, an empty or
unreadable file, a chat id that is not an integer — falls back to the no-op
config and logs a `WARNING`; it never stops Alertmanager from starting.

## Where each alert goes

| Alert | Receiver | First message after | Repeats while firing | In Telegram |
|---|---|---|---|---|
| `PipelineTaskFailed` (posted by the Airflow DAG) | `pipeline` | 10 s, grouped per DAG | every 12 h, if it is still active | with sound; no "resolved" message |
| `severity="critical"`: `ApiDown`, `ModelNotLoaded` | `critical` | 10 s | every 1 h | with sound; "resolved" when it clears |
| `severity="warning"`: `HighErrorRate`, `Slow*Predictions`, `HighRiskShareShift`, `FeatureDriftHigh`, `FairnessGapExceeded` | `warning` | 2 min, grouped per component (serving, model, fairness) | every 4 h | silent; "resolved" when it clears |
| `severity="info"` (`SlowExplanations`) and anything without a severity | `null` | — | — | never sent |

`PipelineTaskFailed` gets no "resolved" message because a task failure is an
event, not a state: the callback posts it once, and unless the post carries an
end time Alertmanager times it out after `resolve_timeout` (5 minutes). A "RESOLVED" at that point would claim the
pipeline had been fixed when nothing happened.

Two inhibit rules keep a cause from arriving with its symptoms: `ApiDown`
suppresses every warning and info alert, and `ModelNotLoaded` suppresses
`HighErrorRate`.

The routing lives in `monitoring/alertmanager/alertmanager.yml` (no-op receivers)
and `alertmanager.telegram.yml` (Telegram receivers). Everything from `route:`
to `receivers:` must be identical in the two; CI checks that.

## Turn on Telegram delivery

You need the Telegram app and, for step 2, `curl` and `python3`.

### 1. Create a bot

In Telegram, open a chat with **@BotFather**, send `/newbot`, and answer its two
questions (a display name, then a username ending in `bot`). It replies with a
token of the form `123456789:AAH...`. Treat the token as a password: anyone
holding it can post as the bot.

### 2. Find the chat id

Decide where alerts should land:

- **A group** (recommended, so more than one person sees them): create the
  group, add the bot as a member, and send any message in the group, for example
  `/start@<your_bot_username>`.
- **A direct chat with you**: open the bot, press **Start**, send any message.

Then ask Telegram which chats the bot has seen. `read -s` keeps the token out of
your shell history and off the screen:

```bash
read -rs TG_TOKEN        # paste the token, press Enter
curl -fsS "https://api.telegram.org/bot${TG_TOKEN}/getUpdates" |
  python3 -c 'import json,sys; [print(c["id"], c.get("type"), c.get("title") or c.get("username")) for c in {u[k]["chat"]["id"]: u[k]["chat"] for u in json.load(sys.stdin)["result"] for k in ("message","my_chat_member") if k in u}.values()]'
```

Each line is `chat_id type name`. A group's id is negative (a supergroup's
starts with `-100`); a direct chat's is positive. If nothing is printed, send
another message in the chat and run it again.

### 3. Put the two files in the secrets directory

From the repository root, in the same shell:

```bash
printf '%s' "$TG_TOKEN" > monitoring/alertmanager/secrets/telegram_bot_token
printf '%s' '-1001234567890' > monitoring/alertmanager/secrets/telegram_chat_id   # your chat id
chmod 600 monitoring/alertmanager/secrets/telegram_*
unset TG_TOKEN
git status --short --ignored monitoring/alertmanager/secrets   # both show as "!!" (ignored)
```

Everything in `monitoring/alertmanager/secrets/` except its `.gitignore` is
ignored by git. Compose mounts the directory read-only at
`/run/secrets/alertmanager`; Alertmanager reads the token from there itself, on
every notification, and the chat id is filled into a copy of the config inside
the container. Neither value is written to any tracked file or to `.env`.

**On a Linux host**, Alertmanager runs as uid 65534 and a bind mount keeps the
host's owner, so a `600` file owned by you is unreadable to it (the log says
so, and delivery stays off). Hand the two files to that uid instead:
`sudo chown 65534:65534 monitoring/alertmanager/secrets/telegram_*`. With Colima
on macOS the mount presents the files as owned by the reading user, and `600`
works as it is.

**To keep the secrets outside the checkout** — on the deploy host, for example
(see [`RUNNER.md`](RUNNER.md)) — put the two files in any directory and set its
absolute path in `.env`:

```bash
ALERTMANAGER_SECRETS_DIR=/Users/you/.config/credit-risk/alertmanager
```

### 4. Restart Alertmanager

```bash
docker compose up -d alertmanager
```

`up -d` recreates the container if its definition changed — needed once, the
first time after pulling this change, because the entrypoint and the secrets
mount are new. After that, `docker compose restart alertmanager` is enough to
pick up a new chat id. A new token needs no restart at all.

### 5. Check it

```bash
docker compose logs alertmanager | grep 'alert delivery'
# entrypoint: alert delivery: Telegram (bot token from /run/secrets/alertmanager/telegram_bot_token)
```

Then send a test alert straight to Alertmanager's API:

```bash
curl -fsS -X POST -H 'Content-Type: application/json' http://127.0.0.1:19093/api/v2/alerts -d '[{
  "labels": {"alertname": "DeliveryTest", "severity": "critical"},
  "annotations": {"summary": "Test message from docs/ALERTING.md"}
}]'
```

A `[FIRING x1] critical DeliveryTest` message should arrive within about 15
seconds, and a `[RESOLVED]` one five to seven minutes later, when the test alert
times out. If nothing arrives:

```bash
docker compose logs alertmanager | grep -i -e warning -e telegram
curl -s http://127.0.0.1:19093/metrics | grep 'alertmanager_notifications_failed_total{integration="telegram"}'
```

`Unauthorized` means a wrong token; `chat not found` means a wrong chat id, or a
bot that is not a member of the group.

## Turn it off

Delete the two files (or point `ALERTMANAGER_SECRETS_DIR` somewhere empty) and
`docker compose restart alertmanager`. The log then reads
`alert delivery: off`.

## Quieting a flapping alert

Silence it; do not edit the rule. A silence has an expiry, a reason and an
author, and the alert stays visible in the UI while it is silenced:

```bash
docker compose exec alertmanager amtool silence add alertname=FairnessGapExceeded \
  --alertmanager.url=http://localhost:9093 --duration=4h \
  --author="$(git config user.name)" --comment="flapping at the 0.05 limit; mix changed, model unchanged"
```

`FairnessGapExceeded` in particular sits close to its threshold under normal
traffic, deliberately; [`ETHICS.md`](../ETHICS.md) section 4 says why, and what to
check before silencing it.

## What leaves the machine

With delivery on, alert names, labels, summaries and descriptions go to
Telegram's servers. The rules publish aggregates only — group selection rates,
PSI per feature, error ratios — never a request payload, an identifier or a raw
feature value. Keep it that way when adding a rule or an annotation, and keep
data values out of the text the Airflow failure callback posts.

## Changing the routing

Edit the section from `route:` to `receivers:` in `alertmanager.yml`, copy it
unchanged into `alertmanager.telegram.yml`, and run the checks — the same ones
CI runs:

```bash
docker run --rm --entrypoint /bin/sh \
  -v "$PWD/monitoring/alertmanager:/etc/alertmanager:ro" \
  prom/alertmanager:v0.27.0 /etc/alertmanager/tests/test_config.sh
```

They load both configs, compare their routing, check which receiver a pipeline
failure, a critical, a warning and an info alert reach, and run the entrypoint
against complete, partial and malformed secrets.
