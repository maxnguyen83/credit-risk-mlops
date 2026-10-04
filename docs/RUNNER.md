# Self-hosted runner for continuous deployment

`deploy.yml` runs on one machine: the one that runs the compose stack. This page
registers a GitHub Actions runner there, labelled `credit-risk`, and removes it
again. Why CI stays on GitHub's machines while deployment does not is
[`ARCHITECTURE.md` D8](../ARCHITECTURE.md#d8--ci-on-github-hosted-runners-cd-on-a-self-hosted-runner).

The commands assume macOS on Apple silicon with Docker through Colima, which is
what the stack is developed on. Nothing here needs `sudo`.

## What a deploy does

Triggered when CI succeeds on a push to `main`, or by hand from `main`:

1. checks out the exact commit CI tested;
2. `docker compose build`, then `docker compose up -d` — containers are updated
   in place and the named volumes, which hold the MLflow registry and every
   model artifact, are never removed;
3. waits for Postgres, the object store, MLflow and Airflow to report healthy;
4. downloads and splits the dataset if `data/processed/` is empty;
5. if the registry serves nothing yet -- no version carries the `champion`
   alias and none is in Production -- runs the pipeline once (`airflow dags
   test`, the same as `make dag-test`), which promotes the first model that
   passes the gates; otherwise trains nothing;
6. restarts `credit-api` so it loads the current champion, and waits for every
   service;
7. runs `scripts/verify_deploy.py`: the version the `champion` alias names (the
   Production stage when no version carries it), the version `/health` reports
   and the version that scored `docs/examples/high_risk.json` must all be the
   same, and `/version` must report the deployed commit.

The job summary shows the commit, the model version served, whether the alias
or the stage named it, and the health of every container.

A deploy never replaces a champion with a newly trained model. A retrain is the
DAG's job; its register step promotes a candidate only if it is no worse than
the champion beyond `PROMOTION_PR_AUC_TOLERANCE`, and otherwise registers it as
the `challenger` for someone to review.

## Before you start

- **A private repository.** On a public one, a pull request from a fork can
  change a workflow to `runs-on: [self-hosted, credit-risk]` and run its code on
  this machine. Keep *Settings → Actions → General → Fork pull request
  workflows* disabled (the default for private repositories).
- **Docker answers as your user.** `docker info` must work in the shell you
  register from. With Colima: `colima start --cpu 4 --memory 8`. The deploy job
  checks this first and fails with a message if Docker is down; it never starts
  Colima itself.
- **`python3` (3.9 or newer), `git` and `curl` on `PATH`.** The verification
  script uses the standard library only, so the system `python3` is enough.
- **The stack's `.env`, outside the repository.** Postgres keeps the password it
  was first started with, so the deploy must use the same `.env` as the stack it
  is taking over:

  ```bash
  mkdir -p ~/.config/credit-risk
  cp /path/to/your/clone/.env ~/.config/credit-risk/deploy.env
  chmod 600 ~/.config/credit-risk/deploy.env
  ```

  On a machine that has never run the stack this file is optional: the job falls
  back to `.env.example`, as CI does. To keep it elsewhere, add
  `DEPLOY_ENV_FILE=/absolute/path` to the runner's `.env` file.
- **Optional: alert delivery.** Without Telegram secrets the deployed stack
  starts and sends no alerts. To have it deliver, keep the two files outside
  the runner's checkout and point the deploy's `.env` at them:

  ```bash
  mkdir -p ~/.config/credit-risk/alertmanager
  # telegram_bot_token and telegram_chat_id go here: docs/ALERTING.md, steps 1-3
  echo "ALERTMANAGER_SECRETS_DIR=$HOME/.config/credit-risk/alertmanager" \
    >> ~/.config/credit-risk/deploy.env
  ```

  The next deploy picks them up. Changing the chat id later needs
  `docker compose restart alertmanager` in the runner's checkout, since a
  deploy that changes nothing in the service definition leaves it running.
- **One runner directory per repository.** A runner registration belongs to a
  single repository. If this machine already has a runner for another one (for
  example `~/actions-runner` from Lab 5), leave it alone and use a new directory.

## Register the runner

The registration token is short-lived (one hour) and is never written to a
file. Fetch it with `gh` (repository admin rights needed), or copy it from
*Settings → Actions → Runners → New self-hosted runner*, which also shows the
current runner version and its SHA-256.

```bash
# 1. Download the runner into its own directory.
mkdir -p ~/actions-runner-credit-risk && cd ~/actions-runner-credit-risk
RUNNER_VERSION=2.337.0            # the version shown on the "New self-hosted runner" page
curl -fsSLo actions-runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
shasum -a 256 actions-runner.tar.gz   # compare with the hash on that page
tar xzf actions-runner.tar.gz

# 2. Register it against this repository with the credit-risk label.
RUNNER_TOKEN=$(gh api -X POST repos/maxnguyen83/credit-risk-mlops/actions/runners/registration-token --jq .token)
./config.sh --unattended \
  --url https://github.com/maxnguyen83/credit-risk-mlops \
  --token "$RUNNER_TOKEN" \
  --name "$(hostname -s)-credit-risk" \
  --labels credit-risk \
  --work _work
unset RUNNER_TOKEN

# 3. Run it as a service, so it survives closing the terminal and logging in again.
./svc.sh install                  # macOS: a LaunchAgent for the current user, no sudo
./svc.sh start
./svc.sh status
```

`config.sh` records the shell's `PATH` in `.path`, and the service runs with that
`PATH`. Register from a shell where `docker` and `python3` resolve; if they move
later, run `./env.sh` and then `./svc.sh stop && ./svc.sh start`.

A macOS LaunchAgent runs while you are logged in. For a short demo, `./run.sh` in
a terminal works too, but the runner stops when that terminal closes and any
deploy queues until it is back.

On a Linux host the same steps apply with the `linux-x64` (or `linux-arm64`)
archive, and `sudo ./svc.sh install "$USER"` to install a systemd unit.

Check that GitHub sees it, online and labelled:

```bash
gh api repos/maxnguyen83/credit-risk-mlops/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```

## Trigger the first deploy

`deploy.yml` has to be on `main` before either trigger exists: GitHub reads
`workflow_run` and `workflow_dispatch` workflows from the default branch.

```bash
# Automatic: every push to main that CI passes is deployed.
# By hand, from main (refused unless CI passed on that commit):
gh workflow run deploy.yml --ref main
gh run list --workflow deploy.yml --limit 3
gh run watch "$(gh run list --workflow deploy.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

Check what is live at any time, from the repository root on the host:

```bash
python3 scripts/verify_deploy.py              # registry vs API, plus one prediction
./scripts/stack_health.sh table               # every container's state and health
```

## Changing the model that serves

The API loads the champion once, at startup, so changing the model is three
steps: promote, restart, verify. From the repository root on the host:

```bash
# 1. promote: a reviewed challenger, a rollback, or a version promoted before
#    the alias existed. Idempotent; --dry-run prints what it would change.
.venv/bin/python -m credit_risk.models.registry set-champion --version <N>

# 2. restart the API so it resolves the alias again -- or deploy, which restarts
#    and verifies in one go
docker compose restart credit-api        # or: gh workflow run deploy.yml --ref main

# 3. verify: registry, /health and a live prediction agree
python3 scripts/verify_deploy.py
```

`/health` then reports `"model_ref": "alias"` and the new `model_version`. If it
reports `"stage"`, no version carries the alias yet: run step 1 with the version
it serves. A version registered before versions carried their decision
threshold also needs `.venv/bin/python -m credit_risk.models.registry
tag-threshold --version <N>` before the restart, or `/health` reports
`"threshold_source": "fallback"`.

## Living with the runner

- **The deploy owns the stack.** The compose project name is fixed
  (`ddm501-credit-risk`), so the stack runs from the runner's checkout under
  `~/actions-runner-credit-risk/_work/`. `make up` from another clone takes the
  containers back to that clone's files. `make down` from any clone runs
  `docker compose down -v` and **deletes the registry volume**; use
  `docker compose stop` to pause the stack instead.
- **Disk.** Every rebuild leaves untagged image layers behind;
  `docker image prune` reclaims them.
- **A deploy that queues forever** means the runner is offline
  (`./svc.sh status`) or its labels do not include `credit-risk`.

## Remove the runner

```bash
cd ~/actions-runner-credit-risk
./svc.sh stop
./svc.sh uninstall
RUNNER_TOKEN=$(gh api -X POST repos/maxnguyen83/credit-risk-mlops/actions/runners/remove-token --jq .token)
./config.sh remove --token "$RUNNER_TOKEN"
unset RUNNER_TOKEN
```

Then delete the `~/actions-runner-credit-risk` directory. Removing the runner
stops future deploys only: the containers and volumes keep running. If the
machine is gone and `config.sh remove` cannot run, delete the runner under
*Settings → Actions → Runners* instead.
