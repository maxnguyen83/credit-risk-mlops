#!/usr/bin/env bash
#
# Health of the compose stack as Docker sees it. Used by the deploy workflow,
# and by anyone asking "is it up yet" without squinting at `docker compose ps`.
#
#   scripts/stack_health.sh wait [SERVICE...]   block until ready, or fail
#   scripts/stack_health.sh table               Markdown table of every container
#
#   WAIT_SECONDS   how long `wait` waits before giving up (300)
#   POLL_SECONDS   how often it looks (5)
#
# With no SERVICE, `wait` waits for every service in docker-compose.yml.
#
# "Ready" is decided per container, from what that container declares:
#
#   has a healthcheck       must be `healthy`. Running is not enough: credit-api
#                           answers on its port long before it has a model, and
#                           that gap is exactly what its healthcheck is for.
#   no healthcheck          must be `running`.
#   one-shot (restart: no)  must have exited 0, like createbucket. A non-zero
#                           exit is final, so `wait` fails at once instead of
#                           spending the whole timeout on it.
#
# Read-only: nothing here starts, stops or restarts a container.
#
# Written for bash 3.2 as well as 5, because a macOS runner's /bin/bash is 3.2:
# no associative arrays, no mapfile, no "${empty[@]}" under set -u.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WAIT_SECONDS="${WAIT_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-5}"

# service|status|health|exit code|restart policy -- one line per container.
FORMAT='{{index .Config.Labels "com.docker.compose.service"}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}|{{.State.ExitCode}}|{{.HostConfig.RestartPolicy.Name}}'

usage() {
  sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}

# Container states for the named services, or for all of them with no argument.
# `ps -a` so a one-shot that has already exited is still seen.
#
# Never fails: a container removed between `ps` and `inspect`, or a daemon that
# blinks, reads as "not there yet" and `wait` looks again on its next pass.
states() {
  local ids
  ids=$(docker compose ps -a -q "$@") || true
  if [[ -z "$ids" ]]; then
    return 0
  fi
  # Word splitting on purpose: one container id per word.
  # shellcheck disable=SC2086
  docker inspect --format "$FORMAT" $ids 2>/dev/null | sort || true
}

# ready | waiting | failed, from one container's status, health, exit code and
# restart policy.
verdict() {
  local status=$1 health=$2 code=$3 restart=$4
  if [[ "$health" != "-" ]]; then
    if [[ "$health" == "healthy" ]]; then echo ready; else echo waiting; fi
  elif [[ "$status" == "running" ]]; then
    echo ready
  elif [[ "$status" == "exited" && ( "$restart" == "no" || -z "$restart" ) ]]; then
    if [[ "$code" == "0" ]]; then echo ready; else echo failed; fi
  else
    echo waiting
  fi
}

wait_ready() {
  local services=() service lines name status health code restart
  local pending failed last='' deadline
  if [[ $# -gt 0 ]]; then
    services=("$@")
  else
    while IFS= read -r service; do
      services+=("$service")
    done < <(docker compose config --services)
  fi
  if [[ ${#services[@]} -eq 0 ]]; then
    echo 'FAIL no services to wait for (is docker-compose.yml here?)'
    return 1
  fi

  printf 'waiting up to %ss for: %s\n' "$WAIT_SECONDS" "${services[*]}"
  deadline=$(( $(date +%s) + WAIT_SECONDS ))
  while :; do
    pending=''
    failed=''
    for service in "${services[@]}"; do
      lines=$(states "$service")
      if [[ -z "$lines" ]]; then
        pending+=" ${service}(no container)"
        continue
      fi
      while IFS='|' read -r name status health code restart; do
        case "$(verdict "$status" "$health" "$code" "$restart")" in
          ready) ;;
          failed) failed+=" ${name}(exited ${code})" ;;
          *) pending+=" ${name}(${status}/${health})" ;;
        esac
      done <<<"$lines"
    done

    if [[ -n "$failed" ]]; then
      printf 'FAIL one-shot service(s) exited non-zero:%s\n' "$failed"
      print_table
      return 1
    fi
    if [[ -z "$pending" ]]; then
      printf 'ready: %s\n' "${services[*]}"
      return 0
    fi
    if [[ $(date +%s) -ge $deadline ]]; then
      printf 'FAIL not ready after %ss:%s\n' "$WAIT_SECONDS" "$pending"
      print_table
      return 1
    fi
    # Only print when something changed: a 600-second wait otherwise fills the
    # job log with 120 identical lines.
    if [[ "$pending" != "$last" ]]; then
      printf '  still waiting:%s\n' "$pending"
      last=$pending
    fi
    sleep "$POLL_SECONDS"
  done
}

# Deliberately no command line, environment or ports: the mlflow command line
# carries the Postgres password, and this table goes into the job summary.
print_table() {
  local lines name status health code
  lines=$(states)
  echo '| service | state | health |'
  echo '|---|---|---|'
  if [[ -z "$lines" ]]; then
    echo '| _no containers_ | | |'
    return 0
  fi
  while IFS='|' read -r name status health code _; do
    if [[ "$status" == "exited" ]]; then
      status="exited (${code})"
    fi
    if [[ "$health" == "-" ]]; then
      health='no healthcheck'
    fi
    printf '| %s | %s | %s |\n' "$name" "$status" "$health"
  done <<<"$lines"
}

case "${1:-}" in
  wait)
    shift
    wait_ready "$@"
    ;;
  table)
    print_table
    ;;
  *)
    usage
    ;;
esac
