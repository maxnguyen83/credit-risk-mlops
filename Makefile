# Every command a reviewer needs, discoverable with `make help`.
.DEFAULT_GOAL := help
SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
# Logical date for `make dag-test`; override with DAG_DATE=YYYY-MM-DD.
DAG_DATE ?= $(shell date -u +%Y-%m-%d)

.PHONY: help venv install lint fmt typecheck openapi test test-fast data train dag-test up down ps logs smoke deploy-check traffic drift bias broken demo clean

help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

venv: ## create the local virtualenv
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)

install: venv ## install the package and dev tools
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

lint: ## ruff check + format check
	$(VENV)/bin/ruff check src tests dags scripts
	$(VENV)/bin/ruff format --check src tests dags scripts

fmt: ## apply ruff formatting
	$(VENV)/bin/ruff format src tests dags scripts
	$(VENV)/bin/ruff check --fix src tests dags scripts

typecheck: ## mypy over src
	$(VENV)/bin/mypy

openapi: ## regenerate docs/openapi.json from the app (no stack needed)
	$(PY) -c 'import json; from credit_risk.serving.main import app; print(json.dumps(app.openapi(), indent=2))' > docs/openapi.json

test: ## full test suite with coverage gate
	$(VENV)/bin/pytest --cov=src/credit_risk --cov-report=term-missing --cov-report=xml --cov-fail-under=80

test-fast: ## skip slow tests
	$(VENV)/bin/pytest -m "not slow"

data: ## download + clean + split the dataset
	$(PY) -m credit_risk.data.download
	$(PY) -m credit_risk.data.split

train: ## run the training pipeline locally (needs MLflow up)
	$(PY) -m credit_risk.models.train

dag-test: ## run the whole Airflow DAG once in the running stack (registers a model if gates pass)
	docker compose exec airflow airflow dags test credit_risk_pipeline $(DAG_DATE)

up: ## start the whole stack
	GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose up -d --build

down: ## stop the stack and remove volumes it created
	docker compose down -v

ps: ## show service health
	docker compose ps

logs: ## follow api logs
	docker compose logs -f credit-api

smoke: ## assert the running stack answers correctly
	./scripts/smoke.sh

deploy-check: ## the API serves the registry's champion model (deploy.yml's last check)
	python3 scripts/verify_deploy.py

traffic: ## normal traffic
	$(PY) scripts/traffic.py --rps 20 --seconds 120

drift: ## covariate drift -- PSI 0.03 -> 12; FeatureDriftHigh reaches pending (for: 20m > 300s run)
	$(PY) scripts/traffic.py --drift 3.0 --rps 20 --seconds 300

bias: ## skewed group mix -- watch the per-group selection rates diverge
	$(PY) scripts/traffic.py --bias 0.95 --rps 20 --seconds 300

broken: ## malformed payloads -- watch HighErrorRate fire
	$(PY) scripts/traffic.py --broken 0.3 --rps 20 --seconds 300

demo: ## the full presentation script
	./scripts/demo.sh

clean: ## remove caches and coverage output
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
