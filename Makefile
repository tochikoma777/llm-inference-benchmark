.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV ?= .venv
ifeq ($(wildcard $(VENV)/bin/python),)
  PY := python3
else
  PY := $(VENV)/bin/python
endif
BENCH := $(PY) -m llmbench

.PHONY: help venv install install-gpu install-viz test lint fmt check demo run run-gpu report dashboard clean

help:  ## Show this help
	@echo "llmbench — common tasks"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the isolated virtual environment
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip

install: venv  ## Install core + dashboard + dev dependencies
	$(VENV)/bin/pip install -e ".[viz,dev]"

install-gpu:  ## Install GPU runtime dependencies (torch, transformers, vLLM)
	$(VENV)/bin/pip install -r requirements-gpu.txt

install-viz:  ## Install the dashboard stack only
	$(VENV)/bin/pip install -r requirements-viz.txt

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Static checks (ruff + mypy)
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src/llmbench

fmt:  ## Format the code
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/black src tests

check: lint test  ## Lint + test

demo:  ## Run the three simulated backends and build a combined report
	bash scripts/demo.sh

run:  ## Run the default (mock) configuration
	$(BENCH) run -c config/default.yaml

run-gpu:  ## Run the vLLM configuration (needs GPU + model)
	$(BENCH) run -c config/vllm.yaml

report:  ## Rebuild the HTML + Markdown report for the latest run
	$(BENCH) report --latest --open

dashboard:  ## Launch the interactive dashboard
	$(BENCH) dashboard

clean:  ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf reports/generated
