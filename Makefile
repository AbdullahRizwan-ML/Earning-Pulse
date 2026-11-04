# One-command reproducible setup:  make reproduce
# Needs Python 3.11 (the pins in requirements.txt do not install on 3.13).

PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: reproduce setup test verify-data check-results results clean

## Set up the environment, run tests, verify the dataset and every README Results number
reproduce: test verify-data check-results

setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	touch $@

test: setup
	$(BIN)/python -m pytest tests -q

## Rebuild data/earnings_dataset.csv from data/cache/ (offline) and require a byte-identical match
verify-data: setup
	$(BIN)/python -m src.data_loader --rebuild-from-cache --check

## Recompute every README Results number and fail if any drifts from results/metrics.json
check-results: setup
	$(BIN)/python -m src.evaluate --check

## Regenerate results/metrics.json and the README Results block (after an intentional change)
results: setup
	$(BIN)/python -m src.evaluate

clean:
	rm -rf $(VENV)
