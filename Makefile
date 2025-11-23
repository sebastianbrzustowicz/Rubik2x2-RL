PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: venv install run-experiment run-pipeline dev clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel

install: venv
	$(BIN)/pip install -e .

install-dev:
	$(BIN)/pip install -e ".[dev]"

run-experiment:
	$(BIN)/python src/rubik2x2/training/rl_experiment_planner.py

run-pipeline:
	$(BIN)/python run_pipeline.py --scramble "$(SCRAMBLE)"

train-il:
	$(BIN)/python src/rubik2x2/training/train_il.py

evaluate-rl:
	$(BIN)/python src/rubik2x2/scripts/evaluate_rl_model.py

evaluate-il:
	$(BIN)/python src/rubik2x2/scripts/evaluate_il_model.py

clean:
	rm -rf $(VENV)
