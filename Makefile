PY ?= python

.PHONY: help install install-dev test lint data train evaluate demo clean clean-artifacts

help:
	@echo "MindSwarm JEPA - make targets"
	@echo "  install         install runtime dependencies"
	@echo "  install-dev     install runtime + dev deps (pytest, ruff)"
	@echo "  test            run the test suite"
	@echo "  lint            run ruff lint checks"
	@echo "  data            generate the dataset (500 episodes)"
	@echo "  train           train the JEPA model and the LSTM baseline"
	@echo "  evaluate        compare both models, write plots to results/"
	@echo "  demo            launch the live visual demo"
	@echo "  clean           remove caches (__pycache__, .pytest_cache, .ruff_cache)"
	@echo "  clean-artifacts remove generated data/, checkpoints/, results/"

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

data:
	$(PY) generate_dataset.py

train:
	$(PY) train_jepa.py
	$(PY) train_baseline.py

evaluate:
	$(PY) evaluate.py

demo:
	$(PY) demo.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-artifacts:
	rm -rf data checkpoints results
