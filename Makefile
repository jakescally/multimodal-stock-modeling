# Makefile for multimodal stock modeling project

.PHONY: help install install-dev test lint format clean data train evaluate

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install basic dependencies
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev]"

install-full:  ## Install with all optional dependencies
	pip install -e ".[full,dev]"

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linting
	flake8 models/ data/ utils/ tests/
	mypy models/ data/ utils/

format:  ## Format code
	black models/ data/ utils/ tests/ main.py

clean:  ## Clean up cache and build files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/

setup-env:  ## Set up virtual environment and install dependencies
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e ".[dev]"

activate:  ## Show activation command
	@echo "Run: source venv/bin/activate"

data:  ## Download and preprocess data
	python scripts/download_data.py
	python scripts/preprocess_data.py

train:  ## Train the model
	python scripts/train_model.py

evaluate:  ## Evaluate model performance
	python scripts/evaluate_model.py

notebook:  ## Start Jupyter notebook
	jupyter notebook

tensorboard:  ## Start TensorBoard
	tensorboard --logdir=logs/

# Docker commands (if you want to add containerization later)
docker-build:  ## Build Docker image
	docker build -t multimodal-stock-model .

docker-run:  ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace multimodal-stock-model