.PHONY: help clean install dev-install test test-cov lint format type-check docs docs-serve doctest build ci

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .coverage htmlcov/
	rm -rf docs/_build/ _site/ _doctest/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

install: ## Install the package into the project environment
	uv sync

dev-install: ## Install every dependency group and the pre-commit hooks
	uv sync --all-groups
	uv run pre-commit install

test: ## Run the test suite
	uv run pytest

test-cov: ## Run the test suite with a coverage report
	uv run pytest --cov --cov-report=term-missing

lint: ## Lint and check formatting the way CI does
	uv run ruff check .
	uv run ruff format --check .

format: ## Apply formatting and safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run the type checker
	uv run pyright

docs: ## Build the documentation with warnings treated as errors
	uv run sphinx-build -W -b html docs _site

doctest: ## Run the examples embedded in docstrings and docs pages
	uv run sphinx-build -b doctest docs _doctest

docs-serve: docs ## Build and serve the documentation locally
	cd _site && python -m http.server 8000

build: ## Build the sdist and wheel
	uv build

ci: lint type-check test docs doctest ## Run the checks CI runs
	uv run pydoclint fewlab
	uvx preen check --strict --skip tests
