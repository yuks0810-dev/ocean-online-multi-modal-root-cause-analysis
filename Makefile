# OCEAN Model Makefile

.PHONY: help install test lint format docker-build docker-test docker-dev clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-test - Run tests in Docker"
	@echo "  docker-dev  - Start development container"
	@echo "  clean       - Clean build artifacts"

# Python environment setup
install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-coverage:
	python -m pytest tests/ --cov=ocean --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 ocean tests
	mypy ocean

format:
	black ocean tests
	isort ocean tests

# Docker commands
docker-build:
	docker-compose build

docker-test:
	docker-compose run --rm ocean-test

docker-integration:
	docker-compose run --rm ocean-integration-test

docker-dev:
	docker-compose run --rm ocean-dev

docker-clean:
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# Development helpers
dev-setup: install
	pip install jupyter ipython
	pip install pre-commit
	pre-commit install

# Quick verification
verify: docker-build docker-test
	@echo "✅ All tests passed in Docker environment"

# Full pipeline test
full-test: docker-build docker-test docker-integration
	@echo "🎉 Complete test suite passed!"