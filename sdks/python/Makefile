# Makefile for Statecharts Python SDK

.PHONY: help install install-dev test test-cov lint format check clean build publish examples

# Default target
help:
	@echo "Statecharts Python SDK - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  check        - Run all checks (lint, format, type)"
	@echo ""
	@echo "Building:"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  publish      - Publish to PyPI (requires credentials)"
	@echo ""
	@echo "Examples:"
	@echo "  examples     - Run all example scripts"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

# Testing
test:
	pytest

test-cov:
	pytest --cov=statecharts --cov-report=html --cov-report=term

# Code quality
lint:
	ruff statecharts tests
	mypy statecharts

format:
	black statecharts tests

format-check:
	black --check statecharts tests

check: format-check lint test

# Building and publishing
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Examples
examples:
	@echo "Running simple statechart example..."
	python -m statecharts.examples.simple_statechart
	@echo ""
	@echo "Running hierarchical statechart example..."
	python -m statecharts.examples.hierarchical_statechart
	@echo ""
	@echo "Running orthogonal statechart example..."
	python -m statecharts.examples.orthogonal_statechart

# Development helpers
proto-gen:
	@echo "Regenerating Protocol Buffer files..."
	cd ../../proto && make generate

validate-examples:
	@echo "Validating all example statecharts..."
	python -c "
from statecharts.examples.simple_statechart import create_simple_toggle_statechart
from statecharts.examples.hierarchical_statechart import create_alarm_system_statechart
from statecharts.examples.orthogonal_statechart import create_smartphone_statechart
from statecharts import validate_statechart

charts = [
    ('Simple Toggle', create_simple_toggle_statechart()),
    ('Alarm System', create_alarm_system_statechart()),
    ('Smartphone', create_smartphone_statechart()),
]

for name, chart in charts:
    result = validate_statechart(chart)
    status = '✓' if result.is_valid else '✗'
    print(f'{status} {name}: {result}')
"

# Quick development cycle
dev-cycle: format lint test
	@echo "✓ Development cycle complete"

# Setup development environment
setup-dev:
	pip install --upgrade pip
	pip install -e .[dev]
	@echo "✓ Development environment setup complete"