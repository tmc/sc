.PHONY: generate
generate:
	@$(MAKE) -C proto generate

.PHONY: generate-sdk-rust
generate-sdk-rust:
	@$(MAKE) -C proto generate-rust

.PHONY: build-sdk-rust
build-sdk-rust: generate-sdk-rust
	@$(MAKE) -C sdks/rust build

.PHONY: test-sdk-rust
test-sdk-rust:
	@$(MAKE) -C sdks/rust test

.PHONY: sdks
sdks: build-sdk-rust
	@echo "All SDKs built successfully"

# Docker development commands
.PHONY: docker-setup
docker-setup: ## Setup Docker development environment
	@./scripts/docker-dev.sh setup

.PHONY: docker-start
docker-start: ## Start Docker services
	@./scripts/docker-dev.sh start

.PHONY: docker-stop
docker-stop: ## Stop Docker services
	@./scripts/docker-dev.sh stop

.PHONY: docker-shell
docker-shell: ## Open Docker development shell
	@./scripts/docker-dev.sh shell

.PHONY: docker-test-env
docker-test-env: ## Test Docker environment
	@./scripts/test-docker.sh

.PHONY: docker-clean
docker-clean: ## Clean Docker environment
	@./scripts/docker-dev.sh clean

.PHONY: help
help: ## Show help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
