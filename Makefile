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


# -----------------------------------------------------------------------------
# Test Targets
# -----------------------------------------------------------------------------
.PHONY: test test-unit test-integration test-e2e test-benchmarks test-all
.PHONY: test-coverage test-performance clean-test setup-test

# Default target
all: test-all

# Test Targets
test: test-unit
	@echo "Running unit tests..."

test-unit:
	@echo "🧪 Running unit tests..."
	go test -v -race -short ./...

test-integration:
	@echo "🔗 Running integration tests..."
	go test -v -tags=integration ./...

test-e2e:
	@echo "🔄 Running end-to-end tests..."
	go test -v -run="TestE2E" ./...

test-benchmarks:
	@echo "⚡ Running performance benchmarks..."
	go test -bench=. -benchmem -run=^$$ ./...

test-coverage:
	@echo "📊 Generating coverage report..."
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	go tool cover -func=coverage.out

test-performance:
	@echo "🚀 Running performance tests..."
	go test -v -run="Performance|Benchmark" ./...

test-all: clean-test setup-test test-unit test-integration test-benchmarks
	@echo "✅ All tests completed!"

clean-test:
	@echo "🧹 Cleaning test artifacts..."
	@rm -f coverage.out coverage.html

setup-test:
	@echo "🔧 Setting up test environment..."
	@go mod tidy


# -----------------------------------------------------------------------------
# States iOS App Automation
# -----------------------------------------------------------------------------
.PHONY: ios-version-bump ios-build-release ios-upload-testflight ios-release-full
.PHONY: ios-setup ios-clean ios-verify ios-archive ios-export ios-help
.PHONY: app-open app-cloud-trigger

# iOS Project Configuration
# Using APP_ROOT from existing makefile to maintain consistency but aliasing for the new targets
APP_ROOT = ios/States
IOS_PROJECT_DIR = $(APP_ROOT)
IOS_PROJECT = $(IOS_PROJECT_DIR)/States.xcodeproj
IOS_SCHEME = States
IOS_ARCHIVE_PATH = $(IOS_PROJECT_DIR)/build/States.xcarchive
IOS_EXPORT_PATH = $(IOS_PROJECT_DIR)/build/TestFlight
IOS_EXPORT_OPTIONS = $(IOS_PROJECT_DIR)/ExportOptions.plist

# Get current version information from Xcode project
ios-get-version:
	@echo "📱 Current iOS App Version Info:"
	@cd $(IOS_PROJECT_DIR) && xcodebuild -project States.xcodeproj -showBuildSettings -configuration Release | grep -E "MARKETING_VERSION|CURRENT_PROJECT_VERSION" | head -2

# Bump build version (increment CURRENT_PROJECT_VERSION)
ios-version-bump:
	@echo "🔢 Bumping iOS app build version..."
	@cd $(IOS_PROJECT_DIR) && \
	CURRENT_BUILD=$$(xcodebuild -project States.xcodeproj -showBuildSettings -configuration Release | grep "CURRENT_PROJECT_VERSION" | head -1 | sed 's/.*= //'); \
	NEW_BUILD=$$(($$CURRENT_BUILD + 1)); \
	echo "Incrementing build from $$CURRENT_BUILD to $$NEW_BUILD"; \
	xcrun agvtool new-version $$NEW_BUILD
	@echo "✅ Build version bumped successfully"
	@$(MAKE) ios-get-version

# Build release archive
ios-build-release: ios-verify
	@echo "🏗️  Building iOS release archive..."
	@cd $(IOS_PROJECT_DIR) && \
	xcodebuild clean -project States.xcodeproj -scheme $(IOS_SCHEME) -configuration Release && \
	xcodebuild archive \
		-project States.xcodeproj \
		-scheme $(IOS_SCHEME) \
		-configuration Release \
		-destination "generic/platform=iOS" \
		-archivePath "$(shell pwd)/$(IOS_ARCHIVE_PATH)" \
		SKIP_INSTALL=NO \
		BUILD_LIBRARY_FOR_DISTRIBUTION=YES
	@echo "✅ Archive built successfully at $(IOS_ARCHIVE_PATH)"

# Export IPA for TestFlight
ios-export: ios-build-release
	@echo "📦 Exporting IPA for TestFlight..."
	@mkdir -p $(IOS_EXPORT_PATH)
	@cd $(IOS_PROJECT_DIR) && \
	xcodebuild -exportArchive \
		-archivePath "$(shell pwd)/$(IOS_ARCHIVE_PATH)" \
		-exportPath "$(shell pwd)/$(IOS_EXPORT_PATH)" \
		-exportOptionsPlist "$(shell pwd)/$(IOS_EXPORT_OPTIONS)"
	@echo "✅ IPA exported successfully to $(IOS_EXPORT_PATH)"

# Upload to TestFlight Internal
ios-upload-testflight: ios-export
	@echo "🚀 Uploading to TestFlight..."
	@if [ -z "$$APPLE_ID" ] || [ -z "$$APPLE_APP_PASSWORD" ]; then \
		echo "❌ Error: APPLE_ID and APPLE_APP_PASSWORD environment variables must be set"; \
		exit 1; \
	fi
	@cd $(IOS_PROJECT_DIR) && \
	xcrun altool --upload-app \
		--type ios \
		--file "$(shell pwd)/$(IOS_EXPORT_PATH)/States.ipa" \
		--username "$$APPLE_ID" \
		--password "$$APPLE_APP_PASSWORD" \
		--verbose
	@echo "✅ Upload to TestFlight completed successfully"

# Complete release pipeline (bump version + build + upload)
ios-release-full: ios-version-bump ios-upload-testflight
	@echo "🎉 Complete iOS release pipeline finished!"

ios-verify:
	@echo "🔍 Running pre-flight checks..."
	@if [ ! -f "$(IOS_EXPORT_OPTIONS)" ]; then \
		echo "❌ Error: ExportOptions.plist not found at $(IOS_EXPORT_OPTIONS)"; \
		exit 1; \
	fi

ios-clean:
	@echo "🧹 Cleaning iOS build artifacts..."
	@cd $(IOS_PROJECT_DIR) && rm -rf build/ DerivedData/ .build/
	@echo "✅ iOS build artifacts cleaned"

app-cloud-trigger: ## Trigger Xcode Cloud via empty commit
	git commit --allow-empty -m "Trigger Xcode Cloud Build [skip ci]"
	@echo "Commit created. Run 'git push' to trigger configured Xcode Cloud workflow."

app-open: ## Open States project in Xcode
	open $(IOS_PROJECT)

ios-help:
	@echo "🍎 iOS Build and Deployment Targets:"
	@echo "  ios-get-version        - Show current app version info"
	@echo "  ios-version-bump       - Increment build number"
	@echo "  ios-build-release      - Build release archive"
	@echo "  ios-export             - Export IPA from archive"
	@echo "  ios-upload-testflight  - Upload to TestFlight"
	@echo "  ios-release-full       - Complete release pipeline"
	@echo "  ios-clean              - Clean build artifacts"
