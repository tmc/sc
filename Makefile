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
