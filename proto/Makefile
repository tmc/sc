
.PHONY: deps
tools: ## Install tools
	@go install github.com/bufbuild/buf/cmd/buf
	@go install github.com/tmc/protoc-gen-apidocs
	@echo "Installing Rust protobuf plugins (requires Cargo)..."
	@cargo install protoc-gen-prost protoc-gen-tonic || echo "Failed to install Rust protobuf plugins. Make sure Cargo is installed."

.PHONY: lint
lint: deps ## Lint proto files.
	@buf lint

.PHONY: generate
generate: tools ## Generate code.
	@buf generate

.PHONY: generate-rust
generate-rust: tools ## Generate Rust code only.
	@buf generate --template buf.gen.rust.yaml

.PHONY: build-rust
build-rust: generate-rust ## Build the Rust SDK.
	@cd ../sdks/rust && cargo build

.PHONY: test-rust
test-rust: build-rust ## Test the Rust SDK.
	@cd ../sdks/rust && cargo test

