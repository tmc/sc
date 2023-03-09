
.PHONY: deps
tools: ## Install tools
	@go install github.com/bufbuild/buf/cmd/buf
	@go install github.com/tmc/protoc-gen-apidocs

.PHONY: lint
lint: deps ## Lint proto files.
	@buf lint

.PHONY: generate
generate: tools ## Generate code.
	@buf generate

