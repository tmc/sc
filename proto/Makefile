
.PHONY: tools
tools:
	@go install github.com/bufbuild/buf/cmd/buf

.PHONY: lint
lint: tools
	buf lint

.PHONY: generate
generate: tools
	buf generate

