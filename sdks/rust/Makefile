.PHONY: all
all: build

.PHONY: build
build:
	cargo build

.PHONY: clean
clean:
	cargo clean
	rm -rf src/generated/*.rs

.PHONY: test
test:
	cargo test

.PHONY: example
example:
	cargo run --example simple_statechart