//go:build tools
// +build tools

package tools

import (
	// buf
	_ "github.com/bufbuild/buf/cmd/buf"
	_ "github.com/tmc/protoc-gen-apidocs"
)
