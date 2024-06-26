// errors.go
package semantics

import (
	"errors"
)

// Errors
var (
	ErrSemanticsInconsistent = errors.New("semantics: inconsistent statechart")
	ErrSemanticsNotFound     = errors.New("semantics: state not found")
)
