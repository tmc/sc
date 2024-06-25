// errors.go
package semantics

import (
    "errors"
    "fmt"
)

// Errors
var (
    ErrSemanticsInconsistent = errors.New("semantics: inconsistent statechart")
    ErrSemanticsNotFound     = errors.New("semantics: state not found")
)

// StatechartError represents an error that occurred during a statechart operation.
type StatechartError struct {
    Op  string
    Err error
}

func (e *StatechartError) Error() string {
    return fmt.Sprintf("semantics: %s: %v", e.Op, e.Err)
}