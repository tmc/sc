package common

import (
	"fmt"

	"github.com/tmc/sc"
)

// Converter defines the interface for bidirectional format conversion.
type Converter[T any] interface {
	// Import converts external format to Harel core semantics
	Import(external T) (*sc.Statechart, error)
	
	// Export converts Harel core semantics to external format
	Export(statechart *sc.Statechart) (T, error)
	
	// Validate checks semantic consistency of conversion
	Validate(external T) error
	
	// FormatName returns the name of the external format
	FormatName() string
}

// ConversionError represents errors during format conversion.
type ConversionError struct {
	Format    string
	Operation string
	Field     string
	Cause     error
}

func (e *ConversionError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("%s %s failed on field '%s': %v", e.Format, e.Operation, e.Field, e.Cause)
	}
	return fmt.Sprintf("%s %s failed: %v", e.Format, e.Operation, e.Cause)
}

func (e *ConversionError) Unwrap() error {
	return e.Cause
}

// NewConversionError creates a new conversion error.
func NewConversionError(format, operation, field string, cause error) *ConversionError {
	return &ConversionError{
		Format:    format,
		Operation: operation,
		Field:     field,
		Cause:     cause,
	}
}

// SemanticMapping handles differences between state machine formalisms.
type SemanticMapping struct {
	// Maps external state types to Harel core types
	StateTypeMapping map[string]sc.StateType
	
	// Maps external event types to Harel core events
	EventMapping map[string]string
	
	// Maps external action types to Harel core actions
	ActionMapping map[string]string
	
	// Tracks unsupported features for warning/error reporting
	UnsupportedFeatures []string
}

// MapStateType converts external state type to Harel core type.
func (sm *SemanticMapping) MapStateType(externalType string) (sc.StateType, error) {
	if coreType, exists := sm.StateTypeMapping[externalType]; exists {
		return coreType, nil
	}
	
	// Default mapping based on common conventions
	switch externalType {
	case "basic", "atomic", "leaf", "simple":
		return sc.StateTypeBasic, nil
	case "compound", "composite", "hierarchical", "normal":
		return sc.StateTypeNormal, nil
	case "parallel", "orthogonal", "concurrent", "and":
		return sc.StateTypeParallel, nil
	default:
		return sc.StateTypeUnspecified, fmt.Errorf("unknown state type: %s", externalType)
	}
}

// ValidationResult captures validation issues during conversion.
type ValidationResult struct {
	Valid    bool
	Warnings []string
	Errors   []string
}

// AddWarning adds a warning to the validation result.
func (vr *ValidationResult) AddWarning(msg string, args ...interface{}) {
	vr.Warnings = append(vr.Warnings, fmt.Sprintf(msg, args...))
}

// AddError adds an error to the validation result.
func (vr *ValidationResult) AddError(msg string, args ...interface{}) {
	vr.Errors = append(vr.Errors, fmt.Sprintf(msg, args...))
	vr.Valid = false
}

// HasIssues returns true if there are any warnings or errors.
func (vr *ValidationResult) HasIssues() bool {
	return len(vr.Warnings) > 0 || len(vr.Errors) > 0
}