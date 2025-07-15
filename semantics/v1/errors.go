// errors.go
package semantics

import (
	"fmt"
)

// ErrorCode represents a unique identifier for error types
type ErrorCode int

// Error categories and codes
const (
	// Validation errors (1000-1999)
	ErrCodeValidationBase                = 1000
	ErrCodeStatechartNil                 = 1001
	ErrCodeStatechartMissingRoot         = 1002
	ErrCodeStateEmpty                    = 1003
	ErrCodeStateDuplicate                = 1004
	ErrCodeStateCircularContainment      = 1005
	ErrCodeStateInvalidType              = 1006
	ErrCodeStateInvalidInitial           = 1007
	ErrCodeStateInvalidChildren          = 1008
	ErrCodeStateInvalidParent            = 1009
	ErrCodeTransitionInvalidSource       = 1010
	ErrCodeTransitionInvalidTarget       = 1011
	ErrCodeTransitionMissingLabel        = 1012
	ErrCodeTransitionMalformed           = 1013
	ErrCodeEventUndeclared               = 1014
	ErrCodeEventInvalidLabel             = 1015
	ErrCodeGuardInvalidExpression        = 1016
	ErrCodeActionInvalidLabel            = 1017
	ErrCodeConfigurationInvalid          = 1018
	ErrCodeConfigurationInconsistent     = 1019
	ErrCodeOrthogonalStatesOverlap       = 1020
	ErrCodeParallelStateInsufficientRegions = 1021
	ErrCodeFinalStateOutgoingTransitions = 1022
	ErrCodeHistoryStateInvalidProperties = 1023

	// Runtime errors (2000-2999)
	ErrCodeRuntimeBase                = 2000
	ErrCodeMachineNotRunning          = 2001
	ErrCodeMachineAlreadyRunning      = 2002
	ErrCodeMachineAlreadyStopped      = 2003
	ErrCodeMachineInvalidState        = 2004
	ErrCodeMachineInvalidID           = 2005
	ErrCodeTransitionExecutionFailed  = 2006
	ErrCodeStateTransitionFailed      = 2007
	ErrCodeGuardEvaluationFailed      = 2008
	ErrCodeActionExecutionFailed      = 2009
	ErrCodeConfigurationUpdateFailed  = 2010
	ErrCodeDefaultCompletionFailed    = 2011
	ErrCodeConflictResolutionFailed   = 2012
	ErrCodeStateNotFound              = 2013
	ErrCodeStateDepthCalculationFailed = 2014
	ErrCodeParentLookupFailed         = 2015

	// Conversion errors (3000-3999)
	ErrCodeConversionBase       = 3000
	ErrCodeConversionImportFailed = 3001
	ErrCodeConversionExportFailed = 3002
	ErrCodeConversionValidationFailed = 3003
	ErrCodeConversionFormatUnsupported = 3004
	ErrCodeConversionFieldMapping = 3005

	// Input errors (4000-4999)
	ErrCodeInputBase          = 4000
	ErrCodeInputNil           = 4001
	ErrCodeInputEmpty         = 4002
	ErrCodeInputInvalidFormat = 4003
	ErrCodeInputOutOfRange    = 4004
	ErrCodeInputMissingRequired = 4005

	// System errors (5000-5999)
	ErrCodeSystemBase         = 5000
	ErrCodeSystemInternal     = 5001
	ErrCodeSystemUnsupported  = 5002
	ErrCodeSystemResourceLimit = 5003
	ErrCodeSystemTimeout      = 5004
)

// ErrorCategory represents broad categories of errors
type ErrorCategory string

const (
	CategoryValidation ErrorCategory = "validation"
	CategoryRuntime    ErrorCategory = "runtime"
	CategoryConversion ErrorCategory = "conversion"
	CategoryInput      ErrorCategory = "input"
	CategorySystem     ErrorCategory = "system"
)

// StatechartError represents a standardized error in the statechart system
type StatechartError struct {
	code     ErrorCode
	category ErrorCategory
	message  string
	context  map[string]interface{}
	cause    error
}

// Error implements the error interface
func (e *StatechartError) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("%s: %s", e.message, e.cause.Error())
	}
	return e.message
}

// Code returns the error code
func (e *StatechartError) Code() ErrorCode {
	return e.code
}

// Category returns the error category
func (e *StatechartError) Category() ErrorCategory {
	return e.category
}

// Context returns the error context
func (e *StatechartError) Context() map[string]interface{} {
	return e.context
}

// Unwrap returns the underlying error
func (e *StatechartError) Unwrap() error {
	return e.cause
}

// Is implements error matching
func (e *StatechartError) Is(target error) bool {
	if se, ok := target.(*StatechartError); ok {
		return e.code == se.code
	}
	return false
}

// ErrorBuilder provides a fluent interface for creating standardized errors
type ErrorBuilder struct {
	code     ErrorCode
	category ErrorCategory
	message  string
	context  map[string]interface{}
	cause    error
}

// NewErrorBuilder creates a new error builder
func NewErrorBuilder(code ErrorCode, category ErrorCategory, message string) *ErrorBuilder {
	return &ErrorBuilder{
		code:     code,
		category: category,
		message:  message,
		context:  make(map[string]interface{}),
	}
}

// WithContext adds context to the error
func (b *ErrorBuilder) WithContext(key string, value interface{}) *ErrorBuilder {
	b.context[key] = value
	return b
}

// WithCause sets the underlying cause
func (b *ErrorBuilder) WithCause(cause error) *ErrorBuilder {
	b.cause = cause
	return b
}

// Build creates the final error
func (b *ErrorBuilder) Build() *StatechartError {
	return &StatechartError{
		code:     b.code,
		category: b.category,
		message:  b.message,
		context:  b.context,
		cause:    b.cause,
	}
}

// getCategoryForCode determines the category based on error code
func getCategoryForCode(code ErrorCode) ErrorCategory {
	switch {
	case code >= 1000 && code < 2000:
		return CategoryValidation
	case code >= 2000 && code < 3000:
		return CategoryRuntime
	case code >= 3000 && code < 4000:
		return CategoryConversion
	case code >= 4000 && code < 5000:
		return CategoryInput
	case code >= 5000 && code < 6000:
		return CategorySystem
	default:
		return CategorySystem
	}
}

// Common error creation functions

// NewValidationError creates a validation error
func NewValidationError(code ErrorCode, message string) *StatechartError {
	return NewErrorBuilder(code, CategoryValidation, message).Build()
}

// NewRuntimeError creates a runtime error
func NewRuntimeError(code ErrorCode, message string) *StatechartError {
	return NewErrorBuilder(code, CategoryRuntime, message).Build()
}

// NewConversionError creates a conversion error
func NewConversionError(code ErrorCode, message string) *StatechartError {
	return NewErrorBuilder(code, CategoryConversion, message).Build()
}

// NewInputError creates an input error
func NewInputError(code ErrorCode, message string) *StatechartError {
	return NewErrorBuilder(code, CategoryInput, message).Build()
}

// NewSystemError creates a system error
func NewSystemError(code ErrorCode, message string) *StatechartError {
	return NewErrorBuilder(code, CategorySystem, message).Build()
}

// Convenience functions for common error patterns

// ValidationErrorf creates a validation error with formatted message
func ValidationErrorf(code ErrorCode, format string, args ...interface{}) *StatechartError {
	return NewValidationError(code, fmt.Sprintf(format, args...))
}

// RuntimeErrorf creates a runtime error with formatted message
func RuntimeErrorf(code ErrorCode, format string, args ...interface{}) *StatechartError {
	return NewRuntimeError(code, fmt.Sprintf(format, args...))
}

// ConversionErrorf creates a conversion error with formatted message
func ConversionErrorf(code ErrorCode, format string, args ...interface{}) *StatechartError {
	return NewConversionError(code, fmt.Sprintf(format, args...))
}

// InputErrorf creates an input error with formatted message
func InputErrorf(code ErrorCode, format string, args ...interface{}) *StatechartError {
	return NewInputError(code, fmt.Sprintf(format, args...))
}

// SystemErrorf creates a system error with formatted message
func SystemErrorf(code ErrorCode, format string, args ...interface{}) *StatechartError {
	return NewSystemError(code, fmt.Sprintf(format, args...))
}

// Legacy error variables for backward compatibility
var (
	ErrSemanticsInconsistent = NewValidationError(ErrCodeConfigurationInconsistent, "inconsistent statechart")
	ErrSemanticsNotFound     = NewRuntimeError(ErrCodeStateNotFound, "state not found")
)

// Helper functions for common error patterns

// IsValidationError checks if an error is a validation error
func IsValidationError(err error) bool {
	if se, ok := err.(*StatechartError); ok {
		return se.Category() == CategoryValidation
	}
	return false
}

// IsRuntimeError checks if an error is a runtime error
func IsRuntimeError(err error) bool {
	if se, ok := err.(*StatechartError); ok {
		return se.Category() == CategoryRuntime
	}
	return false
}

// IsConversionError checks if an error is a conversion error
func IsConversionError(err error) bool {
	if se, ok := err.(*StatechartError); ok {
		return se.Category() == CategoryConversion
	}
	return false
}

// WrapError wraps an existing error with statechart error context
func WrapError(code ErrorCode, message string, cause error) *StatechartError {
	category := getCategoryForCode(code)
	return NewErrorBuilder(code, category, message).WithCause(cause).Build()
}

// WrapErrorf wraps an existing error with formatted message
func WrapErrorf(code ErrorCode, cause error, format string, args ...interface{}) *StatechartError {
	return WrapError(code, fmt.Sprintf(format, args...), cause)
}
