package semantics

import (
	"errors"
	"testing"
)

func TestErrorBuilder(t *testing.T) {
	// Test basic error creation
	err := NewErrorBuilder(ErrCodeStatechartNil, CategoryValidation, "statechart is nil").Build()
	
	if err.Code() != ErrCodeStatechartNil {
		t.Errorf("Expected code %d, got %d", ErrCodeStatechartNil, err.Code())
	}
	
	if err.Category() != CategoryValidation {
		t.Errorf("Expected category %s, got %s", CategoryValidation, err.Category())
	}
	
	if err.Error() != "statechart is nil" {
		t.Errorf("Expected message 'statechart is nil', got '%s'", err.Error())
	}
}

func TestErrorBuilderWithContext(t *testing.T) {
	err := NewErrorBuilder(ErrCodeStateNotFound, CategoryRuntime, "state not found").
		WithContext("state_label", "TestState").
		WithContext("statechart_id", "TestChart").
		Build()
	
	context := err.Context()
	
	if context["state_label"] != "TestState" {
		t.Errorf("Expected context state_label 'TestState', got '%v'", context["state_label"])
	}
	
	if context["statechart_id"] != "TestChart" {
		t.Errorf("Expected context statechart_id 'TestChart', got '%v'", context["statechart_id"])
	}
}

func TestErrorBuilderWithCause(t *testing.T) {
	cause := errors.New("underlying error")
	err := NewErrorBuilder(ErrCodeGuardEvaluationFailed, CategoryRuntime, "guard evaluation failed").
		WithCause(cause).
		Build()
	
	if err.Unwrap() != cause {
		t.Errorf("Expected unwrapped error to be the cause")
	}
	
	expectedMessage := "guard evaluation failed: underlying error"
	if err.Error() != expectedMessage {
		t.Errorf("Expected message '%s', got '%s'", expectedMessage, err.Error())
	}
}

func TestConvenienceFunctions(t *testing.T) {
	// Test ValidationErrorf
	err := ValidationErrorf(ErrCodeStateEmpty, "state '%s' has empty label", "TestState")
	if err.Code() != ErrCodeStateEmpty {
		t.Errorf("Expected code %d, got %d", ErrCodeStateEmpty, err.Code())
	}
	if err.Category() != CategoryValidation {
		t.Errorf("Expected category %s, got %s", CategoryValidation, err.Category())
	}
	expectedMessage := "state 'TestState' has empty label"
	if err.Error() != expectedMessage {
		t.Errorf("Expected message '%s', got '%s'", expectedMessage, err.Error())
	}
	
	// Test RuntimeErrorf
	runtimeErr := RuntimeErrorf(ErrCodeMachineNotRunning, "machine '%s' is not running", "TestMachine")
	if runtimeErr.Category() != CategoryRuntime {
		t.Errorf("Expected category %s, got %s", CategoryRuntime, runtimeErr.Category())
	}
}

func TestErrorMatching(t *testing.T) {
	err1 := NewValidationError(ErrCodeStatechartNil, "statechart is nil")
	err2 := NewValidationError(ErrCodeStatechartNil, "different message")
	err3 := NewValidationError(ErrCodeStateEmpty, "state empty")
	
	// Test Is method
	if !errors.Is(err1, err2) {
		t.Errorf("Expected errors with same code to match")
	}
	
	if errors.Is(err1, err3) {
		t.Errorf("Expected errors with different codes to not match")
	}
}

func TestCategoryHelpers(t *testing.T) {
	validationErr := NewValidationError(ErrCodeStatechartNil, "validation error")
	runtimeErr := NewRuntimeError(ErrCodeMachineNotRunning, "runtime error")
	conversionErr := NewConversionError(ErrCodeConversionImportFailed, "conversion error")
	
	// Test IsValidationError
	if !IsValidationError(validationErr) {
		t.Errorf("Expected IsValidationError to return true for validation error")
	}
	if IsValidationError(runtimeErr) {
		t.Errorf("Expected IsValidationError to return false for runtime error")
	}
	
	// Test IsRuntimeError
	if !IsRuntimeError(runtimeErr) {
		t.Errorf("Expected IsRuntimeError to return true for runtime error")
	}
	if IsRuntimeError(validationErr) {
		t.Errorf("Expected IsRuntimeError to return false for validation error")
	}
	
	// Test IsConversionError
	if !IsConversionError(conversionErr) {
		t.Errorf("Expected IsConversionError to return true for conversion error")
	}
	if IsConversionError(validationErr) {
		t.Errorf("Expected IsConversionError to return false for validation error")
	}
}

func TestWrapError(t *testing.T) {
	cause := errors.New("original error")
	wrapped := WrapError(ErrCodeStateNotFound, "state lookup failed", cause)
	
	if wrapped.Code() != ErrCodeStateNotFound {
		t.Errorf("Expected code %d, got %d", ErrCodeStateNotFound, wrapped.Code())
	}
	
	if wrapped.Unwrap() != cause {
		t.Errorf("Expected unwrapped error to be the cause")
	}
	
	expectedMessage := "state lookup failed: original error"
	if wrapped.Error() != expectedMessage {
		t.Errorf("Expected message '%s', got '%s'", expectedMessage, wrapped.Error())
	}
}

func TestWrapErrorf(t *testing.T) {
	cause := errors.New("original error")
	wrapped := WrapErrorf(ErrCodeStateNotFound, cause, "state '%s' not found in statechart '%s'", "TestState", "TestChart")
	
	expectedMessage := "state 'TestState' not found in statechart 'TestChart': original error"
	if wrapped.Error() != expectedMessage {
		t.Errorf("Expected message '%s', got '%s'", expectedMessage, wrapped.Error())
	}
}

func TestGetCategoryForCode(t *testing.T) {
	testCases := []struct {
		code     ErrorCode
		expected ErrorCategory
	}{
		{ErrCodeStatechartNil, CategoryValidation},
		{ErrCodeMachineNotRunning, CategoryRuntime},
		{ErrCodeConversionImportFailed, CategoryConversion},
		{ErrCodeInputNil, CategoryInput},
		{ErrCodeSystemInternal, CategorySystem},
		{ErrorCode(9999), CategorySystem}, // Unknown code defaults to system
	}
	
	for _, tc := range testCases {
		actual := getCategoryForCode(tc.code)
		if actual != tc.expected {
			t.Errorf("Expected category %s for code %d, got %s", tc.expected, tc.code, actual)
		}
	}
}

func TestLegacyErrorVariables(t *testing.T) {
	// Test that legacy variables still work
	if !IsValidationError(ErrSemanticsInconsistent) {
		t.Errorf("Expected ErrSemanticsInconsistent to be a validation error")
	}
	
	if !IsRuntimeError(ErrSemanticsNotFound) {
		t.Errorf("Expected ErrSemanticsNotFound to be a runtime error")
	}
	
	// Test that they have correct codes
	if ErrSemanticsInconsistent.Code() != ErrCodeConfigurationInconsistent {
		t.Errorf("Expected code %d, got %d", ErrCodeConfigurationInconsistent, ErrSemanticsInconsistent.Code())
	}
}