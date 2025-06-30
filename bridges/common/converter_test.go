package common

import (
	"errors"
	"testing"

	"github.com/tmc/sc"
)

// MockConverter implements Converter for testing
type MockConverter struct {
	formatName string
	importErr  error
	exportErr  error
	validateErr error
}

func (m *MockConverter) Import(external string) (*sc.Statechart, error) {
	if m.importErr != nil {
		return nil, m.importErr
	}
	return &sc.Statechart{}, nil
}

func (m *MockConverter) Export(statechart *sc.Statechart) (string, error) {
	if m.exportErr != nil {
		return "", m.exportErr
	}
	return "exported", nil
}

func (m *MockConverter) Validate(external string) error {
	return m.validateErr
}

func (m *MockConverter) FormatName() string {
	return m.formatName
}

func TestConversionError_Error(t *testing.T) {
	err := &ConversionError{
		Format:    "SCXML",
		Operation: "import",
		Field:     "state.id",
		Cause:     errors.New("invalid id"),
	}
	
	expected := "SCXML import failed on field 'state.id': invalid id"
	if err.Error() != expected {
		t.Errorf("Expected %q, got %q", expected, err.Error())
	}
}

func TestConversionError_ErrorWithoutField(t *testing.T) {
	err := &ConversionError{
		Format:    "XState",
		Operation: "export",
		Cause:     errors.New("network error"),
	}
	
	expected := "XState export failed: network error"
	if err.Error() != expected {
		t.Errorf("Expected %q, got %q", expected, err.Error())
	}
}

func TestMockConverter_Interface(t *testing.T) {
	// Test that MockConverter implements Converter interface
	var converter Converter[string] = &MockConverter{
		formatName: "Mock",
	}
	
	if converter.FormatName() != "Mock" {
		t.Errorf("Expected format name 'Mock', got %q", converter.FormatName())
	}
}

func TestMockConverter_Import(t *testing.T) {
	converter := &MockConverter{}
	
	statechart, err := converter.Import("test")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if statechart == nil {
		t.Error("Expected non-nil statechart")
	}
}

func TestMockConverter_ImportError(t *testing.T) {
	expectedErr := errors.New("import failed")
	converter := &MockConverter{importErr: expectedErr}
	
	_, err := converter.Import("test")
	if err != expectedErr {
		t.Errorf("Expected error %v, got %v", expectedErr, err)
	}
}

func TestMockConverter_Export(t *testing.T) {
	converter := &MockConverter{}
	
	result, err := converter.Export(&sc.Statechart{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if result != "exported" {
		t.Errorf("Expected 'exported', got %q", result)
	}
}

func TestMockConverter_ExportError(t *testing.T) {
	expectedErr := errors.New("export failed")
	converter := &MockConverter{exportErr: expectedErr}
	
	_, err := converter.Export(&sc.Statechart{})
	if err != expectedErr {
		t.Errorf("Expected error %v, got %v", expectedErr, err)
	}
}

func TestMockConverter_Validate(t *testing.T) {
	converter := &MockConverter{}
	
	err := converter.Validate("test")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestMockConverter_ValidateError(t *testing.T) {
	expectedErr := errors.New("validation failed")
	converter := &MockConverter{validateErr: expectedErr}
	
	err := converter.Validate("test")
	if err != expectedErr {
		t.Errorf("Expected error %v, got %v", expectedErr, err)
	}
}