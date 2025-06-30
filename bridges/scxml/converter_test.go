package scxml

import (
	"testing"

	"github.com/tmc/sc"
)

func TestNewConverter(t *testing.T) {
	converter := NewConverter()
	if converter == nil {
		t.Fatal("Expected non-nil converter")
	}
	
	if converter.FormatName() != "SCXML" {
		t.Errorf("Expected format name 'SCXML', got %q", converter.FormatName())
	}
}

func TestConverter_FormatName(t *testing.T) {
	converter := &Converter{}
	if converter.FormatName() != "SCXML" {
		t.Errorf("Expected format name 'SCXML', got %q", converter.FormatName())
	}
}

func TestConverter_Validate_EmptyDocument(t *testing.T) {
	converter := NewConverter()
	
	err := converter.Validate(Document{})
	if err == nil {
		t.Error("Expected validation error for empty document")
	}
}

func TestConverter_Validate_ValidDocument(t *testing.T) {
	converter := NewConverter()
	
	doc := Document{
		Namespace: "http://www.w3.org/2005/07/scxml",
		Version:   "1.0",
		Initial:   "start",
		States: []*State{
			{
				ID: "start",
			},
		},
	}
	
	err := converter.Validate(doc)
	if err != nil {
		t.Errorf("Expected no validation error, got: %v", err)
	}
}

func TestConverter_Export_Nil(t *testing.T) {
	converter := NewConverter()
	
	_, err := converter.Export(nil)
	if err == nil {
		t.Error("Expected error for nil statechart")
	}
}

func TestConverter_Import_BasicStatechart(t *testing.T) {
	converter := NewConverter()
	
	doc := Document{
		Namespace: "http://www.w3.org/2005/07/scxml",
		Version:   "1.0",
		Initial:   "idle",
		States: []*State{
			{
				ID: "idle",
			},
			{
				ID: "active",
			},
		},
	}
	
	statechart, err := converter.Import(doc)
	if err != nil {
		t.Fatalf("Import failed: %v", err)
	}
	
	if statechart == nil {
		t.Fatal("Expected non-nil statechart")
	}
	
	if statechart.RootState == nil {
		t.Error("Expected non-nil root state")
	}
}

func TestConverter_Export_BasicStatechart(t *testing.T) {
	converter := NewConverter()
	
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "root",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "idle",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "active",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				From:  []string{"idle"},
				To:    []string{"active"},
				Event: "start",
			},
		},
	}
	
	doc, err := converter.Export(statechart)
	if err != nil {
		t.Fatalf("Export failed: %v", err)
	}
	
	if len(doc.States) == 0 {
		t.Error("Expected at least one state in exported document")
	}
}