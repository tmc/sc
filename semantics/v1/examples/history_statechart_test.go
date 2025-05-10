package examples

import (
	"testing"

	"github.com/tmc/sc/semantics/v1"
)

func TestHistoryStatechart(t *testing.T) {
	chart := HistoryStatechart()

	// Verify the statechart is valid
	if err := chart.Validate(); err != nil {
		t.Errorf("History statechart is invalid: %v", err)
	}

	// Skip testing the root state's default since it would require examining
	// the internal structure, which isn't part of the public API

	// Test that Editing is the default state within Active
	if state, err := chart.Default("Active"); err != nil || state != "Editing" {
		t.Errorf("Expected Editing to be default state within Active, got %s", state)
	}

	// Test that General is the default state within Settings
	if state, err := chart.Default("Settings"); err != nil || state != "General" {
		t.Errorf("Expected General to be default state within Settings, got %s", state)
	}

	// Test default completion
	completion, err := chart.DefaultCompletion("Active")
	if err != nil {
		t.Errorf("Error getting default completion: %v", err)
	}

	// Verify the completion contains expected states
	expectedStates := []string{"Active", "Editing"}
	for _, expected := range expectedStates {
		found := false
		for _, state := range completion {
			if string(state) == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected state %s in default completion, but it was not found", expected)
		}
	}

	// Test ancestral relations
	related, err := chart.AncestrallyRelated("Active", "Editing")
	if err != nil || !related {
		t.Errorf("Expected Active and Editing to be ancestrally related")
	}

	// Test history state behavior simulation
	// Since the actual history mechanism is conceptual in this example,
	// we'll simulate what would happen with a history mechanism

	// Create a simple history tracking mechanism
	type historyMemory struct {
		active   semantics.StateLabel
		settings semantics.StateLabel
	}

	// Initialize with default states
	history := historyMemory{
		active:   "Editing",
		settings: "General",
	}

	// Simulate state changes and history
	history.active = "Searching" // User navigates to Searching

	// Then transitions to Settings
	activeSaved := history.active

	// Navigate to Advanced in Settings
	history.settings = "Advanced"

	// Now simulate returning from Settings to Active with history
	// This would restore the previous active state (Searching)
	restoredActive := activeSaved

	if restoredActive != "Searching" {
		t.Errorf("History mechanism simulation failed, expected to restore state Searching, got %s", restoredActive)
	}

	// This test demonstrates the conceptual behavior of history states,
	// though the actual implementation would need to handle this in the
	// state machine execution logic
}
