package examples

import (
	"testing"
)

func TestHierarchicalStatechart(t *testing.T) {
	chart := HierarchicalStatechart()

	// Verify the statechart is valid according to semantic rules
	if err := chart.Validate(); err != nil {
		t.Errorf("Hierarchical statechart is invalid: %v", err)
	}

	// Skip testing the root state's default since it would require examining
	// the internal structure, which isn't part of the public API

	// Test that Idle is the default state within On
	if state, err := chart.Default("On"); err != nil || state != "Idle" {
		t.Errorf("Expected Idle to be default state within On, got %s", state)
	}

	// Test that Monitoring is the default state within Armed
	if state, err := chart.Default("Armed"); err != nil || state != "Monitoring" {
		t.Errorf("Expected Monitoring to be default state within Armed, got %s", state)
	}

	// Test default completion
	completion, err := chart.DefaultCompletion("On")
	if err != nil {
		t.Errorf("Error getting default completion: %v", err)
	}

	// Verify the completion contains expected states
	expectedStates := []string{"On", "Idle"}
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
	related, err := chart.AncestrallyRelated("On", "Monitoring")
	if err != nil || !related {
		t.Errorf("Expected On and Monitoring to be ancestrally related")
	}

	// Test orthogonality (should be false for hierarchical states)
	orthogonal, err := chart.Orthogonal("Idle", "Armed")
	if err != nil || orthogonal {
		t.Errorf("Expected Idle and Armed to NOT be orthogonal")
	}
}
