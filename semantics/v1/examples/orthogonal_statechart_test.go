package examples

import (
	"testing"
)

func TestOrthogonalStatechart(t *testing.T) {
	chart := OrthogonalStatechart()

	// Verify the statechart is valid
	if err := chart.Validate(); err != nil {
		t.Errorf("Orthogonal statechart is invalid: %v", err)
	}

	// Test orthogonality relationship between the two regions
	orthogonal, err := chart.Orthogonal("Playing", "Muted")
	if err != nil {
		t.Errorf("Error checking orthogonality: %v", err)
	}
	if !orthogonal {
		t.Errorf("Expected Playing and Muted to be orthogonal")
	}

	// Test orthogonality relationship between states in the same region
	orthogonal, err = chart.Orthogonal("Playing", "Paused")
	if err != nil {
		t.Errorf("Error checking orthogonality: %v", err)
	}
	if orthogonal {
		t.Errorf("Expected Playing and Paused to NOT be orthogonal (they're in the same region)")
	}

	// Test default completion includes states from both regions
	completion, err := chart.DefaultCompletion("PlaybackControl")
	if err != nil {
		t.Errorf("Error getting default completion: %v", err)
	}

	// Convert completion to string slice for easier checking
	completionStrings := make([]string, len(completion))
	for i, state := range completion {
		completionStrings[i] = string(state)
	}

	// The default completion should include both Paused and Normal
	// (the initial states from both orthogonal regions)
	expectedStates := []string{"PlaybackControl", "PlaybackState", "Paused", "VolumeControl", "Normal"}
	for _, expected := range expectedStates {
		found := false
		for _, state := range completionStrings {
			if state == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected state %s in default completion, but it was not found", expected)
		}
	}

	// Test consistent state configurations
	consistent, err := chart.Consistent("PlaybackControl", "Playing", "Normal")
	if err != nil {
		t.Errorf("Error checking consistency: %v", err)
	}
	if !consistent {
		t.Errorf("Expected PlaybackControl, Playing, and Normal to be consistent")
	}

	// Test inconsistent state configurations
	consistent, err = chart.Consistent("Playing", "Paused")
	if err != nil {
		t.Errorf("Error checking consistency: %v", err)
	}
	if consistent {
		t.Errorf("Expected Playing and Paused to be inconsistent (XOR siblings)")
	}

	// Verify orthogonal state through orthogonality test (since findState is not exposed)
	// We already tested orthogonality relations which confirms the states are properly defined
}
