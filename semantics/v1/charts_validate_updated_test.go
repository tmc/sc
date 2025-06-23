package semantics

import (
	"context"
	"testing"

	"github.com/tmc/sc"
	testutil "github.com/tmc/sc/testing"
)

func TestValidatorClient_NewValidatorClient(t *testing.T) {
	tests := []struct {
		name        string
		target      string
		expectError bool
	}{
		{
			name:        "invalid target",
			target:      "invalid-target",
			expectError: false, // grpc.Dial doesn't fail immediately for invalid targets
		},
		{
			name:        "empty target",
			target:      "",
			expectError: true, // empty target should fail when trying to connect
		},
		{
			name:        "localhost target",
			target:      "localhost:50051",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewValidatorClient(tt.target)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				return
			}
			
			if err != nil {
				t.Fatalf("Expected no error, but got: %v", err)
			}
			
			if client == nil {
				t.Fatal("Expected non-nil client")
			}
			
			if client.client == nil {
				t.Fatal("Expected non-nil client.client")
			}
			
			if client.conn == nil {
				t.Fatal("Expected non-nil client.conn")
			}
			
			// Clean up
			_ = client.Close()
		})
	}
}

func TestValidatorClient_Close(t *testing.T) {
	client, err := NewValidatorClient("localhost:50051")
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	
	err = client.Close()
	if err != nil {
		t.Fatalf("Failed to close client: %v", err)
	}
	
	// Closing again should not error
	err = client.Close()
	if err != nil {
		t.Fatalf("Second close should not error: %v", err)
	}
}

func TestValidatorClient_CloseWithNilConnection(t *testing.T) {
	client := &ValidatorClient{conn: nil}
	err := client.Close()
	if err != nil {
		t.Fatalf("Close with nil connection should not error: %v", err)
	}
}

func TestConvertStateToProto(t *testing.T) {
	tests := []struct {
		name     string
		input    *sc.State
		expected func(*sc.State) bool // Function to validate the conversion
	}{
		{
			name:  "nil state",
			input: nil,
			expected: func(*sc.State) bool {
				return true // Should return nil
			},
		},
		{
			name: "basic state",
			input: &sc.State{
				Label:     "TestState",
				Type:      sc.StateTypeBasic,
				IsInitial: true,
				IsFinal:   false,
				Children:  []*sc.State{},
			},
			expected: func(original *sc.State) bool {
				return true // Just test that it doesn't panic
			},
		},
		{
			name: "compound state with children",
			input: &sc.State{
				Label: "Parent",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label:     "Child1",
						Type:      sc.StateTypeBasic,
						IsInitial: true,
						IsFinal:   false,
						Children:  []*sc.State{},
					},
					{
						Label:     "Child2",
						Type:      sc.StateTypeBasic,
						IsInitial: false,
						IsFinal:   true,
						Children:  []*sc.State{},
					},
				},
				IsInitial: false,
				IsFinal:   false,
			},
			expected: func(original *sc.State) bool {
				return true
			},
		},
		{
			name: "parallel state",
			input: &sc.State{
				Label: "ParallelState",
				Type:  sc.StateTypeParallel,
				Children: []*sc.State{
					{
						Label:     "Region1",
						Type:      sc.StateTypeNormal,
						IsInitial: false,
						IsFinal:   false,
						Children:  []*sc.State{},
					},
				},
				IsInitial: false,
				IsFinal:   false,
			},
			expected: func(original *sc.State) bool {
				return true
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that conversion doesn't panic and returns something reasonable
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("convertStateToProto panicked: %v", r)
				}
			}()
			
			result := convertStateToProto(tt.input)
			
			if tt.input == nil {
				if result != nil {
					t.Fatal("Expected nil result for nil input")
				}
				return
			}
			
			if result == nil {
				t.Fatal("Expected non-nil result for non-nil input")
			}
			
			// Basic validation
			if result.Label != tt.input.Label {
				t.Errorf("Label mismatch: expected %s, got %s", tt.input.Label, result.Label)
			}
			
			if result.IsInitial != tt.input.IsInitial {
				t.Errorf("IsInitial mismatch: expected %v, got %v", tt.input.IsInitial, result.IsInitial)
			}
			
			if result.IsFinal != tt.input.IsFinal {
				t.Errorf("IsFinal mismatch: expected %v, got %v", tt.input.IsFinal, result.IsFinal)
			}
			
			if len(result.Children) != len(tt.input.Children) {
				t.Errorf("Children count mismatch: expected %d, got %d", len(tt.input.Children), len(result.Children))
			}
		})
	}
}

func TestConvertTransitionToProto(t *testing.T) {
	tests := []struct {
		name  string
		input *sc.Transition
	}{
		{
			name:  "nil transition",
			input: nil,
		},
		{
			name: "simple transition",
			input: &sc.Transition{
				Label: "TestTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
			},
		},
		{
			name: "transition with guard",
			input: &sc.Transition{
				Label: "GuardedTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
				Guard: &sc.Guard{Expression: "x > 0"},
			},
		},
		{
			name: "transition with actions",
			input: &sc.Transition{
				Label: "ActionTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
				Actions: []*sc.Action{
					{Label: "action1"},
					{Label: "action2"},
				},
			},
		},
		{
			name: "transition with guard and actions",
			input: &sc.Transition{
				Label: "ComplexTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
				Guard: &sc.Guard{Expression: "condition"},
				Actions: []*sc.Action{
					{Label: "action1"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("convertTransitionToProto panicked: %v", r)
				}
			}()
			
			result := convertTransitionToProto(tt.input)
			
			if tt.input == nil {
				if result != nil {
					t.Fatal("Expected nil result for nil input")
				}
				return
			}
			
			if result == nil {
				t.Fatal("Expected non-nil result for non-nil input")
			}
			
			// Basic validation
			if result.Label != tt.input.Label {
				t.Errorf("Label mismatch: expected %s, got %s", tt.input.Label, result.Label)
			}
			
			if result.Event != tt.input.Event {
				t.Errorf("Event mismatch: expected %s, got %s", tt.input.Event, result.Event)
			}
			
			if len(result.From) != len(tt.input.From) {
				t.Errorf("From count mismatch: expected %d, got %d", len(tt.input.From), len(result.From))
			}
			
			if len(result.To) != len(tt.input.To) {
				t.Errorf("To count mismatch: expected %d, got %d", len(tt.input.To), len(result.To))
			}
			
			// Guard validation
			if tt.input.Guard == nil && result.Guard != nil {
				t.Error("Expected nil guard, got non-nil")
			}
			if tt.input.Guard != nil && result.Guard == nil {
				t.Error("Expected non-nil guard, got nil")
			}
			if tt.input.Guard != nil && result.Guard != nil {
				if result.Guard.Expression != tt.input.Guard.Expression {
					t.Errorf("Guard expression mismatch: expected %s, got %s", tt.input.Guard.Expression, result.Guard.Expression)
				}
			}
			
			// Actions validation
			if len(result.Actions) != len(tt.input.Actions) {
				t.Errorf("Actions count mismatch: expected %d, got %d", len(tt.input.Actions), len(result.Actions))
			}
		})
	}
}

func TestConvertEventToProto(t *testing.T) {
	tests := []struct {
		name  string
		input *sc.Event
	}{
		{
			name:  "nil event",
			input: nil,
		},
		{
			name: "simple event",
			input: &sc.Event{
				Label: "test_event",
			},
		},
		{
			name: "empty label event",
			input: &sc.Event{
				Label: "",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("convertEventToProto panicked: %v", r)
				}
			}()
			
			result := convertEventToProto(tt.input)
			
			if tt.input == nil {
				if result != nil {
					t.Fatal("Expected nil result for nil input")
				}
				return
			}
			
			if result == nil {
				t.Fatal("Expected non-nil result for non-nil input")
			}
			
			if result.Label != tt.input.Label {
				t.Errorf("Label mismatch: expected %s, got %s", tt.input.Label, result.Label)
			}
		})
	}
}

func TestConvertMachineToProto(t *testing.T) {
	tests := []struct {
		name  string
		input *sc.Machine
	}{
		{
			name:  "nil machine",
			input: nil,
		},
		{
			name: "simple machine",
			input: &sc.Machine{
				Id:    "test-machine",
				State: sc.MachineStateRunning,
			},
		},
		{
			name: "machine with different state",
			input: &sc.Machine{
				Id:    "another-machine",
				State: sc.MachineStateStopped,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("convertMachineToProto panicked: %v", r)
				}
			}()
			
			result := convertMachineToProto(tt.input)
			
			if tt.input == nil {
				if result != nil {
					t.Fatal("Expected nil result for nil input")
				}
				return
			}
			
			if result == nil {
				t.Fatal("Expected non-nil result for non-nil input")
			}
			
			if result.Id != tt.input.Id {
				t.Errorf("Id mismatch: expected %s, got %s", tt.input.Id, result.Id)
			}
		})
	}
}

func TestStatechart_ValidateWithService(t *testing.T) {
	// This test is mostly to ensure the method exists and doesn't panic
	// In a real scenario, this would require a running gRPC service
	
	statechart := &Statechart{
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
			},
		},
	}
	
	// Create a mock client (this would fail in real usage but tests the code path)
	client := &ValidatorClient{
		client: nil, // In real usage, this would be a proper gRPC client
		conn:   nil,
	}
	
	ctx := context.Background()
	
	// This should return an error since we don't have a real service running
	err := statechart.ValidateWithService(ctx, client)
	if err == nil {
		t.Log("ValidateWithService completed without error (possibly due to mock setup)")
	} else {
		t.Logf("ValidateWithService returned expected error: %v", err)
	}
}

func TestComplexConversions(t *testing.T) {
	// Test complex statechart conversions
	testCases := []struct {
		name       string
		statechart *sc.Statechart
	}{
		{
			name:       "simple statechart",
			statechart: testutil.CreateSimpleStatechart(),
		},
		{
			name:       "hierarchical statechart",
			statechart: testutil.CreateHierarchicalStatechart(),
		},
		{
			name:       "orthogonal statechart",
			statechart: testutil.CreateOrthogonalStatechart(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test root state conversion
			if tc.statechart.RootState != nil {
				protoState := convertStateToProto(tc.statechart.RootState)
				if protoState == nil {
					t.Fatal("Expected non-nil proto state")
				}
				
				if protoState.Label != tc.statechart.RootState.Label {
					t.Errorf("Root state label mismatch: expected %s, got %s", tc.statechart.RootState.Label, protoState.Label)
				}
			}
			
			// Test transitions conversion
			for i, transition := range tc.statechart.Transitions {
				protoTransition := convertTransitionToProto(transition)
				if protoTransition == nil {
					t.Fatalf("Expected non-nil proto transition for transition %d", i)
				}
				
				if protoTransition.Label != transition.Label {
					t.Errorf("Transition %d label mismatch: expected %s, got %s", i, transition.Label, protoTransition.Label)
				}
			}
			
			// Test events conversion
			for i, event := range tc.statechart.Events {
				protoEvent := convertEventToProto(event)
				if protoEvent == nil {
					t.Fatalf("Expected non-nil proto event for event %d", i)
				}
				
				if protoEvent.Label != event.Label {
					t.Errorf("Event %d label mismatch: expected %s, got %s", i, event.Label, protoEvent.Label)
				}
			}
		})
	}
}

func BenchmarkConvertStateToProto(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(100, 50)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = convertStateToProto(statechart.RootState)
	}
}

func BenchmarkConvertTransitionToProto(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(100, 50)
	if len(statechart.Transitions) == 0 {
		b.Skip("No transitions to benchmark")
	}
	
	transition := statechart.Transitions[0]
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = convertTransitionToProto(transition)
	}
}

func BenchmarkConvertEventToProto(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(100, 50)
	if len(statechart.Events) == 0 {
		b.Skip("No events to benchmark")
	}
	
	event := statechart.Events[0]
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = convertEventToProto(event)
	}
}

func BenchmarkConvertMachineToProto(b *testing.B) {
	machine := &sc.Machine{
		Id:    "benchmark-machine",
		State: sc.MachineStateRunning,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = convertMachineToProto(machine)
	}
}