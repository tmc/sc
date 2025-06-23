package validation

import (
	"testing"

	"github.com/tmc/sc"
	testutil "github.com/tmc/sc/testing"
)

func TestValidateUniqueStateLabels(t *testing.T) {
	tests := []struct {
		name          string
		statechart    *sc.Statechart
		expectError   bool
		expectedError string
	}{
		{
			name:          "nil root state",
			statechart:    &sc.Statechart{RootState: nil},
			expectError:   true,
			expectedError: "root state is nil",
		},
		{
			name:        "unique labels",
			statechart:  testutil.CreateSimpleStatechart(),
			expectError: false,
		},
		{
			name: "duplicate labels at same level",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").AsInitial().Build(),
							testutil.NewStateBuilder("A").Build(), // Duplicate
						).Build(),
				).Build(),
			expectError:   true,
			expectedError: "duplicate state label: A",
		},
		{
			name: "duplicate labels at different levels",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Parent").
								AsInitial().
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Child").AsInitial().Build(),
								).Build(),
							testutil.NewStateBuilder("Child").Build(), // Duplicate at different level
						).Build(),
				).Build(),
			expectError:   true,
			expectedError: "duplicate state label: Child",
		},
		{
			name: "complex hierarchy with unique labels",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").
								AsInitial().
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("A1").AsInitial().Build(),
									testutil.NewStateBuilder("A2").Build(),
								).Build(),
							testutil.NewStateBuilder("B").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("B1").AsInitial().Build(),
									testutil.NewStateBuilder("B2").Build(),
								).Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateUniqueStateLabels(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Fatalf("Expected error '%s', got '%s'", tt.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestValidateSingleDefaultChild(t *testing.T) {
	tests := []struct {
		name          string
		statechart    *sc.Statechart
		expectError   bool
		expectedError string
	}{
		{
			name:          "nil root state",
			statechart:    &sc.Statechart{RootState: nil},
			expectError:   true,
			expectedError: "root state is nil",
		},
		{
			name: "normal state with single default child",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").AsInitial().Build(),
							testutil.NewStateBuilder("B").Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
		{
			name: "normal state with no default child",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").Build(),
							testutil.NewStateBuilder("B").Build(),
						).Build(),
				).Build(),
			expectError:   true,
			expectedError: "state __root__ has 0 default states, should have exactly 1",
		},
		{
			name: "normal state with multiple default children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").AsInitial().Build(),
							testutil.NewStateBuilder("B").AsInitial().Build(), // Multiple defaults
						).Build(),
				).Build(),
			expectError:   true,
			expectedError: "state __root__ has 2 default states, should have exactly 1",
		},
		{
			name: "parallel state with multiple default children (should pass)",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeParallel).
						WithChildren(
							testutil.NewStateBuilder("RegionA").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("A1").AsInitial().Build(),
								).Build(),
							testutil.NewStateBuilder("RegionB").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("B1").AsInitial().Build(),
								).Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
		{
			name: "nested normal states with correct defaults",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Parent").
								AsInitial().
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Child1").AsInitial().Build(),
									testutil.NewStateBuilder("Child2").Build(),
								).Build(),
							testutil.NewStateBuilder("Other").Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
		{
			name: "nested normal state with invalid defaults",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Parent").
								AsInitial().
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Child1").Build(), // No default child
									testutil.NewStateBuilder("Child2").Build(),
								).Build(),
							testutil.NewStateBuilder("Other").Build(),
						).Build(),
				).Build(),
			expectError:   true,
			expectedError: "state Parent has 0 default states, should have exactly 1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateSingleDefaultChild(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Fatalf("Expected error '%s', got '%s'", tt.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestValidateBasicHasNoChildren(t *testing.T) {
	tests := []struct {
		name          string
		statechart    *sc.Statechart
		expectError   bool
		expectedError string
	}{
		{
			name:          "nil root state",
			statechart:    &sc.Statechart{RootState: nil},
			expectError:   true,
			expectedError: "root state is nil",
		},
		{
			name: "basic state with no children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").AsInitial().WithType(sc.StateTypeBasic).Build(),
							testutil.NewStateBuilder("B").WithType(sc.StateTypeBasic).Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
		{
			name: "basic state with children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").
								AsInitial().
								WithType(sc.StateTypeBasic).
								WithChildren(
									testutil.NewStateBuilder("Child").Build(), // Basic state should not have children
								).Build(),
						).Build(),
				).Build(),
			expectError:   true,
			expectedError: "basic state A has children",
		},
		{
			name: "mixed state types - valid",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Compound").
								AsInitial().
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Basic1").AsInitial().WithType(sc.StateTypeBasic).Build(),
									testutil.NewStateBuilder("Basic2").WithType(sc.StateTypeBasic).Build(),
								).Build(),
							testutil.NewStateBuilder("BasicTop").WithType(sc.StateTypeBasic).Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateBasicHasNoChildren(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Fatalf("Expected error '%s', got '%s'", tt.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestValidateCompoundHasChildren(t *testing.T) {
	tests := []struct {
		name          string
		statechart    *sc.Statechart
		expectError   bool
		expectedError string
	}{
		{
			name:          "nil root state",
			statechart:    &sc.Statechart{RootState: nil},
			expectError:   true,
			expectedError: "root state is nil",
		},
		{
			name: "normal state with children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("A").AsInitial().Build(),
							testutil.NewStateBuilder("B").Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
		{
			name: "normal state with no children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("BadCompound").
						WithType(sc.StateTypeNormal).
						Build(), // No children for compound state
				).Build(),
			expectError:   true,
			expectedError: "compound state BadCompound has no children",
		},
		{
			name: "parallel state with children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("__root__").
						WithType(sc.StateTypeParallel).
						WithChildren(
							testutil.NewStateBuilder("RegionA").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("A1").AsInitial().Build(),
								).Build(),
							testutil.NewStateBuilder("RegionB").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("B1").AsInitial().Build(),
								).Build(),
						).Build(),
				).Build(),
			expectError: false,
		},
		{
			name: "parallel state with no children",
			statechart: testutil.NewStatechartBuilder().
				WithRootState(
					testutil.NewStateBuilder("EmptyParallel").
						WithType(sc.StateTypeParallel).
						Build(), // No children for parallel state
				).Build(),
			expectError:   true,
			expectedError: "compound state EmptyParallel has no children",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCompoundHasChildren(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Fatalf("Expected error '%s', got '%s'", tt.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestValidateRootState(t *testing.T) {
	tests := []struct {
		name          string
		statechart    *sc.Statechart
		expectError   bool
		expectedError string
	}{
		{
			name:          "nil root state",
			statechart:    &sc.Statechart{RootState: nil},
			expectError:   true,
			expectedError: "root state is nil",
		},
		{
			name: "correct root state label",
			statechart: &sc.Statechart{
				RootState: &sc.State{Label: "__root__"},
			},
			expectError: false,
		},
		{
			name: "incorrect root state label",
			statechart: &sc.Statechart{
				RootState: &sc.State{Label: "WrongRoot"},
			},
			expectError:   true,
			expectedError: "root state has an unexpected label of 'WrongRoot' (expected '__root__')",
		},
		{
			name: "empty root state label",
			statechart: &sc.Statechart{
				RootState: &sc.State{Label: ""},
			},
			expectError:   true,
			expectedError: "root state has an unexpected label of '' (expected '__root__')",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateRootState(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Fatalf("Expected error '%s', got '%s'", tt.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestValidateDeterministicTransitionSelection(t *testing.T) {
	tests := []struct {
		name          string
		statechart    *sc.Statechart
		expectError   bool
		expectedError string
	}{
		{
			name: "deterministic transitions",
			statechart: testutil.NewStatechartBuilder().
				WithTransition(
					testutil.NewTransitionBuilder("T1").
						From("A").
						To("B").
						OnEvent("event1").
						Build(),
				).
				WithTransition(
					testutil.NewTransitionBuilder("T2").
						From("B").
						To("C").
						OnEvent("event1").
						Build(),
				).
				Build(),
			expectError: false,
		},
		{
			name: "non-deterministic transitions - same source and event",
			statechart: testutil.NewStatechartBuilder().
				WithTransition(
					testutil.NewTransitionBuilder("T1").
						From("A").
						To("B").
						OnEvent("event1").
						Build(),
				).
				WithTransition(
					testutil.NewTransitionBuilder("T2").
						From("A").
						To("C").
						OnEvent("event1").
						Build(),
				).
				Build(),
			expectError:   true,
			expectedError: "non-deterministic transitions: multiple transitions from state 'A' on event 'event1'",
		},
		{
			name: "empty event transitions ignored",
			statechart: testutil.NewStatechartBuilder().
				WithTransition(
					testutil.NewTransitionBuilder("T1").
						From("A").
						To("B").
						OnEvent("").
						Build(),
				).
				WithTransition(
					testutil.NewTransitionBuilder("T2").
						From("A").
						To("C").
						OnEvent("").
						Build(),
				).
				Build(),
			expectError: false,
		},
		{
			name: "mixed transitions - some with same event",
			statechart: testutil.NewStatechartBuilder().
				WithTransition(
					testutil.NewTransitionBuilder("T1").
						From("A").
						To("B").
						OnEvent("event1").
						Build(),
				).
				WithTransition(
					testutil.NewTransitionBuilder("T2").
						From("A").
						To("C").
						OnEvent("event2").
						Build(),
				).
				WithTransition(
					testutil.NewTransitionBuilder("T3").
						From("B").
						To("A").
						OnEvent("event1").
						Build(),
				).
				Build(),
			expectError: false,
		},
		{
			name: "multiple sources with same event",
			statechart: testutil.NewStatechartBuilder().
				WithTransition(
					testutil.NewTransitionBuilder("T1").
						From("A", "B").
						To("C").
						OnEvent("event1").
						Build(),
				).
				WithTransition(
					testutil.NewTransitionBuilder("T2").
						From("A").
						To("D").
						OnEvent("event1").
						Build(),
				).
				Build(),
			expectError:   true,
			expectedError: "non-deterministic transitions: multiple transitions from state 'A' on event 'event1'",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateDeterministicTransitionSelection(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
				if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Fatalf("Expected error '%s', got '%s'", tt.expectedError, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func TestValidateNoEventBroadcastCycles(t *testing.T) {
	tests := []struct {
		name        string
		statechart  *sc.Statechart
		expectError bool
	}{
		{
			name:        "simple statechart",
			statechart:  testutil.CreateSimpleStatechart(),
			expectError: false,
		},
		{
			name:        "complex statechart",
			statechart:  testutil.CreateHierarchicalStatechart(),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateNoEventBroadcastCycles(tt.statechart)
			
			if tt.expectError {
				if err == nil {
					t.Fatal("Expected error, but got nil")
				}
			} else {
				if err != nil {
					t.Fatalf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}

func BenchmarkValidateUniqueStateLabels(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(1000, 500)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validateUniqueStateLabels(statechart)
	}
}

func BenchmarkValidateSingleDefaultChild(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(1000, 500)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validateSingleDefaultChild(statechart)
	}
}

func BenchmarkValidateBasicHasNoChildren(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(1000, 500)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validateBasicHasNoChildren(statechart)
	}
}

func BenchmarkValidateCompoundHasChildren(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(1000, 500)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validateCompoundHasChildren(statechart)
	}
}

func BenchmarkValidateDeterministicTransitionSelection(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(1000, 2000)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validateDeterministicTransitionSelection(statechart)
	}
}