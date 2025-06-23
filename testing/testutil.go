// Package testing provides comprehensive testing utilities for the statecharts project.
package testing

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/sc"
)

// StatechartBuilder provides a fluent interface for building test statecharts.
type StatechartBuilder struct {
	statechart *sc.Statechart
}

// NewStatechartBuilder creates a new builder for constructing statecharts in tests.
func NewStatechartBuilder() *StatechartBuilder {
	return &StatechartBuilder{
		statechart: &sc.Statechart{
			RootState:   nil,
			Transitions: []*sc.Transition{},
			Events:      []*sc.Event{},
		},
	}
}

// WithRootState sets the root state of the statechart.
func (b *StatechartBuilder) WithRootState(state *sc.State) *StatechartBuilder {
	b.statechart.RootState = state
	return b
}

// WithTransition adds a transition to the statechart.
func (b *StatechartBuilder) WithTransition(transition *sc.Transition) *StatechartBuilder {
	b.statechart.Transitions = append(b.statechart.Transitions, transition)
	return b
}

// WithEvent adds an event to the statechart.
func (b *StatechartBuilder) WithEvent(event *sc.Event) *StatechartBuilder {
	b.statechart.Events = append(b.statechart.Events, event)
	return b
}

// Build returns the constructed statechart.
func (b *StatechartBuilder) Build() *sc.Statechart {
	return b.statechart
}

// StateBuilder provides a fluent interface for building test states.
type StateBuilder struct {
	state *sc.State
}

// NewStateBuilder creates a new builder for constructing states in tests.
func NewStateBuilder(label string) *StateBuilder {
	return &StateBuilder{
		state: &sc.State{
			Label:     label,
			Type:      sc.StateTypeBasic,
			Children:  []*sc.State{},
			IsInitial: false,
			IsFinal:   false,
		},
	}
}

// WithType sets the type of the state.
func (b *StateBuilder) WithType(stateType sc.StateType) *StateBuilder {
	b.state.Type = stateType
	return b
}

// WithChildren adds children to the state.
func (b *StateBuilder) WithChildren(children ...*sc.State) *StateBuilder {
	b.state.Children = append(b.state.Children, children...)
	return b
}

// AsInitial marks the state as initial.
func (b *StateBuilder) AsInitial() *StateBuilder {
	b.state.IsInitial = true
	return b
}

// AsFinal marks the state as final.
func (b *StateBuilder) AsFinal() *StateBuilder {
	b.state.IsFinal = true
	return b
}

// Build returns the constructed state.
func (b *StateBuilder) Build() *sc.State {
	return b.state
}

// TransitionBuilder provides a fluent interface for building test transitions.
type TransitionBuilder struct {
	transition *sc.Transition
}

// NewTransitionBuilder creates a new builder for constructing transitions in tests.
func NewTransitionBuilder(label string) *TransitionBuilder {
	return &TransitionBuilder{
		transition: &sc.Transition{
			Label:   label,
			From:    []string{},
			To:      []string{},
			Event:   "",
			Guard:   nil,
			Actions: []*sc.Action{},
		},
	}
}

// From sets the source states of the transition.
func (b *TransitionBuilder) From(states ...string) *TransitionBuilder {
	b.transition.From = states
	return b
}

// To sets the target states of the transition.
func (b *TransitionBuilder) To(states ...string) *TransitionBuilder {
	b.transition.To = states
	return b
}

// OnEvent sets the triggering event for the transition.
func (b *TransitionBuilder) OnEvent(event string) *TransitionBuilder {
	b.transition.Event = event
	return b
}

// WithGuard adds a guard condition to the transition.
func (b *TransitionBuilder) WithGuard(expression string) *TransitionBuilder {
	b.transition.Guard = &sc.Guard{Expression: expression}
	return b
}

// WithAction adds an action to the transition.
func (b *TransitionBuilder) WithAction(label string) *TransitionBuilder {
	b.transition.Actions = append(b.transition.Actions, &sc.Action{Label: label})
	return b
}

// Build returns the constructed transition.
func (b *TransitionBuilder) Build() *sc.Transition {
	return b.transition
}

// AssertStatechartValid is a test helper that validates a statechart and fails the test if invalid.
func AssertStatechartValid(t *testing.T, statechart *sc.Statechart) {
	t.Helper()
	if err := sc.BasicValidate(statechart); err != nil {
		t.Fatalf("Statechart validation failed: %v", err)
	}
}

// AssertStatechartInvalid is a test helper that expects a statechart to be invalid.
func AssertStatechartInvalid(t *testing.T, statechart *sc.Statechart, expectedError string) {
	t.Helper()
	err := sc.BasicValidate(statechart)
	if err == nil {
		t.Fatal("Expected statechart to be invalid, but validation passed")
	}
	if expectedError != "" && err.Error() != expectedError {
		t.Fatalf("Expected error '%s', got '%s'", expectedError, err.Error())
	}
}

// AssertConfigurationEqual compares two configurations and fails if they differ.
func AssertConfigurationEqual(t *testing.T, expected, actual []string) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("Configuration mismatch.\nExpected: %v\nActual: %v", expected, actual)
	}
}

// AssertConfigurationContains checks if a configuration contains specific states.
func AssertConfigurationContains(t *testing.T, configuration []string, states ...string) {
	t.Helper()
	configMap := make(map[string]bool)
	for _, state := range configuration {
		configMap[state] = true
	}
	
	for _, state := range states {
		if !configMap[state] {
			t.Fatalf("Configuration %v does not contain state '%s'", configuration, state)
		}
	}
}

// AssertConfigurationNotContains checks if a configuration does not contain specific states.
func AssertConfigurationNotContains(t *testing.T, configuration []string, states ...string) {
	t.Helper()
	configMap := make(map[string]bool)
	for _, state := range configuration {
		configMap[state] = true
	}
	
	for _, state := range states {
		if configMap[state] {
			t.Fatalf("Configuration %v should not contain state '%s'", configuration, state)
		}
	}
}

// CreateSimpleStatechart creates a basic statechart for testing purposes.
func CreateSimpleStatechart() *sc.Statechart {
	return NewStatechartBuilder().
		WithRootState(
			NewStateBuilder("__root__").
				WithType(sc.StateTypeNormal).
				WithChildren(
					NewStateBuilder("A").AsInitial().Build(),
					NewStateBuilder("B").Build(),
				).Build(),
		).
		WithTransition(
			NewTransitionBuilder("A_to_B").
				From("A").
				To("B").
				OnEvent("go").
				Build(),
		).
		WithEvent(&sc.Event{Label: "go"}).
		Build()
}

// CreateHierarchicalStatechart creates a hierarchical statechart for testing.
func CreateHierarchicalStatechart() *sc.Statechart {
	return NewStatechartBuilder().
		WithRootState(
			NewStateBuilder("__root__").
				WithType(sc.StateTypeNormal).
				WithChildren(
					NewStateBuilder("Active").
						AsInitial().
						WithType(sc.StateTypeNormal).
						WithChildren(
							NewStateBuilder("Idle").AsInitial().Build(),
							NewStateBuilder("Processing").Build(),
						).Build(),
					NewStateBuilder("Inactive").Build(),
				).Build(),
		).
		WithTransition(
			NewTransitionBuilder("idle_to_processing").
				From("Idle").
				To("Processing").
				OnEvent("start").
				Build(),
		).
		WithTransition(
			NewTransitionBuilder("processing_to_idle").
				From("Processing").
				To("Idle").
				OnEvent("finish").
				Build(),
		).
		WithTransition(
			NewTransitionBuilder("active_to_inactive").
				From("Active").
				To("Inactive").
				OnEvent("stop").
				Build(),
		).
		WithEvent(&sc.Event{Label: "start"}).
		WithEvent(&sc.Event{Label: "finish"}).
		WithEvent(&sc.Event{Label: "stop"}).
		Build()
}

// CreateOrthogonalStatechart creates an orthogonal statechart for testing.
func CreateOrthogonalStatechart() *sc.Statechart {
	return NewStatechartBuilder().
		WithRootState(
			NewStateBuilder("__root__").
				WithType(sc.StateTypeParallel).
				WithChildren(
					NewStateBuilder("RegionA").
						WithType(sc.StateTypeNormal).
						WithChildren(
							NewStateBuilder("A1").AsInitial().Build(),
							NewStateBuilder("A2").Build(),
						).Build(),
					NewStateBuilder("RegionB").
						WithType(sc.StateTypeNormal).
						WithChildren(
							NewStateBuilder("B1").AsInitial().Build(),
							NewStateBuilder("B2").Build(),
						).Build(),
				).Build(),
		).
		WithTransition(
			NewTransitionBuilder("A1_to_A2").
				From("A1").
				To("A2").
				OnEvent("switch_a").
				Build(),
		).
		WithTransition(
			NewTransitionBuilder("B1_to_B2").
				From("B1").
				To("B2").
				OnEvent("switch_b").
				Build(),
		).
		WithEvent(&sc.Event{Label: "switch_a"}).
		WithEvent(&sc.Event{Label: "switch_b"}).
		Build()
}

// TestCase represents a generic test case for statechart operations.
type TestCase struct {
	Name        string
	Description string
	Setup       func() *sc.Statechart
	Execute     func(*sc.Statechart) error
	Validate    func(*testing.T, *sc.Statechart, error)
}

// RunTestCases executes a slice of test cases.
func RunTestCases(t *testing.T, testCases []TestCase) {
	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			statechart := tc.Setup()
			err := tc.Execute(statechart)
			tc.Validate(t, statechart, err)
		})
	}
}

// ValidationError represents an expected validation error for testing.
type ValidationError struct {
	Message string
	Rule    string
}

// ExpectedValidationErrors represents a set of expected validation errors.
type ExpectedValidationErrors []ValidationError

// Contains checks if the expected errors contain a specific error message.
func (e ExpectedValidationErrors) Contains(message string) bool {
	for _, err := range e {
		if err.Message == message {
			return true
		}
	}
	return false
}

// AssertValidationErrors checks that validation produces expected errors.
func AssertValidationErrors(t *testing.T, statechart *sc.Statechart, expected ExpectedValidationErrors) {
	t.Helper()
	err := sc.BasicValidate(statechart)
	
	if len(expected) == 0 {
		if err != nil {
			t.Fatalf("Expected no validation errors, but got: %v", err)
		}
		return
	}
	
	if err == nil {
		t.Fatal("Expected validation errors, but validation passed")
	}
	
	// For simplicity, we'll just check if the error message contains expected text
	// A more sophisticated implementation would parse structured errors
	errorMessage := err.Error()
	for _, expectedErr := range expected {
		if expectedErr.Message != "" && !contains(errorMessage, expectedErr.Message) {
			t.Fatalf("Expected error message to contain '%s', but got: %s", expectedErr.Message, errorMessage)
		}
	}
}

// contains is a simple string contains check.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || (len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsAt(s, substr))))
}

func containsAt(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// BenchmarkHelper provides utilities for benchmarking statechart operations.
type BenchmarkHelper struct {
	statechart *sc.Statechart
}

// NewBenchmarkHelper creates a new benchmark helper.
func NewBenchmarkHelper(statechart *sc.Statechart) *BenchmarkHelper {
	return &BenchmarkHelper{statechart: statechart}
}

// BenchmarkValidation benchmarks statechart validation.
func (h *BenchmarkHelper) BenchmarkValidation(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sc.BasicValidate(h.statechart)
	}
}

// CreateLargeStatechart creates a large statechart for performance testing.
func CreateLargeStatechart(numStates, numTransitions int) *sc.Statechart {
	builder := NewStatechartBuilder()
	
	// Create root state
	rootBuilder := NewStateBuilder("__root__").WithType(sc.StateTypeNormal)
	
	// Create states
	states := make([]*sc.State, numStates)
	for i := 0; i < numStates; i++ {
		stateBuilder := NewStateBuilder(fmt.Sprintf("State%d", i))
		if i == 0 {
			stateBuilder.AsInitial()
		}
		states[i] = stateBuilder.Build()
	}
	
	rootBuilder.WithChildren(states...)
	builder.WithRootState(rootBuilder.Build())
	
	// Create transitions
	for i := 0; i < numTransitions && i < numStates-1; i++ {
		transition := NewTransitionBuilder(fmt.Sprintf("Transition%d", i)).
			From(fmt.Sprintf("State%d", i)).
			To(fmt.Sprintf("State%d", (i+1)%numStates)).
			OnEvent(fmt.Sprintf("event%d", i)).
			Build()
		builder.WithTransition(transition)
		builder.WithEvent(&sc.Event{Label: fmt.Sprintf("event%d", i)})
	}
	
	return builder.Build()
}