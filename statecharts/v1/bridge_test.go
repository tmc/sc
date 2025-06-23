package v1

import (
	"reflect"
	"testing"

	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
	testutil "github.com/tmc/sc/testing"
)

func TestFromNative(t *testing.T) {
	tests := []struct {
		name     string
		input    *sc.Statechart
		expected *Statechart
	}{
		{
			name:     "nil statechart",
			input:    nil,
			expected: nil,
		},
		{
			name:  "simple statechart",
			input: testutil.CreateSimpleStatechart(),
			expected: &Statechart{
				RootState: &State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_OR,
					Children: []*State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
							IsFinal:   false,
							Children:  []*State{},
						},
						{
							Label:     "B",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: false,
							IsFinal:   false,
							Children:  []*State{},
						},
					},
					IsInitial: false,
					IsFinal:   false,
				},
				Transitions: []*Transition{
					{
						Label: "A_to_B",
						From:  []string{"A"},
						To:    []string{"B"},
						Event: "go",
					},
				},
				Events: []*Event{
					{Label: "go"},
				},
			},
		},
		{
			name:  "hierarchical statechart",
			input: testutil.CreateHierarchicalStatechart(),
			expected: &Statechart{
				RootState: &State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_OR,
					Children: []*State{
						{
							Label:     "Active",
							Type:      pb.StateType_STATE_TYPE_OR,
							IsInitial: true,
							IsFinal:   false,
							Children: []*State{
								{
									Label:     "Idle",
									Type:      pb.StateType_STATE_TYPE_BASIC,
									IsInitial: true,
									IsFinal:   false,
									Children:  []*State{},
								},
								{
									Label:     "Processing",
									Type:      pb.StateType_STATE_TYPE_BASIC,
									IsInitial: false,
									IsFinal:   false,
									Children:  []*State{},
								},
							},
						},
						{
							Label:     "Inactive",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: false,
							IsFinal:   false,
							Children:  []*State{},
						},
					},
					IsInitial: false,
					IsFinal:   false,
				},
				Transitions: []*Transition{
					{
						Label: "idle_to_processing",
						From:  []string{"Idle"},
						To:    []string{"Processing"},
						Event: "start",
					},
					{
						Label: "processing_to_idle",
						From:  []string{"Processing"},
						To:    []string{"Idle"},
						Event: "finish",
					},
					{
						Label: "active_to_inactive",
						From:  []string{"Active"},
						To:    []string{"Inactive"},
						Event: "stop",
					},
				},
				Events: []*Event{
					{Label: "start"},
					{Label: "finish"},
					{Label: "stop"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FromNative(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("FromNative() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestToNative(t *testing.T) {
	tests := []struct {
		name     string
		input    *Statechart
		expected *sc.Statechart
	}{
		{
			name:     "nil statechart",
			input:    nil,
			expected: nil,
		},
		{
			name: "simple statechart",
			input: &Statechart{
				RootState: &State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_OR,
					Children: []*State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
							IsFinal:   false,
							Children:  []*State{},
						},
						{
							Label:     "B",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: false,
							IsFinal:   false,
							Children:  []*State{},
						},
					},
					IsInitial: false,
					IsFinal:   false,
				},
				Transitions: []*Transition{
					{
						Label: "A_to_B",
						From:  []string{"A"},
						To:    []string{"B"},
						Event: "go",
					},
				},
				Events: []*Event{
					{Label: "go"},
				},
			},
			expected: testutil.CreateSimpleStatechart(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ToNative(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("ToNative() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestRoundTripConversion(t *testing.T) {
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
			// Convert native -> protobuf -> native
			proto := FromNative(tc.statechart)
			roundTrip := ToNative(proto)

			if !reflect.DeepEqual(tc.statechart, roundTrip) {
				t.Errorf("Round trip conversion failed.\nOriginal: %+v\nRound trip: %+v", tc.statechart, roundTrip)
			}
		})
	}
}

func TestFromNativeState(t *testing.T) {
	tests := []struct {
		name     string
		input    *sc.State
		expected *State
	}{
		{
			name:     "nil state",
			input:    nil,
			expected: nil,
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
			expected: &State{
				Label:     "TestState",
				Type:      pb.StateType_STATE_TYPE_BASIC,
				IsInitial: true,
				IsFinal:   false,
				Children:  []*State{},
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
			expected: &State{
				Label: "Parent",
				Type:  pb.StateType_STATE_TYPE_OR,
				Children: []*State{
					{
						Label:     "Child1",
						Type:      pb.StateType_STATE_TYPE_BASIC,
						IsInitial: true,
						IsFinal:   false,
						Children:  []*State{},
					},
					{
						Label:     "Child2",
						Type:      pb.StateType_STATE_TYPE_BASIC,
						IsInitial: false,
						IsFinal:   true,
						Children:  []*State{},
					},
				},
				IsInitial: false,
				IsFinal:   false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fromNativeState(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("fromNativeState() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestToNativeState(t *testing.T) {
	tests := []struct {
		name     string
		input    *State
		expected *sc.State
	}{
		{
			name:     "nil state",
			input:    nil,
			expected: nil,
		},
		{
			name: "basic state",
			input: &State{
				Label:     "TestState",
				Type:      pb.StateType_STATE_TYPE_BASIC,
				IsInitial: true,
				IsFinal:   false,
				Children:  []*State{},
			},
			expected: &sc.State{
				Label:     "TestState",
				Type:      sc.StateTypeBasic,
				IsInitial: true,
				IsFinal:   false,
				Children:  []*sc.State{},
			},
		},
		{
			name: "parallel state",
			input: &State{
				Label: "ParallelState",
				Type:  pb.StateType_STATE_TYPE_AND,
				Children: []*State{
					{
						Label:     "Region1",
						Type:      pb.StateType_STATE_TYPE_OR,
						IsInitial: false,
						IsFinal:   false,
						Children:  []*State{},
					},
				},
				IsInitial: false,
				IsFinal:   false,
			},
			expected: &sc.State{
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
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := toNativeState(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("toNativeState() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestFromNativeTransition(t *testing.T) {
	tests := []struct {
		name     string
		input    *sc.Transition
		expected *Transition
	}{
		{
			name:     "nil transition",
			input:    nil,
			expected: nil,
		},
		{
			name: "simple transition",
			input: &sc.Transition{
				Label: "TestTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
			},
			expected: &Transition{
				Label:   "TestTransition",
				From:    []string{"A"},
				To:      []string{"B"},
				Event:   "test_event",
				Guard:   nil,
				Actions: nil,
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
			expected: &Transition{
				Label: "GuardedTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
				Guard: &pb.Guard{Expression: "x > 0"},
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
			expected: &Transition{
				Label: "ActionTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
				Actions: []*pb.Action{
					{Label: "action1"},
					{Label: "action2"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fromNativeTransition(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("fromNativeTransition() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestToNativeTransition(t *testing.T) {
	tests := []struct {
		name     string
		input    *Transition
		expected *sc.Transition
	}{
		{
			name:     "nil transition",
			input:    nil,
			expected: nil,
		},
		{
			name: "simple transition",
			input: &Transition{
				Label: "TestTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
			},
			expected: &sc.Transition{
				Label:   "TestTransition",
				From:    []string{"A"},
				To:      []string{"B"},
				Event:   "test_event",
				Guard:   nil,
				Actions: []*sc.Action{},
			},
		},
		{
			name: "transition with guard and actions",
			input: &Transition{
				Label: "ComplexTransition",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "test_event",
				Guard: &pb.Guard{Expression: "condition"},
				Actions: []*pb.Action{
					{Label: "action1"},
				},
			},
			expected: &sc.Transition{
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
			result := toNativeTransition(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("toNativeTransition() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestFromNativeEvent(t *testing.T) {
	tests := []struct {
		name     string
		input    *sc.Event
		expected *Event
	}{
		{
			name:     "nil event",
			input:    nil,
			expected: nil,
		},
		{
			name: "simple event",
			input: &sc.Event{
				Label: "test_event",
			},
			expected: &Event{
				Label: "test_event",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fromNativeEvent(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("fromNativeEvent() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func TestToNativeEvent(t *testing.T) {
	tests := []struct {
		name     string
		input    *Event
		expected *sc.Event
	}{
		{
			name:     "nil event",
			input:    nil,
			expected: nil,
		},
		{
			name: "simple event",
			input: &Event{
				Label: "test_event",
			},
			expected: &sc.Event{
				Label: "test_event",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := toNativeEvent(tt.input)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("toNativeEvent() result mismatch.\nExpected: %+v\nGot: %+v", tt.expected, result)
			}
		})
	}
}

func BenchmarkFromNative(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(100, 50)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FromNative(statechart)
	}
}

func BenchmarkToNative(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(100, 50)
	proto := FromNative(statechart)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ToNative(proto)
	}
}

func BenchmarkRoundTripConversion(b *testing.B) {
	statechart := testutil.CreateLargeStatechart(100, 50)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto := FromNative(statechart)
		_ = ToNative(proto)
	}
}