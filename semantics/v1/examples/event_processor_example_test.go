package examples

import (
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

func TestEventProcessorExample(t *testing.T) {
	// This test ensures the example function runs without panic
	// and demonstrates proper turnstile behavior
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("EventProcessorExample panicked: %v", r)
		}
	}()

	// Run the example (it will print to stdout)
	EventProcessorExample()
}

func TestEventProcessorExampleLogic(t *testing.T) {
	// Test the turnstile logic step by step
	machine := &sc.Machine{
		Id:    "turnstile-test",
		State: sc.MachineStateRunning,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{Label: "Locked", Type: sc.StateTypeBasic},
					{Label: "Unlocked", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{Label: "InsertCoin", From: []string{"Locked"}, To: []string{"Unlocked"}, Event: "COIN"},
				{Label: "Push", From: []string{"Unlocked"}, To: []string{"Locked"}, Event: "PUSH"},
				{Label: "PushLocked", From: []string{"Locked"}, To: []string{"Locked"}, Event: "PUSH"},
			},
			Events: []*sc.Event{
				{Label: "COIN"}, {Label: "PUSH"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Locked"}},
		},
	}

	processor := semantics.NewEventProcessor(machine)
	processor.EnableTracing()
	processor.Start()
	defer processor.Stop()

	// Test initial state
	config := processor.GetConfiguration()
	if len(config.States) != 1 || config.States[0].Label != "Locked" {
		t.Errorf("Expected initial state to be Locked, got %v", config.States)
	}

	// Test pushing without coin (should stay locked)
	processor.SendEvent("PUSH", nil)
	time.Sleep(10 * time.Millisecond)

	config = processor.GetConfiguration()
	if len(config.States) != 1 || config.States[0].Label != "Locked" {
		t.Errorf("Expected state to remain Locked after push, got %v", config.States)
	}

	// Test inserting coin (should unlock)
	processor.SendEvent("COIN", nil)
	time.Sleep(10 * time.Millisecond)

	config = processor.GetConfiguration()
	if len(config.States) != 1 || config.States[0].Label != "Unlocked" {
		t.Errorf("Expected state to be Unlocked after coin, got %v", config.States)
	}

	// Test pushing through (should lock again)
	processor.SendEvent("PUSH", nil)
	time.Sleep(10 * time.Millisecond)

	config = processor.GetConfiguration()
	if len(config.States) != 1 || config.States[0].Label != "Locked" {
		t.Errorf("Expected state to be Locked after push, got %v", config.States)
	}

	// Verify trace has events
	trace := processor.GetTrace()
	if len(trace) == 0 {
		t.Error("Expected trace to contain events")
	}
}

func TestEventProcessorMaintenanceFilter(t *testing.T) {
	machine := &sc.Machine{
		Id:    "turnstile-filter-test",
		State: sc.MachineStateRunning,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{Label: "Locked", Type: sc.StateTypeBasic},
					{Label: "Unlocked", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{Label: "InsertCoin", From: []string{"Locked"}, To: []string{"Unlocked"}, Event: "COIN"},
			},
			Events: []*sc.Event{{Label: "COIN"}},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Locked"}},
		},
	}

	processor := semantics.NewEventProcessor(machine)
	processor.EnableTracing()

	// Add maintenance filter
	maintenanceMode := true
	maintenanceFilter := semantics.NewConditionalFilter("maintenance",
		func(event semantics.ProcessedEvent, machine *sc.Machine) bool {
			return !maintenanceMode
		})
	processor.AddFilter(maintenanceFilter)

	processor.Start()
	defer processor.Stop()

	// Test that events are filtered when maintenance mode is on
	processor.SendEvent("COIN", nil)
	time.Sleep(10 * time.Millisecond)

	// Should still be locked because event was filtered
	config := processor.GetConfiguration()
	if len(config.States) != 1 || config.States[0].Label != "Locked" {
		t.Errorf("Expected state to remain Locked when maintenance mode is on, got %v", config.States)
	}

	// Turn off maintenance mode
	maintenanceMode = false
	processor.SendEvent("COIN", nil)
	time.Sleep(10 * time.Millisecond)

	// Should now be unlocked
	config = processor.GetConfiguration()
	if len(config.States) != 1 || config.States[0].Label != "Unlocked" {
		t.Errorf("Expected state to be Unlocked when maintenance mode is off, got %v", config.States)
	}
}

func TestOrthogonalEventExample(t *testing.T) {
	// This test ensures the orthogonal example runs without panic
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("OrthogonalEventExample panicked: %v", r)
		}
	}()

	OrthogonalEventExample()
}

func TestOrthogonalEventExampleLogic(t *testing.T) {
	// Test the media player orthogonal logic
	machine := &sc.Machine{
		Id:    "media-player-test",
		State: sc.MachineStateRunning,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{
						Label: "MediaPlayer",
						Type:  sc.StateTypeOrthogonal,
						Children: []*sc.State{
							{
								Label: "PlaybackState",
								Type:  sc.StateTypeNormal,
								Children: []*sc.State{
									{Label: "Paused", Type: sc.StateTypeBasic},
									{Label: "Playing", Type: sc.StateTypeBasic},
									{Label: "Stopped", Type: sc.StateTypeBasic},
								},
							},
							{
								Label: "VolumeControl",
								Type:  sc.StateTypeNormal,
								Children: []*sc.State{
									{Label: "Normal", Type: sc.StateTypeBasic},
									{Label: "Muted", Type: sc.StateTypeBasic},
								},
							},
						},
					},
				},
			},
			Transitions: []*sc.Transition{
				{Label: "Play", From: []string{"Paused", "Stopped"}, To: []string{"Playing"}, Event: "PLAY"},
				{Label: "Pause", From: []string{"Playing"}, To: []string{"Paused"}, Event: "PAUSE"},
				{Label: "Stop", From: []string{"Playing", "Paused"}, To: []string{"Stopped"}, Event: "STOP"},
				{Label: "Mute", From: []string{"Normal"}, To: []string{"Muted"}, Event: "MUTE"},
				{Label: "Unmute", From: []string{"Muted"}, To: []string{"Normal"}, Event: "UNMUTE"},
			},
			Events: []*sc.Event{
				{Label: "PLAY"}, {Label: "PAUSE"}, {Label: "STOP"},
				{Label: "MUTE"}, {Label: "UNMUTE"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{
				{Label: "Paused"},
				{Label: "Normal"},
			},
		},
	}

	processor := semantics.NewEventProcessor(machine)
	processor.EnableTracing()
	processor.Start()
	defer processor.Stop()

	// Test initial state (should have both Paused and Normal)
	config := processor.GetConfiguration()
	actualStates := make([]string, len(config.States))
	for i, state := range config.States {
		actualStates[i] = state.Label
	}

	if len(actualStates) != 2 {
		t.Errorf("Expected 2 initial states, got %d: %v", len(actualStates), actualStates)
	}

	hasState := func(states []string, target string) bool {
		for _, state := range states {
			if state == target {
				return true
			}
		}
		return false
	}

	if !hasState(actualStates, "Paused") || !hasState(actualStates, "Normal") {
		t.Errorf("Expected initial states [Paused, Normal], got %v", actualStates)
	}

	// Test playing (should change playback but not volume)
	processor.SendEvent("PLAY", nil)
	time.Sleep(10 * time.Millisecond)

	config = processor.GetConfiguration()
	actualStates = make([]string, len(config.States))
	for i, state := range config.States {
		actualStates[i] = state.Label
	}

	if !hasState(actualStates, "Playing") || !hasState(actualStates, "Normal") {
		t.Errorf("Expected states [Playing, Normal] after PLAY, got %v", actualStates)
	}

	// Test muting (should change volume but not playback)
	processor.SendEvent("MUTE", nil)
	time.Sleep(10 * time.Millisecond)

	config = processor.GetConfiguration()
	actualStates = make([]string, len(config.States))
	for i, state := range config.States {
		actualStates[i] = state.Label
	}

	if !hasState(actualStates, "Playing") || !hasState(actualStates, "Muted") {
		t.Errorf("Expected states [Playing, Muted] after MUTE, got %v", actualStates)
	}
}

func TestPrintCurrentState(t *testing.T) {
	tests := []struct {
		name        string
		machine     *sc.Machine
		description string
	}{
		{
			name: "no configuration",
			machine: &sc.Machine{
				Configuration: nil,
			},
			description: "should handle nil configuration",
		},
		{
			name: "empty states",
			machine: &sc.Machine{
				Configuration: &sc.Configuration{
					States: []*sc.StateRef{},
				},
			},
			description: "should handle empty states",
		},
		{
			name: "single state",
			machine: &sc.Machine{
				Configuration: &sc.Configuration{
					States: []*sc.StateRef{{Label: "TestState"}},
				},
			},
			description: "should handle single state",
		},
		{
			name: "multiple states",
			machine: &sc.Machine{
				Configuration: &sc.Configuration{
					States: []*sc.StateRef{
						{Label: "State1"},
						{Label: "State2"},
						{Label: "State3"},
					},
				},
			},
			description: "should handle multiple states",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that printCurrentState doesn't panic
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("printCurrentState panicked: %v", r)
				}
			}()

			printCurrentState(tt.machine.Configuration)
		})
	}
}

func TestFormatStates(t *testing.T) {
	tests := []struct {
		name     string
		states   []string
		expected string
	}{
		{
			name:     "empty states",
			states:   []string{},
			expected: "[]",
		},
		{
			name:     "single state",
			states:   []string{"State1"},
			expected: "State1",
		},
		{
			name:     "two states",
			states:   []string{"State1", "State2"},
			expected: "[State1, State2]",
		},
		{
			name:     "three states",
			states:   []string{"A", "B", "C"},
			expected: "[A, B, C]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatStates(tt.states)
			if result != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestEventPriorityHandling(t *testing.T) {
	machine := &sc.Machine{
		Id:    "priority-test",
		State: sc.MachineStateRunning,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{Label: "Idle", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{Label: "LowPrio", From: []string{"Idle"}, To: []string{"Idle"}, Event: "LOW"},
				{Label: "HighPrio", From: []string{"Idle"}, To: []string{"Idle"}, Event: "HIGH"},
				{Label: "CriticalPrio", From: []string{"Idle"}, To: []string{"Idle"}, Event: "CRITICAL"},
			},
			Events: []*sc.Event{
				{Label: "LOW"}, {Label: "HIGH"}, {Label: "CRITICAL"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Idle"}},
		},
	}

	processor := semantics.NewEventProcessor(machine)
	processor.EnableTracing()
	processor.Start()
	defer processor.Stop()

	// Send events with different priorities
	processor.SendEventWithPriority("LOW", semantics.PriorityLow, nil)
	processor.SendEventWithPriority("CRITICAL", semantics.PriorityCritical, nil)
	processor.SendEventWithPriority("HIGH", semantics.PriorityHigh, nil)

	time.Sleep(20 * time.Millisecond)

	trace := processor.GetTrace()
	if len(trace) < 3 {
		t.Errorf("Expected at least 3 trace entries, got %d", len(trace))
	}

	// Verify that events were processed (order might vary based on priority handling)
	eventsSeen := make(map[string]bool)
	for _, entry := range trace {
		eventsSeen[entry.Event.Event.Label] = true
	}

	expectedEvents := []string{"LOW", "HIGH", "CRITICAL"}
	for _, event := range expectedEvents {
		if !eventsSeen[event] {
			t.Errorf("Expected to see event %s in trace", event)
		}
	}
}
