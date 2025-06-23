// Package examples provides usage examples for the event processing system.
package examples

import (
	"fmt"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// EventProcessorExample demonstrates how to use the event processing system
// with a simple state machine that models a turnstile.
func EventProcessorExample() {
	// Create a turnstile state machine
	machine := &sc.Machine{
		Id:    "turnstile-example",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"coins_inserted": structpb.NewNumberValue(0),
				"people_passed":  structpb.NewNumberValue(0),
			},
		},
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{
						Label: "Locked",
						Type:  sc.StateTypeBasic,
					},
					{
						Label: "Unlocked",
						Type:  sc.StateTypeBasic,
					},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "InsertCoin",
					From:  []string{"Locked"},
					To:    []string{"Unlocked"},
					Event: "COIN",
				},
				{
					Label: "Push",
					From:  []string{"Unlocked"},
					To:    []string{"Locked"},
					Event: "PUSH",
				},
				{
					Label: "PushLocked",
					From:  []string{"Locked"},
					To:    []string{"Locked"},
					Event: "PUSH",
				},
			},
			Events: []*sc.Event{
				{Label: "COIN"},
				{Label: "PUSH"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Locked"}},
		},
	}

	// Create event processor
	processor := semantics.NewEventProcessor(machine)
	
	// Enable tracing for debugging
	processor.EnableTracing()
	
	// Add a maintenance filter
	maintenanceMode := false
	maintenanceFilter := semantics.NewConditionalFilter("maintenance", 
		func(event semantics.ProcessedEvent, machine *sc.Machine) bool {
			return !maintenanceMode
		})
	processor.AddFilter(maintenanceFilter)
	
	// Start the processor
	processor.Start()
	defer processor.Stop()
	
	fmt.Println("=== Turnstile Event Processing Example ===")
	
	// Simulate turnstile usage
	fmt.Println("\n1. Initial state:")
	printCurrentState(machine)
	
	// Try to push without coin (should stay locked)
	fmt.Println("\n2. Pushing without coin:")
	processor.SendEvent("PUSH", nil)
	time.Sleep(10 * time.Millisecond) // Wait for processing
	printCurrentState(machine)
	
	// Insert coin (should unlock)
	fmt.Println("\n3. Inserting coin:")
	processor.SendEvent("COIN", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	// Push to go through (should lock again)
	fmt.Println("\n4. Pushing through:")
	processor.SendEvent("PUSH", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	// Test priority handling
	fmt.Println("\n5. Testing priority handling:")
	processor.SendEventWithPriority("COIN", semantics.PriorityLow, nil)
	processor.SendEventWithPriority("EMERGENCY_STOP", semantics.PriorityCritical, nil)
	processor.SendEventWithPriority("PUSH", semantics.PriorityHigh, nil)
	time.Sleep(20 * time.Millisecond)
	printCurrentState(machine)
	
	// Test maintenance mode
	fmt.Println("\n6. Enabling maintenance mode:")
	maintenanceMode = true
	processor.SendEvent("COIN", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	fmt.Println("\n7. Disabling maintenance mode:")
	maintenanceMode = false
	processor.SendEvent("COIN", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	// Print event trace
	fmt.Println("\n=== Event Trace ===")
	trace := processor.GetTrace()
	for i, entry := range trace {
		fmt.Printf("%d. %s: %s -> %s (Action: %s)\n", 
			i+1, 
			entry.Event.Event.Label,
			formatStates(entry.FromStates),
			formatStates(entry.ToStates),
			entry.Action)
		if entry.Error != nil {
			fmt.Printf("   Error: %v\n", entry.Error)
		}
	}
	
	fmt.Println("\n=== Example Complete ===")
}

// OrthogonalEventExample demonstrates event processing with orthogonal states
func OrthogonalEventExample() {
	// Create a media player with orthogonal regions
	machine := &sc.Machine{
		Id:    "media-player-example",
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
				// Playback transitions
				{Label: "Play", From: []string{"Paused", "Stopped"}, To: []string{"Playing"}, Event: "PLAY"},
				{Label: "Pause", From: []string{"Playing"}, To: []string{"Paused"}, Event: "PAUSE"},
				{Label: "Stop", From: []string{"Playing", "Paused"}, To: []string{"Stopped"}, Event: "STOP"},
				
				// Volume transitions
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

	fmt.Println("\n=== Orthogonal States Event Processing Example ===")
	
	fmt.Println("\n1. Initial state (Paused + Normal):")
	printCurrentState(machine)
	
	// Test independent transitions in orthogonal regions
	fmt.Println("\n2. Start playing:")
	processor.SendEvent("PLAY", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	fmt.Println("\n3. Mute audio (independent of playback):")
	processor.SendEvent("MUTE", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	fmt.Println("\n4. Pause playback (audio still muted):")
	processor.SendEvent("PAUSE", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	fmt.Println("\n5. Unmute audio:")
	processor.SendEvent("UNMUTE", nil)
	time.Sleep(10 * time.Millisecond)
	printCurrentState(machine)
	
	fmt.Println("\n=== Orthogonal Example Complete ===")
}

// Helper functions for the examples

func printCurrentState(machine *sc.Machine) {
	if machine.Configuration == nil || len(machine.Configuration.States) == 0 {
		fmt.Println("  No active states")
		return
	}
	
	fmt.Print("  Current states: ")
	for i, state := range machine.Configuration.States {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Print(state.Label)
	}
	fmt.Println()
}

func formatStates(states []string) string {
	if len(states) == 0 {
		return "[]"
	}
	if len(states) == 1 {
		return states[0]
	}
	result := "["
	for i, state := range states {
		if i > 0 {
			result += ", "
		}
		result += state
	}
	result += "]"
	return result
}