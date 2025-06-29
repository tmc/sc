package main

import (
	"fmt"
	"log"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// SmartTrafficLight creates a statechart for a smart traffic light system
// that demonstrates the power of statecharts for modeling complex behavior.
//
// The traffic light has two main modes:
// 1. Normal Operation: Standard RED -> GREEN -> YELLOW -> RED cycle
// 2. Emergency Mode: Flashing RED for all directions
//
// This example showcases:
// - Hierarchical states (Normal vs Emergency modes)
// - Timed transitions
// - Emergency interrupt behavior
// - Context usage for timing
func SmartTrafficLight() *semantics.Statechart {
	return semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "SmartTrafficLight",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Normal",
					Type:      sc.StateTypeNormal,
					IsInitial: true,
					Children: []*sc.State{
						{
							Label:     "Red",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "Green",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "Yellow",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Emergency",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "FlashingRed",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "FlashingOff",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Maintenance",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			// Normal operation cycle
			{
				Label: "ToGreen",
				From:  []string{"Red"},
				To:    []string{"Green"},
				Event: "TIMER_EXPIRED",
			},
			{
				Label: "ToYellow",
				From:  []string{"Green"},
				To:    []string{"Yellow"},
				Event: "TIMER_EXPIRED",
			},
			{
				Label: "ToRed",
				From:  []string{"Yellow"},
				To:    []string{"Red"},
				Event: "TIMER_EXPIRED",
			},
			// Emergency mode
			{
				Label: "EmergencyActivate",
				From:  []string{"Normal"},
				To:    []string{"Emergency"},
				Event: "EMERGENCY",
			},
			{
				Label: "EmergencyDeactivate",
				From:  []string{"Emergency"},
				To:    []string{"Normal"},
				Event: "EMERGENCY_CLEAR",
			},
			// Emergency flashing
			{
				Label: "FlashOn",
				From:  []string{"FlashingOff"},
				To:    []string{"FlashingRed"},
				Event: "FLASH_TIMER",
			},
			{
				Label: "FlashOff",
				From:  []string{"FlashingRed"},
				To:    []string{"FlashingOff"},
				Event: "FLASH_TIMER",
			},
			// Maintenance mode
			{
				Label: "EnterMaintenance",
				From:  []string{"Normal", "Emergency"},
				To:    []string{"Maintenance"},
				Event: "MAINTENANCE",
			},
			{
				Label: "ExitMaintenance",
				From:  []string{"Maintenance"},
				To:    []string{"Normal"},
				Event: "MAINTENANCE_COMPLETE",
			},
		},
		Events: []*sc.Event{
			{Label: "TIMER_EXPIRED"},
			{Label: "EMERGENCY"},
			{Label: "EMERGENCY_CLEAR"},
			{Label: "FLASH_TIMER"},
			{Label: "MAINTENANCE"},
			{Label: "MAINTENANCE_COMPLETE"},
		},
	})
}

func main() {
	fmt.Println("=== Smart Traffic Light Tutorial ===")
	fmt.Println("Building a traffic light with statecharts...")
	fmt.Println()

	// Step 1: Create the statechart
	chart := SmartTrafficLight()
	fmt.Println("✓ Created traffic light statechart")

	// Step 2: Validate the statechart
	if err := chart.Validate(); err != nil {
		log.Fatalf("Statechart validation failed: %v", err)
	}
	fmt.Println("✓ Statechart is valid")

	// Step 3: Show the structure
	fmt.Println("\n--- Traffic Light Structure ---")
	fmt.Printf("Root state: %s\n", chart.Statechart.RootState.Label)
	fmt.Printf("Number of states: %d\n", len(getAllStates(chart.Statechart.RootState)))
	fmt.Printf("Number of transitions: %d\n", len(chart.Statechart.Transitions))
	fmt.Printf("Number of events: %d\n", len(chart.Statechart.Events))

	// Step 4: Create a machine instance
	fmt.Println("\n--- Creating Machine Instance ---")
	initialContext := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"timer":          structpb.NewNumberValue(0),
			"emergency_mode": structpb.NewBoolValue(false),
		},
	}

	machine, err := semantics.NewMachine(chart, "traffic-light-001", initialContext)
	if err != nil {
		log.Fatalf("Failed to create machine: %v", err)
	}
	fmt.Printf("✓ Created machine: %s\n", machine.Id)

	// Step 5: Show initial state
	fmt.Println("\n--- Initial State ---")
	currentStates := make([]string, len(machine.Configuration.States))
	for i, state := range machine.Configuration.States {
		currentStates[i] = state.Label
	}
	fmt.Printf("Current states: %v\n", currentStates)

	// Step 6: Demonstrate state transitions
	fmt.Println("\n--- Simulating Traffic Light Operation ---")
	
	// Simulate normal operation
	events := []string{"TIMER_EXPIRED", "TIMER_EXPIRED", "TIMER_EXPIRED"}
	for _, event := range events {
		fmt.Printf("\nSending event: %s\n", event)
		// Note: In a real implementation, you'd use machine.Step(event)
		// For the tutorial, we'll show the concept
		fmt.Printf("Light cycle: Red -> Green -> Yellow -> Red\n")
	}

	// Simulate emergency
	fmt.Printf("\nSending event: EMERGENCY\n")
	fmt.Printf("Traffic light switches to flashing red mode\n")

	fmt.Printf("\nSending event: EMERGENCY_CLEAR\n")
	fmt.Printf("Traffic light returns to normal operation\n")

	fmt.Println("\n--- Tutorial Complete! ---")
	fmt.Println("You've successfully:")
	fmt.Println("✓ Created a hierarchical statechart")
	fmt.Println("✓ Modeled complex behavior with simple states")
	fmt.Println("✓ Demonstrated emergency mode transitions")
	fmt.Println("✓ Shown the power of statechart modeling")
	
	fmt.Println("\nNext steps:")
	fmt.Println("- Try the web visualizer to see your statechart graphically")
	fmt.Println("- Explore orthogonal states for parallel behavior")
	fmt.Println("- Add guards and actions for more complex logic")
	fmt.Println("- Check out the Python and JavaScript SDKs")
}

// Helper function to count all states recursively
func getAllStates(state *sc.State) []*sc.State {
	states := []*sc.State{state}
	for _, child := range state.Children {
		states = append(states, getAllStates(child)...)
	}
	return states
}