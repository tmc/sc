// Command event_demo demonstrates the statechart event processing system.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/internal/version"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

func main() {
	versionFlag := flag.Bool("version", false, "Print version information")
	flag.Parse()

	if *versionFlag {
		fmt.Println(version.Info())
		os.Exit(0)
	}

	fmt.Println("=== Statechart Event Processing Demo ===")

	// Create a simple turnstile state machine
	machine := createTurnstileStateMachine()
	
	// Create and configure event processor
	processor := semantics.NewEventProcessor(machine)
	processor.EnableTracing()
	
	// Add event filter
	filter := semantics.NewConditionalFilter("demo_filter", 
		func(event semantics.ProcessedEvent, machine *sc.Machine) bool {
			// Allow all events in this demo
			return true
		})
	processor.AddFilter(filter)
	
	// Start the processor
	processor.Start()
	defer func() {
		fmt.Println("\nShutting down event processor...")
		processor.Stop()
		fmt.Println("Event processor stopped.")
	}()
	
	// Run the demo
	runTurnstileDemo(processor, machine)
	
	// Show trace
	showEventTrace(processor)
}

func createTurnstileStateMachine() *sc.Machine {
	return &sc.Machine{
		Id:    "turnstile-demo",
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
					{Label: "Locked", Type: sc.StateTypeBasic},
					{Label: "Unlocked", Type: sc.StateTypeBasic},
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
					Label: "PushThrough",
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
}

func runTurnstileDemo(processor *semantics.EventProcessor, machine *sc.Machine) {
	fmt.Println("\n1. Initial state:")
	printMachineState(machine)
	
	fmt.Println("\n2. Attempting to push while locked:")
	sendEventAndWait(processor, "PUSH", "Should remain locked")
	printMachineState(machine)
	
	fmt.Println("\n3. Inserting coin:")
	sendEventAndWait(processor, "COIN", "Should unlock")
	printMachineState(machine)
	
	fmt.Println("\n4. Pushing through:")
	sendEventAndWait(processor, "PUSH", "Should lock again")
	printMachineState(machine)
	
	fmt.Println("\n5. Testing priority events:")
	// Send multiple events with different priorities
	processor.SendEventWithPriority("COIN", semantics.PriorityLow, nil)
	processor.SendEventWithPriority("EMERGENCY", semantics.PriorityCritical, nil)
	processor.SendEventWithPriority("PUSH", semantics.PriorityHigh, nil)
	time.Sleep(50 * time.Millisecond)
	printMachineState(machine)
}

func sendEventAndWait(processor *semantics.EventProcessor, event, description string) {
	fmt.Printf("   Sending %s event (%s)\n", event, description)
	err := processor.SendEvent(event, nil)
	if err != nil {
		log.Printf("Failed to send event %s: %v", event, err)
		return
	}
	time.Sleep(20 * time.Millisecond) // Wait for processing
}

func printMachineState(machine *sc.Machine) {
	if machine.Configuration == nil || len(machine.Configuration.States) == 0 {
		fmt.Println("   State: No active states")
		return
	}
	
	fmt.Printf("   State: %s\n", machine.Configuration.States[0].Label)
	
	// Print context if available
	if machine.Context != nil && machine.Context.Fields != nil {
		fmt.Printf("   Context: ")
		first := true
		for key, value := range machine.Context.Fields {
			if !first {
				fmt.Printf(", ")
			}
			fmt.Printf("%s=%s", key, value.String())
			first = false
		}
		fmt.Println()
	}
}

func showEventTrace(processor *semantics.EventProcessor) {
	fmt.Println("\n=== Event Processing Trace ===")
	trace := processor.GetTrace()
	
	if len(trace) == 0 {
		fmt.Println("No events processed.")
		return
	}
	
	for i, entry := range trace {
		fmt.Printf("%d. Event: %s (Type: %s, Priority: %s)\n", 
			i+1, 
			entry.Event.Event.Label,
			entry.Event.Type.String(),
			entry.Event.Priority.String())
		fmt.Printf("   Action: %s\n", entry.Action)
		fmt.Printf("   From: %v -> To: %v\n", entry.FromStates, entry.ToStates)
		if len(entry.Transitions) > 0 {
			fmt.Printf("   Transitions: %v\n", entry.Transitions)
		}
		if entry.Error != nil {
			fmt.Printf("   Error: %v\n", entry.Error)
		}
		fmt.Printf("   Timestamp: %s\n", entry.Timestamp.Format("15:04:05.000"))
		fmt.Println()
	}
	
	fmt.Printf("Total events processed: %d\n", len(trace))
}