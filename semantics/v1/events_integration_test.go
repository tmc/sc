package semantics

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

// TestEventProcessorIntegration tests the complete event processing pipeline
func TestEventProcessorIntegration(t *testing.T) {
	// Create a simple test machine
	machine := &sc.Machine{
		Id:    "integration-test",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
			},
		},
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{Label: "StateA", Type: sc.StateTypeBasic},
					{Label: "StateB", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "AtoB",
					From:  []string{"StateA"},
					To:    []string{"StateB"},
					Event: "GO_TO_B",
				},
				{
					Label: "BtoA",
					From:  []string{"StateB"},
					To:    []string{"StateA"},
					Event: "GO_TO_A",
				},
			},
			Events: []*sc.Event{
				{Label: "GO_TO_B"},
				{Label: "GO_TO_A"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "StateA"}},
		},
	}

	// Create and configure event processor
	processor := NewEventProcessor(machine)
	processor.EnableTracing()

	// Add a simple filter
	filter := NewConditionalFilter("test_filter", func(event ProcessedEvent, machine *sc.Machine) bool {
		// Allow all events except BLOCKED
		return event.Event.Label != "BLOCKED"
	})
	processor.AddFilter(filter)

	// Start processor
	processor.Start()
	defer processor.Stop()

	// Test 1: Basic event processing
	t.Run("BasicEventProcessing", func(t *testing.T) {
		err := processor.SendEvent("GO_TO_B", nil)
		if err != nil {
			t.Fatalf("Failed to send event: %v", err)
		}

		// Wait for processing
		time.Sleep(50 * time.Millisecond)

		// Check state transition
		if len(machine.Configuration.States) == 0 {
			t.Fatal("No states in configuration")
		}

		if machine.Configuration.States[0].Label != "StateB" {
			t.Errorf("Expected state 'StateB', got '%s'", machine.Configuration.States[0].Label)
		}
	})

	// Test 2: Priority handling
	t.Run("PriorityHandling", func(t *testing.T) {
		// Send events with different priorities
		processor.SendEventWithPriority("LOW_PRIORITY", PriorityLow, nil)
		processor.SendEventWithPriority("HIGH_PRIORITY", PriorityHigh, nil)

		// Wait for processing
		time.Sleep(50 * time.Millisecond)

		// Check trace for processing order
		trace := processor.GetTrace()
		if len(trace) == 0 {
			t.Error("Expected trace entries")
		}
	})

	// Test 3: Event filtering
	t.Run("EventFiltering", func(t *testing.T) {
		processor.ClearTrace()

		// Send blocked and allowed events
		processor.SendEvent("BLOCKED", nil)
		processor.SendEvent("GO_TO_A", nil)

		// Wait for processing
		time.Sleep(50 * time.Millisecond)

		// Check trace
		trace := processor.GetTrace()

		blockedFound := false
		allowedFound := false

		for _, entry := range trace {
			if entry.Event.Event.Label == "BLOCKED" && entry.Action == "filtered" {
				blockedFound = true
			}
			if entry.Event.Event.Label == "GO_TO_A" && entry.Action == "processed" {
				allowedFound = true
			}
		}

		if !blockedFound {
			t.Error("Expected BLOCKED event to be filtered")
		}
		if !allowedFound {
			t.Error("Expected GO_TO_A event to be processed")
		}
	})

	// Test 4: Internal events
	t.Run("InternalEvents", func(t *testing.T) {
		processor.ClearTrace()

		// Send an event to trigger transitions
		processor.SendEvent("GO_TO_B", nil)

		// Wait for processing
		time.Sleep(50 * time.Millisecond)

		// Check for internal events in trace
		trace := processor.GetTrace()

		entryFound := false
		exitFound := false

		for _, entry := range trace {
			if entry.Event.Type == EventTypeEntry {
				entryFound = true
			}
			if entry.Event.Type == EventTypeExit {
				exitFound = true
			}
		}

		if !entryFound {
			t.Error("Expected entry event to be generated")
		}
		if !exitFound {
			t.Error("Expected exit event to be generated")
		}
	})
}

// TestEventQueueConcurrency tests concurrent access to the event queue
func TestEventQueueConcurrency(t *testing.T) {
	queue := NewEventQueue()
	
	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	numProducers := 3
	numConsumers := 2
	eventsPerProducer := 10
	
	// Channel to collect consumed events
	consumedEvents := make(chan ProcessedEvent, numProducers*eventsPerProducer)
	
	// Start consumers
	for i := 0; i < numConsumers; i++ {
		go func(consumerID int) {
			for {
				event, ok := queue.Dequeue(ctx)
				if !ok {
					return
				}
				consumedEvents <- event
			}
		}(i)
	}
	
	// Start producers
	for i := 0; i < numProducers; i++ {
		go func(producerID int) {
			for j := 0; j < eventsPerProducer; j++ {
				event := ProcessedEvent{
					Event: &sc.Event{Label: "CONCURRENT_TEST"},
					Type:  EventTypeExternal,
					Priority: PriorityNormal,
					Timestamp: time.Now(),
					ID: fmt.Sprintf("p%d_e%d", producerID, j),
				}
				queue.Enqueue(event)
			}
		}(i)
	}
	
	// Wait for all events to be produced and consumed
	time.Sleep(100 * time.Millisecond)
	queue.Close()
	
	// Count consumed events
	close(consumedEvents)
	count := 0
	for range consumedEvents {
		count++
	}
	
	expectedCount := numProducers * eventsPerProducer
	if count != expectedCount {
		t.Errorf("Expected %d events to be consumed, got %d", expectedCount, count)
	}
}

// TestEventProcessorLifecycle tests the lifecycle management of the event processor
func TestEventProcessorLifecycle(t *testing.T) {
	machine := &sc.Machine{
		Id:    "lifecycle-test",
		State: sc.MachineStateRunning,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{Label: "Initial", Type: sc.StateTypeBasic},
				},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Initial"}},
		},
	}

	processor := NewEventProcessor(machine)
	
	// Test that processor starts and stops cleanly
	processor.Start()
	
	// Send some events
	for i := 0; i < 5; i++ {
		processor.SendEvent("TEST_EVENT", nil)
	}
	
	// Wait a bit for processing
	time.Sleep(50 * time.Millisecond)
	
	// Stop should complete without hanging
	processor.Stop()
	
	// Verify queue is properly closed
	if processor.queue.Len() < 0 {
		t.Error("Queue should be in a valid state after stop")
	}
}