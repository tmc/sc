package semantics

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestEventQueue_BasicOperations(t *testing.T) {
	queue := NewEventQueue()
	
	// Test empty queue
	if queue.Len() != 0 {
		t.Errorf("Expected empty queue length 0, got %d", queue.Len())
	}
	
	// Test enqueue
	event := ProcessedEvent{
		Event: &sc.Event{Label: "TEST"},
		Type:  EventTypeExternal,
		Priority: PriorityNormal,
		Timestamp: time.Now(),
		ID: "test-1",
	}
	
	err := queue.Enqueue(event)
	if err != nil {
		t.Fatalf("Failed to enqueue event: %v", err)
	}
	
	if queue.Len() != 1 {
		t.Errorf("Expected queue length 1, got %d", queue.Len())
	}
	
	// Test dequeue
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	
	dequeuedEvent, ok := queue.Dequeue(ctx)
	if !ok {
		t.Fatal("Failed to dequeue event")
	}
	
	if dequeuedEvent.Event.Label != "TEST" {
		t.Errorf("Expected event label 'TEST', got '%s'", dequeuedEvent.Event.Label)
	}
	
	if queue.Len() != 0 {
		t.Errorf("Expected empty queue after dequeue, got length %d", queue.Len())
	}
}

func TestEventQueue_PriorityOrdering(t *testing.T) {
	queue := NewEventQueue()
	
	// Add events with different priorities
	events := []ProcessedEvent{
		{Event: &sc.Event{Label: "LOW"}, Priority: PriorityLow, Timestamp: time.Now(), ID: "1"},
		{Event: &sc.Event{Label: "HIGH"}, Priority: PriorityHigh, Timestamp: time.Now(), ID: "2"},
		{Event: &sc.Event{Label: "NORMAL"}, Priority: PriorityNormal, Timestamp: time.Now(), ID: "3"},
		{Event: &sc.Event{Label: "CRITICAL"}, Priority: PriorityCritical, Timestamp: time.Now(), ID: "4"},
	}
	
	// Enqueue in random order
	for _, event := range events {
		queue.Enqueue(event)
	}
	
	// Dequeue and check priority order
	ctx := context.Background()
	expectedOrder := []string{"CRITICAL", "HIGH", "NORMAL", "LOW"}
	
	for i, expected := range expectedOrder {
		event, ok := queue.Dequeue(ctx)
		if !ok {
			t.Fatalf("Failed to dequeue event %d", i)
		}
		if event.Event.Label != expected {
			t.Errorf("Expected event %d to be '%s', got '%s'", i, expected, event.Event.Label)
		}
	}
}

func TestEventProcessor_BasicEventHandling(t *testing.T) {
	machine := createTestMachine()
	processor := NewEventProcessor(machine)
	processor.EnableTracing()
	
	// Start processor
	processor.Start()
	defer processor.Stop()
	
	// Send event
	err := processor.SendEvent("TURN_ON", nil)
	if err != nil {
		t.Fatalf("Failed to send event: %v", err)
	}
	
	// Wait for processing
	time.Sleep(100 * time.Millisecond)
	
	// Check state transition
	if len(machine.Configuration.States) == 0 {
		t.Fatal("No states in configuration")
	}
	
	if machine.Configuration.States[0].Label != "On" {
		t.Errorf("Expected state 'On', got '%s'", machine.Configuration.States[0].Label)
	}
	
	// Check trace
	trace := processor.GetTrace()
	if len(trace) == 0 {
		t.Error("Expected trace entries, got none")
	}
	
	// Verify event was processed
	found := false
	for _, entry := range trace {
		if entry.Event.Event.Label == "TURN_ON" && entry.Action == "processed" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected to find TURN_ON event in trace")
	}
}

func TestEventProcessor_PriorityHandling(t *testing.T) {
	machine := createTestMachine()
	processor := NewEventProcessor(machine)
	processor.EnableTracing()
	
	processor.Start()
	defer processor.Stop()
	
	// Send events with different priorities
	processor.SendEventWithPriority("LOW_PRIORITY", PriorityLow, nil)
	processor.SendEventWithPriority("HIGH_PRIORITY", PriorityHigh, nil)
	processor.SendEventWithPriority("NORMAL_PRIORITY", PriorityNormal, nil)
	
	// Wait for processing
	time.Sleep(100 * time.Millisecond)
	
	// Check trace for processing order
	trace := processor.GetTrace()
	if len(trace) < 3 {
		t.Fatalf("Expected at least 3 trace entries, got %d", len(trace))
	}
	
	// High priority should be processed first (assuming they're all processed)
	processedEvents := make([]string, 0)
	for _, entry := range trace {
		if entry.Action == "processed" {
			processedEvents = append(processedEvents, entry.Event.Event.Label)
		}
	}
	
	if len(processedEvents) > 0 && processedEvents[0] != "HIGH_PRIORITY" {
		t.Errorf("Expected HIGH_PRIORITY to be processed first, got %s", processedEvents[0])
	}
}

func TestEventProcessor_Filtering(t *testing.T) {
	machine := createTestMachine()
	processor := NewEventProcessor(machine)
	processor.EnableTracing()
	
	// Add a filter that blocks events with "BLOCKED" in the label
	filter := NewConditionalFilter("block_test", func(event ProcessedEvent, machine *sc.Machine) bool {
		return event.Event.Label != "BLOCKED_EVENT"
	})
	processor.AddFilter(filter)
	
	processor.Start()
	defer processor.Stop()
	
	// Send normal and blocked events
	processor.SendEvent("TURN_ON", nil)
	processor.SendEvent("BLOCKED_EVENT", nil)
	
	time.Sleep(100 * time.Millisecond)
	
	// Check trace
	trace := processor.GetTrace()
	
	blockedFound := false
	processedFound := false
	
	for _, entry := range trace {
		if entry.Event.Event.Label == "BLOCKED_EVENT" && entry.Action == "filtered" {
			blockedFound = true
		}
		if entry.Event.Event.Label == "TURN_ON" && entry.Action == "processed" {
			processedFound = true
		}
	}
	
	if !blockedFound {
		t.Error("Expected BLOCKED_EVENT to be filtered")
	}
	if !processedFound {
		t.Error("Expected TURN_ON to be processed")
	}
}

func TestEventProcessor_InternalEvents(t *testing.T) {
	machine := createTestMachine()
	processor := NewEventProcessor(machine)
	processor.EnableTracing()
	
	processor.Start()
	defer processor.Stop()
	
	// Send an event that should trigger internal events
	processor.SendEvent("TURN_ON", nil)
	
	time.Sleep(100 * time.Millisecond)
	
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
}

func TestEventProcessor_ConcurrentAccess(t *testing.T) {
	machine := createTestMachine()
	processor := NewEventProcessor(machine)
	
	processor.Start()
	defer processor.Stop()
	
	// Test concurrent event sending
	var wg sync.WaitGroup
	numGoroutines := 10
	eventsPerGoroutine := 10
	
	wg.Add(numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(goroutineID int) {
			defer wg.Done()
			for j := 0; j < eventsPerGoroutine; j++ {
				err := processor.SendEvent("TEST_EVENT", nil)
				if err != nil {
					t.Errorf("Failed to send event from goroutine %d: %v", goroutineID, err)
				}
			}
		}(i)
	}
	
	wg.Wait()
	
	// Wait for all events to be processed
	time.Sleep(500 * time.Millisecond)
	
	// Verify queue is empty
	if processor.queue.Len() != 0 {
		t.Errorf("Expected empty queue after processing, got %d events", processor.queue.Len())
	}
}

func TestEventProcessor_OrthogonalStates(t *testing.T) {
	machine := createOrthogonalTestMachine()
	processor := NewEventProcessor(machine)
	processor.EnableTracing()
	
	processor.Start()
	defer processor.Stop()
	
	// Send events to different orthogonal regions
	processor.SendEvent("PLAY", nil)
	processor.SendEvent("MUTE", nil)
	
	time.Sleep(200 * time.Millisecond)
	
	// Check that both regions transitioned independently
	states := processor.getCurrentStateLabels()
	
	// Should have states from both orthogonal regions
	if len(states) < 2 {
		t.Errorf("Expected at least 2 active states in orthogonal regions, got %d", len(states))
	}
	
	// Check trace for parallel processing
	trace := processor.GetTrace()
	
	playFound := false
	muteFound := false
	
	for _, entry := range trace {
		if entry.Event.Event.Label == "PLAY" && entry.Action == "processed" {
			playFound = true
		}
		if entry.Event.Event.Label == "MUTE" && entry.Action == "processed" {
			muteFound = true
		}
	}
	
	if !playFound {
		t.Error("Expected PLAY event to be processed")
	}
	if !muteFound {
		t.Error("Expected MUTE event to be processed")
	}
}

func TestConditionalFilter(t *testing.T) {
	filter := NewConditionalFilter("test_filter", func(event ProcessedEvent, machine *sc.Machine) bool {
		return event.Event.Label != "FILTERED"
	})
	
	if filter.Name() != "test_filter" {
		t.Errorf("Expected filter name 'test_filter', got '%s'", filter.Name())
	}
	
	// Test filtering
	allowedEvent := ProcessedEvent{Event: &sc.Event{Label: "ALLOWED"}}
	filteredEvent := ProcessedEvent{Event: &sc.Event{Label: "FILTERED"}}
	
	if !filter.Filter(allowedEvent, nil) {
		t.Error("Expected ALLOWED event to pass filter")
	}
	
	if filter.Filter(filteredEvent, nil) {
		t.Error("Expected FILTERED event to be blocked by filter")
	}
}

func TestEventTypeString(t *testing.T) {
	tests := []struct {
		eventType EventType
		expected  string
	}{
		{EventTypeExternal, "external"},
		{EventTypeInternal, "internal"},
		{EventTypeTimeout, "timeout"},
		{EventTypeEntry, "entry"},
		{EventTypeExit, "exit"},
	}
	
	for _, test := range tests {
		if test.eventType.String() != test.expected {
			t.Errorf("Expected %v.String() to be '%s', got '%s'", test.eventType, test.expected, test.eventType.String())
		}
	}
}

func TestEventPriorityString(t *testing.T) {
	tests := []struct {
		priority EventPriority
		expected string
	}{
		{PriorityLow, "low"},
		{PriorityNormal, "normal"},
		{PriorityHigh, "high"},
		{PriorityCritical, "critical"},
	}
	
	for _, test := range tests {
		if test.priority.String() != test.expected {
			t.Errorf("Expected %v.String() to be '%s', got '%s'", test.priority, test.expected, test.priority.String())
		}
	}
}

// Helper function to create a test machine
func createTestMachine() *sc.Machine {
	return &sc.Machine{
		Id:    "test-machine",
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
					{Label: "Off", Type: sc.StateTypeBasic},
					{Label: "On", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "turn_on",
					From:  []string{"Off"},
					To:    []string{"On"},
					Event: "TURN_ON",
				},
				{
					Label: "turn_off",
					From:  []string{"On"},
					To:    []string{"Off"},
					Event: "TURN_OFF",
				},
			},
			Events: []*sc.Event{
				{Label: "TURN_ON"},
				{Label: "TURN_OFF"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Off"}},
		},
	}
}

// Helper function to create an orthogonal test machine
func createOrthogonalTestMachine() *sc.Machine {
	return &sc.Machine{
		Id:    "orthogonal-test-machine",
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
									{Label: "Paused", Type: sc.StateTypeBasic, IsInitial: true},
									{Label: "Playing", Type: sc.StateTypeBasic},
								},
							},
							{
								Label: "VolumeControl",
								Type:  sc.StateTypeNormal,
								Children: []*sc.State{
									{Label: "Normal", Type: sc.StateTypeBasic, IsInitial: true},
									{Label: "Muted", Type: sc.StateTypeBasic},
								},
							},
						},
					},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "Play",
					From:  []string{"Paused"},
					To:    []string{"Playing"},
					Event: "PLAY",
				},
				{
					Label: "Mute",
					From:  []string{"Normal"},
					To:    []string{"Muted"},
					Event: "MUTE",
				},
			},
			Events: []*sc.Event{
				{Label: "PLAY"},
				{Label: "MUTE"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{
				{Label: "Paused"},
				{Label: "Normal"},
			},
		},
	}
}

func TestEventQueue_ConcurrentAccess(t *testing.T) {
	queue := NewEventQueue()
	
	var producerWg sync.WaitGroup
	var consumerWg sync.WaitGroup
	numProducers := 5
	numConsumers := 3
	eventsPerProducer := 20
	
	// Start consumers
	consumedEvents := make(chan ProcessedEvent, numProducers*eventsPerProducer)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	consumerWg.Add(numConsumers)
	for i := 0; i < numConsumers; i++ {
		go func() {
			defer consumerWg.Done()
			for {
				event, ok := queue.Dequeue(ctx)
				if !ok {
					return
				}
				consumedEvents <- event
			}
		}()
	}
	
	// Start producers
	producerWg.Add(numProducers)
	for i := 0; i < numProducers; i++ {
		go func(producerID int) {
			defer producerWg.Done()
			for j := 0; j < eventsPerProducer; j++ {
				event := ProcessedEvent{
					Event: &sc.Event{Label: fmt.Sprintf("P%d_E%d", producerID, j)},
					Priority: PriorityNormal,
					Timestamp: time.Now(),
					ID: fmt.Sprintf("p%d_e%d", producerID, j),
				}
				queue.Enqueue(event)
			}
		}(i)
	}
	
	// Wait for all producers to finish
	producerWg.Wait()
	
	// Close the queue to signal consumers
	queue.Close()
	
	// Wait for consumers to finish
	consumerWg.Wait()
	
	// Collect consumed events - close after consumers finish
	close(consumedEvents)
	var events []ProcessedEvent
	for event := range consumedEvents {
		events = append(events, event)
	}
	
	// Verify all events were consumed
	expectedCount := numProducers * eventsPerProducer
	if len(events) != expectedCount {
		t.Errorf("Expected %d events to be consumed, got %d", expectedCount, len(events))
	}
}