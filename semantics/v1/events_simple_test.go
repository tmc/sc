package semantics

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/sc"
)

func TestSimpleEventQueue(t *testing.T) {
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

func TestEventPriorityQueue(t *testing.T) {
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

