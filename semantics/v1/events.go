// Package semantics provides event processing functionality for statecharts.
// This file implements a comprehensive event system including queuing, broadcasting,
// internal event generation, priority handling, and thread-safe operations.
package semantics

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
	"google.golang.org/protobuf/types/known/structpb"
)

// EventType represents the type of event
type EventType int

const (
	// EventTypeExternal represents events originating from outside the statechart
	EventTypeExternal EventType = iota
	// EventTypeInternal represents events generated internally by the statechart
	EventTypeInternal
	// EventTypeTimeout represents timeout events
	EventTypeTimeout
	// EventTypeEntry represents entry action events
	EventTypeEntry
	// EventTypeExit represents exit action events
	EventTypeExit
)

// EventPriority represents the priority level of an event
type EventPriority int

const (
	// PriorityLow for low priority events
	PriorityLow EventPriority = iota
	// PriorityNormal for normal priority events (default)
	PriorityNormal
	// PriorityHigh for high priority events
	PriorityHigh
	// PriorityCritical for critical system events
	PriorityCritical
)

// ProcessedEvent represents an event with additional processing metadata
type ProcessedEvent struct {
	// Event is the original event
	Event *sc.Event
	// Type is the event type
	Type EventType
	// Priority is the event priority
	Priority EventPriority
	// Timestamp when the event was created
	Timestamp time.Time
	// Data contains additional event data
	Data *structpb.Struct
	// Source identifies the source of the event
	Source string
	// TargetStates are the specific states this event targets (optional)
	TargetStates []string
	// ID is a unique identifier for this event instance
	ID string
}

// EventProcessor manages event processing for a statechart machine
type EventProcessor struct {
	// machine is the statechart machine this processor handles
	machine *sc.Machine
	// queue is the priority event queue
	queue *EventQueue
	// tracing enables event tracing
	tracing bool
	// trace stores event trace information
	trace []EventTrace
	// traceMutex protects trace operations
	traceMutex sync.RWMutex
	// filters are event filters
	filters []EventFilter
	// filterMutex protects filter operations
	filterMutex sync.RWMutex
	// ctx is the context for cancellation
	ctx context.Context
	// cancel cancels the processor
	cancel context.CancelFunc
	// wg tracks running goroutines
	wg sync.WaitGroup
}

// EventQueue implements a thread-safe priority queue for events
type EventQueue struct {
	events []ProcessedEvent
	mutex  sync.RWMutex
	cond   *sync.Cond
	closed bool
}

// EventTrace represents trace information for event processing
type EventTrace struct {
	Event       ProcessedEvent
	Timestamp   time.Time
	Action      string
	FromStates  []string
	ToStates    []string
	Transitions []string
	Error       error
}

// EventFilter represents a filter for events
type EventFilter interface {
	// Filter returns true if the event should be processed
	Filter(event ProcessedEvent, machine *sc.Machine) bool
	// Name returns the name of the filter
	Name() string
}

// ConditionalFilter implements a conditional event filter
type ConditionalFilter struct {
	name      string
	condition func(ProcessedEvent, *sc.Machine) bool
}

// Filter implements EventFilter
func (f *ConditionalFilter) Filter(event ProcessedEvent, machine *sc.Machine) bool {
	return f.condition(event, machine)
}

// Name implements EventFilter
func (f *ConditionalFilter) Name() string {
	return f.name
}

// NewEventProcessor creates a new event processor for a machine
func NewEventProcessor(machine *sc.Machine) *EventProcessor {
	ctx, cancel := context.WithCancel(context.Background())
	queue := NewEventQueue()
	
	return &EventProcessor{
		machine: machine,
		queue:   queue,
		ctx:     ctx,
		cancel:  cancel,
		trace:   make([]EventTrace, 0),
		filters: make([]EventFilter, 0),
	}
}

// NewEventQueue creates a new event queue
func NewEventQueue() *EventQueue {
	eq := &EventQueue{
		events: make([]ProcessedEvent, 0),
	}
	eq.cond = sync.NewCond(&eq.mutex)
	return eq
}

// Start starts the event processor
func (ep *EventProcessor) Start() {
	ep.wg.Add(1)
	go ep.processLoop()
}

// Stop stops the event processor and waits for completion
func (ep *EventProcessor) Stop() {
	ep.cancel()
	ep.queue.Close()
	ep.wg.Wait()
}

// EnableTracing enables event tracing
func (ep *EventProcessor) EnableTracing() {
	ep.traceMutex.Lock()
	defer ep.traceMutex.Unlock()
	ep.tracing = true
}

// DisableTracing disables event tracing
func (ep *EventProcessor) DisableTracing() {
	ep.traceMutex.Lock()
	defer ep.traceMutex.Unlock()
	ep.tracing = false
}

// GetTrace returns a copy of the event trace
func (ep *EventProcessor) GetTrace() []EventTrace {
	ep.traceMutex.RLock()
	defer ep.traceMutex.RUnlock()
	traceCopy := make([]EventTrace, len(ep.trace))
	copy(traceCopy, ep.trace)
	return traceCopy
}

// ClearTrace clears the event trace
func (ep *EventProcessor) ClearTrace() {
	ep.traceMutex.Lock()
	defer ep.traceMutex.Unlock()
	ep.trace = ep.trace[:0]
}

// AddFilter adds an event filter
func (ep *EventProcessor) AddFilter(filter EventFilter) {
	ep.filterMutex.Lock()
	defer ep.filterMutex.Unlock()
	ep.filters = append(ep.filters, filter)
}

// RemoveFilter removes an event filter by name
func (ep *EventProcessor) RemoveFilter(name string) {
	ep.filterMutex.Lock()
	defer ep.filterMutex.Unlock()
	for i, filter := range ep.filters {
		if filter.Name() == name {
			ep.filters = append(ep.filters[:i], ep.filters[i+1:]...)
			break
		}
	}
}

// SendEvent sends an external event to the processor
func (ep *EventProcessor) SendEvent(eventLabel string, data *structpb.Struct) error {
	event := &ProcessedEvent{
		Event: &sc.Event{Label: eventLabel},
		Type:  EventTypeExternal,
		Priority: PriorityNormal,
		Timestamp: time.Now(),
		Data: data,
		Source: "external",
		ID: fmt.Sprintf("ext_%d_%s", time.Now().UnixNano(), eventLabel),
	}
	
	return ep.queue.Enqueue(*event)
}

// SendEventWithPriority sends an event with specified priority
func (ep *EventProcessor) SendEventWithPriority(eventLabel string, priority EventPriority, data *structpb.Struct) error {
	event := &ProcessedEvent{
		Event: &sc.Event{Label: eventLabel},
		Type:  EventTypeExternal,
		Priority: priority,
		Timestamp: time.Now(),
		Data: data,
		Source: "external",
		ID: fmt.Sprintf("ext_%d_%s", time.Now().UnixNano(), eventLabel),
	}
	
	return ep.queue.Enqueue(*event)
}

// generateInternalEvent creates internal events for state entry/exit
func (ep *EventProcessor) generateInternalEvent(eventType EventType, stateName string, data *structpb.Struct) {
	var eventLabel string
	switch eventType {
	case EventTypeEntry:
		eventLabel = fmt.Sprintf("entry_%s", stateName)
	case EventTypeExit:
		eventLabel = fmt.Sprintf("exit_%s", stateName)
	default:
		eventLabel = fmt.Sprintf("internal_%s", stateName)
	}
	
	event := ProcessedEvent{
		Event: &sc.Event{Label: eventLabel},
		Type:  eventType,
		Priority: PriorityHigh,
		Timestamp: time.Now(),
		Data: data,
		Source: "internal",
		TargetStates: []string{stateName},
		ID: fmt.Sprintf("int_%d_%s", time.Now().UnixNano(), eventLabel),
	}
	
	// Internal events bypass filtering
	ep.queue.EnqueueInternal(event)
}

// processLoop is the main event processing loop
func (ep *EventProcessor) processLoop() {
	defer ep.wg.Done()
	
	for {
		select {
		case <-ep.ctx.Done():
			return
		default:
			event, ok := ep.queue.Dequeue(ep.ctx)
			if !ok {
				return // Queue is closed
			}
			
			ep.processEvent(event)
		}
	}
}

// processEvent processes a single event
func (ep *EventProcessor) processEvent(event ProcessedEvent) {
	// Apply filters
	if !ep.applyFilters(event) {
		ep.addTrace(event, "filtered", nil, nil, nil, nil)
		return
	}
	
	// Record initial state
	initialStates := ep.getCurrentStateLabels()
	
	// Process the event based on type
	var transitions []string
	var finalStates []string
	var err error
	
	switch event.Type {
	case EventTypeExternal:
		transitions, finalStates, err = ep.handleExternalEvent(event)
	case EventTypeInternal, EventTypeEntry, EventTypeExit:
		transitions, finalStates, err = ep.handleInternalEvent(event)
	case EventTypeTimeout:
		transitions, finalStates, err = ep.handleTimeoutEvent(event)
	default:
		err = fmt.Errorf("unknown event type: %v", event.Type)
	}
	
	// Add trace
	ep.addTrace(event, "processed", initialStates, finalStates, transitions, err)
}

// handleExternalEvent processes external events
func (ep *EventProcessor) handleExternalEvent(event ProcessedEvent) ([]string, []string, error) {
	currentStates := ep.getCurrentStateLabels()
	executedTransitions := make([]string, 0)
	
	// Find enabled transitions for this event
	enabledTransitions := ep.findEnabledTransitions(event.Event.Label, currentStates)
	
	if len(enabledTransitions) == 0 {
		return nil, currentStates, nil // No transition, stay in current state
	}
	
	// Execute transitions based on priority and consistency
	for _, transition := range enabledTransitions {
		if ep.canExecuteTransition(transition, currentStates) {
			err := ep.executeTransition(transition, event)
			if err != nil {
				return executedTransitions, ep.getCurrentStateLabels(), err
			}
			executedTransitions = append(executedTransitions, transition.Label)
			
			// Update current states for next iteration
			currentStates = ep.getCurrentStateLabels()
		}
	}
	
	return executedTransitions, ep.getCurrentStateLabels(), nil
}

// handleInternalEvent processes internal events
func (ep *EventProcessor) handleInternalEvent(event ProcessedEvent) ([]string, []string, error) {
	// Internal events are processed differently based on their type
	switch event.Type {
	case EventTypeEntry:
		// Handle entry actions
		return ep.handleEntryEvent(event)
	case EventTypeExit:
		// Handle exit actions
		return ep.handleExitEvent(event)
	default:
		// General internal events
		return ep.handleExternalEvent(event) // Reuse external event logic
	}
}

// handleTimeoutEvent processes timeout events
func (ep *EventProcessor) handleTimeoutEvent(event ProcessedEvent) ([]string, []string, error) {
	// Timeout events are treated like external events but with special handling
	return ep.handleExternalEvent(event)
}

// handleEntryEvent processes entry events
func (ep *EventProcessor) handleEntryEvent(event ProcessedEvent) ([]string, []string, error) {
	currentStates := ep.getCurrentStateLabels()
	// Entry events don't cause transitions, they execute entry actions
	// For now, we just return the current states
	return nil, currentStates, nil
}

// handleExitEvent processes exit events
func (ep *EventProcessor) handleExitEvent(event ProcessedEvent) ([]string, []string, error) {
	currentStates := ep.getCurrentStateLabels()
	// Exit events don't cause transitions, they execute exit actions
	// For now, we just return the current states
	return nil, currentStates, nil
}

// findEnabledTransitions finds all transitions enabled by the given event
func (ep *EventProcessor) findEnabledTransitions(eventLabel string, currentStates []string) []*sc.Transition {
	var enabled []*sc.Transition
	
	for _, transition := range ep.machine.Statechart.Transitions {
		if transition.Event == eventLabel {
			// Check if any of the transition's source states are currently active
			for _, fromState := range transition.From {
				if slices.Contains(currentStates, fromState) {
					enabled = append(enabled, transition)
					break
				}
			}
		}
	}
	
	// Sort by priority (first defined transition has higher priority)
	sort.Slice(enabled, func(i, j int) bool {
		// Find indices in the original transitions slice
		iIndex := ep.findTransitionIndex(enabled[i])
		jIndex := ep.findTransitionIndex(enabled[j])
		return iIndex < jIndex
	})
	
	return enabled
}

// findTransitionIndex finds the index of a transition in the statechart
func (ep *EventProcessor) findTransitionIndex(transition *sc.Transition) int {
	for i, t := range ep.machine.Statechart.Transitions {
		if t == transition {
			return i
		}
	}
	return -1
}

// canExecuteTransition checks if a transition can be executed
func (ep *EventProcessor) canExecuteTransition(transition *sc.Transition, currentStates []string) bool {
	// Check if source state is active
	sourceActive := false
	for _, fromState := range transition.From {
		if slices.Contains(currentStates, fromState) {
			sourceActive = true
			break
		}
	}
	
	if !sourceActive {
		return false
	}
	
	// Check guard condition if present
	if transition.Guard != nil && transition.Guard.Expression != "" {
		// For now, we'll assume all guards are true
		// In a full implementation, this would evaluate the guard expression
		return true
	}
	
	return true
}

// executeTransition executes a transition
func (ep *EventProcessor) executeTransition(transition *sc.Transition, event ProcessedEvent) error {
	if ep.machine.Configuration == nil || len(ep.machine.Configuration.States) == 0 {
		return fmt.Errorf("invalid machine configuration")
	}
	
	// Find the current state that matches the transition source
	var sourceStateIndex = -1
	for i, stateRef := range ep.machine.Configuration.States {
		for _, fromState := range transition.From {
			if stateRef.Label == fromState {
				sourceStateIndex = i
				break
			}
		}
		if sourceStateIndex >= 0 {
			break
		}
	}
	
	if sourceStateIndex < 0 {
		return fmt.Errorf("source state not found in configuration")
	}
	
	// Generate exit events for the source state
	sourceState := ep.machine.Configuration.States[sourceStateIndex].Label
	ep.generateInternalEvent(EventTypeExit, sourceState, event.Data)
	
	// Update the machine configuration
	if len(transition.To) > 0 {
		ep.machine.Configuration.States[sourceStateIndex].Label = transition.To[0]
		
		// Generate entry events for the target state
		targetState := transition.To[0]
		ep.generateInternalEvent(EventTypeEntry, targetState, event.Data)
	}
	
	// Execute transition actions
	for _, action := range transition.Actions {
		err := ep.executeAction(action, event)
		if err != nil {
			return fmt.Errorf("failed to execute action %s: %w", action.Label, err)
		}
	}
	
	// Update context if present
	if ep.machine.Context != nil && ep.machine.Context.Fields != nil {
		if countValue, exists := ep.machine.Context.Fields["count"]; exists {
			if count, ok := countValue.GetKind().(*structpb.Value_NumberValue); ok {
				newCount := structpb.NewNumberValue(count.NumberValue + 1)
				ep.machine.Context.Fields["count"] = newCount
			}
		}
	}
	
	return nil
}

// executeAction executes a transition action
func (ep *EventProcessor) executeAction(action *sc.Action, event ProcessedEvent) error {
	// For now, actions are just logged
	// In a full implementation, this would execute the actual action logic
	return nil
}

// getCurrentStateLabels returns the labels of currently active states
func (ep *EventProcessor) getCurrentStateLabels() []string {
	if ep.machine.Configuration == nil {
		return nil
	}
	
	labels := make([]string, len(ep.machine.Configuration.States))
	for i, state := range ep.machine.Configuration.States {
		labels[i] = state.Label
	}
	return labels
}

// applyFilters applies all registered filters to an event
func (ep *EventProcessor) applyFilters(event ProcessedEvent) bool {
	ep.filterMutex.RLock()
	defer ep.filterMutex.RUnlock()
	
	for _, filter := range ep.filters {
		if !filter.Filter(event, ep.machine) {
			return false
		}
	}
	return true
}

// addTrace adds a trace entry if tracing is enabled
func (ep *EventProcessor) addTrace(event ProcessedEvent, action string, fromStates, toStates, transitions []string, err error) {
	if !ep.tracing {
		return
	}
	
	ep.traceMutex.Lock()
	defer ep.traceMutex.Unlock()
	
	trace := EventTrace{
		Event:       event,
		Timestamp:   time.Now(),
		Action:      action,
		FromStates:  fromStates,
		ToStates:    toStates,
		Transitions: transitions,
		Error:       err,
	}
	
	ep.trace = append(ep.trace, trace)
}

// Enqueue adds an event to the queue
func (eq *EventQueue) Enqueue(event ProcessedEvent) error {
	eq.mutex.Lock()
	defer eq.mutex.Unlock()
	
	if eq.closed {
		return fmt.Errorf("queue is closed")
	}
	
	eq.events = append(eq.events, event)
	eq.sortEvents()
	eq.cond.Signal()
	return nil
}

// EnqueueInternal adds an internal event to the queue (bypasses normal queuing rules)
func (eq *EventQueue) EnqueueInternal(event ProcessedEvent) error {
	eq.mutex.Lock()
	defer eq.mutex.Unlock()
	
	if eq.closed {
		return fmt.Errorf("queue is closed")
	}
	
	// Internal events go to the front based on priority
	eq.events = append(eq.events, event)
	eq.sortEvents()
	eq.cond.Signal()
	return nil
}

// Dequeue removes and returns the highest priority event from the queue
func (eq *EventQueue) Dequeue(ctx context.Context) (ProcessedEvent, bool) {
	eq.mutex.Lock()
	defer eq.mutex.Unlock()
	
	for len(eq.events) == 0 && !eq.closed {
		// Check if context is done before waiting
		select {
		case <-ctx.Done():
			return ProcessedEvent{}, false
		default:
		}
		
		// Wait for an event to be available
		eq.cond.Wait()
		
		// Check context again after waking up
		select {
		case <-ctx.Done():
			return ProcessedEvent{}, false
		default:
		}
	}
	
	if len(eq.events) == 0 || eq.closed {
		return ProcessedEvent{}, false
	}
	
	event := eq.events[0]
	eq.events = eq.events[1:]
	return event, true
}

// Close closes the event queue
func (eq *EventQueue) Close() {
	eq.mutex.Lock()
	defer eq.mutex.Unlock()
	eq.closed = true
	eq.cond.Broadcast()
}

// Len returns the number of events in the queue
func (eq *EventQueue) Len() int {
	eq.mutex.RLock()
	defer eq.mutex.RUnlock()
	return len(eq.events)
}

// sortEvents sorts events by priority and timestamp
func (eq *EventQueue) sortEvents() {
	sort.Slice(eq.events, func(i, j int) bool {
		// Higher priority first
		if eq.events[i].Priority != eq.events[j].Priority {
			return eq.events[i].Priority > eq.events[j].Priority
		}
		// Earlier timestamp first for same priority
		return eq.events[i].Timestamp.Before(eq.events[j].Timestamp)
	})
}

// NewConditionalFilter creates a new conditional filter
func NewConditionalFilter(name string, condition func(ProcessedEvent, *sc.Machine) bool) EventFilter {
	return &ConditionalFilter{
		name:      name,
		condition: condition,
	}
}

// String returns a string representation of EventType
func (et EventType) String() string {
	switch et {
	case EventTypeExternal:
		return "external"
	case EventTypeInternal:
		return "internal"
	case EventTypeTimeout:
		return "timeout"
	case EventTypeEntry:
		return "entry"
	case EventTypeExit:
		return "exit"
	default:
		return "unknown"
	}
}

// String returns a string representation of EventPriority
func (ep EventPriority) String() string {
	switch ep {
	case PriorityLow:
		return "low"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}