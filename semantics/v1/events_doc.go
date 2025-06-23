// Package semantics provides comprehensive event processing for statecharts.
//
// # Event Processing System Documentation
//
// This document describes the event processing system implemented for statecharts,
// providing a robust foundation for handling complex event scenarios in hierarchical
// and orthogonal state machines.
//
// ## Architecture Overview
//
// The event processing system consists of several key components:
//
// 1. **EventProcessor**: The main coordinator that manages event processing
// 2. **EventQueue**: A thread-safe priority queue for event ordering
// 3. **ProcessedEvent**: Enhanced event structure with metadata
// 4. **EventFilter**: Pluggable filtering system for conditional processing
// 5. **EventTrace**: Debugging and monitoring capabilities
//
// ## Event Types
//
// The system supports five distinct event types:
//
// - **EventTypeExternal**: Events originating from outside the statechart
// - **EventTypeInternal**: Events generated internally by the statechart
// - **EventTypeTimeout**: Timer-based events
// - **EventTypeEntry**: Events triggered when entering a state
// - **EventTypeExit**: Events triggered when exiting a state
//
// ## Priority System
//
// Events are processed based on priority levels:
//
// - **PriorityCritical**: Highest priority, processed immediately
// - **PriorityHigh**: High priority, processed before normal events
// - **PriorityNormal**: Default priority for most events
// - **PriorityLow**: Lowest priority, processed last
//
// ## Thread Safety
//
// The entire event processing system is designed to be thread-safe:
//
// - EventQueue uses RWMutex for concurrent access
// - EventProcessor manages goroutines safely
// - Event tracing is protected with mutex
// - Event filters are thread-safe
//
// ## Usage Examples
//
// ### Basic Event Processing
//
//	machine := createYourMachine()
//	processor := NewEventProcessor(machine)
//	processor.Start()
//	defer processor.Stop()
//	
//	// Send external event
//	processor.SendEvent("USER_INPUT", nil)
//	
//	// Send high priority event
//	processor.SendEventWithPriority("CRITICAL_ERROR", PriorityCritical, nil)
//
// ### Event Tracing
//
//	processor.EnableTracing()
//	
//	// Process some events...
//	
//	trace := processor.GetTrace()
//	for _, entry := range trace {
//		fmt.Printf("Event: %s, Action: %s, Error: %v\n", 
//			entry.Event.Event.Label, entry.Action, entry.Error)
//	}
//
// ### Event Filtering
//
//	// Block events during maintenance mode
//	maintenanceFilter := NewConditionalFilter("maintenance", 
//		func(event ProcessedEvent, machine *sc.Machine) bool {
//			return !isMaintenanceMode(machine)
//		})
//	processor.AddFilter(maintenanceFilter)
//
// ## Event Lifecycle
//
// Events follow a well-defined lifecycle:
//
// 1. **Creation**: Event is created and enqueued
// 2. **Filtering**: Event passes through registered filters
// 3. **Processing**: Event is processed according to its type
// 4. **Transition**: State transitions are executed if applicable
// 5. **Internal Events**: Entry/exit events are generated as needed
// 6. **Tracing**: Event processing is recorded if tracing is enabled
//
// ## Hierarchical Event Processing
//
// The system supports hierarchical statecharts by:
//
// - Processing events at the appropriate level in the state hierarchy
// - Generating internal events for state entry and exit
// - Handling event propagation through the state tree
// - Supporting event broadcasting to multiple states
//
// ## Orthogonal State Support
//
// For orthogonal (parallel) states, the system:
//
// - Processes events independently in each orthogonal region
// - Maintains separate state configurations for each region
// - Ensures proper synchronization between parallel regions
// - Handles complex event scenarios with multiple active states
//
// ## Error Handling
//
// The event processing system provides comprehensive error handling:
//
// - Graceful degradation when events cannot be processed
// - Error reporting through the tracing system
// - Continuation of processing even when individual events fail
// - Proper cleanup and resource management
//
// ## Performance Considerations
//
// The system is designed for high performance:
//
// - Efficient priority queue implementation
// - Minimal memory allocation during processing
// - Optimized state lookup and transition execution
// - Configurable tracing to reduce overhead when not needed
//
// ## Extensibility
//
// The system is designed to be extensible:
//
// - Pluggable event filters for custom logic
// - Extensible event types for domain-specific events
// - Configurable priority levels
// - Customizable tracing and debugging
//
// ## Best Practices
//
// When using the event processing system:
//
// 1. Always call Stop() to properly clean up resources
// 2. Use appropriate priority levels to ensure correct ordering
// 3. Enable tracing during development and debugging
// 4. Use filters judiciously to avoid performance impact
// 5. Handle errors appropriately in your event handlers
// 6. Test thoroughly with concurrent event scenarios
//
// ## Integration with Existing Code
//
// The event processing system is designed to integrate with existing
// statechart implementations:
//
// - Compatible with existing Machine and Statechart structures
// - Preserves existing event handling behavior
// - Provides enhanced functionality as an opt-in feature
// - Maintains backward compatibility
//
// This event processing system provides a solid foundation for building
// complex statechart-based applications with reliable event handling,
// proper concurrency support, and comprehensive debugging capabilities.
package semantics