package semantics

import (
	"fmt"

	"github.com/tmc/sc"
)

// HandleEvent processes an event for a machine using the complete transition execution semantics.
// It returns true if any transitions were executed, false otherwise.
func HandleEvent(machine *sc.Machine, event string) (bool, error) {
	if machine == nil || machine.Statechart == nil || machine.Configuration == nil {
		return false, fmt.Errorf("machine, statechart, and configuration cannot be nil")
	}

	// Create a Statechart wrapper for semantic operations
	statechart := NewStatechart(machine.Statechart)

	// Execute transitions for the event
	result, err := statechart.ExecuteTransitions(machine.Configuration, machine.Context, event)
	if err != nil {
		return false, fmt.Errorf("failed to execute transitions: %w", err)
	}

	if result.Executed {
		// Update machine state
		machine.Configuration = result.NewConfig
		machine.Context = result.NewContext

		// Record the step in machine history
		if len(result.Steps) > 0 {
			step := result.Steps[0] // For now, we only handle single steps
			machineStep := &sc.Step{
				Events: []*sc.Event{{Label: event}},
				Transitions: step.Transitions,
				StartingConfiguration: step.SourceConfiguration,
				ResultingConfiguration: step.TargetConfiguration,
				Context: result.NewContext,
			}
			machine.StepHistory = append(machine.StepHistory, machineStep)
		}
	}

	return result.Executed, nil
}

// ProcessEvents processes a sequence of events for a machine.
func ProcessEvents(machine *sc.Machine, events []string) ([]bool, error) {
	results := make([]bool, len(events))
	
	for i, event := range events {
		handled, err := HandleEvent(machine, event)
		if err != nil {
			return results, fmt.Errorf("failed to handle event %s at index %d: %w", event, i, err)
		}
		results[i] = handled
	}
	
	return results, nil
}

// GetActiveEvents returns all events that have enabled transitions in the current configuration.
func GetActiveEvents(machine *sc.Machine) ([]string, error) {
	if machine == nil || machine.Statechart == nil || machine.Configuration == nil {
		return nil, fmt.Errorf("machine, statechart, and configuration cannot be nil")
	}

	statechart := NewStatechart(machine.Statechart)
	var activeEvents []string
	eventMap := make(map[string]bool)

	// Check all events in the statechart
	for _, event := range machine.Statechart.Events {
		if eventMap[event.Label] {
			continue // Already processed
		}

		enabled, err := statechart.FindEnabledTransitions(machine.Configuration, machine.Context, event.Label)
		if err != nil {
			return nil, fmt.Errorf("failed to find enabled transitions for event %s: %w", event.Label, err)
		}

		if len(enabled) > 0 {
			activeEvents = append(activeEvents, event.Label)
			eventMap[event.Label] = true
		}
	}

	return activeEvents, nil
}
