// Package main generates ground truth traces from statechart execution.
// These traces can be used to validate that ML predictions match Go semantics.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/sc"
	semantics "github.com/tmc/sc/semantics/v1"
)

// TraceStep represents a single step in a trace.
type TraceStep struct {
	StateBefore     []string `json:"state_before"`
	Event           string   `json:"event"`
	StateAfter      []string `json:"state_after"`
	GuardsEvaluated []string `json:"guards_evaluated,omitempty"`
}

// Trace represents a complete execution trace.
type Trace struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Statechart  string      `json:"statechart"`
	Steps       []TraceStep `json:"steps"`
}

// configToLabels extracts state labels from a configuration.
func configToLabels(config *sc.Configuration) []string {
	if config == nil {
		return []string{}
	}
	labels := make([]string, 0, len(config.States))
	for _, state := range config.States {
		if state != nil {
			labels = append(labels, state.Label)
		}
	}
	return labels
}

// generateToggleTrace creates a trace for a simple 2-state toggle.
func generateToggleTrace() (*Trace, error) {
	// Create toggle statechart
	chart := semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "Toggle",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "Off", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "On", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "turn_on",
				From:  []string{"Off"},
				To:    []string{"On"},
				Event: "TOGGLE",
			},
			{
				Label: "turn_off",
				From:  []string{"On"},
				To:    []string{"Off"},
				Event: "TOGGLE",
			},
		},
		Events: []*sc.Event{
			{Label: "TOGGLE"},
		},
	})

	// Create and start machine
	machine, err := semantics.NewMachine(chart, "toggle-1", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create machine: %w", err)
	}
	if err := machine.Start(); err != nil {
		return nil, fmt.Errorf("failed to start machine: %w", err)
	}

	trace := &Trace{
		Name:        "two_state_toggle",
		Description: "Simple toggle between Off and On states",
		Statechart:  "Toggle: Off <--TOGGLE--> On",
		Steps:       []TraceStep{},
	}

	// Generate events
	events := []string{"TOGGLE", "TOGGLE", "TOGGLE", "TOGGLE"}

	for _, eventLabel := range events {
		stateBefore := configToLabels(machine.Configuration)

		fired, err := machine.Step(eventLabel)
		if err != nil {
			return nil, fmt.Errorf("step failed: %w", err)
		}

		stateAfter := configToLabels(machine.Configuration)

		trace.Steps = append(trace.Steps, TraceStep{
			StateBefore:     stateBefore,
			Event:           eventLabel,
			StateAfter:      stateAfter,
			GuardsEvaluated: []string{fmt.Sprintf("fired=%v", fired)},
		})
	}

	return trace, nil
}

// generateHierarchyTrace creates a trace for a 3-level hierarchical statechart.
func generateHierarchyTrace() (*Trace, error) {
	// Create hierarchical statechart
	// Root (OR)
	// ├── A (OR)
	// │   ├── A1 (BASIC, initial)
	// │   └── A2 (BASIC)
	// └── B (OR)
	//     ├── B1 (BASIC, initial)
	//     └── B2 (BASIC)
	chart := semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "A",
					Type:      sc.StateTypeNormal,
					IsInitial: true,
					Children: []*sc.State{
						{Label: "A1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "A2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "B1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			// Within A
			{
				Label: "a1_to_a2",
				From:  []string{"A1"},
				To:    []string{"A2"},
				Event: "NEXT",
			},
			{
				Label: "a2_to_a1",
				From:  []string{"A2"},
				To:    []string{"A1"},
				Event: "PREV",
			},
			// Cross-hierarchy
			{
				Label: "a_to_b",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "SWITCH",
			},
			{
				Label: "b_to_a",
				From:  []string{"B"},
				To:    []string{"A"},
				Event: "SWITCH",
			},
			// Within B
			{
				Label: "b1_to_b2",
				From:  []string{"B1"},
				To:    []string{"B2"},
				Event: "NEXT",
			},
			{
				Label: "b2_to_b1",
				From:  []string{"B2"},
				To:    []string{"B1"},
				Event: "PREV",
			},
		},
		Events: []*sc.Event{
			{Label: "NEXT"},
			{Label: "PREV"},
			{Label: "SWITCH"},
		},
	})

	// Create and start machine
	machine, err := semantics.NewMachine(chart, "hierarchy-1", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create machine: %w", err)
	}
	if err := machine.Start(); err != nil {
		return nil, fmt.Errorf("failed to start machine: %w", err)
	}

	trace := &Trace{
		Name:        "three_level_hierarchy",
		Description: "Hierarchical states with nested transitions",
		Statechart:  "Root -> A(A1,A2), B(B1,B2)",
		Steps:       []TraceStep{},
	}

	// Generate event sequence
	events := []string{
		"NEXT",   // A1 -> A2
		"SWITCH", // A -> B (enters B1)
		"NEXT",   // B1 -> B2
		"SWITCH", // B -> A (enters A1)
		"PREV",   // A1 -> A1 (no change)
		"NEXT",   // A1 -> A2
		"NEXT",   // A2 -> A2 (no change)
	}

	for _, eventLabel := range events {
		stateBefore := configToLabels(machine.Configuration)

		fired, err := machine.Step(eventLabel)
		if err != nil {
			// Record the step even if error
			trace.Steps = append(trace.Steps, TraceStep{
				StateBefore:     stateBefore,
				Event:           eventLabel,
				StateAfter:      stateBefore,
				GuardsEvaluated: []string{fmt.Sprintf("error: %v", err)},
			})
			continue
		}

		stateAfter := configToLabels(machine.Configuration)

		trace.Steps = append(trace.Steps, TraceStep{
			StateBefore:     stateBefore,
			Event:           eventLabel,
			StateAfter:      stateAfter,
			GuardsEvaluated: []string{fmt.Sprintf("fired=%v", fired)},
		})
	}

	return trace, nil
}

func main() {
	fmt.Println("Generating ground truth traces...")

	outputDir := filepath.Join("testdata", "traces")

	// Generate toggle trace
	toggleTrace, err := generateToggleTrace()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to generate toggle trace: %v\n", err)
		os.Exit(1)
	}

	togglePath := filepath.Join(outputDir, "toggle_trace.json")
	toggleData, _ := json.MarshalIndent(toggleTrace, "", "  ")
	if err := os.WriteFile(togglePath, toggleData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write toggle trace: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Generated: %s (%d steps)\n", togglePath, len(toggleTrace.Steps))

	// Generate hierarchy trace
	hierarchyTrace, err := generateHierarchyTrace()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to generate hierarchy trace: %v\n", err)
		os.Exit(1)
	}

	hierarchyPath := filepath.Join(outputDir, "hierarchy_trace.json")
	hierarchyData, _ := json.MarshalIndent(hierarchyTrace, "", "  ")
	if err := os.WriteFile(hierarchyPath, hierarchyData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write hierarchy trace: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Generated: %s (%d steps)\n", hierarchyPath, len(hierarchyTrace.Steps))

	fmt.Println("\nDone! Traces can be used to validate ML predictions.")
}
