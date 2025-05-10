// Package examples provides academic examples of statechart implementations.
// This file demonstrates hierarchical statechart composition (OR-states).
package examples

import (
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// HierarchicalStatechart creates a statechart that demonstrates hierarchical state composition.
// It models a simple alarm system with nested states:
//   - Off (initial)
//   - On
//   - Idle (initial)
//   - Armed
//   - Monitoring (initial)
//   - Triggered
//
// The example demonstrates:
// 1. State hierarchy with OR-state composition
// 2. Default/initial state selection at each level
// 3. Transitions between states at different hierarchical levels
//
// This follows Harel's original formulation where hierarchical states encapsulate
// behavioral refinements.
func HierarchicalStatechart() *semantics.Statechart {
	// Root contains two top-level states: Off and On
	return semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "AlarmSystem",
			Children: []*sc.State{
				{
					Label:     "Off",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "On",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "Idle",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "Armed",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "Monitoring",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "Triggered",
									Type:  sc.StateTypeBasic,
								},
							},
						},
					},
				},
			},
		},
		// Transitions are defined separately from the state hierarchy
		Transitions: []*sc.Transition{
			{
				Label: "PowerOn",
				From:  []string{"Off"},
				To:    []string{"On"},
				Event: "POWER_ON",
			},
			{
				Label: "PowerOff",
				From:  []string{"On"},
				To:    []string{"Off"},
				Event: "POWER_OFF",
			},
			{
				Label: "Arm",
				From:  []string{"Idle"},
				To:    []string{"Armed"},
				Event: "ARM",
			},
			{
				Label: "Disarm",
				From:  []string{"Armed"},
				To:    []string{"Idle"},
				Event: "DISARM",
			},
			{
				Label: "Trigger",
				From:  []string{"Monitoring"},
				To:    []string{"Triggered"},
				Event: "MOTION_DETECTED",
			},
			{
				Label: "Reset",
				From:  []string{"Triggered"},
				To:    []string{"Monitoring"},
				Event: "RESET",
			},
		},
		// Define the events in the statechart alphabet
		Events: []*sc.Event{
			{Label: "POWER_ON"},
			{Label: "POWER_OFF"},
			{Label: "ARM"},
			{Label: "DISARM"},
			{Label: "MOTION_DETECTED"},
			{Label: "RESET"},
		},
	})
}
