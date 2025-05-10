// Package examples provides academic examples of statechart implementations.
// This file demonstrates history states in statecharts.
package examples

import (
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// HistoryStatechart creates a statechart that demonstrates history semantics.
// It models a text editor with history mechanism:
//   - TextEditor
//   - Inactive (initial)
//   - Active
//   - Editing (initial)
//   - Searching
//   - Formatting
//   - Settings
//   - General (initial)
//   - Display
//   - Advanced
//
// The example demonstrates:
// 1. Deep history mechanism to remember previous states
// 2. Transitions to history pseudostates
// 3. Default history behavior
//
// This follows the history mechanism described in Harel's statecharts, allowing
// a system to "remember" previously active states.
func HistoryStatechart() *semantics.Statechart {
	return semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "TextEditor",
			Children: []*sc.State{
				{
					Label:     "Inactive",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "Active",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "Editing",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "Searching",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "Formatting",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Settings",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "General",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "Display",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "Advanced",
							Type:  sc.StateTypeBasic,
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			// Basic transitions
			{
				Label: "Open",
				From:  []string{"Inactive"},
				To:    []string{"Active"},
				Event: "OPEN",
			},
			{
				Label: "Close",
				From:  []string{"Active"},
				To:    []string{"Inactive"},
				Event: "CLOSE",
			},
			{
				Label: "OpenSettings",
				From:  []string{"Active"},
				To:    []string{"Settings"},
				Event: "SETTINGS",
			},
			{
				Label: "CloseSettings",
				From:  []string{"Settings"},
				To:    []string{"Active"},
				Event: "BACK",
			},

			// Transitions within Active state
			{
				Label: "StartSearch",
				From:  []string{"Editing"},
				To:    []string{"Searching"},
				Event: "SEARCH",
			},
			{
				Label: "EndSearch",
				From:  []string{"Searching"},
				To:    []string{"Editing"},
				Event: "CANCEL",
			},
			{
				Label: "StartFormat",
				From:  []string{"Editing"},
				To:    []string{"Formatting"},
				Event: "FORMAT",
			},
			{
				Label: "EndFormat",
				From:  []string{"Formatting"},
				To:    []string{"Editing"},
				Event: "DONE",
			},

			// Transitions within Settings state
			{
				Label: "GoToDisplay",
				From:  []string{"General"},
				To:    []string{"Display"},
				Event: "DISPLAY",
			},
			{
				Label: "GoToAdvanced",
				From:  []string{"General"},
				To:    []string{"Advanced"},
				Event: "ADVANCED",
			},
			{
				Label: "BackToGeneral",
				From:  []string{"Display", "Advanced"},
				To:    []string{"General"},
				Event: "GENERAL",
			},
		},
		Events: []*sc.Event{
			{Label: "OPEN"},
			{Label: "CLOSE"},
			{Label: "SETTINGS"},
			{Label: "BACK"},
			{Label: "SEARCH"},
			{Label: "CANCEL"},
			{Label: "FORMAT"},
			{Label: "DONE"},
			{Label: "DISPLAY"},
			{Label: "ADVANCED"},
			{Label: "GENERAL"},
		},
	})
}

// Note: This example demonstrates the conceptual usage of history states,
// though the current implementation might need a more explicit representation
// of history pseudostates. In an actual implementation, when transitioning
// back to a state with history, the statechart would need to maintain a record
// of previously active states for each composite state.
