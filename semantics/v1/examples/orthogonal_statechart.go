// Package examples provides academic examples of statechart implementations.
// This file demonstrates orthogonal (parallel/AND) statechart composition.
package examples

import (
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// OrthogonalStatechart creates a statechart that demonstrates orthogonal regions (AND-states).
// It models a media player with concurrent regions for playback and volume control:
//   - MediaPlayer
//   - PlaybackControl (parallel/orthogonal)
//   - PlaybackState
//   - Playing
//   - Paused (initial)
//   - Stopped
//   - VolumeControl
//   - Normal (initial)
//   - Muted
//
// The example demonstrates:
// 1. Orthogonal/parallel regions with AND-state composition
// 2. Concurrent state configurations
// 3. Independent transitions within orthogonal regions
//
// This follows Harel's original formalism where AND-decomposition (orthogonal regions)
// allows for concurrency and synchronization within a statechart.
func OrthogonalStatechart() *semantics.Statechart {
	sc := &sc.Statechart{
		RootState: &sc.State{
			Label: "MediaPlayer",
			Children: []*sc.State{
				{
					Label: "PlaybackControl",
					// Use the ORTHOGONAL alias for demonstrating academic terminology compatibility
					Type: sc.StateTypeOrthogonal,
					Children: []*sc.State{
						{
							Label: "PlaybackState",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label: "Playing",
									Type:  sc.StateTypeBasic,
								},
								{
									Label:     "Paused",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "Stopped",
									Type:  sc.StateTypeBasic,
								},
							},
						},
						{
							Label: "VolumeControl",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "Normal",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "Muted",
									Type:  sc.StateTypeBasic,
								},
							},
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			// Playback state transitions
			{
				Label: "Play",
				From:  []string{"Paused"},
				To:    []string{"Playing"},
				Event: "PLAY",
			},
			{
				Label: "Pause",
				From:  []string{"Playing"},
				To:    []string{"Paused"},
				Event: "PAUSE",
			},
			{
				Label: "Stop",
				From:  []string{"Playing", "Paused"},
				To:    []string{"Stopped"},
				Event: "STOP",
			},
			{
				Label: "Resume",
				From:  []string{"Stopped"},
				To:    []string{"Playing"},
				Event: "PLAY",
			},

			// Volume control transitions - these occur independently
			{
				Label: "Mute",
				From:  []string{"Normal"},
				To:    []string{"Muted"},
				Event: "MUTE",
			},
			{
				Label: "Unmute",
				From:  []string{"Muted"},
				To:    []string{"Normal"},
				Event: "UNMUTE",
			},
		},
		Events: []*sc.Event{
			{Label: "PLAY"},
			{Label: "PAUSE"},
			{Label: "STOP"},
			{Label: "MUTE"},
			{Label: "UNMUTE"},
		},
	}

	return semantics.NewStatechart(sc)
}
