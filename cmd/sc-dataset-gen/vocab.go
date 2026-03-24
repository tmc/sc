package main

import (
	"sort"

	"github.com/tmc/sc"
)

type vocabulary struct {
	States map[string]int `json:"states"`
	Events map[string]int `json:"events"`
}

func buildVocabulary(chart *sc.Statechart) *vocabulary {
	v := &vocabulary{
		States: make(map[string]int),
		Events: make(map[string]int),
	}

	// Collect state labels, excluding __root__.
	var stateLabels []string
	var walkStates func(s *sc.State)
	walkStates = func(s *sc.State) {
		if s == nil {
			return
		}
		if s.Label != "__root__" {
			stateLabels = append(stateLabels, s.Label)
		}
		for _, child := range s.Children {
			walkStates(child)
		}
	}
	walkStates(chart.RootState)
	sort.Strings(stateLabels)

	for i, label := range stateLabels {
		v.States[label] = i
	}

	// Collect event labels from events and transitions.
	seen := make(map[string]bool)
	var eventLabels []string
	for _, e := range chart.Events {
		if e != nil && e.Label != "" && !seen[e.Label] {
			seen[e.Label] = true
			eventLabels = append(eventLabels, e.Label)
		}
	}
	for _, t := range chart.Transitions {
		if t != nil && t.Event != "" && !seen[t.Event] {
			seen[t.Event] = true
			eventLabels = append(eventLabels, t.Event)
		}
	}
	sort.Strings(eventLabels)

	for i, label := range eventLabels {
		v.Events[label] = i
	}

	return v
}
