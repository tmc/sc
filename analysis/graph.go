// Package analysis provides graph analysis algorithms for statecharts.
package analysis

import (
	"sort"

	sc "github.com/tmc/sc/gen/statecharts/v1"
)

// StateGraph represents a statechart as a directed graph for analysis.
type StateGraph struct {
	// States indexed by label
	States map[string]*StateNode

	// All transition edges
	Transitions []*TransitionEdge

	// Quick lookups
	rootLabel    string
	leafStates   []string
	initialState string
	finalStates  []string
}

// StateNode represents a state in the graph.
type StateNode struct {
	Label     string
	Type      sc.StateType
	IsInitial bool
	IsFinal   bool
	IsLeaf    bool
	Parent    string
	Children  []string

	// Transition edges
	Outgoing []*TransitionEdge
	Incoming []*TransitionEdge
}

// TransitionEdge represents a transition in the graph.
type TransitionEdge struct {
	From  string
	To    string
	Event string
	Guard string
	Label string // Transition label if any
}

// BuildGraph constructs a StateGraph from a statechart.
func BuildGraph(chart *sc.Statechart) *StateGraph {
	g := &StateGraph{
		States: make(map[string]*StateNode),
	}
	if chart != nil && chart.RootState != nil {
		g.rootLabel = chart.RootState.Label
	}

	// Collect all states via DFS
	var collectStates func(s *sc.State, parent string)
	collectStates = func(s *sc.State, parent string) {
		if s == nil {
			return
		}

		node := &StateNode{
			Label:     s.Label,
			Type:      s.Type,
			IsInitial: s.IsInitial,
			IsFinal:   s.IsFinal,
			IsLeaf:    len(s.Children) == 0,
			Parent:    parent,
		}

		// Collect children labels
		for _, child := range s.Children {
			node.Children = append(node.Children, child.Label)
		}

		g.States[s.Label] = node

		// Track special states
		if node.IsLeaf {
			g.leafStates = append(g.leafStates, s.Label)
		}
		if node.IsInitial && parent == "" {
			g.initialState = s.Label
		} else if node.IsInitial && parent == "__root__" {
			g.initialState = s.Label
		}
		if node.IsFinal {
			g.finalStates = append(g.finalStates, s.Label)
		}

		// Recurse
		for _, child := range s.Children {
			collectStates(child, s.Label)
		}
	}
	collectStates(chart.RootState, "")

	// Build transition edges
	for _, t := range chart.Transitions {
		for _, from := range t.From {
			for _, to := range t.To {
				guard := ""
				if t.Guard != nil {
					guard = t.Guard.Expression
				}
				edge := &TransitionEdge{
					From:  from,
					To:    to,
					Event: t.Event,
					Guard: guard,
					Label: t.Label,
				}
				g.Transitions = append(g.Transitions, edge)

				// Add to adjacency lists
				if fromNode, ok := g.States[from]; ok {
					fromNode.Outgoing = append(fromNode.Outgoing, edge)
				}
				if toNode, ok := g.States[to]; ok {
					toNode.Incoming = append(toNode.Incoming, edge)
				}
			}
		}
	}

	return g
}

// Orphans returns states with no incoming transitions.
// By default excludes initial states (they're expected to have no incoming).
func (g *StateGraph) Orphans(includeInitial bool) []string {
	var orphans []string
	for label, node := range g.States {
		if label == "__root__" {
			continue
		}
		if len(node.Incoming) == 0 {
			// Skip initial state unless explicitly included
			if !includeInitial && node.IsInitial {
				continue
			}
			orphans = append(orphans, label)
		}
	}
	sort.Strings(orphans)
	return orphans
}

// Sinks returns states with no outgoing transitions.
// By default excludes final states (they're expected to have no outgoing).
func (g *StateGraph) Sinks(includeFinal bool) []string {
	var sinks []string
	for label, node := range g.States {
		if label == "__root__" {
			continue
		}
		if len(node.Outgoing) == 0 {
			// Skip final state unless explicitly included
			if !includeFinal && node.IsFinal {
				continue
			}
			sinks = append(sinks, label)
		}
	}
	sort.Strings(sinks)
	return sinks
}

// Disconnected returns states with neither incoming nor outgoing transitions.
func (g *StateGraph) Disconnected() []string {
	var disconnected []string
	for label, node := range g.States {
		if label == "__root__" {
			continue
		}
		if len(node.Incoming) == 0 && len(node.Outgoing) == 0 {
			disconnected = append(disconnected, label)
		}
	}
	sort.Strings(disconnected)
	return disconnected
}

// TransitionsFrom returns all transitions originating from a state.
func (g *StateGraph) TransitionsFrom(state string) []*TransitionEdge {
	if node, ok := g.States[state]; ok {
		return node.Outgoing
	}
	return nil
}

// TransitionsTo returns all transitions targeting a state.
func (g *StateGraph) TransitionsTo(state string) []*TransitionEdge {
	if node, ok := g.States[state]; ok {
		return node.Incoming
	}
	return nil
}

// FilterTransitions returns transitions matching the given criteria.
// Empty string means no filter for that field.
func (g *StateGraph) FilterTransitions(from, to, event string) []*TransitionEdge {
	var result []*TransitionEdge
	for _, t := range g.Transitions {
		if from != "" && t.From != from {
			continue
		}
		if to != "" && t.To != to {
			continue
		}
		if event != "" && t.Event != event {
			continue
		}
		result = append(result, t)
	}
	return result
}

// AllStates returns all state labels (excluding __root__).
func (g *StateGraph) AllStates() []string {
	var states []string
	for label := range g.States {
		if label != "__root__" {
			states = append(states, label)
		}
	}
	sort.Strings(states)
	return states
}

// LeafStates returns all leaf state labels.
func (g *StateGraph) LeafStates() []string {
	result := make([]string, len(g.leafStates))
	copy(result, g.leafStates)
	sort.Strings(result)
	return result
}

// InitialState returns the initial state label.
func (g *StateGraph) InitialState() string {
	return g.initialState
}

// FinalStates returns all final state labels.
func (g *StateGraph) FinalStates() []string {
	result := make([]string, len(g.finalStates))
	copy(result, g.finalStates)
	return result
}

// StateCount returns the number of states (excluding __root__).
func (g *StateGraph) StateCount() int {
	count := len(g.States)
	if _, ok := g.States["__root__"]; ok {
		count--
	}
	return count
}

// TransitionCount returns the number of transitions.
func (g *StateGraph) TransitionCount() int {
	return len(g.Transitions)
}
