package analysis

import (
	"sort"

	sc "github.com/tmc/sc/gen/statecharts/v1"
)

// DiffResult captures differences between two statecharts.
type DiffResult struct {
	FromName string
	ToName   string

	// State changes
	StatesAdded    []string
	StatesRemoved  []string
	StatesModified []StateMod

	// Transition changes
	TransitionsAdded   []*TransitionEdge
	TransitionsRemoved []*TransitionEdge
	TransitionsModified []TransitionMod

	// Event changes
	EventsAdded   []string
	EventsRemoved []string

	// Summary
	IsCompatible   bool
	BreakingChanges []string
}

// StateMod describes a modified state.
type StateMod struct {
	Label          string
	Changes        []string // e.g., "type: OR -> AND", "initial: true -> false"
	OldType        sc.StateType
	NewType        sc.StateType
	InitialChanged bool
	FinalChanged   bool
	ParentChanged  bool
	OldParent      string
	NewParent      string
}

// TransitionMod describes a modified transition.
type TransitionMod struct {
	From         string
	To           string
	Event        string
	Changes      []string
	GuardChanged bool
	OldGuard     string
	NewGuard     string
}

// Diff computes differences between two statecharts.
func Diff(chart1, chart2 *sc.Statechart) *DiffResult {
	g1 := BuildGraph(chart1)
	g2 := BuildGraph(chart2)

	result := &DiffResult{
		FromName:     chart1.Name,
		ToName:       chart2.Name,
		IsCompatible: true,
	}

	// Compare states
	states1 := make(map[string]*StateNode)
	states2 := make(map[string]*StateNode)

	for label, node := range g1.States {
		if label != "__root__" {
			states1[label] = node
		}
	}
	for label, node := range g2.States {
		if label != "__root__" {
			states2[label] = node
		}
	}

	// Find added, removed, modified states
	for label := range states2 {
		if _, exists := states1[label]; !exists {
			result.StatesAdded = append(result.StatesAdded, label)
		}
	}
	for label, node1 := range states1 {
		if node2, exists := states2[label]; !exists {
			result.StatesRemoved = append(result.StatesRemoved, label)
			result.IsCompatible = false
			result.BreakingChanges = append(result.BreakingChanges,
				"State removed: "+label)
		} else {
			// Check for modifications
			mod := compareStates(node1, node2)
			if len(mod.Changes) > 0 {
				result.StatesModified = append(result.StatesModified, mod)
				// Type changes are breaking
				if mod.OldType != mod.NewType {
					result.IsCompatible = false
					result.BreakingChanges = append(result.BreakingChanges,
						"State type changed: "+label)
				}
			}
		}
	}

	// Compare transitions
	trans1 := make(map[string]*TransitionEdge)
	trans2 := make(map[string]*TransitionEdge)

	for _, t := range g1.Transitions {
		key := t.From + "->" + t.To + "[" + t.Event + "]"
		trans1[key] = t
	}
	for _, t := range g2.Transitions {
		key := t.From + "->" + t.To + "[" + t.Event + "]"
		trans2[key] = t
	}

	for key, t := range trans2 {
		if _, exists := trans1[key]; !exists {
			result.TransitionsAdded = append(result.TransitionsAdded, t)
		}
	}
	for key, t1 := range trans1 {
		if t2, exists := trans2[key]; !exists {
			result.TransitionsRemoved = append(result.TransitionsRemoved, t1)
			result.IsCompatible = false
			result.BreakingChanges = append(result.BreakingChanges,
				"Transition removed: "+key)
		} else {
			// Check for guard changes
			if t1.Guard != t2.Guard {
				mod := TransitionMod{
					From:         t1.From,
					To:           t1.To,
					Event:        t1.Event,
					GuardChanged: true,
					OldGuard:     t1.Guard,
					NewGuard:     t2.Guard,
				}
				if t1.Guard == "" && t2.Guard != "" {
					mod.Changes = append(mod.Changes, "guard added: "+t2.Guard)
					result.IsCompatible = false
					result.BreakingChanges = append(result.BreakingChanges,
						"Guard added to transition: "+key)
				} else if t1.Guard != "" && t2.Guard == "" {
					mod.Changes = append(mod.Changes, "guard removed")
				} else {
					mod.Changes = append(mod.Changes, "guard changed: "+t1.Guard+" -> "+t2.Guard)
				}
				result.TransitionsModified = append(result.TransitionsModified, mod)
			}
		}
	}

	// Compare events
	events1 := make(map[string]bool)
	events2 := make(map[string]bool)

	for _, t := range g1.Transitions {
		if t.Event != "" {
			events1[t.Event] = true
		}
	}
	for _, t := range g2.Transitions {
		if t.Event != "" {
			events2[t.Event] = true
		}
	}

	for e := range events2 {
		if !events1[e] {
			result.EventsAdded = append(result.EventsAdded, e)
		}
	}
	for e := range events1 {
		if !events2[e] {
			result.EventsRemoved = append(result.EventsRemoved, e)
			result.IsCompatible = false
			result.BreakingChanges = append(result.BreakingChanges,
				"Event removed: "+e)
		}
	}

	// Sort for consistent output
	sort.Strings(result.StatesAdded)
	sort.Strings(result.StatesRemoved)
	sort.Strings(result.EventsAdded)
	sort.Strings(result.EventsRemoved)
	sort.Strings(result.BreakingChanges)

	return result
}

func compareStates(n1, n2 *StateNode) StateMod {
	mod := StateMod{
		Label:   n1.Label,
		OldType: n1.Type,
		NewType: n2.Type,
	}

	if n1.Type != n2.Type {
		mod.Changes = append(mod.Changes, "type: "+stateTypeName(n1.Type)+" -> "+stateTypeName(n2.Type))
	}
	if n1.IsInitial != n2.IsInitial {
		mod.InitialChanged = true
		mod.Changes = append(mod.Changes, "initial changed")
	}
	if n1.IsFinal != n2.IsFinal {
		mod.FinalChanged = true
		mod.Changes = append(mod.Changes, "final changed")
	}
	if n1.Parent != n2.Parent {
		mod.ParentChanged = true
		mod.OldParent = n1.Parent
		mod.NewParent = n2.Parent
		mod.Changes = append(mod.Changes, "parent: "+n1.Parent+" -> "+n2.Parent)
	}

	return mod
}

func stateTypeName(t sc.StateType) string {
	switch t {
	case sc.StateType_STATE_TYPE_BASIC:
		return "BASIC"
	case sc.StateType_STATE_TYPE_OR:
		return "OR"
	case sc.StateType_STATE_TYPE_AND:
		return "AND"
	default:
		return "UNKNOWN"
	}
}

// HasChanges returns true if there are any differences.
func (d *DiffResult) HasChanges() bool {
	return len(d.StatesAdded) > 0 ||
		len(d.StatesRemoved) > 0 ||
		len(d.StatesModified) > 0 ||
		len(d.TransitionsAdded) > 0 ||
		len(d.TransitionsRemoved) > 0 ||
		len(d.TransitionsModified) > 0 ||
		len(d.EventsAdded) > 0 ||
		len(d.EventsRemoved) > 0
}

// Summary returns a short summary of changes.
func (d *DiffResult) Summary() string {
	if !d.HasChanges() {
		return "No changes"
	}

	parts := []string{}
	if n := len(d.StatesAdded); n > 0 {
		parts = append(parts, "+"+itoa(n)+" states")
	}
	if n := len(d.StatesRemoved); n > 0 {
		parts = append(parts, "-"+itoa(n)+" states")
	}
	if n := len(d.StatesModified); n > 0 {
		parts = append(parts, "~"+itoa(n)+" states")
	}
	if n := len(d.TransitionsAdded); n > 0 {
		parts = append(parts, "+"+itoa(n)+" transitions")
	}
	if n := len(d.TransitionsRemoved); n > 0 {
		parts = append(parts, "-"+itoa(n)+" transitions")
	}
	if n := len(d.TransitionsModified); n > 0 {
		parts = append(parts, "~"+itoa(n)+" transitions")
	}

	result := ""
	for i, p := range parts {
		if i > 0 {
			result += ", "
		}
		result += p
	}
	return result
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}

// Evolution represents the evolution of a statechart across versions.
type Evolution struct {
	Versions []EvolutionVersion
	Timeline []EvolutionStep
}

// EvolutionVersion describes one version in the evolution.
type EvolutionVersion struct {
	Name       string
	StateCount int
	TransCount int
}

// EvolutionStep describes changes between consecutive versions.
type EvolutionStep struct {
	FromVersion string
	ToVersion   string
	Diff        *DiffResult
}

// DeriveEvolution computes the evolution across a series of charts.
func DeriveEvolution(charts []*sc.Statechart, names []string) *Evolution {
	if len(charts) < 2 {
		return nil
	}

	ev := &Evolution{}

	for i, chart := range charts {
		g := BuildGraph(chart)
		name := ""
		if i < len(names) {
			name = names[i]
		} else {
			name = chart.Name
		}
		ev.Versions = append(ev.Versions, EvolutionVersion{
			Name:       name,
			StateCount: g.StateCount(),
			TransCount: g.TransitionCount(),
		})
	}

	for i := 1; i < len(charts); i++ {
		diff := Diff(charts[i-1], charts[i])
		diff.FromName = ev.Versions[i-1].Name
		diff.ToName = ev.Versions[i].Name
		ev.Timeline = append(ev.Timeline, EvolutionStep{
			FromVersion: ev.Versions[i-1].Name,
			ToVersion:   ev.Versions[i].Name,
			Diff:        diff,
		})
	}

	return ev
}
