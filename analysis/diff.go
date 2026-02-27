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
	TransitionsAdded    []*TransitionEdge
	TransitionsRemoved  []*TransitionEdge
	TransitionsModified []TransitionMod

	// Event changes
	EventsAdded   []string
	EventsRemoved []string

	// Summary
	IsCompatible    bool
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
	buckets1 := bucketTransitionsByBase(g1.Transitions)
	buckets2 := bucketTransitionsByBase(g2.Transitions)
	consumed1 := make(map[*TransitionEdge]bool)
	consumed2 := make(map[*TransitionEdge]bool)

	// Detect true one-to-one guard edits as modifications.
	baseKeys := make(map[string]bool)
	for key := range buckets1 {
		baseKeys[key] = true
	}
	for key := range buckets2 {
		baseKeys[key] = true
	}
	for key := range baseKeys {
		list1 := buckets1[key]
		list2 := buckets2[key]
		if len(list1) != 1 || len(list2) != 1 {
			continue
		}
		t1 := list1[0]
		t2 := list2[0]
		if t1.Guard == t2.Guard {
			continue
		}
		consumed1[t1] = true
		consumed2[t2] = true

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
				"Guard added to transition: "+transitionDisplayKey(t1))
		} else if t1.Guard != "" && t2.Guard == "" {
			mod.Changes = append(mod.Changes, "guard removed")
		} else {
			mod.Changes = append(mod.Changes, "guard changed: "+t1.Guard+" -> "+t2.Guard)
		}
		result.TransitionsModified = append(result.TransitionsModified, mod)
	}

	counts1, sample1 := countTransitionMultiset(g1.Transitions, consumed1)
	counts2, sample2 := countTransitionMultiset(g2.Transitions, consumed2)
	allTransitionKeys := make(map[string]bool)
	for key := range counts1 {
		allTransitionKeys[key] = true
	}
	for key := range counts2 {
		allTransitionKeys[key] = true
	}
	for key := range allTransitionKeys {
		c1 := counts1[key]
		c2 := counts2[key]
		if c2 > c1 {
			for i := 0; i < c2-c1; i++ {
				result.TransitionsAdded = append(result.TransitionsAdded, sample2[key])
			}
		}
		if c1 > c2 {
			for i := 0; i < c1-c2; i++ {
				t := sample1[key]
				result.TransitionsRemoved = append(result.TransitionsRemoved, t)
				result.IsCompatible = false
				result.BreakingChanges = append(result.BreakingChanges,
					"Transition removed: "+transitionDisplayKey(t))
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
	sortTransitionEdges(result.TransitionsAdded)
	sortTransitionEdges(result.TransitionsRemoved)
	sortTransitionMods(result.TransitionsModified)

	return result
}

func bucketTransitionsByBase(transitions []*TransitionEdge) map[string][]*TransitionEdge {
	buckets := make(map[string][]*TransitionEdge)
	for _, t := range transitions {
		key := transitionBaseKey(t)
		buckets[key] = append(buckets[key], t)
	}
	return buckets
}

func countTransitionMultiset(transitions []*TransitionEdge, consumed map[*TransitionEdge]bool) (map[string]int, map[string]*TransitionEdge) {
	counts := make(map[string]int)
	sample := make(map[string]*TransitionEdge)
	for _, t := range transitions {
		if consumed[t] {
			continue
		}
		key := transitionFullKey(t)
		counts[key]++
		if sample[key] == nil {
			sample[key] = t
		}
	}
	return counts, sample
}

func transitionBaseKey(t *TransitionEdge) string {
	if t == nil {
		return ""
	}
	return t.From + "->" + t.To + "[" + t.Event + "]#" + t.Label
}

func transitionFullKey(t *TransitionEdge) string {
	if t == nil {
		return ""
	}
	return transitionBaseKey(t) + "{" + t.Guard + "}"
}

func transitionDisplayKey(t *TransitionEdge) string {
	if t == nil {
		return ""
	}
	return t.From + "->" + t.To + "[" + t.Event + "]"
}

func sortTransitionEdges(edges []*TransitionEdge) {
	sort.Slice(edges, func(i, j int) bool {
		return transitionFullKey(edges[i]) < transitionFullKey(edges[j])
	})
}

func sortTransitionMods(mods []TransitionMod) {
	sort.Slice(mods, func(i, j int) bool {
		if mods[i].From != mods[j].From {
			return mods[i].From < mods[j].From
		}
		if mods[i].To != mods[j].To {
			return mods[i].To < mods[j].To
		}
		if mods[i].Event != mods[j].Event {
			return mods[i].Event < mods[j].Event
		}
		if mods[i].OldGuard != mods[j].OldGuard {
			return mods[i].OldGuard < mods[j].OldGuard
		}
		return mods[i].NewGuard < mods[j].NewGuard
	})
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
