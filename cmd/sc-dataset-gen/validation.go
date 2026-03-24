package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math/rand"
	"sort"

	"github.com/tmc/sc"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	validation "github.com/tmc/sc/validation/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// validationRow records the result of running the full validator against a chart.
type validationRow struct {
	ChartID        string             `json:"chart_id"`
	ChartHash      string             `json:"chart_hash"`
	IsMutated      bool               `json:"is_mutated"`
	MutationFamily string             `json:"mutation_family,omitempty"`
	MutationOp     string             `json:"mutation_op,omitempty"`
	NStates        int                `json:"n_states"`
	NTransitions   int                `json:"n_transitions"`
	Families       []string           `json:"families"`
	Violations     []violationEntry   `json:"violations"`
	ViolatedRuleIDs []int             `json:"violated_rule_ids"`
	NViolations    int                `json:"n_violations"`
	IsWellFormed   bool               `json:"is_well_formed"`
}

type violationEntry struct {
	RuleID   int    `json:"rule_id"`
	RuleName string `json:"rule_name"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}

// buildValidationRows runs the validator against the clean chart and targeted
// mutations, returning one row per chart variant.
func buildValidationRows(chart *sc.Statechart, chartID string, families []string, seed int64, mutationCount int) []validationRow {
	var rows []validationRow

	// Clean chart row.
	rows = append(rows, runValidation(chart, chartID, families, false, "", ""))

	// Targeted mutations that exercise specific rule violations.
	rng := rand.New(rand.NewSource(seed))
	mutators := targetedMutators()
	for i := 0; i < mutationCount; i++ {
		m := mutators[rng.Intn(len(mutators))]
		mutated := proto.Clone(chart).(*sc.Statechart)
		op, ok := m.fn(mutated, rng)
		if !ok {
			continue
		}
		rows = append(rows, runValidation(mutated, chartID, families, true, m.family, op))
	}

	return rows
}

func runValidation(chart *sc.Statechart, chartID string, families []string, isMutated bool, mutFamily, mutOp string) validationRow {
	v := validation.NewSemanticValidator()
	resp, err := v.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{Chart: chart})

	row := validationRow{
		ChartID:      chartID,
		ChartHash:    validationChartHash(chart),
		IsMutated:    isMutated,
		MutationFamily: mutFamily,
		MutationOp:   mutOp,
		NStates:      countStates(chart.RootState),
		NTransitions: len(chart.Transitions),
		Families:     families,
		IsWellFormed: true,
	}

	if err == nil && resp != nil {
		seen := map[int]bool{}
		for _, viol := range resp.Violations {
			id := int(viol.Rule)
			entry := violationEntry{
				RuleID:   id,
				RuleName: viol.Rule.String(),
				Severity: viol.Severity.String(),
				Message:  viol.Message,
			}
			row.Violations = append(row.Violations, entry)
			if !seen[id] {
				seen[id] = true
				row.ViolatedRuleIDs = append(row.ViolatedRuleIDs, id)
			}
			if viol.Severity == validationv1.Severity_ERROR {
				row.IsWellFormed = false
			}
		}
		sort.Ints(row.ViolatedRuleIDs)
	}
	row.NViolations = len(row.Violations)
	return row
}

func validationChartHash(chart *sc.Statechart) string {
	b, err := protojson.MarshalOptions{UseProtoNames: true}.Marshal(chart)
	if err != nil {
		b = []byte(chart.Name)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func countStates(s *sc.State) int {
	if s == nil {
		return 0
	}
	n := 1
	for _, c := range s.Children {
		n += countStates(c)
	}
	return n
}

// targetedMutator describes a mutation operator designed to trigger specific
// validation rule violations.
type targetedMutator struct {
	family string
	fn     func(*sc.Statechart, *rand.Rand) (string, bool)
}

func targetedMutators() []targetedMutator {
	return []targetedMutator{
		// Rule 1: UNIQUE_STATE_LABELS
		{"duplicate_label", mutateDuplicateLabel},
		// Rule 2: SINGLE_DEFAULT_CHILD — add extra initial
		{"extra_initial", mutateExtraInitial},
		// Rule 2: SINGLE_DEFAULT_CHILD — remove initial
		{"remove_initial", mutateRemoveInitial},
		// Rule 3: BASIC_HAS_NO_CHILDREN
		{"basic_with_children", mutateBasicWithChildren},
		// Rule 4: COMPOUND_HAS_CHILDREN
		{"compound_no_children", mutateCompoundNoChildren},
		// Rule 5: DETERMINISTIC_TRANSITION_SELECTION
		{"nondeterministic_transition", mutateNondeterministicTransition},
		// Rule 7: HISTORY_STATES_WELL_FORMED
		{"invalid_history", mutateInvalidHistory},
		// Rule 10: CHOICE_GUARDS_COMPLETE — remove transition leaving dead end
		{"create_dead_end", mutateCreateDeadEnd},
		// Rule 13: GUARD_EXPRESSIONS_VALID
		{"invalid_guard", mutateInvalidGuard},
		// Rule 14: EVENT_PARAMETERS_CONSISTENT
		{"undeclared_event", mutateUndeclaredEvent},
		// Rule 15: INTERNAL_TRANSITIONS_VALID — cross-boundary source
		{"cross_boundary_source", mutateCrossBoundarySource},
		// Structural: remove all transitions
		{"remove_all_transitions", mutateRemoveAllTransitions},
		// Structural: add self-loop on final state
		{"final_with_outgoing", mutateFinalWithOutgoing},
		// Structural: change state type without adjusting children
		{"wrong_state_type", mutateWrongStateType},
	}
}

// mutateDuplicateLabel renames a random state to share a label with another state.
func mutateDuplicateLabel(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	states := collectAllStates(chart.RootState)
	if len(states) < 3 { // need root + at least 2 others
		return "", false
	}
	// Pick two non-root states.
	nonRoot := states[1:] // skip __root__
	if len(nonRoot) < 2 {
		return "", false
	}
	a := nonRoot[rng.Intn(len(nonRoot))]
	b := nonRoot[rng.Intn(len(nonRoot))]
	for b == a {
		b = nonRoot[rng.Intn(len(nonRoot))]
	}
	old := b.Label
	b.Label = a.Label
	// Also fix transition references that pointed to old label.
	for _, t := range chart.Transitions {
		for i, f := range t.From {
			if f == old {
				t.From[i] = a.Label
			}
		}
		for i, to := range t.To {
			if to == old {
				t.To[i] = a.Label
			}
		}
	}
	return "duplicate label " + a.Label, true
}

// mutateExtraInitial marks a second child as initial in an OR state.
func mutateExtraInitial(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	if len(composites) == 0 {
		return "", false
	}
	parent := composites[rng.Intn(len(composites))]
	for _, c := range parent.Children {
		if !c.IsInitial && !c.IsHistory {
			c.IsInitial = true
			return "extra initial in " + parent.Label, true
		}
	}
	return "", false
}

// mutateRemoveInitial clears the initial flag from the default child.
func mutateRemoveInitial(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	if len(composites) == 0 {
		return "", false
	}
	parent := composites[rng.Intn(len(composites))]
	for _, c := range parent.Children {
		if c.IsInitial {
			c.IsInitial = false
			return "remove initial from " + parent.Label, true
		}
	}
	return "", false
}

// mutateBasicWithChildren turns a basic state into one with children.
func mutateBasicWithChildren(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	basics := collectBasicStates(chart.RootState)
	if len(basics) == 0 {
		return "", false
	}
	s := basics[rng.Intn(len(basics))]
	s.Children = []*sc.State{
		{Label: s.Label + "_illegal_child", Type: sc.StateTypeBasic, IsInitial: true},
	}
	return "add child to basic " + s.Label, true
}

// mutateCompoundNoChildren removes all children from a composite state.
func mutateCompoundNoChildren(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	composites = append(composites, collectCompositeStates(chart.RootState, sc.StateTypeParallel)...)
	if len(composites) == 0 {
		return "", false
	}
	s := composites[rng.Intn(len(composites))]
	s.Children = nil
	return "remove children from " + s.Label, true
}

// mutateNondeterministicTransition duplicates a transition creating nondeterminism.
func mutateNondeterministicTransition(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	dup := proto.Clone(t).(*sc.Transition)
	// Change target to create a genuine conflict.
	states := collectAllStates(chart.RootState)
	nonRoot := filterNonRoot(states)
	if len(nonRoot) < 2 {
		return "", false
	}
	dup.To = []string{nonRoot[rng.Intn(len(nonRoot))].Label}
	dup.Label = t.Label + "_dup"
	chart.Transitions = append(chart.Transitions, dup)
	return "duplicate transition " + t.Label, true
}

// mutateInvalidHistory creates a badly configured history pseudostate.
func mutateInvalidHistory(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	basics := collectBasicStates(chart.RootState)
	if len(basics) == 0 {
		return "", false
	}
	s := basics[rng.Intn(len(basics))]
	s.IsHistory = true
	s.HistoryType = sc.HistoryType_HISTORY_TYPE_SHALLOW
	// History in a basic state (no composite parent with proper structure) is invalid.
	return "invalid history on " + s.Label, true
}

// mutateCreateDeadEnd removes all outgoing transitions from a random state.
func mutateCreateDeadEnd(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	states := collectAllStates(chart.RootState)
	nonRoot := filterNonRoot(states)
	if len(nonRoot) == 0 {
		return "", false
	}
	target := nonRoot[rng.Intn(len(nonRoot))].Label
	kept := chart.Transitions[:0]
	removed := 0
	for _, t := range chart.Transitions {
		isFrom := false
		for _, f := range t.From {
			if f == target {
				isFrom = true
				break
			}
		}
		if isFrom {
			removed++
		} else {
			kept = append(kept, t)
		}
	}
	chart.Transitions = kept
	if removed == 0 {
		return "", false
	}
	return "create dead end at " + target, true
}

// mutateInvalidGuard adds a syntactically invalid guard expression.
func mutateInvalidGuard(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	t.Guard = &sc.Guard{Expression: ";;invalid;;"}
	label := t.Label
	if label == "" {
		label = t.Event
	}
	return "invalid guard on " + label, true
}

// mutateUndeclaredEvent changes a transition's event to one not in the Events list.
func mutateUndeclaredEvent(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	chart.Transitions[idx].Event = "__UNDECLARED_EVENT__"
	return "undeclared event on transition", true
}

// mutateCrossBoundarySource changes a transition source to a state in a different branch.
func mutateCrossBoundarySource(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	states := collectAllStates(chart.RootState)
	nonRoot := filterNonRoot(states)
	if len(nonRoot) < 2 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	newSrc := nonRoot[rng.Intn(len(nonRoot))].Label
	t.From = []string{newSrc}
	return "cross boundary source " + newSrc, true
}

// mutateRemoveAllTransitions removes every transition.
func mutateRemoveAllTransitions(chart *sc.Statechart, _ *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	chart.Transitions = nil
	return "remove all transitions", true
}

// mutateFinalWithOutgoing adds an outgoing transition from a final state (or marks
// a state with outgoing transitions as final).
func mutateFinalWithOutgoing(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	// Pick a random transition source and mark it final.
	t := chart.Transitions[rng.Intn(len(chart.Transitions))]
	if len(t.From) == 0 {
		return "", false
	}
	src := t.From[0]
	s := findState(chart.RootState, src)
	if s == nil {
		return "", false
	}
	s.IsFinal = true
	return "final with outgoing " + src, true
}

// mutateWrongStateType flips a state type without adjusting structure.
func mutateWrongStateType(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	if len(composites) == 0 {
		// Try flipping a basic state to compound.
		basics := collectBasicStates(chart.RootState)
		if len(basics) == 0 {
			return "", false
		}
		s := basics[rng.Intn(len(basics))]
		s.Type = sc.StateTypeNormal
		return "basic->normal without children " + s.Label, true
	}
	s := composites[rng.Intn(len(composites))]
	s.Type = sc.StateTypeBasic
	return "normal->basic with children " + s.Label, true
}

// --- helpers ---

func collectAllStates(root *sc.State) []*sc.State {
	if root == nil {
		return nil
	}
	result := []*sc.State{root}
	for _, c := range root.Children {
		result = append(result, collectAllStates(c)...)
	}
	return result
}

func collectBasicStates(root *sc.State) []*sc.State {
	var result []*sc.State
	for _, s := range collectAllStates(root) {
		if s.Type == sc.StateTypeBasic && s.Label != "__root__" {
			result = append(result, s)
		}
	}
	return result
}

func collectCompositeStates(root *sc.State, typ sc.StateType) []*sc.State {
	var result []*sc.State
	for _, s := range collectAllStates(root) {
		if s.Type == typ && len(s.Children) > 0 && s.Label != "__root__" {
			result = append(result, s)
		}
	}
	return result
}

func filterNonRoot(states []*sc.State) []*sc.State {
	var result []*sc.State
	for _, s := range states {
		if s.Label != "__root__" {
			result = append(result, s)
		}
	}
	return result
}

func findState(root *sc.State, label string) *sc.State {
	if root == nil {
		return nil
	}
	if root.Label == label {
		return root
	}
	for _, c := range root.Children {
		if s := findState(c, label); s != nil {
			return s
		}
	}
	return nil
}
