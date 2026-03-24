package semantics

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/sc"
)

// ReconciledConstraintID identifies one of the constraints from
// "Reconciling statechart semantics".
type ReconciledConstraintID string

const (
	ReconciledConstraintC1  ReconciledConstraintID = "C1"
	ReconciledConstraintC2  ReconciledConstraintID = "C2"
	ReconciledConstraintC3  ReconciledConstraintID = "C3"
	ReconciledConstraintC4  ReconciledConstraintID = "C4"
	ReconciledConstraintC5  ReconciledConstraintID = "C5"
	ReconciledConstraintC6  ReconciledConstraintID = "C6"
	ReconciledConstraintC7  ReconciledConstraintID = "C7"
	ReconciledConstraintC8  ReconciledConstraintID = "C8"
	ReconciledConstraintC9  ReconciledConstraintID = "C9"
	ReconciledConstraintC10 ReconciledConstraintID = "C10"
	ReconciledConstraintC11 ReconciledConstraintID = "C11"
	ReconciledConstraintC12 ReconciledConstraintID = "C12"
	ReconciledConstraintC13 ReconciledConstraintID = "C13"
	ReconciledConstraintC14 ReconciledConstraintID = "C14"
	ReconciledConstraintC15 ReconciledConstraintID = "C15"
	ReconciledConstraintC16 ReconciledConstraintID = "C16"
	ReconciledConstraintC17 ReconciledConstraintID = "C17"
)

// ReconciledConstraintViolation reports one violated paper constraint.
type ReconciledConstraintViolation struct {
	Constraint ReconciledConstraintID
	Message    string
}

// CheckReconciledConstraints evaluates constraints C1-C17 from
// "Reconciling statechart semantics" against the current statechart.
//
// The checks follow the repository's restricted syntax:
// a transition may generate internal events via action labels starting with
// `raise:`, `emit:`, or `send:`.
func (s *Statechart) CheckReconciledConstraints(opts ReconciledOptions) ([]ReconciledConstraintViolation, error) {
	if s == nil || s.Statechart == nil {
		return nil, fmt.Errorf("statechart is nil")
	}

	normalized, err := s.Normalize()
	if err != nil {
		return nil, err
	}

	analyzer, err := newConstraintAnalyzer(normalized, opts)
	if err != nil {
		return nil, err
	}

	var violations []ReconciledConstraintViolation
	violations = append(violations, analyzer.checkC1()...)
	violations = append(violations, analyzer.checkC2()...)
	violations = append(violations, analyzer.checkC3()...)
	violations = append(violations, analyzer.checkC4()...)
	violations = append(violations, analyzer.checkC5()...)
	violations = append(violations, analyzer.checkC6()...)
	violations = append(violations, analyzer.checkC7()...)
	violations = append(violations, analyzer.checkC8()...)
	violations = append(violations, analyzer.checkC9()...)
	violations = append(violations, analyzer.checkC10()...)
	violations = append(violations, analyzer.checkC11()...)
	violations = append(violations, analyzer.checkC12()...)
	violations = append(violations, analyzer.checkC13()...)
	violations = append(violations, analyzer.checkC14()...)
	violations = append(violations, analyzer.checkC15()...)
	violations = append(violations, analyzer.checkC16()...)
	violations = append(violations, analyzer.checkC17()...)

	sort.Slice(violations, func(i, j int) bool {
		if violations[i].Constraint != violations[j].Constraint {
			return violations[i].Constraint < violations[j].Constraint
		}
		return violations[i].Message < violations[j].Message
	})
	return dedupeConstraintViolations(violations), nil
}

type transitionKind int

const (
	transitionKindExternal transitionKind = iota
	transitionKindInternal
	transitionKindCompletion
)

type constraintAnalyzer struct {
	engine           *reconciledEngine
	options          ReconciledOptions
	transitions      []*sc.Transition
	scopes           map[*sc.Transition]StateLabel
	entered          map[*sc.Transition]map[string]bool
	generated        map[*sc.Transition][]string
	directTriggers   map[*sc.Transition][]*sc.Transition
	eventTransitions map[string][]*sc.Transition
	internalEvents   map[string]bool
}

type pairKey struct {
	left  *sc.Transition
	right *sc.Transition
}

func newConstraintAnalyzer(statechart *Statechart, opts ReconciledOptions) (*constraintAnalyzer, error) {
	engine := &reconciledEngine{
		statechart: statechart,
		options:    opts,
	}

	analyzer := &constraintAnalyzer{
		engine:           engine,
		options:          opts,
		transitions:      append([]*sc.Transition{}, statechart.Transitions...),
		scopes:           make(map[*sc.Transition]StateLabel),
		entered:          make(map[*sc.Transition]map[string]bool),
		generated:        make(map[*sc.Transition][]string),
		directTriggers:   make(map[*sc.Transition][]*sc.Transition),
		eventTransitions: make(map[string][]*sc.Transition),
		internalEvents:   make(map[string]bool),
	}

	for _, transition := range analyzer.transitions {
		scope, err := engine.scope(transition)
		if err != nil {
			return nil, fmt.Errorf("compute scope for %s: %w", transitionName(transition), err)
		}
		analyzer.scopes[transition] = scope

		entered, err := engine.enteredStates(transition)
		if err != nil {
			return nil, fmt.Errorf("compute entered states for %s: %w", transitionName(transition), err)
		}
		enteredSet := make(map[string]bool, len(entered))
		for _, state := range entered {
			enteredSet[string(state)] = true
		}
		analyzer.entered[transition] = enteredSet

		generated := generatedEventsForTransition(transition)
		analyzer.generated[transition] = generated
		for _, label := range generated {
			analyzer.internalEvents[label] = true
		}
		if transition.Event != "" {
			analyzer.eventTransitions[transition.Event] = append(analyzer.eventTransitions[transition.Event], transition)
		}
	}

	for _, transition := range analyzer.transitions {
		for _, label := range analyzer.generated[transition] {
			analyzer.directTriggers[transition] = append(analyzer.directTriggers[transition], analyzer.eventTransitions[label]...)
		}
	}

	return analyzer, nil
}

func (a *constraintAnalyzer) kind(transition *sc.Transition) transitionKind {
	if transition.Event == "" {
		return transitionKindCompletion
	}
	if a.internalEvents[transition.Event] {
		return transitionKindInternal
	}
	return transitionKindExternal
}

func (a *constraintAnalyzer) touches(left, right *sc.Transition) bool {
	for _, source := range right.From {
		if a.entered[left][source] {
			return true
		}
	}
	return false
}

func (a *constraintAnalyzer) consistent(left, right *sc.Transition) (bool, error) {
	return a.engine.transitionsPairConsistent(left, right)
}

func (a *constraintAnalyzer) conflict(left, right *sc.Transition) (bool, error) {
	return a.engine.conflict(left, right)
}

func (a *constraintAnalyzer) indirectlyTriggers(left, right *sc.Transition) bool {
	visited := make(map[*sc.Transition]bool)
	var walk func(*sc.Transition) bool
	walk = func(current *sc.Transition) bool {
		for _, next := range a.directTriggers[current] {
			if next == right {
				return true
			}
			if visited[next] {
				continue
			}
			visited[next] = true
			if walk(next) {
				return true
			}
		}
		return false
	}
	visited[left] = true
	return walk(left)
}

func (a *constraintAnalyzer) checkC1() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, transition := range a.transitions {
		if transition.Event == "" {
			violations = append(violations, violation(ReconciledConstraintC1,
				"%s is a completion transition", transitionName(transition)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC2() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, transition := range a.transitions {
		if a.indirectlyTriggers(transition, transition) {
			violations = append(violations, violation(ReconciledConstraintC2,
				"%s indirectly triggers itself", transitionName(transition)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC3() []ReconciledConstraintViolation {
	return a.checkConflictKinds(ReconciledConstraintC3, transitionKindExternal, transitionKindInternal,
		"external transition %s conflicts with internal transition %s")
}

func (a *constraintAnalyzer) checkC4() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, transition := range a.transitions {
		for _, triggered := range a.directTriggers[transition] {
			consistent, err := a.consistent(transition, triggered)
			if err != nil || consistent {
				continue
			}
			violations = append(violations, violation(ReconciledConstraintC4,
				"%s triggers inconsistent transition %s", transitionName(transition), transitionName(triggered)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC5() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, touched := range a.transitions {
		if a.kind(touched) != transitionKindInternal {
			continue
		}
		for _, external := range a.transitions {
			if a.kind(external) != transitionKindExternal || !a.touches(external, touched) {
				continue
			}
			for _, trigger := range a.transitions {
				if !containsTransition(a.directTriggers[trigger], touched) {
					continue
				}
				consistent, err := a.consistent(trigger, external)
				if err != nil || !consistent {
					continue
				}
				violations = append(violations, violation(ReconciledConstraintC5,
					"%s touches internal transition %s but is consistent with triggering transition %s",
					transitionName(external), transitionName(touched), transitionName(trigger)))
			}
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC6() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for i, left := range a.transitions {
		for _, right := range a.transitions[i+1:] {
			consistent, err := a.consistent(left, right)
			if err != nil || !consistent {
				continue
			}
			for _, leftTriggered := range a.directTriggers[left] {
				for _, rightTriggered := range a.directTriggers[right] {
					triggeredConsistent, err := a.consistent(leftTriggered, rightTriggered)
					if err != nil || triggeredConsistent {
						continue
					}
					violations = append(violations, violation(ReconciledConstraintC6,
						"consistent transitions %s and %s trigger inconsistent transitions %s and %s",
						transitionName(left), transitionName(right), transitionName(leftTriggered), transitionName(rightTriggered)))
				}
			}
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC7() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	visited := make(map[*sc.Transition]int)
	stack := make([]*sc.Transition, 0)

	var walk func(*sc.Transition)
	walk = func(current *sc.Transition) {
		visited[current] = len(stack) + 1
		stack = append(stack, current)
		for _, next := range a.transitions {
			if a.kind(current) != transitionKindCompletion || a.kind(next) != transitionKindCompletion || !a.touches(current, next) {
				continue
			}
			if pos := visited[next]; pos > 0 {
				cycle := append([]*sc.Transition{}, stack[pos-1:]...)
				violations = append(violations, violation(ReconciledConstraintC7,
					"completion cycle detected: %s", transitionList(cycle)))
				continue
			}
			walk(next)
		}
		stack = stack[:len(stack)-1]
		visited[current] = -1
	}

	for _, transition := range a.transitions {
		if a.kind(transition) == transitionKindCompletion && visited[transition] == 0 {
			walk(transition)
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC8() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, completion := range a.transitions {
		if a.kind(completion) != transitionKindCompletion {
			continue
		}
		for _, internal := range a.transitions {
			if a.kind(internal) != transitionKindInternal || !a.touches(completion, internal) {
				continue
			}
			violations = append(violations, violation(ReconciledConstraintC8,
				"completion transition %s touches internal transition %s",
				transitionName(completion), transitionName(internal)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC9() []ReconciledConstraintViolation {
	return a.checkConflictKinds(ReconciledConstraintC9, transitionKindExternal, transitionKindCompletion,
		"external transition %s conflicts with completion transition %s")
}

func (a *constraintAnalyzer) checkC10() []ReconciledConstraintViolation {
	return a.checkConflictKinds(ReconciledConstraintC10, transitionKindCompletion, transitionKindInternal,
		"completion transition %s conflicts with internal transition %s")
}

func (a *constraintAnalyzer) checkC11() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for i, left := range a.transitions {
		if a.kind(left) != transitionKindCompletion {
			continue
		}
		for _, right := range a.transitions[i+1:] {
			if a.kind(right) != transitionKindCompletion {
				continue
			}
			conflict, err := a.conflict(left, right)
			if err != nil || !conflict || sameStrings(left.From, right.From) {
				continue
			}
			violations = append(violations, violation(ReconciledConstraintC11,
				"conflicting completion transitions %s and %s have different sources",
				transitionName(left), transitionName(right)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC12() []ReconciledConstraintViolation {
	graph, err := a.precGraph()
	if err != nil {
		return []ReconciledConstraintViolation{violation(ReconciledConstraintC12, "%v", err)}
	}

	var violations []ReconciledConstraintViolation
	visited := make(map[string]int)
	stack := make([]string, 0)

	var walk func(string)
	walk = func(event string) {
		visited[event] = len(stack) + 1
		stack = append(stack, event)
		for next := range graph[event] {
			if pos := visited[next]; pos > 0 {
				cycle := append([]string{}, stack[pos-1:]...)
				violations = append(violations, violation(ReconciledConstraintC12,
					"prec relation is cyclic: %s", strings.Join(cycle, " -> ")))
				continue
			}
			if visited[next] == 0 {
				walk(next)
			}
		}
		stack = stack[:len(stack)-1]
		visited[event] = -1
	}

	var events []string
	for event := range graph {
		events = append(events, event)
	}
	sort.Strings(events)
	for _, event := range events {
		if visited[event] == 0 {
			walk(event)
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC13() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for i, left := range a.transitions {
		if left.Event == "" {
			continue
		}
		for _, right := range a.transitions[i+1:] {
			if left.Event != right.Event {
				continue
			}
			conflict, err := a.conflict(left, right)
			if err != nil || !conflict {
				continue
			}
			if sameStrings(left.From, right.From) && a.scopes[left] == a.scopes[right] {
				continue
			}
			violations = append(violations, violation(ReconciledConstraintC13,
				"conflicting transitions %s and %s share trigger %q but differ in sources or scope",
				transitionName(left), transitionName(right), left.Event))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC14() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, transition := range a.transitions {
		if len(a.generated[transition]) > 1 {
			violations = append(violations, violation(ReconciledConstraintC14,
				"%s generates multiple events: %s",
				transitionName(transition), strings.Join(a.generated[transition], ", ")))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC15() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for i, left := range a.transitions {
		if left.Event == "" {
			continue
		}
		for _, right := range a.transitions[i+1:] {
			if left.Event != right.Event {
				continue
			}
			consistent, err := a.consistent(left, right)
			if err != nil || !consistent || sameStrings(a.generated[left], a.generated[right]) {
				continue
			}
			violations = append(violations, violation(ReconciledConstraintC15,
				"consistent transitions %s and %s share trigger %q but generate different events",
				transitionName(left), transitionName(right), left.Event))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC16() []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, completion := range a.transitions {
		if a.kind(completion) != transitionKindCompletion {
			continue
		}
		for _, internal := range a.transitions {
			if a.kind(internal) != transitionKindInternal {
				continue
			}
			consistent, err := a.consistent(completion, internal)
			if err != nil || !consistent {
				continue
			}
			violations = append(violations, violation(ReconciledConstraintC16,
				"completion transition %s is consistent with internal transition %s",
				transitionName(completion), transitionName(internal)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) checkC17() []ReconciledConstraintViolation {
	if a.options.UMLInternalPriority {
		return nil
	}
	hasExternal := false
	hasInternal := false
	for _, transition := range a.transitions {
		switch a.kind(transition) {
		case transitionKindExternal:
			hasExternal = true
		case transitionKindInternal:
			hasInternal = true
		}
	}
	if !hasExternal || !hasInternal {
		return nil
	}
	return []ReconciledConstraintViolation{violation(ReconciledConstraintC17,
		"UML internal-event priority is disabled; charts with both external and internal transitions do not satisfy C17")}
}

func (a *constraintAnalyzer) checkConflictKinds(id ReconciledConstraintID, leftKind, rightKind transitionKind, format string) []ReconciledConstraintViolation {
	var violations []ReconciledConstraintViolation
	for _, left := range a.transitions {
		if a.kind(left) != leftKind {
			continue
		}
		for _, right := range a.transitions {
			if a.kind(right) != rightKind {
				continue
			}
			conflict, err := a.conflict(left, right)
			if err != nil || !conflict {
				continue
			}
			violations = append(violations, violation(id, format, transitionName(left), transitionName(right)))
		}
	}
	return violations
}

func (a *constraintAnalyzer) makesRelevant(left, target *sc.Transition, memo map[pairKey]bool, visiting map[pairKey]bool) (bool, error) {
	key := pairKey{left: left, right: target}
	if value, ok := memo[key]; ok {
		return value, nil
	}
	if visiting[key] {
		return false, nil
	}
	visiting[key] = true
	defer delete(visiting, key)

	for _, touching := range a.transitions {
		if !a.touches(touching, target) {
			continue
		}
		switch a.kind(touching) {
		case transitionKindExternal:
			if left == touching {
				memo[key] = true
				return true, nil
			}
		case transitionKindInternal:
			indirect := a.indirectlyTriggers(left, touching)
			consistent, err := a.consistent(left, touching)
			if err != nil {
				return false, err
			}
			if indirect && consistent {
				memo[key] = true
				return true, nil
			}
		case transitionKindCompletion:
			relevant, err := a.makesRelevant(left, touching, memo, visiting)
			if err != nil {
				return false, err
			}
			if relevant {
				memo[key] = true
				return true, nil
			}
		}
	}

	memo[key] = false
	return false, nil
}

func (a *constraintAnalyzer) precGraph() (map[string]map[string]bool, error) {
	graph := make(map[string]map[string]bool)
	memo := make(map[pairKey]bool)

	events := a.externalEventLabels()
	for _, leftEvent := range events {
		for _, rightEvent := range events {
			if leftEvent == rightEvent {
				continue
			}
			ok, err := a.prec(leftEvent, rightEvent, memo)
			if err != nil {
				return nil, err
			}
			if !ok {
				continue
			}
			if graph[leftEvent] == nil {
				graph[leftEvent] = make(map[string]bool)
			}
			graph[leftEvent][rightEvent] = true
		}
	}

	return graph, nil
}

func (a *constraintAnalyzer) prec(leftEvent, rightEvent string, memo map[pairKey]bool) (bool, error) {
	for _, left := range a.eventTransitions[leftEvent] {
		if a.kind(left) != transitionKindExternal {
			continue
		}
		for _, right := range a.eventTransitions[rightEvent] {
			if a.kind(right) != transitionKindExternal {
				continue
			}
			makesRelevant, err := a.makesRelevant(right, left, memo, make(map[pairKey]bool))
			if err != nil {
				return false, err
			}
			if makesRelevant {
				return true, nil
			}

			conflict, err := a.conflict(left, right)
			if err != nil || !conflict {
				continue
			}

			for _, witness := range a.eventTransitions[rightEvent] {
				if a.kind(witness) != transitionKindExternal {
					continue
				}
				consistent, err := a.consistent(witness, left)
				if err != nil {
					return false, err
				}
				if consistent {
					return true, nil
				}
				makesWitnessRelevant, err := a.makesRelevant(left, witness, memo, make(map[pairKey]bool))
				if err != nil {
					return false, err
				}
				if makesWitnessRelevant {
					return true, nil
				}
			}
		}
	}
	return false, nil
}

func (a *constraintAnalyzer) externalEventLabels() []string {
	set := make(map[string]bool)
	for event, transitions := range a.eventTransitions {
		for _, transition := range transitions {
			if a.kind(transition) == transitionKindExternal {
				set[event] = true
				break
			}
		}
	}
	var labels []string
	for label := range set {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	return labels
}

func containsTransition(transitions []*sc.Transition, want *sc.Transition) bool {
	for _, transition := range transitions {
		if transition == want {
			return true
		}
	}
	return false
}

func transitionName(transition *sc.Transition) string {
	if transition == nil {
		return "<nil>"
	}
	if transition.Label != "" {
		return transition.Label
	}
	return fmt.Sprintf("%s->%s[%s]", strings.Join(sortedStringsCopy(transition.From), ","), strings.Join(sortedStringsCopy(transition.To), ","), transition.Event)
}

func transitionList(transitions []*sc.Transition) string {
	labels := make([]string, 0, len(transitions))
	for _, transition := range transitions {
		labels = append(labels, transitionName(transition))
	}
	return strings.Join(labels, " -> ")
}

func sameStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	lcopy := append([]string{}, left...)
	rcopy := append([]string{}, right...)
	sort.Strings(lcopy)
	sort.Strings(rcopy)
	for i := range lcopy {
		if lcopy[i] != rcopy[i] {
			return false
		}
	}
	return true
}

func sortedStringsCopy(items []string) []string {
	out := append([]string{}, items...)
	sort.Strings(out)
	return out
}

func violation(id ReconciledConstraintID, format string, args ...interface{}) ReconciledConstraintViolation {
	return ReconciledConstraintViolation{
		Constraint: id,
		Message:    fmt.Sprintf(format, args...),
	}
}

func dedupeConstraintViolations(in []ReconciledConstraintViolation) []ReconciledConstraintViolation {
	seen := make(map[string]bool)
	out := make([]ReconciledConstraintViolation, 0, len(in))
	for _, violation := range in {
		key := string(violation.Constraint) + "|" + violation.Message
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, violation)
	}
	return out
}
