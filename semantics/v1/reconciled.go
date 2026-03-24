package semantics

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

// SemanticsVariant identifies one of the execution models formalized in
// Reconciling Statechart Semantics.
type SemanticsVariant int

const (
	// SemanticsFixpoint models the fixpoint semantics from Section 3.2.
	SemanticsFixpoint SemanticsVariant = iota
	// SemanticsStatemate models the Statemate semantics from Section 3.3.
	SemanticsStatemate
	// SemanticsSingleEventStatemate models the single-event Statemate variant
	// used in Section 4.2 and Section 5.
	SemanticsSingleEventStatemate
	// SemanticsUML models the UML semantics from Section 3.4.
	SemanticsUML
)

func (v SemanticsVariant) String() string {
	switch v {
	case SemanticsFixpoint:
		return "fixpoint"
	case SemanticsStatemate:
		return "statemate"
	case SemanticsSingleEventStatemate:
		return "single-event-statemate"
	case SemanticsUML:
		return "uml"
	default:
		return "unknown"
	}
}

// ReconciledOptions customizes the paper semantics engine.
type ReconciledOptions struct {
	// MaxSteps bounds superstep exploration. Zero selects a conservative default.
	MaxSteps int
	// UMLInternalPriority enables constraint C17 behavior by placing generated
	// internal events ahead of queued external events.
	UMLInternalPriority bool
}

// ReconciledRuntime is the machine snapshot used by the paper semantics engine.
type ReconciledRuntime struct {
	Configuration *sc.Configuration
	Context       *structpb.Struct
	Queue         []string
}

// ReconciledReaction is one possible reaction under a specific semantics.
type ReconciledReaction struct {
	Variant  SemanticsVariant
	Steps    []*sc.Step
	Final    *ReconciledRuntime
	Diverged bool
}

// Reactions enumerates the possible reactions under one of the semantics
// formalized in the paper.
//
// The implementation follows the paper's restricted syntax:
// transitions are triggered by a single event or by completion (empty event).
// Internal event generation is represented by transition actions whose labels
// use one of the prefixes "raise:", "emit:", or "send:".
func (s *Statechart) Reactions(variant SemanticsVariant, runtime *ReconciledRuntime, externalEvents ...string) ([]*ReconciledReaction, error) {
	if s == nil || s.Statechart == nil {
		return nil, fmt.Errorf("statechart is nil")
	}

	rt, err := s.normalizedRuntime(runtime)
	if err != nil {
		return nil, err
	}

	engine := &reconciledEngine{
		statechart: s,
		options:    ReconciledOptions{MaxSteps: 256},
	}

	switch variant {
	case SemanticsFixpoint:
		return engine.runFixpoint(rt, externalEvents)
	case SemanticsStatemate:
		return engine.runStatemateReaction(rt, externalEvents)
	case SemanticsSingleEventStatemate:
		return engine.runSingleEventStatemate(rt, externalEvents)
	case SemanticsUML:
		return engine.runUML(rt, externalEvents)
	default:
		return nil, fmt.Errorf("unsupported semantics variant %q", variant)
	}
}

// ReactionsWithOptions enumerates possible reactions under a semantics variant
// using the provided execution options.
func (s *Statechart) ReactionsWithOptions(variant SemanticsVariant, runtime *ReconciledRuntime, options ReconciledOptions, externalEvents ...string) ([]*ReconciledReaction, error) {
	if options.MaxSteps <= 0 {
		options.MaxSteps = 256
	}

	rt, err := s.normalizedRuntime(runtime)
	if err != nil {
		return nil, err
	}

	engine := &reconciledEngine{
		statechart: s,
		options:    options,
	}

	switch variant {
	case SemanticsFixpoint:
		return engine.runFixpoint(rt, externalEvents)
	case SemanticsStatemate:
		return engine.runStatemateReaction(rt, externalEvents)
	case SemanticsSingleEventStatemate:
		return engine.runSingleEventStatemate(rt, externalEvents)
	case SemanticsUML:
		return engine.runUML(rt, externalEvents)
	default:
		return nil, fmt.Errorf("unsupported semantics variant %q", variant)
	}
}

type reconciledEngine struct {
	statechart *Statechart
	options    ReconciledOptions
}

func (e *reconciledEngine) runFixpoint(runtime *ReconciledRuntime, externalEvents []string) ([]*ReconciledReaction, error) {
	input := uniqueSortedStrings(externalEvents)
	steps, err := e.validSteps(runtime.Configuration, runtime.Context, input, SemanticsFixpoint)
	if err != nil {
		return nil, err
	}

	var reactions []*ReconciledReaction
	for _, transitions := range steps {
		result, err := e.executeStep(runtime, input, transitions)
		if err != nil {
			return nil, err
		}
		reactions = append(reactions, &ReconciledReaction{
			Variant: SemanticsFixpoint,
			Steps:   []*sc.Step{result.step},
			Final: &ReconciledRuntime{
				Configuration: cloneConfiguration(result.configuration),
				Context:       e.statechart.cloneContext(result.context),
			},
		})
	}

	return dedupeReactions(reactions), nil
}

func (e *reconciledEngine) runStatemateReaction(runtime *ReconciledRuntime, externalEvents []string) ([]*ReconciledReaction, error) {
	input := uniqueSortedStrings(externalEvents)
	reactions, err := e.runStatemateSuperstep(runtime, input, nil, 0)
	if err != nil {
		return nil, err
	}
	for _, reaction := range reactions {
		reaction.Variant = SemanticsStatemate
	}
	return dedupeReactions(reactions), nil
}

func (e *reconciledEngine) runStatemateSuperstep(runtime *ReconciledRuntime, input []string, seen map[string]bool, depth int) ([]*ReconciledReaction, error) {
	if depth >= e.options.MaxSteps {
		return []*ReconciledReaction{{
			Diverged: true,
			Final: &ReconciledRuntime{
				Configuration: cloneConfiguration(runtime.Configuration),
				Context:       e.statechart.cloneContext(runtime.Context),
			},
		}}, nil
	}

	if seen == nil {
		seen = make(map[string]bool)
	}
	key := statemateKey(runtime.Configuration, input)
	if seen[key] {
		return []*ReconciledReaction{{
			Diverged: true,
			Final: &ReconciledRuntime{
				Configuration: cloneConfiguration(runtime.Configuration),
				Context:       e.statechart.cloneContext(runtime.Context),
			},
		}}, nil
	}

	if stable, err := e.isStableStatemate(runtime.Configuration, input); err != nil {
		return nil, err
	} else if stable {
		return []*ReconciledReaction{{
			Final: &ReconciledRuntime{
				Configuration: cloneConfiguration(runtime.Configuration),
				Context:       e.statechart.cloneContext(runtime.Context),
			},
		}}, nil
	}

	nextSeen := cloneSeen(seen)
	nextSeen[key] = true

	steps, err := e.validSteps(runtime.Configuration, runtime.Context, input, SemanticsStatemate)
	if err != nil {
		return nil, err
	}

	var reactions []*ReconciledReaction
	for _, transitions := range steps {
		result, err := e.executeStep(runtime, input, transitions)
		if err != nil {
			return nil, err
		}

		subRuntime := &ReconciledRuntime{
			Configuration: result.configuration,
			Context:       result.context,
		}
		subReactions, err := e.runStatemateSuperstep(subRuntime, result.generated, nextSeen, depth+1)
		if err != nil {
			return nil, err
		}
		for _, sub := range subReactions {
			reactions = append(reactions, &ReconciledReaction{
				Steps:    append([]*sc.Step{result.step}, cloneSteps(sub.Steps)...),
				Final:    cloneRuntime(sub.Final),
				Diverged: sub.Diverged,
			})
		}
	}

	return dedupeReactions(reactions), nil
}

func (e *reconciledEngine) runSingleEventStatemate(runtime *ReconciledRuntime, externalEvents []string) ([]*ReconciledReaction, error) {
	if len(externalEvents) == 0 {
		return []*ReconciledReaction{{
			Variant: SemanticsSingleEventStatemate,
			Final:   cloneRuntime(runtime),
		}}, nil
	}

	var reactions []*ReconciledReaction
	for _, permutation := range uniquePermutations(externalEvents) {
		current := []*ReconciledReaction{{
			Variant: SemanticsSingleEventStatemate,
			Final:   cloneRuntime(runtime),
		}}
		for _, event := range permutation {
			var next []*ReconciledReaction
			for _, prefix := range current {
				if prefix.Diverged {
					next = append(next, prefix)
					continue
				}
				part, err := e.runStatemateSuperstep(prefix.Final, []string{event}, nil, 0)
				if err != nil {
					return nil, err
				}
				for _, suffix := range part {
					next = append(next, &ReconciledReaction{
						Variant:  SemanticsSingleEventStatemate,
						Steps:    append(cloneSteps(prefix.Steps), cloneSteps(suffix.Steps)...),
						Final:    cloneRuntime(suffix.Final),
						Diverged: suffix.Diverged,
					})
				}
			}
			current = dedupeReactions(next)
		}
		reactions = append(reactions, current...)
	}

	for _, reaction := range reactions {
		reaction.Variant = SemanticsSingleEventStatemate
	}
	return dedupeReactions(reactions), nil
}

func (e *reconciledEngine) runUML(runtime *ReconciledRuntime, externalEvents []string) ([]*ReconciledReaction, error) {
	var reactions []*ReconciledReaction
	for _, permutation := range uniquePermutations(externalEvents) {
		queued := cloneRuntime(runtime)
		queued.Queue = append(append([]string{}, runtime.Queue...), permutation...)
		part, err := e.runUMLQueue(queued, nil, 0)
		if err != nil {
			return nil, err
		}
		reactions = append(reactions, part...)
	}

	for _, reaction := range reactions {
		reaction.Variant = SemanticsUML
	}
	return dedupeReactions(reactions), nil
}

func (e *reconciledEngine) runUMLQueue(runtime *ReconciledRuntime, seen map[string]bool, depth int) ([]*ReconciledReaction, error) {
	if depth >= e.options.MaxSteps {
		return []*ReconciledReaction{{
			Diverged: true,
			Final:    cloneRuntime(runtime),
		}}, nil
	}

	if seen == nil {
		seen = make(map[string]bool)
	}
	key := umlKey(runtime.Configuration, runtime.Queue)
	if seen[key] {
		return []*ReconciledReaction{{
			Diverged: true,
			Final:    cloneRuntime(runtime),
		}}, nil
	}

	stable, err := e.isStableUML(runtime.Configuration)
	if err != nil {
		return nil, err
	}
	if stable && len(runtime.Queue) == 0 {
		return []*ReconciledReaction{{
			Final: cloneRuntime(runtime),
		}}, nil
	}

	nextSeen := cloneSeen(seen)
	nextSeen[key] = true

	var (
		input []string
		tail  []string
	)
	variant := SemanticsUML
	if stable {
		input = []string{runtime.Queue[0]}
		tail = append([]string{}, runtime.Queue[1:]...)
	} else {
		input = nil
		tail = append([]string{}, runtime.Queue...)
	}

	steps, err := e.validSteps(runtime.Configuration, runtime.Context, input, variant)
	if err != nil {
		return nil, err
	}

	var reactions []*ReconciledReaction
	for _, transitions := range steps {
		result, err := e.executeStep(runtime, input, transitions)
		if err != nil {
			return nil, err
		}

		for _, queue := range e.enqueueGenerated(tail, result.generated) {
			subRuntime := &ReconciledRuntime{
				Configuration: result.configuration,
				Context:       result.context,
				Queue:         queue,
			}
			subReactions, err := e.runUMLQueue(subRuntime, nextSeen, depth+1)
			if err != nil {
				return nil, err
			}
			for _, sub := range subReactions {
				reactions = append(reactions, &ReconciledReaction{
					Steps:    append([]*sc.Step{result.step}, cloneSteps(sub.Steps)...),
					Final:    cloneRuntime(sub.Final),
					Diverged: sub.Diverged,
				})
			}
		}
	}

	return dedupeReactions(reactions), nil
}

func (e *reconciledEngine) enqueueGenerated(existing, generated []string) [][]string {
	if len(generated) == 0 {
		return [][]string{append([]string{}, existing...)}
	}

	var queues [][]string
	for _, permutation := range uniquePermutations(generated) {
		if e.options.UMLInternalPriority {
			queue := append([]string{}, permutation...)
			queue = append(queue, existing...)
			queues = append(queues, queue)
			continue
		}

		queue := append([]string{}, existing...)
		queue = append(queue, permutation...)
		queues = append(queues, queue)
	}
	return dedupeQueues(queues)
}

func (e *reconciledEngine) validSteps(config *sc.Configuration, context *structpb.Struct, input []string, variant SemanticsVariant) ([][]*sc.Transition, error) {
	var (
		candidates []*sc.Transition
		err        error
	)
	switch variant {
	case SemanticsFixpoint:
		candidates, err = e.relevantTransitions(config, context)
	default:
		candidates, err = e.enabledTransitions(config, context, input)
	}
	if err != nil {
		return nil, err
	}

	if len(candidates) == 0 && len(input) == 0 && variant != SemanticsFixpoint {
		return nil, nil
	}

	var steps [][]*sc.Transition
	for _, subset := range transitionSubsets(candidates) {
		effectiveInput := uniqueSortedStrings(input)
		if variant == SemanticsFixpoint {
			effectiveInput = unionStrings(effectiveInput, e.generatedEvents(subset))
		}

		ok, err := e.isStep(subset, config, context, effectiveInput, variant)
		if err != nil {
			return nil, err
		}
		if ok {
			steps = append(steps, cloneTransitions(subset))
		}
	}

	if len(steps) == 0 {
		return nil, fmt.Errorf("no valid step for %s semantics", variant)
	}

	return dedupeTransitionSets(steps), nil
}

func (e *reconciledEngine) isStep(transitions []*sc.Transition, config *sc.Configuration, context *structpb.Struct, input []string, variant SemanticsVariant) (bool, error) {
	enabled, err := e.enabledTransitions(config, context, input)
	if err != nil {
		return false, err
	}
	enabledSet := transitionSet(enabled)
	for _, transition := range transitions {
		if !enabledSet[transition] {
			return false, nil
		}
	}

	consistent, err := e.transitionsConsistent(transitions)
	if err != nil || !consistent {
		return false, err
	}

	maximal, err := e.isMaximal(transitions, enabled)
	if err != nil || !maximal {
		return false, err
	}

	return e.validPriority(transitions, enabled, variant)
}

func (e *reconciledEngine) validPriority(step, enabled []*sc.Transition, variant SemanticsVariant) (bool, error) {
	stepSet := transitionSet(step)
	for _, inStep := range step {
		for _, candidate := range enabled {
			if stepSet[candidate] {
				continue
			}
			higher, err := e.higherPriority(candidate, inStep, variant)
			if err != nil {
				return false, err
			}
			if higher {
				return false, nil
			}
		}
	}
	return true, nil
}

func (e *reconciledEngine) higherPriority(left, right *sc.Transition, variant SemanticsVariant) (bool, error) {
	switch variant {
	case SemanticsFixpoint, SemanticsStatemate, SemanticsSingleEventStatemate:
		leftScope, err := e.scope(left)
		if err != nil {
			return false, err
		}
		rightScope, err := e.scope(right)
		if err != nil {
			return false, err
		}
		if leftScope == rightScope {
			return false, nil
		}
		return e.strictAncestor(leftScope, rightScope)
	case SemanticsUML:
		return e.sourcesNestedInside(left, right)
	default:
		return false, fmt.Errorf("unsupported semantics variant %q", variant)
	}
}

func (e *reconciledEngine) executeStep(runtime *ReconciledRuntime, input []string, transitions []*sc.Transition) (*stepResult, error) {
	newContext := e.statechart.cloneContext(runtime.Context)
	for _, transition := range transitions {
		for _, action := range transition.Actions {
			if isGeneratedEventAction(action) {
				continue
			}
			if err := e.statechart.ExecuteAction(action, newContext); err != nil {
				return nil, fmt.Errorf("execute action %s: %w", action.Label, err)
			}
		}
	}

	nextConfig, err := e.nextConfiguration(runtime.Configuration, transitions)
	if err != nil {
		return nil, err
	}

	step := &sc.Step{
		Events:                 eventsFromLabels(input),
		Transitions:            cloneTransitions(transitions),
		StartingConfiguration:  cloneConfiguration(runtime.Configuration),
		ResultingConfiguration: cloneConfiguration(nextConfig),
		Context:                e.statechart.cloneContext(newContext),
	}

	return &stepResult{
		step:          step,
		configuration: nextConfig,
		context:       newContext,
		generated:     e.generatedEvents(transitions),
	}, nil
}

func (e *reconciledEngine) nextConfiguration(config *sc.Configuration, transitions []*sc.Transition) (*sc.Configuration, error) {
	if len(transitions) == 0 {
		return cloneConfiguration(config), nil
	}

	current := configurationSet(config)
	for _, transition := range transitions {
		scope, err := e.scope(transition)
		if err != nil {
			return nil, err
		}
		descendants, err := e.statechart.ChildrenStar(scope)
		if err != nil {
			return nil, err
		}
		for _, descendant := range descendants {
			delete(current, string(descendant))
		}
	}

	for _, transition := range transitions {
		enters, err := e.enteredStates(transition)
		if err != nil {
			return nil, err
		}
		for _, state := range enters {
			current[string(state)] = true
		}
	}

	return e.configurationFromSet(current)
}

func (e *reconciledEngine) enteredStates(transition *sc.Transition) ([]StateLabel, error) {
	if hasHistoryTargets(transition) {
		return nil, fmt.Errorf("history pseudostates are not part of the paper's core syntax")
	}

	target := &sc.Configuration{
		States: stateRefs(transition.To),
	}
	completed, err := DefaultCompletion(e.statechart, target)
	if err != nil {
		return nil, err
	}

	scope, err := e.scope(transition)
	if err != nil {
		return nil, err
	}
	scopeDescendants, err := e.statechart.ChildrenStar(scope)
	if err != nil {
		return nil, err
	}
	allowed := make(map[string]bool, len(scopeDescendants))
	for _, state := range scopeDescendants {
		allowed[string(state)] = true
	}

	var entered []StateLabel
	for _, state := range completed.States {
		if allowed[state.Label] {
			entered = append(entered, StateLabel(state.Label))
		}
	}
	return entered, nil
}

func (e *reconciledEngine) scope(transition *sc.Transition) (StateLabel, error) {
	all := make([]StateLabel, 0, len(transition.From)+len(transition.To))
	for _, source := range transition.From {
		all = append(all, StateLabel(source))
	}
	for _, target := range transition.To {
		all = append(all, StateLabel(target))
	}

	lca, err := e.statechart.LeastCommonAncestor(all...)
	if err != nil {
		return "", err
	}

	scope := lca
	for {
		state, err := e.statechart.findState(scope)
		if err != nil {
			return "", err
		}
		if state.Type == sc.StateTypeNormal || scope == RootState {
			return scope, nil
		}
		parent, err := e.statechart.GetParent(scope)
		if err != nil {
			return "", err
		}
		if parent == nil {
			return "", nil
		}
		scope = StateLabel(parent.Label)
	}
}

func (e *reconciledEngine) relevantTransitions(config *sc.Configuration, context *structpb.Struct) ([]*sc.Transition, error) {
	active := configurationSet(config)
	var relevant []*sc.Transition
	for _, transition := range e.statechart.Transitions {
		if hasHistoryTargets(transition) {
			continue
		}
		if !allSourcesActive(transition, active) {
			continue
		}
		guardPasses, err := e.guardPasses(transition, context, "")
		if err != nil {
			return nil, err
		}
		if guardPasses {
			relevant = append(relevant, transition)
		}
	}
	return relevant, nil
}

func (e *reconciledEngine) enabledTransitions(config *sc.Configuration, context *structpb.Struct, input []string) ([]*sc.Transition, error) {
	active := configurationSet(config)
	inputSet := stringSet(input)

	var enabled []*sc.Transition
	for _, transition := range e.statechart.Transitions {
		if hasHistoryTargets(transition) {
			continue
		}
		if !allSourcesActive(transition, active) {
			continue
		}
		if transition.Event != "" && !inputSet[transition.Event] {
			continue
		}
		guardPasses, err := e.guardPasses(transition, context, transition.Event)
		if err != nil {
			return nil, err
		}
		if guardPasses {
			enabled = append(enabled, transition)
		}
	}
	return enabled, nil
}

func (e *reconciledEngine) guardPasses(transition *sc.Transition, context *structpb.Struct, event string) (bool, error) {
	var evt *sc.Event
	if event != "" {
		evt = &sc.Event{Label: event}
	}
	return e.statechart.EvaluateGuardWithTransition(transition.Guard, context, evt, transition)
}

func (e *reconciledEngine) transitionsConsistent(transitions []*sc.Transition) (bool, error) {
	for i := range transitions {
		for j := i + 1; j < len(transitions); j++ {
			ok, err := e.transitionsPairConsistent(transitions[i], transitions[j])
			if err != nil {
				return false, err
			}
			if !ok {
				return false, nil
			}
		}
	}
	return true, nil
}

func (e *reconciledEngine) transitionsPairConsistent(left, right *sc.Transition) (bool, error) {
	if left == right {
		return true, nil
	}
	leftScope, err := e.scope(left)
	if err != nil {
		return false, err
	}
	rightScope, err := e.scope(right)
	if err != nil {
		return false, err
	}
	return e.statechart.Orthogonal(leftScope, rightScope)
}

func (e *reconciledEngine) conflict(left, right *sc.Transition) (bool, error) {
	if left == right {
		return false, nil
	}
	var sources []StateLabel
	for _, source := range left.From {
		sources = append(sources, StateLabel(source))
	}
	for _, source := range right.From {
		sources = append(sources, StateLabel(source))
	}
	consistent, err := e.statechart.Consistent(sources...)
	if err != nil || !consistent {
		return false, err
	}

	leftScope, err := e.scope(left)
	if err != nil {
		return false, err
	}
	rightScope, err := e.scope(right)
	if err != nil {
		return false, err
	}
	return e.statechart.AncestrallyRelated(leftScope, rightScope)
}

func (e *reconciledEngine) isMaximal(step, enabled []*sc.Transition) (bool, error) {
	stepSet := transitionSet(step)
	for _, transition := range enabled {
		if stepSet[transition] {
			continue
		}
		withCandidate := append(cloneTransitions(step), transition)
		consistent, err := e.transitionsConsistent(withCandidate)
		if err != nil {
			return false, err
		}
		if consistent {
			return false, nil
		}
	}
	return true, nil
}

func (e *reconciledEngine) isStableStatemate(config *sc.Configuration, input []string) (bool, error) {
	if len(input) != 0 {
		return false, nil
	}
	enabled, err := e.enabledTransitions(config, nil, nil)
	if err != nil {
		return false, err
	}
	return len(enabled) == 0, nil
}

func (e *reconciledEngine) isStableUML(config *sc.Configuration) (bool, error) {
	enabled, err := e.enabledTransitions(config, nil, nil)
	if err != nil {
		return false, err
	}
	return len(enabled) == 0, nil
}

func (e *reconciledEngine) strictAncestor(ancestor, descendant StateLabel) (bool, error) {
	if ancestor == descendant {
		return false, nil
	}
	return e.statechart.Ancestor(ancestor, descendant)
}

func (e *reconciledEngine) sourcesNestedInside(inner, outer *sc.Transition) (bool, error) {
	strict := false
	for _, innerSource := range inner.From {
		nested := false
		for _, outerSource := range outer.From {
			descendant, err := e.statechart.Descendant(StateLabel(innerSource), StateLabel(outerSource))
			if err != nil {
				return false, err
			}
			if descendant {
				nested = true
				if innerSource != outerSource {
					strict = true
				}
				break
			}
		}
		if !nested {
			return false, nil
		}
	}
	return strict, nil
}

func (e *reconciledEngine) generatedEvents(transitions []*sc.Transition) []string {
	generated := make(map[string]bool)
	for _, transition := range transitions {
		for _, event := range generatedEventsForTransition(transition) {
			generated[event] = true
		}
	}

	var events []string
	for event := range generated {
		events = append(events, event)
	}
	sort.Strings(events)
	return events
}

func (s *Statechart) normalizedRuntime(runtime *ReconciledRuntime) (*ReconciledRuntime, error) {
	if runtime == nil {
		initial, err := s.InitialConfiguration()
		if err != nil {
			return nil, err
		}
		return &ReconciledRuntime{
			Configuration: initial,
			Context: &structpb.Struct{
				Fields: make(map[string]*structpb.Value),
			},
		}, nil
	}

	config := cloneConfiguration(runtime.Configuration)
	if config == nil {
		initial, err := s.InitialConfiguration()
		if err != nil {
			return nil, err
		}
		config = initial
	}
	if _, err := DefaultCompletion(s, config); err != nil {
		return nil, err
	}

	context := s.cloneContext(runtime.Context)
	if context == nil {
		context = &structpb.Struct{Fields: make(map[string]*structpb.Value)}
	}

	return &ReconciledRuntime{
		Configuration: config,
		Context:       context,
		Queue:         append([]string{}, runtime.Queue...),
	}, nil
}

func (e *reconciledEngine) configurationFromSet(stateSet map[string]bool) (*sc.Configuration, error) {
	var states []*sc.StateRef
	for label := range stateSet {
		states = append(states, &sc.StateRef{Label: label})
	}

	sorted, err := TopologicalSort(e.statechart, states)
	if err != nil {
		return nil, err
	}
	return &sc.Configuration{States: sorted}, nil
}

type stepResult struct {
	step          *sc.Step
	configuration *sc.Configuration
	context       *structpb.Struct
	generated     []string
}

func cloneRuntime(runtime *ReconciledRuntime) *ReconciledRuntime {
	if runtime == nil {
		return nil
	}
	return &ReconciledRuntime{
		Configuration: cloneConfiguration(runtime.Configuration),
		Context:       cloneContextValue(runtime.Context),
		Queue:         append([]string{}, runtime.Queue...),
	}
}

func cloneContextValue(context *structpb.Struct) *structpb.Struct {
	if context == nil {
		return nil
	}
	fields := make(map[string]*structpb.Value, len(context.Fields))
	for key, value := range context.Fields {
		fields[key] = value
	}
	return &structpb.Struct{Fields: fields}
}

func cloneConfiguration(config *sc.Configuration) *sc.Configuration {
	if config == nil {
		return nil
	}
	states := make([]*sc.StateRef, 0, len(config.States))
	for _, state := range config.States {
		states = append(states, &sc.StateRef{Label: state.Label})
	}
	history := make(map[string]*sc.Configuration, len(config.History))
	for key, value := range config.History {
		history[key] = cloneConfiguration(value)
	}
	return &sc.Configuration{
		States:  states,
		History: history,
	}
}

func cloneTransitions(transitions []*sc.Transition) []*sc.Transition {
	if len(transitions) == 0 {
		return nil
	}
	out := make([]*sc.Transition, len(transitions))
	copy(out, transitions)
	return out
}

func cloneSteps(steps []*sc.Step) []*sc.Step {
	if len(steps) == 0 {
		return nil
	}
	out := make([]*sc.Step, len(steps))
	copy(out, steps)
	return out
}

func allSourcesActive(transition *sc.Transition, active map[string]bool) bool {
	if len(transition.From) == 0 {
		return false
	}
	for _, source := range transition.From {
		if !active[source] {
			return false
		}
	}
	return true
}

func configurationSet(config *sc.Configuration) map[string]bool {
	set := make(map[string]bool)
	if config == nil {
		return set
	}
	for _, state := range config.States {
		set[state.Label] = true
	}
	return set
}

func transitionSet(transitions []*sc.Transition) map[*sc.Transition]bool {
	set := make(map[*sc.Transition]bool, len(transitions))
	for _, transition := range transitions {
		set[transition] = true
	}
	return set
}

func stateRefs(labels []string) []*sc.StateRef {
	refs := make([]*sc.StateRef, 0, len(labels))
	for _, label := range labels {
		refs = append(refs, &sc.StateRef{Label: label})
	}
	return refs
}

func eventsFromLabels(labels []string) []*sc.Event {
	events := make([]*sc.Event, 0, len(labels))
	for _, label := range uniqueSortedStrings(labels) {
		events = append(events, &sc.Event{Label: label})
	}
	return events
}

func generatedEventsForTransition(transition *sc.Transition) []string {
	set := make(map[string]bool)
	for _, action := range transition.Actions {
		label := action.GetLabel()
		for _, prefix := range []string{"raise:", "emit:", "send:"} {
			if !strings.HasPrefix(label, prefix) {
				continue
			}
			event := strings.TrimSpace(strings.TrimPrefix(label, prefix))
			if event != "" {
				set[event] = true
			}
			break
		}
	}

	var events []string
	for event := range set {
		events = append(events, event)
	}
	sort.Strings(events)
	return events
}

func isGeneratedEventAction(action *sc.Action) bool {
	if action == nil {
		return false
	}
	for _, prefix := range []string{"raise:", "emit:", "send:"} {
		if strings.HasPrefix(action.Label, prefix) {
			return true
		}
	}
	return false
}

func hasHistoryTargets(transition *sc.Transition) bool {
	for _, target := range transition.To {
		if target == "" {
			return true
		}
	}
	return false
}

func transitionSubsets(transitions []*sc.Transition) [][]*sc.Transition {
	if len(transitions) == 0 {
		return [][]*sc.Transition{{}}
	}

	limit := 1 << len(transitions)
	subsets := make([][]*sc.Transition, 0, limit)
	for mask := 0; mask < limit; mask++ {
		var subset []*sc.Transition
		for i, transition := range transitions {
			if mask&(1<<i) != 0 {
				subset = append(subset, transition)
			}
		}
		subsets = append(subsets, subset)
	}
	return subsets
}

func uniqueSortedStrings(values []string) []string {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		set[value] = true
	}
	var out []string
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func unionStrings(left, right []string) []string {
	return uniqueSortedStrings(append(append([]string{}, left...), right...))
}

func stringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}

func uniquePermutations(values []string) [][]string {
	sorted := append([]string{}, values...)
	sort.Strings(sorted)

	var (
		out  [][]string
		path []string
		used = make([]bool, len(sorted))
	)

	var walk func()
	walk = func() {
		if len(path) == len(sorted) {
			out = append(out, append([]string{}, path...))
			return
		}
		for i := range sorted {
			if used[i] {
				continue
			}
			if i > 0 && sorted[i] == sorted[i-1] && !used[i-1] {
				continue
			}
			used[i] = true
			path = append(path, sorted[i])
			walk()
			path = path[:len(path)-1]
			used[i] = false
		}
	}

	walk()
	if len(out) == 0 {
		return [][]string{{}}
	}
	return out
}

func statemateKey(config *sc.Configuration, input []string) string {
	return fmt.Sprintf("%s|%s", configKey(config), strings.Join(uniqueSortedStrings(input), ","))
}

func umlKey(config *sc.Configuration, queue []string) string {
	return fmt.Sprintf("%s|%s", configKey(config), strings.Join(queue, ","))
}

func configKey(config *sc.Configuration) string {
	var labels []string
	if config != nil {
		for _, state := range config.States {
			labels = append(labels, state.Label)
		}
	}
	sort.Strings(labels)
	return strings.Join(labels, ",")
}

func cloneSeen(seen map[string]bool) map[string]bool {
	out := make(map[string]bool, len(seen))
	for key, value := range seen {
		out[key] = value
	}
	return out
}

func dedupeReactions(reactions []*ReconciledReaction) []*ReconciledReaction {
	seen := make(map[string]bool)
	var out []*ReconciledReaction
	for _, reaction := range reactions {
		key := reactionKey(reaction)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, reaction)
	}
	return out
}

func dedupeTransitionSets(steps [][]*sc.Transition) [][]*sc.Transition {
	seen := make(map[string]bool)
	var out [][]*sc.Transition
	for _, step := range steps {
		key := transitionSetKey(step)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, step)
	}
	return out
}

func dedupeQueues(queues [][]string) [][]string {
	seen := make(map[string]bool)
	var out [][]string
	for _, queue := range queues {
		key := strings.Join(queue, ",")
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, queue)
	}
	return out
}

func reactionKey(reaction *ReconciledReaction) string {
	var steps []string
	for _, step := range reaction.Steps {
		steps = append(steps, transitionSetKey(step.Transitions))
	}
	queue := ""
	finalConfig := ""
	if reaction.Final != nil {
		queue = strings.Join(reaction.Final.Queue, ",")
		finalConfig = configKey(reaction.Final.Configuration)
	}
	return fmt.Sprintf("%s|%t|%s|%s", strings.Join(steps, "/"), reaction.Diverged, finalConfig, queue)
}

func transitionSetKey(transitions []*sc.Transition) string {
	var labels []string
	for _, transition := range transitions {
		labels = append(labels, transition.Label)
	}
	sort.Strings(labels)
	return strings.Join(labels, ",")
}
