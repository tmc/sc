package semantics

import (
	"fmt"
	"sort"
	"sync"

	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
	"google.golang.org/protobuf/types/known/structpb"
)

// TransitionStep represents a single step in the transition execution process.
type TransitionStep struct {
	Transitions         []*sc.Transition
	SourceConfiguration *sc.Configuration
	TargetConfiguration *sc.Configuration
	ExitedStates        []StateLabel
	EnteredStates       []StateLabel
	ExecutedActions     []*sc.Action
}

// TransitionResult represents the result of executing transitions.
type TransitionResult struct {
	Executed    bool
	Steps       []*TransitionStep
	NewConfig   *sc.Configuration
	NewContext  *structpb.Struct
	Error       error
}

// ExecuteTransitions executes all enabled transitions for a given event.
// This implements the complete transition execution algorithm following Harel's semantics.
func (s *Statechart) ExecuteTransitions(config *sc.Configuration, context *structpb.Struct, event string) (*TransitionResult, error) {
	if config == nil {
		return nil, fmt.Errorf("configuration cannot be nil")
	}

	// Find all enabled transitions for the event
	enabledTransitions, err := s.FindEnabledTransitions(config, context, event)
	if err != nil {
		return nil, fmt.Errorf("failed to find enabled transitions: %w", err)
	}

	if len(enabledTransitions) == 0 {
		return &TransitionResult{
			Executed:   false,
			NewConfig:  config,
			NewContext: context,
		}, nil
	}

	// Resolve conflicts among enabled transitions
	selectedTransitions, err := s.ResolveConflicts(enabledTransitions, config)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve transition conflicts: %w", err)
	}

	if len(selectedTransitions) == 0 {
		return &TransitionResult{
			Executed:   false,
			NewConfig:  config,
			NewContext: context,
		}, nil
	}

	// Execute the selected transitions
	result, err := s.executeSelectedTransitions(selectedTransitions, config, context)
	if err != nil {
		return nil, fmt.Errorf("failed to execute transitions: %w", err)
	}

	return result, nil
}

// FindEnabledTransitions finds all transitions that are enabled for the given event and configuration.
func (s *Statechart) FindEnabledTransitions(config *sc.Configuration, context *structpb.Struct, event string) ([]*sc.Transition, error) {
	var enabled []*sc.Transition

	// Get active states from configuration
	activeStates := make(map[string]bool)
	for _, stateRef := range config.States {
		if stateRef != nil {
			activeStates[stateRef.Label] = true
		}
	}

	// Create event object for guard evaluation
	eventObj := &sc.Event{
		Label: event,
		Parameters: &structpb.Struct{
			Fields: make(map[string]*structpb.Value),
		},
	}

	// Check each transition
	for _, transition := range s.Transitions {
		if transition.Event != event {
			continue
		}

		// Check if transition source is active
		isSourceActive := false
		for _, sourceLabel := range transition.From {
			if activeStates[sourceLabel] {
				isSourceActive = true
				break
			}
		}

		if !isSourceActive {
			continue
		}

		// Evaluate guard condition with full context
		guardPasses, err := s.EvaluateGuardWithTransition(transition.Guard, context, eventObj, transition)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate guard for transition %s: %w", transition.Label, err)
		}

		if guardPasses {
			enabled = append(enabled, transition)
		}
	}

	return enabled, nil
}

// ResolveConflicts resolves conflicts among enabled transitions.
// This implements priority-based conflict resolution following Harel's semantics.
func (s *Statechart) ResolveConflicts(transitions []*sc.Transition, config *sc.Configuration) ([]*sc.Transition, error) {
	if len(transitions) <= 1 {
		return transitions, nil
	}

	// Group transitions by their source states for conflict analysis
	var nonConflictingTransitions []*sc.Transition

	// First, we need to identify which transitions are actually in conflict
	// Transitions are in conflict if they share source states or if their execution
	// would result in an inconsistent configuration
	conflicts := s.identifyConflicts(transitions, config)

	if len(conflicts) == 0 {
		// No conflicts, all transitions can be executed
		return transitions, nil
	}

	// Resolve conflicts using priority rules:
	// 1. Transitions from deeper states have higher priority (inner-to-outer precedence)
	// 2. Among transitions at the same level, use source state ordering
	resolved := s.resolveUsingPriority(transitions, conflicts)

	for _, transition := range resolved {
		nonConflictingTransitions = append(nonConflictingTransitions, transition)
	}

	return nonConflictingTransitions, nil
}

// identifyConflicts identifies conflicts between transitions.
func (s *Statechart) identifyConflicts(transitions []*sc.Transition, config *sc.Configuration) [][]int {
	var conflicts [][]int

	for i := 0; i < len(transitions); i++ {
		for j := i + 1; j < len(transitions); j++ {
			if s.areInConflict(transitions[i], transitions[j], config) {
				conflicts = append(conflicts, []int{i, j})
			}
		}
	}

	return conflicts
}

// areInConflict determines if two transitions are in conflict.
func (s *Statechart) areInConflict(t1, t2 *sc.Transition, config *sc.Configuration) bool {
	// Check if transitions share source states
	for _, src1 := range t1.From {
		for _, src2 := range t2.From {
			if src1 == src2 {
				return true
			}
		}
	}

	// Check if source states are ancestrally related
	for _, src1 := range t1.From {
		for _, src2 := range t2.From {
			related, err := s.AncestrallyRelated(StateLabel(src1), StateLabel(src2))
			if err == nil && related {
				return true
			}
		}
	}

	return false
}

// resolveUsingPriority resolves conflicts using priority rules.
func (s *Statechart) resolveUsingPriority(transitions []*sc.Transition, conflicts [][]int) []*sc.Transition {
	if len(conflicts) == 0 {
		return transitions
	}

	// Create priority scores for each transition
	priorities := make([]int, len(transitions))
	for i, transition := range transitions {
		priorities[i] = s.calculateTransitionPriority(transition)
	}

	// Mark transitions to exclude based on conflicts
	excluded := make(map[int]bool)
	for _, conflict := range conflicts {
		i, j := conflict[0], conflict[1]
		if priorities[i] > priorities[j] {
			excluded[j] = true
		} else if priorities[j] > priorities[i] {
			excluded[i] = true
		} else {
			// Same priority, use lexicographic ordering of source states
			if s.compareLexicographically(transitions[i], transitions[j]) < 0 {
				excluded[j] = true
			} else {
				excluded[i] = true
			}
		}
	}

	// Return non-excluded transitions
	var result []*sc.Transition
	for i, transition := range transitions {
		if !excluded[i] {
			result = append(result, transition)
		}
	}

	return result
}

// calculateTransitionPriority calculates the priority of a transition.
// Higher values indicate higher priority.
func (s *Statechart) calculateTransitionPriority(transition *sc.Transition) int {
	maxDepth := 0
	for _, sourceLabel := range transition.From {
		depth := s.getStateDepth(StateLabel(sourceLabel))
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	return maxDepth
}

// getStateDepth calculates the depth of a state in the hierarchy.
func (s *Statechart) getStateDepth(state StateLabel) int {
	depth := 0
	currentState := state

	for currentState != RootState {
		parent, err := s.GetParent(currentState)
		if err != nil {
			break
		}
		depth++
		currentState = StateLabel(parent.Label)
	}

	return depth
}

// compareLexicographically compares two transitions lexicographically.
func (s *Statechart) compareLexicographically(t1, t2 *sc.Transition) int {
	// Sort source states and compare
	sources1 := make([]string, len(t1.From))
	copy(sources1, t1.From)
	sort.Strings(sources1)

	sources2 := make([]string, len(t2.From))
	copy(sources2, t2.From)
	sort.Strings(sources2)

	for i := 0; i < len(sources1) && i < len(sources2); i++ {
		if sources1[i] < sources2[i] {
			return -1
		} else if sources1[i] > sources2[i] {
			return 1
		}
	}

	if len(sources1) < len(sources2) {
		return -1
	} else if len(sources1) > len(sources2) {
		return 1
	}

	return 0
}

// executeSelectedTransitions executes the selected transitions.
func (s *Statechart) executeSelectedTransitions(transitions []*sc.Transition, config *sc.Configuration, context *structpb.Struct) (*TransitionResult, error) {
	// Calculate the scope of the transition
	scope, err := s.calculateTransitionScope(transitions, config)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate transition scope: %w", err)
	}

	// Determine states to exit and enter
	exitStates, enterStates, err := s.calculateStateChanges(transitions, config, scope)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate state changes: %w", err)
	}

	// Create new context for action execution
	newContext := s.cloneContext(context)

	// Execute exit actions
	for _, stateLabel := range exitStates {
		if err := s.executeExitActions(stateLabel, newContext); err != nil {
			return nil, fmt.Errorf("failed to execute exit actions for state %s: %w", stateLabel, err)
		}
	}

	// Execute transition actions
	var executedActions []*sc.Action
	for _, transition := range transitions {
		for _, action := range transition.Actions {
			if err := s.ExecuteAction(action, newContext); err != nil {
				return nil, fmt.Errorf("failed to execute transition action %s: %w", action.Label, err)
			}
			executedActions = append(executedActions, action)
		}
	}

	// Execute entry actions
	for _, stateLabel := range enterStates {
		if err := s.executeEntryActions(stateLabel, newContext); err != nil {
			return nil, fmt.Errorf("failed to execute entry actions for state %s: %w", stateLabel, err)
		}
	}

	// Calculate new configuration
	newConfig, err := s.calculateNewConfiguration(config, exitStates, enterStates)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate new configuration: %w", err)
	}

	// Complete the configuration with default states
	completedConfig, err := DefaultCompletion(s, newConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to complete configuration: %w", err)
	}

	// Filter out the root state from the completed configuration
	var filteredStates []*sc.StateRef
	for _, state := range completedConfig.States {
		if state.Label != s.RootState.Label {
			filteredStates = append(filteredStates, state)
		}
	}
	completedConfig = &sc.Configuration{States: filteredStates}

	step := &TransitionStep{
		Transitions:         transitions,
		SourceConfiguration: config,
		TargetConfiguration: completedConfig,
		ExitedStates:        exitStates,
		EnteredStates:       enterStates,
		ExecutedActions:     executedActions,
	}

	return &TransitionResult{
		Executed:   true,
		Steps:      []*TransitionStep{step},
		NewConfig:  completedConfig,
		NewContext: newContext,
	}, nil
}

// calculateTransitionScope calculates the scope of the transition execution.
func (s *Statechart) calculateTransitionScope(transitions []*sc.Transition, config *sc.Configuration) (StateLabel, error) {
	var allStates []StateLabel

	// Collect all source and target states
	for _, transition := range transitions {
		for _, source := range transition.From {
			allStates = append(allStates, StateLabel(source))
		}
		for _, target := range transition.To {
			allStates = append(allStates, StateLabel(target))
		}
	}

	// Find the least common ancestor of all involved states
	lca, err := s.LeastCommonAncestor(allStates...)
	if err != nil {
		return "", fmt.Errorf("failed to find least common ancestor: %w", err)
	}

	return lca, nil
}

// calculateStateChanges determines which states to exit and enter.
func (s *Statechart) calculateStateChanges(transitions []*sc.Transition, config *sc.Configuration, scope StateLabel) ([]StateLabel, []StateLabel, error) {
	// Get current active states
	activeStates := make(map[StateLabel]bool)
	for _, stateRef := range config.States {
		if stateRef != nil {
			activeStates[StateLabel(stateRef.Label)] = true
		}
	}

	// Determine which states are sources of transitions
	sourceStates := make(map[StateLabel]bool)
	for _, transition := range transitions {
		for _, source := range transition.From {
			sourceStates[StateLabel(source)] = true
		}
	}

	// Calculate states to exit (all states that are active and descendants of sources up to scope)
	var exitStates []StateLabel
	for activeState := range activeStates {
		shouldExit := false
		
		// Check if this state is a source state or descendant of a source state
		for sourceState := range sourceStates {
			if activeState == sourceState {
				shouldExit = true
				break
			}
			
			descendant, err := s.Descendant(activeState, sourceState)
			if err == nil && descendant {
				shouldExit = true
				break
			}
		}

		if shouldExit {
			// Check if state is within scope
			withinScope, err := s.Descendant(activeState, scope)
			if err == nil && (withinScope || activeState == scope) {
				exitStates = append(exitStates, activeState)
			}
		}
	}

	// Calculate states to enter (all target states and their default completions)
	var enterStates []StateLabel
	targetStates := make(map[StateLabel]bool)
	
	for _, transition := range transitions {
		for _, target := range transition.To {
			targetStates[StateLabel(target)] = true
		}
	}

	// Add target states
	for targetState := range targetStates {
		enterStates = append(enterStates, targetState)
	}

	// Sort states for deterministic behavior
	sort.Slice(exitStates, func(i, j int) bool {
		return string(exitStates[i]) < string(exitStates[j])
	})
	sort.Slice(enterStates, func(i, j int) bool {
		return string(enterStates[i]) < string(enterStates[j])
	})

	return exitStates, enterStates, nil
}

// calculateNewConfiguration calculates the new configuration after state changes.
func (s *Statechart) calculateNewConfiguration(oldConfig *sc.Configuration, exitStates, enterStates []StateLabel) (*sc.Configuration, error) {
	// Start with current configuration
	newStates := make(map[string]bool)
	for _, stateRef := range oldConfig.States {
		if stateRef != nil {
			newStates[stateRef.Label] = true
		}
	}

	// Remove exited states
	for _, exitState := range exitStates {
		delete(newStates, string(exitState))
	}

	// Add entered states
	for _, enterState := range enterStates {
		newStates[string(enterState)] = true
	}

	// Convert back to StateRef slice
	var stateRefs []*sc.StateRef
	for label := range newStates {
		stateRefs = append(stateRefs, &sc.StateRef{Label: label})
	}

	return &sc.Configuration{States: stateRefs}, nil
}

// EvaluateGuard evaluates a guard condition using the comprehensive guard evaluator.
func (s *Statechart) EvaluateGuard(guard *sc.Guard, context *structpb.Struct) (bool, error) {
	return s.EvaluateGuardWithEvent(guard, context, nil)
}

// EvaluateGuardWithEvent evaluates a guard condition with event context.
func (s *Statechart) EvaluateGuardWithEvent(guard *sc.Guard, context *structpb.Struct, event *sc.Event) (bool, error) {
	if guard == nil || guard.Expression == "" {
		return true, nil // No guard means always true
	}

	// Create evaluation context
	evalContext := &EvaluationContext{
		Variables: context,
		Event:     event,
		StateData: make(map[string]interface{}),
	}

	// Use the global guard evaluator
	result, err := globalGuardEvaluator.EvaluateGuard(guard, evalContext)
	if err != nil {
		return false, fmt.Errorf("guard evaluation failed: %w", err)
	}

	return result.Value, nil
}

// EvaluateGuardWithTransition evaluates a guard condition with full transition context.
func (s *Statechart) EvaluateGuardWithTransition(guard *sc.Guard, context *structpb.Struct, event *sc.Event, transition *sc.Transition) (bool, error) {
	if guard == nil || guard.Expression == "" {
		return true, nil // No guard means always true
	}

	// Create evaluation context with full information
	evalContext := &EvaluationContext{
		Variables:  context,
		Event:      event,
		StateData:  make(map[string]interface{}),
		Transition: transition,
	}

	// Add transition-specific state data
	if transition != nil {
		evalContext.StateData["sourceStates"] = transition.From
		evalContext.StateData["targetStates"] = transition.To
		evalContext.StateData["transitionLabel"] = transition.Label
	}

	// Use the global guard evaluator
	result, err := globalGuardEvaluator.EvaluateGuard(guard, evalContext)
	if err != nil {
		return false, fmt.Errorf("guard evaluation failed: %w", err)
	}

	return result.Value, nil
}

// evaluateGuardExpression provides backward compatibility.
// This is deprecated in favor of the new guard evaluation system.
func (s *Statechart) evaluateGuardExpression(expression string, context *structpb.Struct) (bool, error) {
	return EvaluateGuardExpression(expression, context)
}

// ExecuteAction executes an action.
func (s *Statechart) ExecuteAction(action *sc.Action, context *structpb.Struct) error {
	if action == nil {
		return nil
	}

	// This is a simplified action execution
	// In a production system, you would have a proper action registry
	return s.executeActionByLabel(action.Label, context)
}

// ActionExecutor represents a function that can execute an action
type ActionExecutor func(context *structpb.Struct) error

// ActionRegistry maintains a registry of action executors
type ActionRegistry struct {
	executors map[string]ActionExecutor
	mu        sync.RWMutex
}

// NewActionRegistry creates a new action registry with default actions
func NewActionRegistry() *ActionRegistry {
	registry := &ActionRegistry{
		executors: make(map[string]ActionExecutor),
	}
	
	// Register default actions
	registry.RegisterAction("increment_count", func(context *structpb.Struct) error {
		if context != nil && context.Fields != nil {
			if countValue, exists := context.Fields["count"]; exists {
				if count, ok := countValue.GetKind().(*structpb.Value_NumberValue); ok {
					context.Fields["count"] = structpb.NewNumberValue(count.NumberValue + 1)
				}
			}
		}
		return nil
	})
	
	registry.RegisterAction("decrement_count", func(context *structpb.Struct) error {
		if context != nil && context.Fields != nil {
			if countValue, exists := context.Fields["count"]; exists {
				if count, ok := countValue.GetKind().(*structpb.Value_NumberValue); ok {
					context.Fields["count"] = structpb.NewNumberValue(count.NumberValue - 1)
				}
			}
		}
		return nil
	})
	
	registry.RegisterAction("reset_count", func(context *structpb.Struct) error {
		if context != nil && context.Fields != nil {
			context.Fields["count"] = structpb.NewNumberValue(0)
		}
		return nil
	})
	
	// Register some example entry/exit actions
	registry.RegisterAction("entry_On", func(context *structpb.Struct) error {
		if context != nil && context.Fields != nil {
			context.Fields["on_entered"] = structpb.NewBoolValue(true)
		}
		return nil
	})
	
	registry.RegisterAction("exit_Off", func(context *structpb.Struct) error {
		if context != nil && context.Fields != nil {
			context.Fields["off_exited"] = structpb.NewBoolValue(true)
		}
		return nil
	})
	
	return registry
}

// RegisterAction registers an action executor for a given label
func (ar *ActionRegistry) RegisterAction(label string, executor ActionExecutor) {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	ar.executors[label] = executor
}

// ExecuteAction executes an action by its label
func (ar *ActionRegistry) ExecuteAction(label string, context *structpb.Struct) error {
	ar.mu.RLock()
	executor, exists := ar.executors[label]
	ar.mu.RUnlock()
	
	if !exists {
		// Unknown action - could log this or just ignore
		return nil
	}
	
	return executor(context)
}

// Global action registry for statechart operations
var globalActionRegistry = NewActionRegistry()

// executeActionByLabel executes an action by its label using the global registry.
func (s *Statechart) executeActionByLabel(label string, context *structpb.Struct) error {
	return globalActionRegistry.ExecuteAction(label, context)
}

// RegisterGlobalAction registers an action in the global action registry
func RegisterGlobalAction(label string, executor ActionExecutor) {
	globalActionRegistry.RegisterAction(label, executor)
}

// SetActionRegistry sets a custom action registry for a statechart
// Note: This would require adding an actionRegistry field to the Statechart struct
func (s *Statechart) SetActionRegistry(registry *ActionRegistry) {
	// TODO: This would require extending the Statechart struct to have an actionRegistry field
	// For now, we use the global registry
}

// executeExitActions executes exit actions for a state.
func (s *Statechart) executeExitActions(state StateLabel, context *structpb.Struct) error {
	// Execute convention-based exit actions
	exitActionLabel := "exit_" + string(state)
	return s.executeActionByLabel(exitActionLabel, context)
}

// executeEntryActions executes entry actions for a state.
func (s *Statechart) executeEntryActions(state StateLabel, context *structpb.Struct) error {
	// Execute convention-based entry actions
	entryActionLabel := "entry_" + string(state)
	return s.executeActionByLabel(entryActionLabel, context)
}

// cloneContext creates a deep copy of the context.
func (s *Statechart) cloneContext(context *structpb.Struct) *structpb.Struct {
	if context == nil {
		return nil
	}

	newContext := &structpb.Struct{
		Fields: make(map[string]*structpb.Value),
	}

	for key, value := range context.Fields {
		// Simple cloning - in production you'd need deep cloning
		newContext.Fields[key] = value
	}

	return newContext
}

// IsTransitionEnabled checks if a specific transition is enabled.
func (s *Statechart) IsTransitionEnabled(transition *sc.Transition, config *sc.Configuration, context *structpb.Struct) (bool, error) {
	return s.IsTransitionEnabledWithEvent(transition, config, context, nil)
}

// IsTransitionEnabledWithEvent checks if a specific transition is enabled with event context.
func (s *Statechart) IsTransitionEnabledWithEvent(transition *sc.Transition, config *sc.Configuration, context *structpb.Struct, event *sc.Event) (bool, error) {
	if transition == nil || config == nil {
		return false, fmt.Errorf("transition and configuration cannot be nil")
	}

	// Check if any source state is active
	activeStates := make(map[string]bool)
	for _, stateRef := range config.States {
		if stateRef != nil {
			activeStates[stateRef.Label] = true
		}
	}

	sourceActive := false
	for _, source := range transition.From {
		if activeStates[source] {
			sourceActive = true
			break
		}
	}

	if !sourceActive {
		return false, nil
	}

	// Evaluate guard with full context
	return s.EvaluateGuardWithTransition(transition.Guard, context, event, transition)
}

// GetTransitionsByEvent returns all transitions triggered by a specific event.
func (s *Statechart) GetTransitionsByEvent(event string) []*sc.Transition {
	var result []*sc.Transition
	for _, transition := range s.Transitions {
		if transition.Event == event {
			result = append(result, transition)
		}
	}
	return result
}

// GetTransitionsFromState returns all transitions originating from a specific state.
func (s *Statechart) GetTransitionsFromState(state StateLabel) []*sc.Transition {
	var result []*sc.Transition
	stateStr := string(state)
	
	for _, transition := range s.Transitions {
		if slices.Contains(transition.From, stateStr) {
			result = append(result, transition)
		}
	}
	return result
}

// GetTransitionsToState returns all transitions targeting a specific state.
func (s *Statechart) GetTransitionsToState(state StateLabel) []*sc.Transition {
	var result []*sc.Transition
	stateStr := string(state)
	
	for _, transition := range s.Transitions {
		if slices.Contains(transition.To, stateStr) {
			result = append(result, transition)
		}
	}
	return result
}

// CompoundTransition represents a compound transition that consists of multiple atomic transitions.
type CompoundTransition struct {
	Label       string
	Transitions []*sc.Transition
	Event       string
	Guard       *sc.Guard
	Actions     []*sc.Action
}

// ExecuteCompoundTransition executes a compound transition atomically.
func (s *Statechart) ExecuteCompoundTransition(compound *CompoundTransition, config *sc.Configuration, context *structpb.Struct) (*TransitionResult, error) {
	if compound == nil || len(compound.Transitions) == 0 {
		return &TransitionResult{
			Executed:   false,
			NewConfig:  config,
			NewContext: context,
		}, nil
	}

	// Check if compound guard passes
	if compound.Guard != nil {
		guardPasses, err := s.EvaluateGuard(compound.Guard, context)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate compound guard: %w", err)
		}
		if !guardPasses {
			return &TransitionResult{
				Executed:   false,
				NewConfig:  config,
				NewContext: context,
			}, nil
		}
	}

	// Check that all constituent transitions are enabled
	for _, transition := range compound.Transitions {
		enabled, err := s.IsTransitionEnabled(transition, config, context)
		if err != nil {
			return nil, fmt.Errorf("failed to check if transition %s is enabled: %w", transition.Label, err)
		}
		if !enabled {
			return &TransitionResult{
				Executed:   false,
				NewConfig:  config,
				NewContext: context,
			}, nil
		}
	}

	// Execute all transitions atomically
	result, err := s.executeSelectedTransitions(compound.Transitions, config, context)
	if err != nil {
		return nil, fmt.Errorf("failed to execute compound transition: %w", err)
	}

	// Execute compound actions after transition actions
	if result.Executed && len(compound.Actions) > 0 {
		for _, action := range compound.Actions {
			if err := s.ExecuteAction(action, result.NewContext); err != nil {
				return nil, fmt.Errorf("failed to execute compound action %s: %w", action.Label, err)
			}
		}
	}

	return result, nil
}

// CrossRegionTransition represents a transition that crosses orthogonal regions.
type CrossRegionTransition struct {
	Label       string
	Sources     map[string]string // region -> source state mapping
	Targets     map[string]string // region -> target state mapping
	Event       string
	Guard       *sc.Guard
	Actions     []*sc.Action
}

// ExecuteCrossRegionTransition executes a transition across multiple orthogonal regions.
func (s *Statechart) ExecuteCrossRegionTransition(crossTransition *CrossRegionTransition, config *sc.Configuration, context *structpb.Struct) (*TransitionResult, error) {
	if crossTransition == nil {
		return &TransitionResult{
			Executed:   false,
			NewConfig:  config,
			NewContext: context,
		}, nil
	}

	// Check guard
	if crossTransition.Guard != nil {
		guardPasses, err := s.EvaluateGuard(crossTransition.Guard, context)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate cross-region guard: %w", err)
		}
		if !guardPasses {
			return &TransitionResult{
				Executed:   false,
				NewConfig:  config,
				NewContext: context,
			}, nil
		}
	}

	// Verify that all source states are active
	activeStates := make(map[string]bool)
	for _, stateRef := range config.States {
		if stateRef != nil {
			activeStates[stateRef.Label] = true
		}
	}

	for _, sourceState := range crossTransition.Sources {
		if !activeStates[sourceState] {
			return &TransitionResult{
				Executed:   false,
				NewConfig:  config,
				NewContext: context,
			}, nil
		}
	}

	// Calculate the scope and state changes for cross-region transition
	var exitStates, enterStates []StateLabel
	
	// Collect all states to exit and enter
	for _, sourceState := range crossTransition.Sources {
		exitStates = append(exitStates, StateLabel(sourceState))
	}
	for _, targetState := range crossTransition.Targets {
		enterStates = append(enterStates, StateLabel(targetState))
	}

	// Create new context for action execution
	newContext := s.cloneContext(context)

	// Execute exit actions for all exited states
	for _, stateLabel := range exitStates {
		if err := s.executeExitActions(stateLabel, newContext); err != nil {
			return nil, fmt.Errorf("failed to execute exit actions for state %s: %w", stateLabel, err)
		}
	}

	// Execute cross-region actions
	var executedActions []*sc.Action
	for _, action := range crossTransition.Actions {
		if err := s.ExecuteAction(action, newContext); err != nil {
			return nil, fmt.Errorf("failed to execute cross-region action %s: %w", action.Label, err)
		}
		executedActions = append(executedActions, action)
	}

	// Execute entry actions for all entered states
	for _, stateLabel := range enterStates {
		if err := s.executeEntryActions(stateLabel, newContext); err != nil {
			return nil, fmt.Errorf("failed to execute entry actions for state %s: %w", stateLabel, err)
		}
	}

	// Calculate new configuration
	newConfig, err := s.calculateNewConfiguration(config, exitStates, enterStates)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate new configuration: %w", err)
	}

	// Complete the configuration with default states
	completedConfig, err := DefaultCompletion(s, newConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to complete configuration: %w", err)
	}

	// Filter out the root state from the completed configuration
	var filteredStates []*sc.StateRef
	for _, state := range completedConfig.States {
		if state.Label != s.RootState.Label {
			filteredStates = append(filteredStates, state)
		}
	}
	completedConfig = &sc.Configuration{States: filteredStates}

	step := &TransitionStep{
		Transitions:         []*sc.Transition{}, // Cross-region transitions don't map directly to single transitions
		SourceConfiguration: config,
		TargetConfiguration: completedConfig,
		ExitedStates:        exitStates,
		EnteredStates:       enterStates,
		ExecutedActions:     executedActions,
	}

	return &TransitionResult{
		Executed:   true,
		Steps:      []*TransitionStep{step},
		NewConfig:  completedConfig,
		NewContext: newContext,
	}, nil
}

// IsOrthogonalTransition checks if a transition crosses orthogonal regions.
func (s *Statechart) IsOrthogonalTransition(transition *sc.Transition) (bool, error) {
	if len(transition.From) != 1 || len(transition.To) != 1 {
		return false, nil // Multi-source/target transitions are handled separately
	}

	sourceState := StateLabel(transition.From[0])
	targetState := StateLabel(transition.To[0])

	// Check if source and target are orthogonal
	return s.Orthogonal(sourceState, targetState)
}

// GetOrthogonalRegions returns all orthogonal regions in the statechart.
func (s *Statechart) GetOrthogonalRegions() ([]*sc.State, error) {
	var regions []*sc.State
	
	var findOrthogonalStates func(*sc.State) error
	findOrthogonalStates = func(state *sc.State) error {
		if state.Type == sc.StateTypeParallel {
			regions = append(regions, state.Children...)
		}
		
		for _, child := range state.Children {
			if err := findOrthogonalStates(child); err != nil {
				return err
			}
		}
		return nil
	}
	
	if err := findOrthogonalStates(s.RootState); err != nil {
		return nil, err
	}
	
	return regions, nil
}

// ValidateOrthogonalTransition validates that a cross-region transition is valid.
func (s *Statechart) ValidateOrthogonalTransition(crossTransition *CrossRegionTransition) error {
	// Get all orthogonal regions
	regions, err := s.GetOrthogonalRegions()
	if err != nil {
		return fmt.Errorf("failed to get orthogonal regions: %w", err)
	}

	// Verify that each region in the transition corresponds to an actual region
	regionLabels := make(map[string]bool)
	for _, region := range regions {
		regionLabels[region.Label] = true
	}

	for regionLabel := range crossTransition.Sources {
		if !regionLabels[regionLabel] {
			return fmt.Errorf("region %s not found in orthogonal regions", regionLabel)
		}
	}

	for regionLabel := range crossTransition.Targets {
		if !regionLabels[regionLabel] {
			return fmt.Errorf("region %s not found in orthogonal regions", regionLabel)
		}
	}

	// Verify that source and target states exist
	for _, sourceState := range crossTransition.Sources {
		if _, err := s.findState(StateLabel(sourceState)); err != nil {
			return fmt.Errorf("source state %s not found: %w", sourceState, err)
		}
	}

	for _, targetState := range crossTransition.Targets {
		if _, err := s.findState(StateLabel(targetState)); err != nil {
			return fmt.Errorf("target state %s not found: %w", targetState, err)
		}
	}

	return nil
}