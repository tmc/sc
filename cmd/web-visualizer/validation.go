package main

import (
	"fmt"
	"strings"

	"github.com/tmc/sc"
	statechartsv1 "github.com/tmc/sc/gen/statecharts/v1"
)

// ValidationEngine provides comprehensive statechart validation
type ValidationEngine struct {
	rules []ValidationRule
}

// ValidationRule defines a validation rule interface
type ValidationRule interface {
	Validate(statechart *sc.Statechart) []ValidationError
	Name() string
	Description() string
	Severity() ValidationSeverity
}

// ValidationError represents a validation error or warning
type ValidationError struct {
	Rule        string              `json:"rule"`
	Severity    ValidationSeverity  `json:"severity"`
	Message     string              `json:"message"`
	Element     string              `json:"element,omitempty"`
	ElementType ValidationElementType `json:"element_type,omitempty"`
	Path        string              `json:"path,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
}

// ValidationSeverity levels
type ValidationSeverity string

const (
	SeverityError   ValidationSeverity = "error"
	SeverityWarning ValidationSeverity = "warning"
	SeverityInfo    ValidationSeverity = "info"
)

// ValidationElementType identifies what type of element has the issue
type ValidationElementType string

const (
	ElementTypeStatechart  ValidationElementType = "statechart"
	ElementTypeState       ValidationElementType = "state"
	ElementTypeTransition  ValidationElementType = "transition"
	ElementTypeEvent       ValidationElementType = "event"
)

// ValidationResult contains the complete validation results
type ValidationResult struct {
	Valid    bool               `json:"valid"`
	Errors   []ValidationError  `json:"errors"`
	Warnings []ValidationError  `json:"warnings"`
	Info     []ValidationError  `json:"info"`
	Summary  ValidationSummary  `json:"summary"`
}

// ValidationSummary provides validation statistics
type ValidationSummary struct {
	TotalErrors   int `json:"total_errors"`
	TotalWarnings int `json:"total_warnings"`
	TotalInfo     int `json:"total_info"`
	RulesChecked  int `json:"rules_checked"`
}

// NewValidationEngine creates a new validation engine with default rules
func NewValidationEngine() *ValidationEngine {
	engine := &ValidationEngine{
		rules: make([]ValidationRule, 0),
	}
	
	// Add default validation rules
	engine.AddRule(&RootStateRule{})
	engine.AddRule(&StateLabelsUniqueRule{})
	engine.AddRule(&TransitionValidityRule{})
	engine.AddRule(&ReachabilityRule{})
	engine.AddRule(&DeadStateRule{})
	engine.AddRule(&NamingConventionRule{})
	engine.AddRule(&HierarchyDepthRule{})
	engine.AddRule(&ParallelStateRule{})
	engine.AddRule(&EventConsistencyRule{})
	engine.AddRule(&PerformanceRule{})
	
	return engine
}

// AddRule adds a validation rule to the engine
func (ve *ValidationEngine) AddRule(rule ValidationRule) {
	ve.rules = append(ve.rules, rule)
}

// Validate runs all validation rules on a statechart
func (ve *ValidationEngine) Validate(statechart *sc.Statechart) ValidationResult {
	var errors, warnings, info []ValidationError
	
	for _, rule := range ve.rules {
		violations := rule.Validate(statechart)
		
		for _, violation := range violations {
			switch violation.Severity {
			case SeverityError:
				errors = append(errors, violation)
			case SeverityWarning:
				warnings = append(warnings, violation)
			case SeverityInfo:
				info = append(info, violation)
			}
		}
	}
	
	return ValidationResult{
		Valid:    len(errors) == 0,
		Errors:   errors,
		Warnings: warnings,
		Info:     info,
		Summary: ValidationSummary{
			TotalErrors:   len(errors),
			TotalWarnings: len(warnings),
			TotalInfo:     len(info),
			RulesChecked:  len(ve.rules),
		},
	}
}

// Validation Rules Implementation

// RootStateRule ensures the statechart has a valid root state
type RootStateRule struct{}

func (r *RootStateRule) Name() string        { return "root_state" }
func (r *RootStateRule) Description() string { return "Statechart must have a valid root state" }
func (r *RootStateRule) Severity() ValidationSeverity { return SeverityError }

func (r *RootStateRule) Validate(statechart *sc.Statechart) []ValidationError {
	var errors []ValidationError
	
	if statechart.RootState == nil {
		errors = append(errors, ValidationError{
			Rule:        r.Name(),
			Severity:    r.Severity(),
			Message:     "Statechart must have a root state",
			ElementType: ElementTypeStatechart,
		})
		return errors
	}
	
	if statechart.RootState.Label == "" {
		errors = append(errors, ValidationError{
			Rule:        r.Name(),
			Severity:    r.Severity(),
			Message:     "Root state must have a label",
			Element:     "root",
			ElementType: ElementTypeState,
		})
	}
	
	return errors
}

// StateLabelsUniqueRule ensures all state labels are unique within their scope
type StateLabelsUniqueRule struct{}

func (r *StateLabelsUniqueRule) Name() string        { return "state_labels_unique" }
func (r *StateLabelsUniqueRule) Description() string { return "State labels must be unique within their scope" }
func (r *StateLabelsUniqueRule) Severity() ValidationSeverity { return SeverityError }

func (r *StateLabelsUniqueRule) Validate(statechart *sc.Statechart) []ValidationError {
	var errors []ValidationError
	
	// Check uniqueness in each scope
	r.checkStateUniqueness(statechart.RootState, "", &errors)
	
	return errors
}

func (r *StateLabelsUniqueRule) checkStateUniqueness(state *sc.State, path string, errors *[]ValidationError) {
	if state == nil {
		return
	}
	
	currentPath := path
	if currentPath != "" {
		currentPath += "."
	}
	currentPath += state.Label
	
	// Check uniqueness among children
	labelCounts := make(map[string]int)
	for _, child := range state.Children {
		if child.Label != "" {
			labelCounts[child.Label]++
		}
	}
	
	for label, count := range labelCounts {
		if count > 1 {
			*errors = append(*errors, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     fmt.Sprintf("Duplicate state label '%s' found %d times", label, count),
				Element:     label,
				ElementType: ElementTypeState,
				Path:        currentPath,
			})
		}
	}
	
	// Recursively check children
	for _, child := range state.Children {
		r.checkStateUniqueness(child, currentPath, errors)
	}
}

// TransitionValidityRule ensures all transitions are valid
type TransitionValidityRule struct{}

func (r *TransitionValidityRule) Name() string        { return "transition_validity" }
func (r *TransitionValidityRule) Description() string { return "Transitions must have valid source and target states" }
func (r *TransitionValidityRule) Severity() ValidationSeverity { return SeverityError }

func (r *TransitionValidityRule) Validate(statechart *sc.Statechart) []ValidationError {
	var errors []ValidationError
	
	// Build map of all states for quick lookup
	stateMap := make(map[string]*sc.State)
	r.buildStateMap(statechart.RootState, "", stateMap)
	
	for _, transition := range statechart.Transitions {
		if len(transition.From) == 0 {
			errors = append(errors, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     "Transition must have a source state",
				Element:     transition.Label,
				ElementType: ElementTypeTransition,
			})
			continue
		}
		
		if len(transition.To) == 0 {
			errors = append(errors, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     "Transition must have a target state",
				Element:     transition.Label,
				ElementType: ElementTypeTransition,
			})
			continue
		}
		
		// Check if source states exist
		for _, fromState := range transition.From {
			if _, exists := stateMap[fromState]; !exists {
				errors = append(errors, ValidationError{
					Rule:        r.Name(),
					Severity:    r.Severity(),
					Message:     fmt.Sprintf("Source state '%s' does not exist", fromState),
					Element:     transition.Label,
					ElementType: ElementTypeTransition,
				})
			}
		}
		
		// Check if target states exist
		for _, toState := range transition.To {
			if _, exists := stateMap[toState]; !exists {
				errors = append(errors, ValidationError{
					Rule:        r.Name(),
					Severity:    r.Severity(),
					Message:     fmt.Sprintf("Target state '%s' does not exist", toState),
					Element:     transition.Label,
					ElementType: ElementTypeTransition,
				})
			}
		}
		
		// Check if event is specified
		if transition.Event == "" {
			errors = append(errors, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     "Transition must have an event",
				Element:     transition.Label,
				ElementType: ElementTypeTransition,
			})
		}
	}
	
	return errors
}

func (r *TransitionValidityRule) buildStateMap(state *sc.State, path string, stateMap map[string]*sc.State) {
	if state == nil {
		return
	}
	
	currentPath := path
	if currentPath != "" {
		currentPath += "."
	}
	currentPath += state.Label
	
	stateMap[currentPath] = state
	
	for _, child := range state.Children {
		r.buildStateMap(child, currentPath, stateMap)
	}
}

// ReachabilityRule checks if all states are reachable from the initial state
type ReachabilityRule struct{}

func (r *ReachabilityRule) Name() string        { return "state_reachability" }
func (r *ReachabilityRule) Description() string { return "All states should be reachable from the initial state" }
func (r *ReachabilityRule) Severity() ValidationSeverity { return SeverityWarning }

func (r *ReachabilityRule) Validate(statechart *sc.Statechart) []ValidationError {
	var warnings []ValidationError
	
	// Build state map and transition graph
	stateMap := make(map[string]*sc.State)
	transitionGraph := make(map[string][]string)
	r.buildStateMap(statechart.RootState, "", stateMap)
	
	for _, transition := range statechart.Transitions {
		if transition.From != nil && transition.To != nil {
			source := strings.Join(transition.From, ",")
			target := strings.Join(transition.To, ",")
			transitionGraph[source] = append(transitionGraph[source], target)
		}
	}
	
	// Find reachable states using DFS
	reachable := make(map[string]bool)
	if statechart.RootState != nil {
		r.markReachable(statechart.RootState.Label, transitionGraph, reachable)
	}
	
	// Check for unreachable states
	for statePath := range stateMap {
		if !reachable[statePath] && statePath != statechart.RootState.Label {
			warnings = append(warnings, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     fmt.Sprintf("State '%s' is unreachable from the initial state", statePath),
				Element:     statePath,
				ElementType: ElementTypeState,
				Path:        statePath,
			})
		}
	}
	
	return warnings
}

func (r *ReachabilityRule) buildStateMap(state *sc.State, path string, stateMap map[string]*sc.State) {
	if state == nil {
		return
	}
	
	currentPath := path
	if currentPath != "" {
		currentPath += "."
	}
	currentPath += state.Label
	
	stateMap[currentPath] = state
	
	for _, child := range state.Children {
		r.buildStateMap(child, currentPath, stateMap)
	}
}

func (r *ReachabilityRule) markReachable(stateName string, graph map[string][]string, reachable map[string]bool) {
	if reachable[stateName] {
		return
	}
	
	reachable[stateName] = true
	
	for _, target := range graph[stateName] {
		r.markReachable(target, graph, reachable)
	}
}

// DeadStateRule identifies states that have no outgoing transitions
type DeadStateRule struct{}

func (r *DeadStateRule) Name() string        { return "dead_states" }
func (r *DeadStateRule) Description() string { return "Identifies states with no outgoing transitions" }
func (r *DeadStateRule) Severity() ValidationSeverity { return SeverityInfo }

func (r *DeadStateRule) Validate(statechart *sc.Statechart) []ValidationError {
	var info []ValidationError
	
	// Build state map
	stateMap := make(map[string]*sc.State)
	r.buildStateMap(statechart.RootState, "", stateMap)
	
	// Track states with outgoing transitions
	hasOutgoing := make(map[string]bool)
	for _, transition := range statechart.Transitions {
		if transition.From != nil {
			hasOutgoing[strings.Join(transition.From, ",")] = true
		}
	}
	
	// Find dead states (leaf states without outgoing transitions)
	for statePath, state := range stateMap {
		if len(state.Children) == 0 && !hasOutgoing[statePath] {
			info = append(info, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     fmt.Sprintf("State '%s' has no outgoing transitions (dead state)", statePath),
				Element:     statePath,
				ElementType: ElementTypeState,
				Path:        statePath,
			})
		}
	}
	
	return info
}

func (r *DeadStateRule) buildStateMap(state *sc.State, path string, stateMap map[string]*sc.State) {
	if state == nil {
		return
	}
	
	currentPath := path
	if currentPath != "" {
		currentPath += "."
	}
	currentPath += state.Label
	
	stateMap[currentPath] = state
	
	for _, child := range state.Children {
		r.buildStateMap(child, currentPath, stateMap)
	}
}

// NamingConventionRule checks for consistent naming conventions
type NamingConventionRule struct{}

func (r *NamingConventionRule) Name() string        { return "naming_convention" }
func (r *NamingConventionRule) Description() string { return "Checks for consistent naming conventions" }
func (r *NamingConventionRule) Severity() ValidationSeverity { return SeverityWarning }

func (r *NamingConventionRule) Validate(statechart *sc.Statechart) []ValidationError {
	var warnings []ValidationError
	
	r.checkStateNaming(statechart.RootState, "", &warnings)
	
	// Check event naming in transitions
	eventNames := make(map[string]bool)
	for _, transition := range statechart.Transitions {
		if transition.Event != "" {
			eventNames[transition.Event] = true
		}
	}
	
	for event := range eventNames {
		if !r.isValidEventName(event) {
			warnings = append(warnings, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     fmt.Sprintf("Event '%s' doesn't follow naming convention (should be UPPER_CASE)", event),
				Element:     event,
				ElementType: ElementTypeEvent,
			})
		}
	}
	
	return warnings
}

func (r *NamingConventionRule) checkStateNaming(state *sc.State, path string, warnings *[]ValidationError) {
	if state == nil {
		return
	}
	
	currentPath := path
	if currentPath != "" {
		currentPath += "."
	}
	currentPath += state.Label
	
	if !r.isValidStateName(state.Label) {
		*warnings = append(*warnings, ValidationError{
			Rule:        r.Name(),
			Severity:    r.Severity(),
			Message:     fmt.Sprintf("State '%s' doesn't follow naming convention (should be camelCase or snake_case)", state.Label),
			Element:     state.Label,
			ElementType: ElementTypeState,
			Path:        currentPath,
		})
	}
	
	for _, child := range state.Children {
		r.checkStateNaming(child, currentPath, warnings)
	}
}

func (r *NamingConventionRule) isValidStateName(name string) bool {
	if name == "" {
		return false
	}
	
	// Allow camelCase or snake_case for states
	return r.isCamelCase(name) || r.isSnakeCase(name)
}

func (r *NamingConventionRule) isValidEventName(name string) bool {
	if name == "" {
		return false
	}
	
	// Events should be UPPER_CASE
	return strings.ToUpper(name) == name && !strings.ContainsAny(name, " -.")
}

func (r *NamingConventionRule) isCamelCase(name string) bool {
	return !strings.ContainsAny(name, " -._") && name[0] >= 'a' && name[0] <= 'z'
}

func (r *NamingConventionRule) isSnakeCase(name string) bool {
	return !strings.ContainsAny(name, " -.") && strings.ToLower(name) == name
}

// HierarchyDepthRule warns about excessively deep state hierarchies
type HierarchyDepthRule struct{}

func (r *HierarchyDepthRule) Name() string        { return "hierarchy_depth" }
func (r *HierarchyDepthRule) Description() string { return "Warns about excessively deep state hierarchies" }
func (r *HierarchyDepthRule) Severity() ValidationSeverity { return SeverityWarning }

func (r *HierarchyDepthRule) Validate(statechart *sc.Statechart) []ValidationError {
	var warnings []ValidationError
	
	maxDepth := r.calculateMaxDepth(statechart.RootState, 0)
	
	if maxDepth > 5 {
		warnings = append(warnings, ValidationError{
			Rule:        r.Name(),
			Severity:    r.Severity(),
			Message:     fmt.Sprintf("State hierarchy depth (%d) is quite deep, consider flattening", maxDepth),
			ElementType: ElementTypeStatechart,
			Details:     map[string]interface{}{"max_depth": maxDepth},
		})
	}
	
	return warnings
}

func (r *HierarchyDepthRule) calculateMaxDepth(state *sc.State, currentDepth int) int {
	if state == nil || len(state.Children) == 0 {
		return currentDepth
	}
	
	maxChildDepth := currentDepth
	for _, child := range state.Children {
		childDepth := r.calculateMaxDepth(child, currentDepth+1)
		if childDepth > maxChildDepth {
			maxChildDepth = childDepth
		}
	}
	
	return maxChildDepth
}

// ParallelStateRule validates parallel state configurations
type ParallelStateRule struct{}

func (r *ParallelStateRule) Name() string        { return "parallel_state_validation" }
func (r *ParallelStateRule) Description() string { return "Validates parallel state configurations" }
func (r *ParallelStateRule) Severity() ValidationSeverity { return SeverityError }

func (r *ParallelStateRule) Validate(statechart *sc.Statechart) []ValidationError {
	var errors []ValidationError
	
	r.checkParallelStates(statechart.RootState, "", &errors)
	
	return errors
}

func (r *ParallelStateRule) checkParallelStates(state *sc.State, path string, errors *[]ValidationError) {
	if state == nil {
		return
	}
	
	currentPath := path
	if currentPath != "" {
		currentPath += "."
	}
	currentPath += state.Label
	
	if state.Type == statechartsv1.StateType_STATE_TYPE_AND {
		if len(state.Children) < 2 {
			*errors = append(*errors, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     "Parallel state must have at least 2 child regions",
				Element:     state.Label,
				ElementType: ElementTypeState,
				Path:        currentPath,
			})
		}
	}
	
	for _, child := range state.Children {
		r.checkParallelStates(child, currentPath, errors)
	}
}

// EventConsistencyRule checks for consistent event usage
type EventConsistencyRule struct{}

func (r *EventConsistencyRule) Name() string        { return "event_consistency" }
func (r *EventConsistencyRule) Description() string { return "Checks for consistent event usage across transitions" }
func (r *EventConsistencyRule) Severity() ValidationSeverity { return SeverityWarning }

func (r *EventConsistencyRule) Validate(statechart *sc.Statechart) []ValidationError {
	var warnings []ValidationError
	
	// Track event usage patterns
	eventUsage := make(map[string][]string) // event -> []sourceStates
	
	for _, transition := range statechart.Transitions {
		if transition.Event != "" && transition.From != nil {
			eventUsage[transition.Event] = append(eventUsage[transition.Event], strings.Join(transition.From, ","))
		}
	}
	
	// Look for events used only once (might be typos)
	for event, sources := range eventUsage {
		if len(sources) == 1 {
			warnings = append(warnings, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     fmt.Sprintf("Event '%s' is only used once (from state '%s')", event, sources[0]),
				Element:     event,
				ElementType: ElementTypeEvent,
				Details:     map[string]interface{}{"source_states": sources},
			})
		}
	}
	
	return warnings
}

// PerformanceRule identifies potential performance issues
type PerformanceRule struct{}

func (r *PerformanceRule) Name() string        { return "performance_optimization" }
func (r *PerformanceRule) Description() string { return "Identifies potential performance issues" }
func (r *PerformanceRule) Severity() ValidationSeverity { return SeverityInfo }

func (r *PerformanceRule) Validate(statechart *sc.Statechart) []ValidationError {
	var info []ValidationError
	
	// Count states and transitions
	stateCount := r.countStates(statechart.RootState)
	transitionCount := len(statechart.Transitions)
	
	if stateCount > 100 {
		info = append(info, ValidationError{
			Rule:        r.Name(),
			Severity:    r.Severity(),
			Message:     fmt.Sprintf("Large number of states (%d) may impact performance", stateCount),
			ElementType: ElementTypeStatechart,
			Details:     map[string]interface{}{"state_count": stateCount},
		})
	}
	
	if transitionCount > 200 {
		info = append(info, ValidationError{
			Rule:        r.Name(),
			Severity:    r.Severity(),
			Message:     fmt.Sprintf("Large number of transitions (%d) may impact performance", transitionCount),
			ElementType: ElementTypeStatechart,
			Details:     map[string]interface{}{"transition_count": transitionCount},
		})
	}
	
	// Check for states with many outgoing transitions
	outgoingCount := make(map[string]int)
	for _, transition := range statechart.Transitions {
		if transition.From != nil {
			outgoingCount[strings.Join(transition.From, ",")]++
		}
	}
	
	for state, count := range outgoingCount {
		if count > 10 {
			info = append(info, ValidationError{
				Rule:        r.Name(),
				Severity:    r.Severity(),
				Message:     fmt.Sprintf("State '%s' has many outgoing transitions (%d)", state, count),
				Element:     state,
				ElementType: ElementTypeState,
				Details:     map[string]interface{}{"transition_count": count},
			})
		}
	}
	
	return info
}

func (r *PerformanceRule) countStates(state *sc.State) int {
	if state == nil {
		return 0
	}
	
	count := 1
	for _, child := range state.Children {
		count += r.countStates(child)
	}
	
	return count
}