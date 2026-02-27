package semantics

import (
	"fmt"
	"sync"
	"time"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

// MachineWrapper wraps a protobuf Machine and provides runtime execution semantics.
type MachineWrapper struct {
	*sc.Machine
	statechart *Statechart
	mu         sync.RWMutex
	errors     []error
}

// NewMachine creates a new machine wrapper from a statechart and optional initial configuration.
func NewMachine(statechart *Statechart, id string, initialContext *structpb.Struct) (*MachineWrapper, error) {
	if statechart == nil {
		return nil, fmt.Errorf("statechart cannot be nil")
	}
	if id == "" {
		return nil, fmt.Errorf("machine id cannot be empty")
	}

	// Validate the statechart
	if err := statechart.Validate(); err != nil {
		return nil, fmt.Errorf("invalid statechart: %w", err)
	}

	// Create initial configuration
	initialConfig, err := statechart.InitialConfiguration()
	if err != nil {
		return nil, fmt.Errorf("failed to compute initial configuration: %w", err)
	}

	// Create default context if none provided
	if initialContext == nil {
		initialContext = &structpb.Struct{
			Fields: make(map[string]*structpb.Value),
		}
	}

	machine := &sc.Machine{
		Id:            id,
		State:         sc.MachineStateStopped,
		Context:       initialContext,
		Statechart:    statechart.Statechart,
		Configuration: initialConfig,
		StepHistory:   []*sc.Step{},
	}

	return &MachineWrapper{
		Machine:    machine,
		statechart: statechart,
		errors:     []error{},
	}, nil
}

// Start starts the machine and transitions it to the running state.
func (m *MachineWrapper) Start() error {
	if m == nil {
		return fmt.Errorf("machine is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.State == sc.MachineStateRunning {
		return fmt.Errorf("machine is already running")
	}

	// Validate current configuration
	if err := ValidateConfiguration(m.statechart, m.Configuration); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	m.State = sc.MachineStateRunning
	m.addStep(&sc.Step{
		Events:                 []*sc.Event{{Label: "__START__"}},
		Transitions:            []*sc.Transition{},
		StartingConfiguration:  &sc.Configuration{States: []*sc.StateRef{}},
		ResultingConfiguration: m.Configuration,
		Context:                m.Context,
	})

	return nil
}

// Stop stops the machine and transitions it to the stopped state.
func (m *MachineWrapper) Stop() error {
	if m == nil {
		return fmt.Errorf("machine is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.State == sc.MachineStateStopped {
		return fmt.Errorf("machine is already stopped")
	}

	m.State = sc.MachineStateStopped
	m.addStep(&sc.Step{
		Events:                 []*sc.Event{{Label: "__STOP__"}},
		Transitions:            []*sc.Transition{},
		StartingConfiguration:  m.Configuration,
		ResultingConfiguration: m.Configuration,
		Context:                m.Context,
	})

	return nil
}

// IsRunning returns true if the machine is in the running state.
func (m *MachineWrapper) IsRunning() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.State == sc.MachineStateRunning
}

// IsStopped returns true if the machine is in the stopped state.
func (m *MachineWrapper) IsStopped() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.State == sc.MachineStateStopped
}

// IsFinal returns true if the machine has reached a final configuration.
// A final configuration is one where all leaf states are marked as final.
func (m *MachineWrapper) IsFinal() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	isFinal, err := m.statechart.IsFinalConfiguration(m.Configuration)
	if err != nil {
		return false
	}
	return isFinal
}

// GetCurrentConfiguration returns a copy of the current configuration.
func (m *MachineWrapper) GetCurrentConfiguration() *sc.Configuration {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Create a deep copy of the configuration
	states := make([]*sc.StateRef, len(m.Configuration.States))
	for i, state := range m.Configuration.States {
		states[i] = &sc.StateRef{Label: state.Label}
	}

	return &sc.Configuration{States: states}
}

// GetContext returns a copy of the current context.
func (m *MachineWrapper) GetContext() *structpb.Struct {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.getContextUnsafe()
}

// getContextUnsafe returns a copy of the current context without acquiring locks.
// This should only be called when the caller already holds a lock.
func (m *MachineWrapper) getContextUnsafe() *structpb.Struct {
	if m.Context == nil {
		return &structpb.Struct{Fields: make(map[string]*structpb.Value)}
	}

	// Create a shallow copy of the context
	fields := make(map[string]*structpb.Value)
	if m.Context.Fields != nil {
		for k, v := range m.Context.Fields {
			fields[k] = v
		}
	}

	return &structpb.Struct{Fields: fields}
}

// GetErrors returns any errors that occurred during machine execution.
func (m *MachineWrapper) GetErrors() []error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	errors := make([]error, len(m.errors))
	copy(errors, m.errors)
	return errors
}

// ClearErrors clears any accumulated errors.
func (m *MachineWrapper) ClearErrors() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.errors = []error{}
}

// GetStepHistory returns a copy of the step history.
func (m *MachineWrapper) GetStepHistory() []*sc.Step {
	m.mu.RLock()
	defer m.mu.RUnlock()

	history := make([]*sc.Step, len(m.StepHistory))
	copy(history, m.StepHistory)
	return history
}

// addStep adds a step to the machine's history.
func (m *MachineWrapper) addStep(step *sc.Step) {
	m.StepHistory = append(m.StepHistory, step)
}

// addError adds an error to the machine's error list.
func (m *MachineWrapper) addError(err error) {
	m.errors = append(m.errors, err)
}

// Reset resets the machine to its initial state.
func (m *MachineWrapper) Reset() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Compute initial configuration
	initialConfig, err := m.statechart.InitialConfiguration()
	if err != nil {
		return fmt.Errorf("failed to compute initial configuration: %w", err)
	}

	// Reset machine state
	m.State = sc.MachineStateStopped
	m.Configuration = initialConfig
	m.StepHistory = []*sc.Step{}
	m.errors = []error{}

	// Reset context to empty
	m.Context = &structpb.Struct{
		Fields: make(map[string]*structpb.Value),
	}

	return nil
}

// Step processes an event and executes transitions using run-to-completion semantics.
// Returns true if any transitions were executed.
func (m *MachineWrapper) Step(eventName string) (bool, error) {
	if m == nil {
		return false, fmt.Errorf("machine is nil")
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.State != sc.MachineStateRunning {
		err := fmt.Errorf("machine is not running")
		m.addError(err)
		return false, err
	}

	return m.step(eventName)
}

// step performs the actual step execution (assumes lock is held).
func (m *MachineWrapper) step(eventName string) (bool, error) {
	initialConfig := m.cloneConfiguration(m.Configuration)

	// Find enabled transitions
	enabledTransitions, err := m.findEnabledTransitions(eventName)
	if err != nil {
		m.addError(err)
		return false, fmt.Errorf("failed to find enabled transitions: %w", err)
	}

	if len(enabledTransitions) == 0 {
		return false, nil // No transitions enabled
	}

	// Resolve conflicts and select transitions to execute
	selectedTransitions, err := m.resolveConflicts(enabledTransitions)
	if err != nil {
		m.addError(err)
		return false, fmt.Errorf("failed to resolve transition conflicts: %w", err)
	}

	// Execute the selected transitions
	if err := m.executeTransitions(selectedTransitions); err != nil {
		m.addError(err)
		return false, fmt.Errorf("failed to execute transitions: %w", err)
	}

	// Record the step
	step := &sc.Step{
		Events:                 []*sc.Event{{Label: eventName}},
		Transitions:            selectedTransitions,
		StartingConfiguration:  initialConfig,
		ResultingConfiguration: m.cloneConfiguration(m.Configuration),
		Context:                m.getContextUnsafe(),
	}
	m.addStep(step)

	// Check if the new configuration is final (all leaf states are final)
	// If so, auto-stop the machine per Harel semantics
	isFinal, err := m.statechart.IsFinalConfiguration(m.Configuration)
	if err != nil {
		m.addError(fmt.Errorf("failed to check final configuration: %w", err))
		// Don't fail the step, just log the error
	} else if isFinal {
		m.State = sc.MachineStateStopped
		// Record the final step
		finalStep := &sc.Step{
			Events:                 []*sc.Event{{Label: "__FINAL__"}},
			Transitions:            []*sc.Transition{},
			StartingConfiguration:  m.cloneConfiguration(m.Configuration),
			ResultingConfiguration: m.cloneConfiguration(m.Configuration),
			Context:                m.getContextUnsafe(),
		}
		m.addStep(finalStep)
	}

	return true, nil
}

// findEnabledTransitions finds all transitions enabled by the given event.
func (m *MachineWrapper) findEnabledTransitions(eventName string) ([]*sc.Transition, error) {
	// Pre-allocate with estimated capacity to reduce allocations
	enabled := make([]*sc.Transition, 0, 8)

	currentStates := make(map[string]bool, len(m.Configuration.States))
	for _, stateRef := range m.Configuration.States {
		currentStates[stateRef.Label] = true
	}

	for _, transition := range m.Statechart.Transitions {
		// Check if any source state of the transition is active
		for _, sourceState := range transition.From {
			if currentStates[sourceState] {
				// Check if the event matches
				if transition.Event == eventName || transition.Event == "" {
					// Evaluate guard
					guardPasses, err := m.evaluateGuard(transition.Guard)
					if err != nil {
						return nil, fmt.Errorf("failed to evaluate guard for transition %s: %w", transition.Label, err)
					}
					if guardPasses {
						enabled = append(enabled, transition)
						break // Only add once per transition
					}
				}
			}
		}
	}

	return enabled, nil
}

// resolveConflicts resolves conflicts between enabled transitions.
// Priority is given to transitions originating from deeper states in the hierarchy.
func (m *MachineWrapper) resolveConflicts(enabled []*sc.Transition) ([]*sc.Transition, error) {
	if len(enabled) <= 1 {
		return enabled, nil
	}

	// For now, implement a simple conflict resolution:
	// 1. Group transitions by their source states
	// 2. Select the transition from the deepest state
	// 3. Remove conflicting transitions

	selected := make(map[string]*sc.Transition)

	for _, transition := range enabled {
		for _, sourceState := range transition.From {
			// Get the depth of the source state
			depth, err := m.statechart.GetDepth(StateLabel(sourceState))
			if err != nil {
				return nil, fmt.Errorf("failed to get depth of state %s: %w", sourceState, err)
			}

			// Check if we have a better transition for this source state
			key := sourceState
			if existing, exists := selected[key]; exists {
				existingDepth, err := m.statechart.GetDepth(StateLabel(existing.From[0]))
				if err != nil {
					return nil, fmt.Errorf("failed to get depth of existing state: %w", err)
				}
				if depth > existingDepth {
					selected[key] = transition
				}
			} else {
				selected[key] = transition
			}
		}
	}

	// Convert map back to slice
	var result []*sc.Transition
	seen := make(map[*sc.Transition]bool)
	for _, transition := range selected {
		if !seen[transition] {
			result = append(result, transition)
			seen[transition] = true
		}
	}

	return result, nil
}

// executeTransitions executes the given transitions.
func (m *MachineWrapper) executeTransitions(transitions []*sc.Transition) error {
	if len(transitions) == 0 {
		return nil
	}

	// Compute the states to exit and enter
	statesToExit, statesToEnter, err := m.computeStateChanges(transitions)
	if err != nil {
		return fmt.Errorf("failed to compute state changes: %w", err)
	}

	// Exit states in reverse hierarchical order
	if err := m.exitStates(statesToExit); err != nil {
		return fmt.Errorf("failed to exit states: %w", err)
	}

	// Execute transition actions
	for _, transition := range transitions {
		for _, action := range transition.Actions {
			if err := m.executeAction(action); err != nil {
				return fmt.Errorf("failed to execute action %s: %w", action.Label, err)
			}
		}
	}

	// Enter states in hierarchical order
	if err := m.enterStates(statesToEnter); err != nil {
		return fmt.Errorf("failed to enter states: %w", err)
	}

	// Update configuration with the new states
	m.updateConfiguration(statesToEnter)

	return nil
}

// computeStateChanges computes which states to exit and enter based on the transitions.
func (m *MachineWrapper) computeStateChanges(transitions []*sc.Transition) ([]string, []string, error) {
	currentStates := make(map[string]bool)
	for _, stateRef := range m.Configuration.States {
		currentStates[stateRef.Label] = true
	}

	statesToExit := make(map[string]bool)
	statesToEnter := make(map[string]bool)

	for _, transition := range transitions {
		// Mark source states for exit
		for _, sourceState := range transition.From {
			if currentStates[sourceState] {
				statesToExit[sourceState] = true
			}
		}

		// Mark target states for entry
		for _, targetState := range transition.To {
			statesToEnter[targetState] = true
		}
	}

	// Resolve history states in target states
	resolvedTargets, err := m.resolveTargetStates(statesToEnter)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to resolve target states: %w", err)
	}
	statesToEnter = make(map[string]bool)
	for _, state := range resolvedTargets {
		statesToEnter[state] = true
	}

	// Convert to slices
	var exitList []string
	for state := range statesToExit {
		exitList = append(exitList, state)
	}

	var enterList []string
	for state := range statesToEnter {
		enterList = append(enterList, state)
	}

	return exitList, enterList, nil
}

// resolveTargetStates resolves history states and default entries in the target set.
func (m *MachineWrapper) resolveTargetStates(targets map[string]bool) ([]string, error) {
	var resolved []string

	// Queue for processing states (handling cascading history/defaults)
	queue := make([]string, 0, len(targets))
	for state := range targets {
		queue = append(queue, state)
	}

	processed := make(map[string]bool)

	for len(queue) > 0 {
		label := queue[0]
		queue = queue[1:]

		if processed[label] {
			continue
		}
		processed[label] = true

		state, err := m.statechart.findState(StateLabel(label))
		if err != nil {
			return nil, fmt.Errorf("state %s not found: %w", label, err)
		}

		if state.IsHistory {
			// Resolve history
			historyConfig := m.resolveHistory(state)
			if len(historyConfig) > 0 {
				for _, histState := range historyConfig {
					if !processed[histState] {
						queue = append(queue, histState)
					}
				}
			} else {
				// Default history behavior (if no history, follow default transition from history state)
				// Note: The proto definition for History states might expect them to have outgoing transitions
				// acting as defaults.
				// For now, we assume if no history, we might fall back to initial state of parent or specific default.
				// This implementation assumes strict history or nothing.
			}
		} else {
			resolved = append(resolved, label)
		}
	}

	return resolved, nil
}

// resolveHistory returns the stored configuration for a history state.
func (m *MachineWrapper) resolveHistory(historyState *sc.State) []string {
	// Find the parent composite state
	parent, err := m.statechart.GetParent(StateLabel(historyState.Label))
	if err != nil || parent == nil {
		return nil
	}

	// Check if we have history for this parent
	if m.Configuration.History == nil {
		return nil
	}

	storedConfig, exists := m.Configuration.History[parent.Label]
	if !exists {
		return nil
	}

	var resolvedStates []string
	for _, ref := range storedConfig.States {
		if historyState.HistoryType == sc.HistoryType_HISTORY_TYPE_DEEP {
			// Deep history: return all stored states
			resolvedStates = append(resolvedStates, ref.Label)
		} else {
			// Shallow history: only return direct children of the parent
			// We need to check if the stored state is a direct child of parent
			storedState, err := m.statechart.findState(StateLabel(ref.Label))
			if err == nil {
				storedParent, _ := m.statechart.GetParent(StateLabel(storedState.Label))
				if storedParent != nil && storedParent.Label == parent.Label {
					resolvedStates = append(resolvedStates, ref.Label)
				}
			}
		}
	}
	return resolvedStates
}

// exitStates executes exit actions and saves history.
func (m *MachineWrapper) exitStates(states []string) error {
	for _, label := range states {
		// Save history if we are exiting a composite state
		state, err := m.statechart.findState(StateLabel(label))
		if err == nil && (len(state.Children) > 0 || state.Type == sc.StateTypeParallel) {
			m.saveHistory(state)
		}

		// Execute exit actions
		// TODO: Implement actual exit action execution logic
	}
	return nil
}

// saveHistory saves the current configuration of a composite state into history.
func (m *MachineWrapper) saveHistory(state *sc.State) {
	if m.Configuration.History == nil {
		m.Configuration.History = make(map[string]*sc.Configuration)
	}

	// Find all active descendants of this state
	var activeDescendants []*sc.StateRef
	for _, activeRef := range m.Configuration.States {
		activeState, err := m.statechart.findState(StateLabel(activeRef.Label))
		if err != nil {
			continue
		}

		// Check if activeState is a descendant of state
		if m.isDescendant(activeState, state) {
			activeDescendants = append(activeDescendants, activeRef)
		}
	}

	if len(activeDescendants) > 0 {
		m.Configuration.History[state.Label] = &sc.Configuration{
			States: activeDescendants,
		}
	}
}

// isDescendant checks if child is a descendant of parent
func (m *MachineWrapper) isDescendant(child, parent *sc.State) bool {
	curr := child
	for curr != nil {
		p, err := m.statechart.GetParent(StateLabel(curr.Label))
		if err != nil || p == nil {
			return false
		}
		if p.Label == parent.Label {
			return true
		}
		curr = p
	}
	return false
}

// enterStates executes entry actions for the given states.
func (m *MachineWrapper) enterStates(states []string) error {
	// TODO: Implement entry actions
	return nil
}

// updateConfiguration updates the machine's configuration with the new states.
func (m *MachineWrapper) updateConfiguration(newStates []string) {
	var stateRefs []*sc.StateRef
	for _, state := range newStates {
		stateRefs = append(stateRefs, &sc.StateRef{Label: state})
	}

	// Compute default completion
	config := &sc.Configuration{States: stateRefs}
	completedConfig, err := DefaultCompletion(m.statechart, config)
	if err != nil {
		m.addError(fmt.Errorf("failed to compute default completion: %w", err))
		return
	}

	// Preserve history from previous configuration
	if m.Configuration != nil && m.Configuration.History != nil {
		completedConfig.History = m.Configuration.History
	}

	m.Configuration = completedConfig
}

// cloneConfiguration creates a deep copy of a configuration.
func (m *MachineWrapper) cloneConfiguration(config *sc.Configuration) *sc.Configuration {
	if config == nil {
		return nil
	}

	states := make([]*sc.StateRef, len(config.States))
	for i, state := range config.States {
		states[i] = &sc.StateRef{Label: state.Label}
	}

	return &sc.Configuration{States: states}
}

// evaluateGuard evaluates a guard condition against the current context.
func (m *MachineWrapper) evaluateGuard(guard *sc.Guard) (bool, error) {
	if guard == nil || guard.Expression == "" {
		return true, nil // No guard means always pass
	}

	// Create evaluation context with current machine state
	evalContext := &EvaluationContext{
		Variables:    m.Context,
		ActiveStates: m.getActiveStateLabels(),
		Statechart:   m.Statechart,
		TimeContext: &TimeContext{
			CurrentTime: time.Now().UnixMilli(),
		},
	}

	// Use the global guard evaluator for comprehensive evaluation
	result, err := globalGuardEvaluator.EvaluateGuard(guard, evalContext)
	if err != nil {
		return false, fmt.Errorf("failed to evaluate guard expression '%s': %w", guard.Expression, err)
	}

	return result.Value, nil
}

// getActiveStateLabels returns the labels of currently active states
func (m *MachineWrapper) getActiveStateLabels() []string {
	var labels []string
	for _, stateRef := range m.Configuration.States {
		if stateRef != nil {
			labels = append(labels, stateRef.Label)
		}
	}
	return labels
}

// executeAction executes an action, potentially modifying the machine context.
func (m *MachineWrapper) executeAction(action *sc.Action) error {
	if action == nil || action.Label == "" {
		return nil // No action to execute
	}

	// For now, implement a simple action executor
	// In a production system, you would have a registry of action handlers
	return m.executeActionByLabel(action.Label)
}

// evaluateExpression evaluates a simple expression against the current context.
// This is a simplified implementation for demonstration purposes.
func (m *MachineWrapper) evaluateExpression(expression string) (bool, error) {
	// Handle some common patterns
	switch expression {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}

	// Handle context field comparisons (simplified)
	if expression == "context.count < 5" {
		if count, ok := m.Context.Fields["count"]; ok {
			if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
				return numValue.NumberValue < 5, nil
			}
		}
		return false, nil
	}

	if expression == "context.count >= 5" {
		if count, ok := m.Context.Fields["count"]; ok {
			if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
				return numValue.NumberValue >= 5, nil
			}
		}
		return false, nil
	}

	// Default to true for unknown expressions (you may want to change this)
	return true, nil
}

// executeActionByLabel executes an action based on its label.
// This is a simplified implementation for demonstration purposes.
func (m *MachineWrapper) executeActionByLabel(label string) error {
	switch label {
	case "increment_count":
		return m.incrementCount()
	case "decrement_count":
		return m.decrementCount()
	case "reset_count":
		return m.resetCount()
	case "log_entry":
		return m.logEntry()
	case "log_exit":
		return m.logExit()
	default:
		// For unknown actions, we just ignore them (you may want to change this)
		return nil
	}
}

// incrementCount increments the count field in the context.
func (m *MachineWrapper) incrementCount() error {
	if count, ok := m.Context.Fields["count"]; ok {
		if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
			m.Context.Fields["count"] = structpb.NewNumberValue(numValue.NumberValue + 1)
			return nil
		}
	}
	// Initialize count if it doesn't exist
	m.Context.Fields["count"] = structpb.NewNumberValue(1)
	return nil
}

// decrementCount decrements the count field in the context.
func (m *MachineWrapper) decrementCount() error {
	if count, ok := m.Context.Fields["count"]; ok {
		if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
			m.Context.Fields["count"] = structpb.NewNumberValue(numValue.NumberValue - 1)
			return nil
		}
	}
	// Initialize count if it doesn't exist
	m.Context.Fields["count"] = structpb.NewNumberValue(-1)
	return nil
}

// resetCount resets the count field in the context to zero.
func (m *MachineWrapper) resetCount() error {
	m.Context.Fields["count"] = structpb.NewNumberValue(0)
	return nil
}

// logEntry logs a state entry (placeholder implementation).
func (m *MachineWrapper) logEntry() error {
	// In a real implementation, this might log to a file or monitoring system
	return nil
}

// logExit logs a state exit (placeholder implementation).
func (m *MachineWrapper) logExit() error {
	// In a real implementation, this might log to a file or monitoring system
	return nil
}

// Validate validates the current state of the machine.
func (m *MachineWrapper) Validate() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Validate the statechart
	if err := m.statechart.Validate(); err != nil {
		return fmt.Errorf("invalid statechart: %w", err)
	}

	// Validate the configuration
	if err := ValidateConfiguration(m.statechart, m.Configuration); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	return nil
}
