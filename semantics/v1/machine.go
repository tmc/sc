package semantics

import (
	"fmt"
	"sync"

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
	
	// Create a shallow copy of the context
	fields := make(map[string]*structpb.Value)
	for k, v := range m.Context.Fields {
		fields[k] = v
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
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.State != sc.MachineStateRunning {
		return false, fmt.Errorf("machine is not running")
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
		Context:                m.GetContext(),
	}
	m.addStep(step)

	return true, nil
}

// findEnabledTransitions finds all transitions enabled by the given event.
func (m *MachineWrapper) findEnabledTransitions(eventName string) ([]*sc.Transition, error) {
	var enabled []*sc.Transition

	currentStates := make(map[string]bool)
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

// exitStates executes exit actions for the given states.
func (m *MachineWrapper) exitStates(states []string) error {
	// TODO: Implement exit actions
	return nil
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

	// For now, implement a simple expression evaluator
	// In a production system, you would want a proper expression parser/evaluator
	result, err := m.evaluateExpression(guard.Expression)
	if err != nil {
		return false, fmt.Errorf("failed to evaluate guard expression '%s': %w", guard.Expression, err)
	}

	return result, nil
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