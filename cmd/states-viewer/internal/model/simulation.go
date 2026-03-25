package model

// HistoryEntry represents a single step in the simulation history.
type HistoryEntry struct {
	ActiveStates []string // Active state IDs after this step
	Event        string   // Event that triggered this step
	Transition   string   // Transition ID that fired
}

// SimulationState manages the simulation state and history.
type SimulationState struct {
	doc          *Document
	activeStates map[string]bool // Currently active state IDs
	history      []HistoryEntry  // Step history
	stepIndex    int             // Current position in history (-1 = initial)
	context      map[string]any  // Variable context
}

// NewSimulationState creates a new simulation state for a document.
func NewSimulationState(doc *Document) *SimulationState {
	s := &SimulationState{
		doc:          doc,
		activeStates: make(map[string]bool),
		history:      make([]HistoryEntry, 0),
		stepIndex:    -1,
		context:      make(map[string]any),
	}
	s.Reset()
	return s
}

// Reset resets the simulation to the initial state.
func (s *SimulationState) Reset() {
	s.activeStates = make(map[string]bool)
	s.history = make([]HistoryEntry, 0)
	s.stepIndex = -1

	// Find and activate initial states
	for _, node := range s.doc.Nodes {
		if node.Initial {
			s.activateState(node.ID)
		}
	}

	// If no initial state found, activate first root node
	if len(s.activeStates) == 0 {
		roots := s.doc.RootNodes()
		if len(roots) > 0 {
			s.activateState(roots[0].ID)
		}
	}

	// Update document active states
	s.syncActiveStates()
}

// activateState activates a state and its parent hierarchy.
func (s *SimulationState) activateState(id string) {
	node, ok := s.doc.Nodes[id]
	if !ok {
		return
	}

	s.activeStates[id] = true

	// Also activate parent hierarchy
	if node.ParentID != "" {
		s.activateState(node.ParentID)
	}
}

// deactivateState deactivates a state.
func (s *SimulationState) deactivateState(id string) {
	delete(s.activeStates, id)
}

// IsActive returns true if the state is currently active.
func (s *SimulationState) IsActive(id string) bool {
	return s.activeStates[id]
}

// ActiveStates returns a slice of currently active state IDs in hierarchical order.
// Parents come before children, siblings are sorted alphabetically.
func (s *SimulationState) ActiveStates() []string {
	if len(s.activeStates) == 0 {
		return nil
	}

	// Build hierarchical order
	var result []string
	var visit func(parentID string, indent int)
	visit = func(parentID string, indent int) {
		// Collect children of this parent that are active
		var children []string
		for _, node := range s.doc.Nodes {
			if node.ParentID == parentID && s.activeStates[node.ID] {
				children = append(children, node.ID)
			}
		}

		// Sort children alphabetically for stable order
		for i := 0; i < len(children); i++ {
			for j := i + 1; j < len(children); j++ {
				if children[i] > children[j] {
					children[i], children[j] = children[j], children[i]
				}
			}
		}

		// Add children and recurse
		for _, childID := range children {
			result = append(result, childID)
			visit(childID, indent+1)
		}
	}

	// Start from root (parentID = "")
	visit("", 0)

	return result
}

// ActiveStatesHierarchy returns active states with indentation levels for tree display.
func (s *SimulationState) ActiveStatesHierarchy() []StateWithLevel {
	if len(s.activeStates) == 0 {
		return nil
	}

	var result []StateWithLevel
	var visit func(parentID string, level int)
	visit = func(parentID string, level int) {
		// Collect children of this parent that are active
		var children []string
		for _, node := range s.doc.Nodes {
			if node.ParentID == parentID && s.activeStates[node.ID] {
				children = append(children, node.ID)
			}
		}

		// Sort children alphabetically for stable order
		for i := 0; i < len(children); i++ {
			for j := i + 1; j < len(children); j++ {
				if children[i] > children[j] {
					children[i], children[j] = children[j], children[i]
				}
			}
		}

		// Add children and recurse
		for _, childID := range children {
			result = append(result, StateWithLevel{ID: childID, Level: level})
			visit(childID, level+1)
		}
	}

	visit("", 0)
	return result
}

// StateWithLevel represents a state ID with its hierarchy level.
type StateWithLevel struct {
	ID    string
	Level int
}

// SendEvent sends an event and executes any matching transitions.
// Returns true if a transition was executed.
func (s *SimulationState) SendEvent(event string) bool {
	// Find enabled transitions for this event
	var enabledEdge *FlowEdge
	for _, edge := range s.doc.Edges {
		if edge.Event == event && s.IsActive(edge.SourceID) {
			enabledEdge = edge
			break
		}
	}

	if enabledEdge == nil {
		return false
	}

	// Execute the transition
	s.executeTransition(enabledEdge, event)
	return true
}

// executeTransition executes a single transition.
func (s *SimulationState) executeTransition(edge *FlowEdge, event string) {
	// Deactivate source state (and descendants)
	s.deactivateState(edge.SourceID)

	// Activate target state
	s.activateState(edge.TargetID)

	// Record in history (truncate future if we've stepped back)
	if s.stepIndex < len(s.history)-1 {
		s.history = s.history[:s.stepIndex+1]
	}

	entry := HistoryEntry{
		ActiveStates: s.ActiveStates(),
		Event:        event,
		Transition:   edge.ID,
	}
	s.history = append(s.history, entry)
	s.stepIndex = len(s.history) - 1

	// Update document
	s.syncActiveStates()
}

// StepBack steps back in history.
// Returns true if successful.
func (s *SimulationState) StepBack() bool {
	if s.stepIndex <= 0 {
		// Can't go back further
		if s.stepIndex == 0 {
			s.stepIndex = -1
			s.Reset()
			return true
		}
		return false
	}

	s.stepIndex--
	entry := s.history[s.stepIndex]

	// Restore active states
	s.activeStates = make(map[string]bool)
	for _, id := range entry.ActiveStates {
		s.activeStates[id] = true
	}

	s.syncActiveStates()
	return true
}

// StepForward steps forward in history.
// Returns true if successful.
func (s *SimulationState) StepForward() bool {
	if s.stepIndex >= len(s.history)-1 {
		return false
	}

	s.stepIndex++
	entry := s.history[s.stepIndex]

	// Restore active states
	s.activeStates = make(map[string]bool)
	for _, id := range entry.ActiveStates {
		s.activeStates[id] = true
	}

	s.syncActiveStates()
	return true
}

// StepCount returns the current step and total steps.
func (s *SimulationState) StepCount() (current, total int) {
	return s.stepIndex + 1, len(s.history)
}

// syncActiveStates updates the document nodes' Active flag.
func (s *SimulationState) syncActiveStates() {
	for _, node := range s.doc.Nodes {
		node.Active = s.IsActive(node.ID)
	}
}

// AvailableEvents returns events that are currently enabled.
func (s *SimulationState) AvailableEvents() []string {
	seen := make(map[string]bool)
	var events []string

	for _, edge := range s.doc.Edges {
		if edge.Event != "" && s.IsActive(edge.SourceID) {
			if !seen[edge.Event] {
				seen[edge.Event] = true
				events = append(events, edge.Event)
			}
		}
	}

	return events
}

// LastEvent returns the event from the most recent history entry.
func (s *SimulationState) LastEvent() string {
	if s.stepIndex >= 0 && s.stepIndex < len(s.history) {
		return s.history[s.stepIndex].Event
	}
	return ""
}

// Context returns the current variable context.
func (s *SimulationState) Context() map[string]any {
	return s.context
}

// SetContext sets a context variable.
func (s *SimulationState) SetContext(key string, value any) {
	s.context[key] = value
}
