package xstate

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/common"
)

// Machine represents an XState machine definition.
type Machine struct {
	ID      string                 `json:"id"`
	Initial string                 `json:"initial,omitempty"`
	States  map[string]*State      `json:"states"`
	Context map[string]interface{} `json:"context,omitempty"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
	Version string                 `json:"version,omitempty"`
}

// State represents an XState state definition.
type State struct {
	Type    string                 `json:"type,omitempty"`
	Initial string                 `json:"initial,omitempty"`
	States  map[string]*State      `json:"states,omitempty"`
	On      map[string]*Transition `json:"on,omitempty"`
	Entry   []string               `json:"entry,omitempty"`
	Exit    []string               `json:"exit,omitempty"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
	Tags    []string               `json:"tags,omitempty"`
}

// Transition represents an XState transition definition.
type Transition struct {
	Target  interface{}            `json:"target,omitempty"` // string or []string
	Cond    string                 `json:"cond,omitempty"`
	Actions []string               `json:"actions,omitempty"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
}

// Converter implements bidirectional conversion between XState and Harel formats.
type Converter struct {
	mapping *common.SemanticMapping
}

// NewConverter creates a new XState converter with default semantic mappings.
func NewConverter() *Converter {
	return &Converter{
		mapping: &common.SemanticMapping{
			StateTypeMapping: map[string]sc.StateType{
				"atomic":   sc.StateTypeBasic,
				"compound": sc.StateTypeNormal,
				"parallel": sc.StateTypeParallel,
				"final":    sc.StateTypeBasic,
				"history":  sc.StateTypeBasic, // Simplified mapping
			},
			EventMapping:  make(map[string]string),
			ActionMapping: make(map[string]string),
			UnsupportedFeatures: []string{
				"invoke", "spawn", "actors", "delays", "activities",
			},
		},
	}
}

// Import converts an XState machine to Harel core semantics.
func (c *Converter) Import(xstateMachine Machine) (*sc.Statechart, error) {
	if xstateMachine.ID == "" {
		return nil, common.NewConversionError("xstate", "import", "id",
			fmt.Errorf("machine ID is required"))
	}

	// Create root state
	rootState := &sc.State{
		Label: "__root__",
		Type:  sc.StateTypeNormal,
	}

	// Convert states
	states, transitions, events, err := c.convertStates(xstateMachine.States, xstateMachine.Initial)
	if err != nil {
		return nil, err
	}

	rootState.Children = states
	if xstateMachine.Initial == "" {
		if len(states) > 1 {
			rootState.Type = sc.StateTypeParallel
		} else if len(states) == 1 {
			states[0].IsInitial = true
		}
	}

	// Create statechart
	statechart := &sc.Statechart{
		RootState:   rootState,
		Transitions: transitions,
		Events:      events,
	}

	return statechart, nil
}

// Export converts Harel core semantics to XState machine format.
func (c *Converter) Export(statechart *sc.Statechart) (Machine, error) {
	if statechart.RootState == nil {
		return Machine{}, common.NewConversionError("xstate", "export", "rootState",
			fmt.Errorf("root state is required"))
	}

	machine := Machine{
		ID:      "exported_machine",
		States:  make(map[string]*State),
		Version: "5.0",
	}

	// Find initial state
	if len(statechart.RootState.Children) > 0 {
		for _, child := range statechart.RootState.Children {
			if child.IsInitial {
				machine.Initial = child.Label
				break
			}
		}
	}

	// Convert states
	err := c.exportStates(statechart.RootState.Children, machine.States, statechart.Transitions)
	if err != nil {
		return Machine{}, err
	}

	return machine, nil
}

// Validate checks semantic consistency of XState machine.
func (c *Converter) Validate(xstateMachine Machine) error {
	result := &common.ValidationResult{Valid: true}

	// Check required fields
	if xstateMachine.ID == "" {
		result.AddError("machine ID is required")
	}

	if len(xstateMachine.States) == 0 {
		result.AddError("machine must have at least one state")
	}

	// Validate state structure
	c.validateStates(xstateMachine.States, result)

	if !result.Valid {
		return fmt.Errorf("validation failed: %v", result.Errors)
	}

	return nil
}

// FormatName returns the name of the XState format.
func (c *Converter) FormatName() string {
	return "XState"
}

// convertStates converts XState states to Harel states.
func (c *Converter) convertStates(xstates map[string]*State, initial string) ([]*sc.State, []*sc.Transition, []*sc.Event, error) {
	var states []*sc.State
	var transitions []*sc.Transition
	var events []*sc.Event
	eventSet := make(map[string]bool)

	for label, xstate := range xstates {
		state := &sc.State{
			Label:     label,
			IsInitial: label == initial,
		}

		// Map state type
		stateType, err := c.mapping.MapStateType(xstate.Type)
		if err != nil {
			// Default to basic if type is unspecified
			if xstate.Type == "" {
				if xstate.States != nil && len(xstate.States) > 0 {
					stateType = sc.StateTypeNormal
				} else {
					stateType = sc.StateTypeBasic
				}
			} else {
				return nil, nil, nil, err
			}
		}
		state.Type = stateType

		// Convert child states recursively
		if xstate.States != nil && len(xstate.States) > 0 {
			childStates, childTransitions, childEvents, err := c.convertStates(xstate.States, xstate.Initial)
			if err != nil {
				return nil, nil, nil, err
			}
			state.Children = childStates
			transitions = append(transitions, childTransitions...)
			events = append(events, childEvents...)
		}

		// Convert transitions
		if xstate.On != nil {
			for eventLabel, transition := range xstate.On {
				// Create event if not seen before
				if !eventSet[eventLabel] {
					events = append(events, &sc.Event{Label: eventLabel})
					eventSet[eventLabel] = true
				}

				// Create transition
				t := &sc.Transition{
					Label: fmt.Sprintf("%s_on_%s", label, eventLabel),
					From:  []string{label},
					Event: eventLabel,
				}

				// Handle target (string or []string)
				switch target := transition.Target.(type) {
				case string:
					t.To = []string{target}
				case []string:
					t.To = target
				case nil:
					// Internal transition
					t.To = []string{label}
				}

				// Handle guard condition
				if transition.Cond != "" {
					t.Guard = &sc.Guard{Expression: transition.Cond}
				}

				// Handle actions
				for _, actionLabel := range transition.Actions {
					t.Actions = append(t.Actions, &sc.Action{Label: actionLabel})
				}

				transitions = append(transitions, t)
			}
		}

		states = append(states, state)
	}

	return states, transitions, events, nil
}

// exportStates converts Harel states to XState states.
func (c *Converter) exportStates(states []*sc.State, xstates map[string]*State, transitions []*sc.Transition) error {
	for _, state := range states {
		xstate := &State{
			Meta: make(map[string]interface{}),
		}

		// Map state type
		switch state.Type {
		case sc.StateTypeBasic:
			xstate.Type = "atomic"
		case sc.StateTypeNormal:
			xstate.Type = "compound"
		case sc.StateTypeParallel:
			xstate.Type = "parallel"
		}

		// Handle child states
		if len(state.Children) > 0 {
			xstate.States = make(map[string]*State)

			// Find initial state
			for _, child := range state.Children {
				if child.IsInitial {
					xstate.Initial = child.Label
					break
				}
			}

			// Convert children recursively
			err := c.exportStates(state.Children, xstate.States, transitions)
			if err != nil {
				return err
			}
		}

		// Convert transitions to XState "on" format
		xstate.On = make(map[string]*Transition)
		for _, transition := range transitions {
			for _, from := range transition.From {
				if from == state.Label {
					xtransition := &Transition{
						Meta: make(map[string]interface{}),
					}

					// Set target
					if len(transition.To) == 1 {
						xtransition.Target = transition.To[0]
					} else if len(transition.To) > 1 {
						xtransition.Target = transition.To
					}

					// Set guard condition
					if transition.Guard != nil {
						xtransition.Cond = transition.Guard.Expression
					}

					// Set actions
					for _, action := range transition.Actions {
						xtransition.Actions = append(xtransition.Actions, action.Label)
					}

					xstate.On[transition.Event] = xtransition
				}
			}
		}

		xstates[state.Label] = xstate
	}

	return nil
}

// validateStates recursively validates XState state structure.
func (c *Converter) validateStates(states map[string]*State, result *common.ValidationResult) {
	for label, state := range states {
		if label == "" {
			result.AddError("state label cannot be empty")
		}

		// Check compound states have initial state
		if state.Type == "compound" && state.States != nil && len(state.States) > 0 {
			if state.Initial == "" {
				result.AddWarning("compound state '%s' should specify initial state", label)
			}
		}

		// Validate child states
		if state.States != nil {
			c.validateStates(state.States, result)
		}
	}
}

// ParseJSON parses XState machine from JSON string.
func ParseJSON(jsonStr string) (Machine, error) {
	var machine Machine
	err := json.Unmarshal([]byte(jsonStr), &machine)
	if err != nil {
		return Machine{}, common.NewConversionError("xstate", "parse", "json", err)
	}
	return machine, nil
}

// ToJSON converts XState machine to JSON string.
func ToJSON(machine Machine) (string, error) {
	jsonBytes, err := json.MarshalIndent(machine, "", "  ")
	if err != nil {
		return "", common.NewConversionError("xstate", "serialize", "json", err)
	}
	return string(jsonBytes), nil
}
