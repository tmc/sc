package runtime

import (
	"fmt"

	"github.com/tmc/sc"
)

// ToProto converts the Starlark MachineDef to a protobuf Machine.
func (m *MachineDef) ToProto() (*sc.Statechart, error) {
	descStates := make([]*sc.State, 0, len(m.States))

	// First pass: create all states
	for _, stateDef := range m.States {
		s, err := stateDef.ToProto()
		if err != nil {
			return nil, fmt.Errorf("failed to convert state %s: %w", stateDef.Label, err)
		}
		descStates = append(descStates, s)
	}

	// For a comprehensive statechart, we need to collect all transitions from all states
	// and flattened them into the top-level Transitions list if we are using the flattened model,
	// OR (more likely based on semantics/v1/statecharts.go) we keep structure.
	// Looking at semantics/v1, it seems it uses sc.Statechart which has a list of States and Transitions.
	// Let's assume a simplified model where generated proto has top-level transitions and hierarchical states.

	// Actually, sc.Statechart has `repeated State states` and `repeated Transition transitions`.
	// Hierarchical states also have children.

	// Let's implement a recursive collector for transitions.
	allTransitions := []*sc.Transition{}
	var collectTransitions func(s *StateDef) error
	collectTransitions = func(s *StateDef) error {
		for _, tDef := range s.Transitions {
			t := &sc.Transition{
				From:  []string{s.Label},
				To:    []string{tDef.To},
				Event: tDef.Event,
				// TODO: Actions, Guards
			}
			allTransitions = append(allTransitions, t)
		}
		for _, child := range s.Children {
			if err := collectTransitions(child); err != nil {
				return err
			}
		}
		return nil
	}

	for _, s := range m.States {
		if err := collectTransitions(s); err != nil {
			return nil, err
		}
	}

	// Create RootState and add top-level states as children
	return &sc.Statechart{
		RootState: &sc.State{
			Label:    "__root__",
			Children: descStates,
			Type:     sc.StateTypeOR, // Root is usually an OR state (containing orthogonal or exclusive states)
		},
		Transitions: allTransitions,
		Name:        m.Name,
	}, nil
}

// ToProto converts Starlark StateDef to proto State.
func (s *StateDef) ToProto() (*sc.State, error) {
	children := make([]*sc.State, 0, len(s.Children))
	for _, childDef := range s.Children {
		child, err := childDef.ToProto()
		if err != nil {
			return nil, err
		}
		children = append(children, child)
	}

	return &sc.State{
		Label:     s.Label,
		Children:  children,
		IsInitial: s.IsInitial,
		// Type: default to basic for now
	}, nil
}
