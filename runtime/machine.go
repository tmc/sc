package runtime

import (
	"fmt"

	"go.starlark.net/starlark"
)

// MachineDef represents a statechart machine definition in Starlark.
type MachineDef struct {
	Name   string
	States []*StateDef
}

func (m *MachineDef) String() string {
	return fmt.Sprintf("<MachineDef %q>", m.Name)
}
func (m *MachineDef) Type() string          { return "MachineDef" }
func (m *MachineDef) Freeze()               {}
func (m *MachineDef) Truth() starlark.Bool  { return starlark.True }
func (m *MachineDef) Hash() (uint32, error) { return 0, fmt.Errorf("unhashable type: MachineDef") }

func newMachine(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	var name string
	var statesList *starlark.List

	if err := starlark.UnpackArgs("machine", args, kwargs, "name", &name, "states", &statesList); err != nil {
		return nil, err
	}

	states := make([]*StateDef, 0, statesList.Len())
	iter := statesList.Iterate()
	defer iter.Done()
	var x starlark.Value
	for iter.Next(&x) {
		s, ok := x.(*StateDef)
		if !ok {
			return nil, fmt.Errorf("expected list of StateDef, got %s", x.Type())
		}
		states = append(states, s)
	}

	return &MachineDef{
		Name:   name,
		States: states,
	}, nil
}

// StateDef represents a state definition in Starlark.
type StateDef struct {
	Label       string
	Transitions []*TransitionDef
	Children    []*StateDef
	IsInitial   bool
}

func (s *StateDef) String() string {
	return fmt.Sprintf("<StateDef %q>", s.Label)
}
func (s *StateDef) Type() string          { return "StateDef" }
func (s *StateDef) Freeze()               {}
func (s *StateDef) Truth() starlark.Bool  { return starlark.True }
func (s *StateDef) Hash() (uint32, error) { return 0, fmt.Errorf("unhashable type: StateDef") }

func newState(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	var label string
	var transitionsList *starlark.List
	var childrenList *starlark.List
	var initial bool

	if err := starlark.UnpackArgs("state", args, kwargs, "label", &label, "transitions?", &transitionsList, "children?", &childrenList, "initial?", &initial); err != nil {
		return nil, err
	}

	var transitions []*TransitionDef
	if transitionsList != nil {
		iter := transitionsList.Iterate()
		defer iter.Done()
		var x starlark.Value
		for iter.Next(&x) {
			t, ok := x.(*TransitionDef)
			if !ok {
				return nil, fmt.Errorf("expected list of TransitionDef, got %s", x.Type())
			}
			transitions = append(transitions, t)
		}
	}

	var children []*StateDef
	if childrenList != nil {
		iter := childrenList.Iterate()
		defer iter.Done()
		var x starlark.Value
		for iter.Next(&x) {
			child, ok := x.(*StateDef)
			if !ok {
				return nil, fmt.Errorf("expected list of StateDef, got %s", x.Type())
			}
			children = append(children, child)
		}
	}

	return &StateDef{
		Label:       label,
		Transitions: transitions,
		Children:    children,
		IsInitial:   initial,
	}, nil
}
