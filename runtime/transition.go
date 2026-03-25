package runtime

import (
	"fmt"

	"go.starlark.net/starlark"
)

// TransitionDef represents a transition definition in Starlark.
type TransitionDef struct {
	To    string
	Event string
}

func (t *TransitionDef) String() string {
	return fmt.Sprintf("<TransitionDef -> %q>", t.To)
}
func (t *TransitionDef) Type() string         { return "TransitionDef" }
func (t *TransitionDef) Freeze()              {}
func (t *TransitionDef) Truth() starlark.Bool { return starlark.True }
func (t *TransitionDef) Hash() (uint32, error) {
	return 0, fmt.Errorf("unhashable type: TransitionDef")
}

func newTransition(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	var to string
	var event string

	if err := starlark.UnpackArgs("transition", args, kwargs, "to", &to, "event?", &event); err != nil {
		return nil, err
	}

	return &TransitionDef{
		To:    to,
		Event: event,
	}, nil
}
