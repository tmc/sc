package runtime

import (
	"fmt"

	"go.starlark.net/starlark"
)

// ModuleName is the name of the Starlark module.
const ModuleName = "sc"

// NewModule returns the Starlark module for statecharts.
func NewModule() *Module {
	return &Module{}
}

// Module represents the sc module.
type Module struct {
	starlark.Value
}

func (m *Module) String() string {
	return "<module 'sc'>"
}

func (m *Module) Type() string {
	return "module"
}

func (m *Module) Freeze() {
}

func (m *Module) Truth() starlark.Bool {
	return starlark.True
}

func (m *Module) Hash() (uint32, error) {
	return 0, fmt.Errorf("unhashable type: module")
}

func (m *Module) Attr(name string) (starlark.Value, error) {
	switch name {
	case "machine":
		return starlark.NewBuiltin("machine", newMachine), nil
	case "state":
		return starlark.NewBuiltin("state", newState), nil
	case "transition":
		return starlark.NewBuiltin("transition", newTransition), nil
	default:
		return nil, nil // TODO: return error or nil? typical module behavior
	}
}

func (m *Module) AttrNames() []string {
	return []string{"machine", "state", "transition"}
}
