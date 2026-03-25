package runtime

import (
	"fmt"

	"github.com/tmc/sc/semantics/v1"
	"go.starlark.net/starlark"
	"google.golang.org/protobuf/types/known/structpb"
)

// MachineInstance represents a running machine instance in Starlark.
type MachineInstance struct {
	Wrapper *semantics.MachineWrapper
}

func (m *MachineInstance) String() string {
	return fmt.Sprintf("<MachineInstance %s>", m.Wrapper.Id)
}
func (m *MachineInstance) Type() string         { return "MachineInstance" }
func (m *MachineInstance) Freeze()              {}
func (m *MachineInstance) Truth() starlark.Bool { return starlark.True }
func (m *MachineInstance) Hash() (uint32, error) {
	return 0, fmt.Errorf("unhashable type: MachineInstance")
}

func (m *MachineInstance) Attr(name string) (starlark.Value, error) {
	switch name {
	case "send":
		return starlark.NewBuiltin("send", m.send), nil
	case "context":
		return m.getContext(), nil
	case "matches":
		return starlark.NewBuiltin("matches", m.matches), nil
	default:
		return nil, nil
	}
}

func (m *MachineInstance) AttrNames() []string {
	return []string{"send", "context", "matches"}
}

func (m *MachineInstance) send(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	var eventName string
	if err := starlark.UnpackArgs("send", args, kwargs, "event", &eventName); err != nil {
		return nil, err
	}

	transitioned, err := m.Wrapper.Step(eventName)
	if err != nil {
		return nil, err
	}

	return starlark.Bool(transitioned), nil
}

func (m *MachineInstance) getContext() starlark.Value {
	// TODO: Return a wrapper around the context that allows detailed inspection/modification
	// For now, simple conversion
	ctx := m.Wrapper.GetContext()
	return fieldsToDict(ctx.Fields)
}

func (m *MachineInstance) matches(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	var stateName string
	if err := starlark.UnpackArgs("matches", args, kwargs, "state", &stateName); err != nil {
		return nil, err
	}

	config := m.Wrapper.GetCurrentConfiguration()
	for _, s := range config.States {
		if s.Label == stateName {
			return starlark.True, nil
		}
		// TODO: Handle hierarchical matches (matches checks if state is active, including ancestors)
		// semantics package might have helper for this.
	}
	return starlark.False, nil
}

// Helper to convert structpb fields to starlark dict
func fieldsToDict(fields map[string]*structpb.Value) *starlark.Dict {
	d := starlark.NewDict(len(fields))
	for k, v := range fields {
		// Simplified conversion
		d.SetKey(starlark.String(k), starlark.String(v.String()))
	}
	return d
}

// Add 'start' method to MachineDef
func (md *MachineDef) Attr(name string) (starlark.Value, error) {
	if name == "start" {
		return starlark.NewBuiltin("start", md.start), nil
	}
	return nil, nil
}

func (md *MachineDef) AttrNames() []string {
	return []string{"start"}
}

func (md *MachineDef) start(thread *starlark.Thread, b *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	// TODO: Handle initial context
	if err := starlark.UnpackArgs("start", args, kwargs); err != nil {
		return nil, err
	}

	protoMachine, err := md.ToProto()
	if err != nil {
		return nil, fmt.Errorf("failed to convert to proto: %w", err)
	}

	scChart := semantics.NewStatechart(protoMachine)
	wrapper, err := semantics.NewMachine(scChart, md.Name+"_instance", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create machine: %w", err)
	}

	if err := wrapper.Start(); err != nil {
		return nil, fmt.Errorf("failed to start machine: %w", err)
	}

	return &MachineInstance{Wrapper: wrapper}, nil
}
