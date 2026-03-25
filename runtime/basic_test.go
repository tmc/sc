package runtime

import (
	"fmt"
	"testing"

	"go.starlark.net/starlark"
)

func TestBasicModule(t *testing.T) {
	// Simple script that uses the sc module
	script := `
load("sc", "sc")

def traffic_light():
    return sc.machine(
        name = "traffic_light",
        states = [
            sc.state("green", initial=True, transitions=[
                sc.transition(to="yellow", event="TIMER"),
            ]),
            sc.state("yellow", transitions=[
                sc.transition(to="red", event="TIMER"),
            ]),
            sc.state("red", transitions=[
                sc.transition(to="green", event="TIMER"),
            ]),
        ],
    )

m = traffic_light()
print(m)
`

	// Create a new thread
	thread := &starlark.Thread{Name: "test"}

	// Define the load function to return our module
	mod := NewModule()
	thread.Load = func(thread *starlark.Thread, module string) (starlark.StringDict, error) {
		if module == "sc" {
			return starlark.StringDict{"sc": mod}, nil
		}
		return nil, fmt.Errorf("module %s not found", module)
	}

	preload := starlark.StringDict{
		"sc": mod,
	}

	// execute the script
	val, err := starlark.ExecFile(thread, "test.star", script, preload)
	if err != nil {
		t.Fatalf("ExecFile failed: %v", err)
	}

	// The script returns 'm', which is a MachineDef
	machineDef, ok := val["m"]
	if !ok {
		t.Fatal("script did not return 'm'")
	}

	def, ok := machineDef.(*MachineDef)
	if !ok {
		t.Fatalf("expected MachineDef, got %T", machineDef)
	}

	// Test conversion to proto
	proto, err := def.ToProto()
	if err != nil {
		t.Fatalf("ToProto failed: %v", err)
	}

	if proto.Name != "traffic_light" {
		t.Errorf("expected name 'traffic_light', got %q", proto.Name)
	}
	if len(proto.RootState.Children) != 3 {
		t.Errorf("expected 3 top-level states, got %d", len(proto.RootState.Children))
	}
	if len(proto.Transitions) != 3 {
		t.Errorf("expected 3 transitions, got %d", len(proto.Transitions))
	}

	// Test runtime
	runtimeScript := script + `
inst = m.start()
initial_green = inst.matches("green")
transitioned = inst.send("TIMER")
now_yellow = inst.matches("yellow")
`
	val, err = starlark.ExecFile(thread, "test_runtime.star", runtimeScript, preload)
	if err != nil {
		t.Fatalf("Runtime ExecFile failed: %v", err)
	}

	if val["initial_green"] != starlark.True {
		t.Error("expected initial state to check matching 'green'")
	}
	if val["transitioned"] != starlark.True {
		t.Error("expected transition to occur")
	}
	if val["now_yellow"] != starlark.True {
		t.Error("expected state to be 'yellow' after transition")
	}
}
