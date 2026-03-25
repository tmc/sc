package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	"google.golang.org/protobuf/encoding/protojson"
)

func TestSimpleTextInput(t *testing.T) {
	input := strings.Join([]string{
		"A -> B [go]",
		"B -> C [step]",
		"C -> A [loop]",
		"",
	}, "\n")

	chart := runAndDecodeChart(t, []string{"-quiet"}, input)
	if chart.GetRootState().GetLabel() != "__root__" {
		t.Fatalf("root label = %q, want __root__", chart.GetRootState().GetLabel())
	}
	if chart.GetRootState().GetType() != sc.StateTypeOR {
		t.Fatalf("root type = %v, want %v", chart.GetRootState().GetType(), sc.StateTypeOR)
	}
	if len(chart.GetTransitions()) != 3 {
		t.Fatalf("len(transitions) = %d, want 3", len(chart.GetTransitions()))
	}

	stateA := findState(chart.GetRootState(), "A")
	if stateA == nil || !stateA.GetIsInitial() {
		t.Fatal("state A was not marked initial")
	}
}

func TestJSONLinesInput(t *testing.T) {
	input := strings.Join([]string{
		`{"from":"Off","to":"On","event":"TOGGLE"}`,
		`{"from":"On","to":"Off","event":"TOGGLE"}`,
		"",
	}, "\n")

	chart := runAndDecodeChart(t, []string{"-quiet"}, input)
	if len(chart.GetTransitions()) != 2 {
		t.Fatalf("len(transitions) = %d, want 2", len(chart.GetTransitions()))
	}
	if findState(chart.GetRootState(), "Off") == nil {
		t.Fatal("missing Off state")
	}
	if findState(chart.GetRootState(), "On") == nil {
		t.Fatal("missing On state")
	}
}

func TestTraceInput(t *testing.T) {
	trace := &statechartspb.ExecutionTrace{
		InitialConfig: config("__root__", "Off"),
		Entries: []*statechartspb.TransitionLogEntry{
			entry("TOGGLE", []string{"__root__", "Off"}, []string{"__root__", "On"},
				ref("Off", "On", "TOGGLE")),
			entry("TOGGLE", []string{"__root__", "On"}, []string{"__root__", "Off"},
				ref("On", "Off", "TOGGLE")),
		},
	}

	chart := runAndDecodeChart(t, []string{"-quiet"}, string(mustTraceDataset(t, trace)))
	if len(chart.GetTransitions()) != 2 {
		t.Fatalf("len(transitions) = %d, want 2", len(chart.GetTransitions()))
	}
	if findState(chart.GetRootState(), "Off") == nil || findState(chart.GetRootState(), "On") == nil {
		t.Fatal("toggle states were not inferred")
	}
}

func TestHierarchyInference(t *testing.T) {
	trace := &statechartspb.ExecutionTrace{
		InitialConfig: config("__root__", "A", "A1"),
		Entries: []*statechartspb.TransitionLogEntry{
			entry("NEXT", []string{"__root__", "A", "A1"}, []string{"__root__", "A", "A2"},
				ref("A1", "A2", "NEXT")),
			entry("SWITCH", []string{"__root__", "A", "A2"}, []string{"__root__", "B", "B1"},
				ref("A2", "B1", "SWITCH")),
			entry("NEXT", []string{"__root__", "B", "B1"}, []string{"__root__", "B", "B2"},
				ref("B1", "B2", "NEXT")),
			entry("SWITCH", []string{"__root__", "B", "B2"}, []string{"__root__", "A", "A1"},
				ref("B2", "A1", "SWITCH")),
		},
	}

	chart := runAndDecodeChart(t, []string{"-quiet"}, string(mustTraceDataset(t, trace)))
	root := chart.GetRootState()
	if root.GetType() != sc.StateTypeOR {
		t.Fatalf("root type = %v, want %v", root.GetType(), sc.StateTypeOR)
	}

	stateA := findState(root, "A")
	stateB := findState(root, "B")
	if stateA == nil || stateB == nil {
		t.Fatal("missing inferred composite states A/B")
	}
	if stateA.GetType() != sc.StateTypeOR || stateB.GetType() != sc.StateTypeOR {
		t.Fatal("A and B should be OR states")
	}
	if !stateA.GetIsInitial() {
		t.Fatal("A should be initial")
	}
	if child := findState(stateA, "A1"); child == nil || !child.GetIsInitial() {
		t.Fatal("A1 should be initial inside A")
	}
	if findState(stateA, "A2") == nil || findState(stateB, "B1") == nil || findState(stateB, "B2") == nil {
		t.Fatal("missing inferred leaf states in hierarchy")
	}
}

func TestParallelDetection(t *testing.T) {
	trace := &statechartspb.ExecutionTrace{
		InitialConfig: config("__root__", "mode", "manual", "status", "idle"),
		Entries: []*statechartspb.TransitionLogEntry{
			entry("TOGGLE_MODE", []string{"__root__", "mode", "manual", "status", "idle"}, []string{"__root__", "mode", "auto", "status", "idle"},
				ref("manual", "auto", "TOGGLE_MODE")),
			entry("START", []string{"__root__", "mode", "auto", "status", "idle"}, []string{"__root__", "mode", "auto", "status", "running"},
				ref("idle", "running", "START")),
			entry("FAULT", []string{"__root__", "mode", "auto", "status", "running"}, []string{"__root__", "mode", "auto", "status", "error"},
				ref("running", "error", "FAULT")),
			entry("RESET", []string{"__root__", "mode", "auto", "status", "error"}, []string{"__root__", "mode", "manual", "status", "idle"},
				ref("auto", "manual", "TOGGLE_MODE"), ref("error", "idle", "RESET")),
		},
	}

	chart := runAndDecodeChart(t, []string{"-quiet"}, string(mustTraceDataset(t, trace)))
	root := chart.GetRootState()
	if root.GetType() != sc.StateTypeAND {
		t.Fatalf("root type = %v, want %v", root.GetType(), sc.StateTypeAND)
	}

	mode := findState(root, "mode")
	status := findState(root, "status")
	if mode == nil || status == nil {
		t.Fatal("missing parallel region states")
	}
	if mode.GetType() != sc.StateTypeOR || status.GetType() != sc.StateTypeOR {
		t.Fatal("parallel region children should be OR states")
	}
	if child := findState(mode, "manual"); child == nil || !child.GetIsInitial() {
		t.Fatal("manual should be initial in mode")
	}
	if child := findState(status, "idle"); child == nil || !child.GetIsInitial() {
		t.Fatal("idle should be initial in status")
	}
}

func TestConfidenceComputation(t *testing.T) {
	input := strings.Join([]string{
		"A -> B [go]",
		"A -> B [go]",
		"A -> C [go]",
		"",
	}, "\n")

	chart := runAndDecodeChart(t, []string{"-quiet"}, input)
	got := transitionMetadataField(chart, []string{"A"}, []string{"B"}, "go", "confidence")
	if got != "0.666667" {
		t.Fatalf("confidence(A->B) = %q, want 0.666667", got)
	}
}

func TestProgressOutput(t *testing.T) {
	input := strings.Join([]string{
		"A -> B [go]",
		"B -> C [step]",
		"",
	}, "\n")

	var stdout, stderr bytes.Buffer
	if err := run(nil, strings.NewReader(input), &stdout, &stderr); err != nil {
		t.Fatalf("run() error: %v", err)
	}
	if !strings.Contains(stderr.String(), "2 observations | 3 states | 2 transitions") {
		t.Fatalf("stderr = %q, want progress line", stderr.String())
	}
}

func runAndDecodeChart(t *testing.T, args []string, input string) *sc.Statechart {
	t.Helper()

	var stdout, stderr bytes.Buffer
	if err := run(args, strings.NewReader(input), &stdout, &stderr); err != nil {
		t.Fatalf("run() error: %v\nstderr=%s", err, stderr.String())
	}

	chart := &sc.Statechart{}
	if err := protojson.Unmarshal(stdout.Bytes(), chart); err != nil {
		t.Fatalf("protojson.Unmarshal() error: %v\noutput=%s", err, stdout.String())
	}
	return chart
}

func findState(root *sc.State, label string) *sc.State {
	if root == nil {
		return nil
	}
	if root.GetLabel() == label {
		return root
	}
	for _, child := range root.GetChildren() {
		if found := findState(child, label); found != nil {
			return found
		}
	}
	return nil
}

func transitionMetadataField(chart *sc.Statechart, from, to []string, event, field string) string {
	for _, transition := range chart.GetTransitions() {
		if !slicesEqual(transition.GetFrom(), from) || !slicesEqual(transition.GetTo(), to) || transition.GetEvent() != event {
			continue
		}
		if transition.GetMetadata() == nil {
			return ""
		}
		value := transition.GetMetadata().GetFields()[field]
		if value == nil {
			return ""
		}
		return value.GetStringValue()
	}
	return ""
}

func config(labels ...string) *sc.Configuration {
	refs := make([]*sc.StateRef, 0, len(labels))
	for _, label := range labels {
		refs = append(refs, &sc.StateRef{Label: label})
	}
	return &sc.Configuration{States: refs}
}

func ref(from, to, event string) *statechartspb.TransitionRef {
	return &statechartspb.TransitionRef{
		From:  []string{from},
		To:    []string{to},
		Event: event,
	}
}

func entry(event string, source, target []string, refs ...*statechartspb.TransitionRef) *statechartspb.TransitionLogEntry {
	return &statechartspb.TransitionLogEntry{
		TriggerEvent:     &statechartspb.Event{Label: event},
		SourceConfig:     config(source...),
		TargetConfig:     config(target...),
		TransitionsFired: refs,
	}
}

func mustTraceDataset(t *testing.T, traces ...*statechartspb.ExecutionTrace) []byte {
	t.Helper()

	raw := make([]json.RawMessage, 0, len(traces))
	for _, trace := range traces {
		data, err := protojson.MarshalOptions{
			UseProtoNames: true,
		}.Marshal(trace)
		if err != nil {
			t.Fatalf("marshal trace: %v", err)
		}
		raw = append(raw, json.RawMessage(data))
	}
	payload, err := json.Marshal(traceDataset{Traces: raw})
	if err != nil {
		t.Fatalf("marshal dataset: %v", err)
	}
	return payload
}

func slicesEqual(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for idx := range left {
		if left[idx] != right[idx] {
			return false
		}
	}
	return true
}
