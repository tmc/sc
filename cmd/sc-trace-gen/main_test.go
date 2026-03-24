package main

import (
	"encoding/json"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	semantics "github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/encoding/protojson"
)

func TestLoadStatechart(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "traffic_light.json")

	chart, err := loadStatechart(path)
	if err != nil {
		t.Fatalf("loadStatechart(%q): %v", path, err)
	}
	if chart.Name != "traffic_light" {
		t.Fatalf("chart.Name = %q, want %q", chart.Name, "traffic_light")
	}

	events := collectEvents(chart)
	if len(events) != 1 || events[0] != "TIMER" {
		t.Fatalf("collectEvents() = %v, want [TIMER]", events)
	}
}

func TestGenerateOutputIncludesTopologyAndTrace(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "history_example.json")

	chart, err := loadStatechart(path)
	if err != nil {
		t.Fatalf("loadStatechart(%q): %v", path, err)
	}
	wrapped := semantics.NewStatechart(chart)
	if err := wrapped.Validate(); err != nil {
		t.Fatalf("wrapped.Validate(): %v", err)
	}

	events := collectEvents(chart)
	data, err := generateOutput(chart, wrapped, path, events, 1, 5, 7, rand.New(rand.NewSource(7)), false, detailStandard)
	if err != nil {
		t.Fatalf("generateOutput(): %v", err)
	}

	var payload traceDataset
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("json.Unmarshal payload: %v", err)
	}
	if payload.StatechartID != "history_example" {
		t.Fatalf("payload.StatechartID = %q, want %q", payload.StatechartID, "history_example")
	}
	if len(payload.Traces) != 1 {
		t.Fatalf("len(payload.Traces) = %d, want 1", len(payload.Traces))
	}

	if !containsHistoryState(payload.GroundTruthTopology) {
		t.Fatal("ground truth topology does not contain a history state")
	}

	var trace statechartspb.ExecutionTrace
	if err := protojson.Unmarshal(payload.Traces[0], &trace); err != nil {
		t.Fatalf("protojson.Unmarshal(trace): %v", err)
	}
	if trace.TraceId == "" {
		t.Fatal("trace.TraceId is empty")
	}
	if trace.InitialConfig == nil || trace.FinalConfig == nil {
		t.Fatal("trace configs were not populated")
	}
	if len(trace.Entries) == 0 {
		t.Fatal("trace has no entries")
	}

	entry := trace.Entries[0]
	if entry.TriggerEvent == nil || entry.TriggerEvent.Label == "" {
		t.Fatal("first entry missing trigger event")
	}
	if entry.SourceConfig == nil || entry.TargetConfig == nil {
		t.Fatal("first entry missing configurations")
	}
}

func containsHistoryState(topology *groundTruthTopology) bool {
	if topology == nil {
		return false
	}
	for _, state := range topology.States {
		if state != nil && state.IsHistory {
			return true
		}
	}
	return false
}

func TestBuildTopologyIncludesRootAndEdges(t *testing.T) {
	chart := &sc.Statechart{
		Name: "tiny",
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "go",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "NEXT",
			},
		},
	}

	topology := buildTopology(chart)
	if topology == nil {
		t.Fatal("buildTopology() returned nil")
	}
	if len(topology.States) != 3 {
		t.Fatalf("len(topology.States) = %d, want 3", len(topology.States))
	}
	if len(topology.Edges) != 1 {
		t.Fatalf("len(topology.Edges) = %d, want 1", len(topology.Edges))
	}
	if topology.Edges[0].Event != "NEXT" {
		t.Fatalf("topology.Edges[0].Event = %q, want %q", topology.Edges[0].Event, "NEXT")
	}
}
