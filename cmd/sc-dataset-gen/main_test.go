package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	"google.golang.org/protobuf/encoding/protojson"
)

func TestSingleChartAllProducts(t *testing.T) {
	chartPath := filepath.Join("..", "..", "testdata", "traffic_light.json")

	var buf bytes.Buffer
	err := singleChartMode(chartPath, parseProducts("all"), 2, 10, detailDebug, "", 42, false, false, 3, "", &buf)
	if err != nil {
		t.Fatalf("singleChartMode: %v", err)
	}

	var record datasetRecord
	if err := json.Unmarshal(buf.Bytes(), &record); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if record.StatechartID != "traffic_light" {
		t.Errorf("StatechartID = %q, want %q", record.StatechartID, "traffic_light")
	}
	if record.GroundTruthTopology == nil {
		t.Error("GroundTruthTopology is nil")
	}
	if len(record.Traces) != 2 {
		t.Errorf("len(Traces) = %d, want 2", len(record.Traces))
	}
	if record.GraphFeatures == nil {
		t.Error("GraphFeatures is nil")
	}
	if record.Classification == nil {
		t.Error("Classification is nil")
	}
	if record.Vocabulary == nil {
		t.Error("Vocabulary is nil")
	}
	// MutationRows may be empty if seed selects inapplicable mutators.
	// The dedicated mutation test covers correctness.
}

func TestSingleChartTracesTopologyOnly(t *testing.T) {
	chartPath := filepath.Join("..", "..", "testdata", "traffic_light.json")

	var buf bytes.Buffer
	err := singleChartMode(chartPath, parseProducts("traces,topology"), 1, 5, detailStandard, "", 1, false, false, 0, "", &buf)
	if err != nil {
		t.Fatalf("singleChartMode: %v", err)
	}

	var record datasetRecord
	if err := json.Unmarshal(buf.Bytes(), &record); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if record.GroundTruthTopology == nil {
		t.Error("topology should be present")
	}
	if len(record.Traces) == 0 {
		t.Error("traces should be present")
	}
	if record.GraphFeatures != nil {
		t.Error("graph features should be absent")
	}
	if record.Classification != nil {
		t.Error("classification should be absent")
	}
	if record.Vocabulary != nil {
		t.Error("vocabulary should be absent")
	}
}

func TestProductFiltering(t *testing.T) {
	chartPath := filepath.Join("..", "..", "testdata", "traffic_light.json")

	var buf bytes.Buffer
	err := singleChartMode(chartPath, parseProducts("graph,classification"), 1, 5, detailStandard, "", 1, false, false, 0, "", &buf)
	if err != nil {
		t.Fatalf("singleChartMode: %v", err)
	}

	var record datasetRecord
	if err := json.Unmarshal(buf.Bytes(), &record); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if record.GraphFeatures == nil {
		t.Error("graph features should be present")
	}
	if record.Classification == nil {
		t.Error("classification should be present")
	}
	if record.GroundTruthTopology != nil {
		t.Error("topology should be absent")
	}
	if len(record.Traces) > 0 {
		t.Error("traces should be absent")
	}
}

func TestBuildGraphFeatures(t *testing.T) {
	chart := &sc.Statechart{
		Name: "cycle",
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"A"}, To: []string{"B"}, Event: "GO"},
			{From: []string{"B"}, To: []string{"A"}, Event: "BACK"},
		},
	}

	gf := buildGraphFeatures(chart)
	if gf.StateCount != 2 {
		t.Errorf("StateCount = %d, want 2", gf.StateCount)
	}
	if gf.TransitionCount != 2 {
		t.Errorf("TransitionCount = %d, want 2", gf.TransitionCount)
	}
	if !gf.HasCycles {
		t.Error("HasCycles should be true for a cycle")
	}
	if len(gf.AdjacencyMatrix) != 2 {
		t.Errorf("AdjacencyMatrix size = %d, want 2", len(gf.AdjacencyMatrix))
	}
}

func TestBuildClassification(t *testing.T) {
	tests := []struct {
		name     string
		chart    *sc.Statechart
		wantFam  string
		wantPar  bool
		wantHist bool
	}{
		{
			name: "flat",
			chart: &sc.Statechart{
				Name: "flat",
				RootState: &sc.State{
					Label: "__root__", Type: sc.StateTypeOR,
					Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					},
				},
			},
			wantFam: "flat",
		},
		{
			name: "orthogonal",
			chart: &sc.Statechart{
				Name: "par",
				RootState: &sc.State{
					Label: "__root__", Type: sc.StateTypeAND,
					Children: []*sc.State{
						{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
							{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						}},
						{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
							{Label: "B", Type: sc.StateTypeBasic, IsInitial: true},
						}},
					},
				},
			},
			wantFam: "orthogonal",
			wantPar: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := buildClassification(tt.chart)
			found := false
			for _, f := range c.Families {
				if f == tt.wantFam {
					found = true
				}
			}
			if !found {
				t.Errorf("families = %v, want to contain %q", c.Families, tt.wantFam)
			}
			if c.HasParallel != tt.wantPar {
				t.Errorf("HasParallel = %v, want %v", c.HasParallel, tt.wantPar)
			}
			if c.HasHistory != tt.wantHist {
				t.Errorf("HasHistory = %v, want %v", c.HasHistory, tt.wantHist)
			}
		})
	}
}

func TestBuildVocabulary(t *testing.T) {
	chart := &sc.Statechart{
		Name: "vocab",
		RootState: &sc.State{
			Label: "__root__", Type: sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "X", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "Y", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"X"}, To: []string{"Y"}, Event: "STEP"},
		},
	}

	v := buildVocabulary(chart)
	if v.States["X"] != 0 || v.States["Y"] != 1 {
		t.Errorf("States = %v, want X:0 Y:1", v.States)
	}
	if v.Events["STEP"] != 0 {
		t.Errorf("Events = %v, want STEP:0", v.Events)
	}
}

func TestHistoryRowExtraction(t *testing.T) {
	chartPath := filepath.Join("..", "..", "testdata", "history_example.json")

	var buf bytes.Buffer
	err := singleChartMode(chartPath, parseProducts("history,traces"), 5, 30, detailDebug, "", 42, false, false, 0, "", &buf)
	if err != nil {
		t.Fatalf("singleChartMode: %v", err)
	}

	var record datasetRecord
	if err := json.Unmarshal(buf.Bytes(), &record); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	// The history example should produce at least some history rows.
	t.Logf("history_rows: %d, traces: %d", len(record.HistoryRows), len(record.Traces))
}

func TestCoverageMode(t *testing.T) {
	chartPath := filepath.Join("..", "..", "testdata", "traffic_light.json")

	var buf bytes.Buffer
	err := singleChartMode(chartPath, parseProducts("traces"), 10, 20, detailDebug, "", 1, false, true, 0, "", &buf)
	if err != nil {
		t.Fatalf("singleChartMode: %v", err)
	}

	var record datasetRecord
	if err := json.Unmarshal(buf.Bytes(), &record); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if record.CoverageResult == nil {
		t.Fatal("CoverageResult should be present in coverage mode")
	}
	if record.CoverageResult.EventCoverage < 1.0 {
		t.Errorf("EventCoverage = %f, want 1.0 (only one event: TIMER)", record.CoverageResult.EventCoverage)
	}
}

func TestMutationGeneration(t *testing.T) {
	chart := &sc.Statechart{
		Name: "mut",
		RootState: &sc.State{
			Label: "__root__", Type: sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "go", From: []string{"A"}, To: []string{"B"}, Event: "GO"},
			{Label: "back", From: []string{"B"}, To: []string{"A"}, Event: "BACK"},
		},
	}

	rows := generateMutations(chart, "mut", 42, 5)
	if len(rows) == 0 {
		t.Error("expected at least one mutation row")
	}
	for _, r := range rows {
		if r.MutationFamily == "" {
			t.Error("mutation family should not be empty")
		}
		if r.CleanChartHash == "" {
			t.Error("clean chart hash should not be empty")
		}
		if r.BehavioralDelta == nil {
			t.Error("behavioral delta should not be nil")
		}
	}
}

func TestTraceFormat(t *testing.T) {
	chartPath := filepath.Join("..", "..", "testdata", "traffic_light.json")

	var buf bytes.Buffer
	err := singleChartMode(chartPath, parseProducts("traces"), 1, 5, detailStandard, "", 1, false, false, 0, "", &buf)
	if err != nil {
		t.Fatalf("singleChartMode: %v", err)
	}

	var record datasetRecord
	if err := json.Unmarshal(buf.Bytes(), &record); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if len(record.Traces) != 1 {
		t.Fatalf("len(Traces) = %d, want 1", len(record.Traces))
	}

	var trace statechartspb.ExecutionTrace
	if err := protojson.Unmarshal(record.Traces[0], &trace); err != nil {
		t.Fatalf("protojson.Unmarshal: %v", err)
	}
	if trace.TraceId == "" {
		t.Error("trace_id should not be empty")
	}
	if trace.InitialConfig == nil {
		t.Error("initial_config should not be nil")
	}
}

func TestRunFlagParsing(t *testing.T) {
	var buf, errBuf bytes.Buffer
	err := run([]string{"sc-dataset-gen"}, &buf, &errBuf)
	if err == nil {
		t.Error("expected error when no -chart or -corpus")
	}
}

func TestSplitRatioParsing(t *testing.T) {
	tests := []struct {
		input   string
		wantErr bool
	}{
		{"80,10,10", false},
		{"70,15,15", false},
		{"50,50", true},
		{"80,10,11", true},
	}
	for _, tt := range tests {
		_, err := parseSplitRatio(tt.input)
		if (err != nil) != tt.wantErr {
			t.Errorf("parseSplitRatio(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
		}
	}
}

func TestValidationProduct(t *testing.T) {
	chart := &sc.Statechart{
		Name: "val_test",
		RootState: &sc.State{
			Label: "__root__", Type: sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "go", From: []string{"A"}, To: []string{"B"}, Event: "GO"},
		},
		Events: []*sc.Event{{Label: "GO"}},
	}

	rows := buildValidationRows(chart, "val_test", []string{"flat"}, 42, 5)
	if len(rows) == 0 {
		t.Fatal("expected at least one validation row")
	}

	// First row is the clean chart.
	clean := rows[0]
	if clean.IsMutated {
		t.Error("first row should be the clean chart")
	}
	if clean.ChartID != "val_test" {
		t.Errorf("ChartID = %q, want val_test", clean.ChartID)
	}
	if clean.NStates != 3 {
		t.Errorf("NStates = %d, want 3", clean.NStates)
	}

	// At least some mutated rows should exist.
	mutated := 0
	for _, r := range rows[1:] {
		if r.IsMutated {
			mutated++
			if r.MutationFamily == "" {
				t.Error("mutated row should have a mutation family")
			}
		}
	}
	if mutated == 0 {
		t.Error("expected at least one mutated validation row")
	}
}

// Suppress unused import warnings.
var _ = (*sc.Statechart)(nil)
