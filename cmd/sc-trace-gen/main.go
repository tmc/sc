package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	semantics "github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("sc-trace-gen", flag.ContinueOnError)
	fs.SetOutput(stderr)

	chartPath := fs.String("chart", "", "path to a statechart JSON or textproto file")
	outPath := fs.String("out", "", "write output to this file instead of stdout")
	traceCount := fs.Int("traces", 1, "number of traces to generate")
	stepCount := fs.Int("steps", 20, "maximum number of random events per trace")
	seed := fs.Int64("seed", 1, "PRNG seed for reproducible trace generation")
	eventsOverride := fs.String("events", "", "comma-separated event labels to sample instead of chart events")
	pretty := fs.Bool("pretty", true, "pretty-print JSON output")
	detail := fs.String("detail", "standard", "trace detail level: minimal, standard, or debug")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *chartPath == "" {
		return fmt.Errorf("chart path is required")
	}
	if *traceCount <= 0 {
		return fmt.Errorf("traces must be > 0")
	}
	if *stepCount <= 0 {
		return fmt.Errorf("steps must be > 0")
	}
	dl, err := parseDetailLevel(*detail)
	if err != nil {
		return err
	}

	chart, err := loadStatechart(*chartPath)
	if err != nil {
		return fmt.Errorf("load statechart: %w", err)
	}
	wrapped := semantics.NewStatechart(chart)
	if err := wrapped.Validate(); err != nil {
		return fmt.Errorf("validate statechart: %w", err)
	}

	events := collectEvents(chart)
	if *eventsOverride != "" {
		events = parseCSV(*eventsOverride)
	}
	if len(events) == 0 {
		return fmt.Errorf("statechart has no events and no --events override was provided")
	}

	rng := rand.New(rand.NewSource(*seed))
	traceBytes, err := generateOutput(chart, wrapped, *chartPath, events, *traceCount, *stepCount, *seed, rng, *pretty, dl)
	if err != nil {
		return err
	}

	if *outPath != "" {
		if err := os.WriteFile(*outPath, traceBytes, 0o644); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
		return nil
	}

	if _, err := stdout.Write(traceBytes); err != nil {
		return fmt.Errorf("write stdout: %w", err)
	}
	if len(traceBytes) == 0 || traceBytes[len(traceBytes)-1] != '\n' {
		if _, err := stdout.Write([]byte("\n")); err != nil {
			return fmt.Errorf("write trailing newline: %w", err)
		}
	}
	return nil
}

type detailLevel int

const (
	detailMinimal  detailLevel = iota // event, configs, fired transitions, summary counts
	detailStandard                    // + guard pass/fail with transition ref, context before/after
	detailDebug                       // + bound_values, no-op steps, errors, full context
)

func parseDetailLevel(s string) (detailLevel, error) {
	switch strings.ToLower(s) {
	case "minimal":
		return detailMinimal, nil
	case "standard", "":
		return detailStandard, nil
	case "debug":
		return detailDebug, nil
	default:
		return 0, fmt.Errorf("invalid -detail level %q: must be minimal, standard, or debug", s)
	}
}

type traceDataset struct {
	StatechartID        string               `json:"statechart_id"`
	SourceFile          string               `json:"source_file"`
	Seed                int64                `json:"seed"`
	Events              []string             `json:"events"`
	GroundTruthTopology *groundTruthTopology `json:"ground_truth_topology"`
	Traces              []json.RawMessage    `json:"traces"`
}

type groundTruthTopology struct {
	States []*topologyState `json:"states"`
	Edges  []*topologyEdge  `json:"edges"`
}

type topologyState struct {
	Label       string `json:"label"`
	Type        string `json:"type"`
	Parent      string `json:"parent,omitempty"`
	Depth       int    `json:"depth"`
	IsInitial   bool   `json:"is_initial,omitempty"`
	IsHistory   bool   `json:"is_history,omitempty"`
	HistoryType string `json:"history_type,omitempty"`
}

type topologyEdge struct {
	Label         string   `json:"label,omitempty"`
	From          []string `json:"from"`
	To            []string `json:"to"`
	Event         string   `json:"event,omitempty"`
	Guard         string   `json:"guard,omitempty"`
	Actions       []string `json:"actions,omitempty"`
	SourceDepth   int      `json:"source_depth"`
	TargetDepth   int      `json:"target_depth"`
	LCA           string   `json:"lca,omitempty"`
	IsInterLevel  bool     `json:"is_inter_level"`
	ExitedStates  []string `json:"exited_states,omitempty"`
	EnteredStates []string `json:"entered_states,omitempty"`
}

// historyActivation records when a history pseudostate resolves during a step.
type historyActivation struct {
	HistoryState    string   `json:"history_state"`
	HistoryType     string   `json:"history_type"`
	RestoredConfig  []string `json:"restored_config"`
	HadPriorHistory bool     `json:"had_prior_history"`
}

// transitionMetadata captures inter-level and history details for a trace step.
type transitionMetadata struct {
	SourceDepth       int                  `json:"source_depth,omitempty"`
	TargetDepth       int                  `json:"target_depth,omitempty"`
	LCA               string               `json:"lca,omitempty"`
	IsInterLevel      bool                 `json:"is_inter_level,omitempty"`
	ExitedStates      []string             `json:"exited_states,omitempty"`
	EnteredStates     []string             `json:"entered_states,omitempty"`
	HistoryActivation *historyActivation   `json:"history_activation,omitempty"`
}

func generateOutput(chart *sc.Statechart, wrapped *semantics.Statechart, sourcePath string, events []string, traceCount, stepCount int, seed int64, rng *rand.Rand, pretty bool, dl detailLevel) ([]byte, error) {
	chartID := chart.Name
	if chartID == "" {
		base := filepath.Base(sourcePath)
		chartID = strings.TrimSuffix(base, filepath.Ext(base))
	}

	traces := make([]json.RawMessage, 0, traceCount)
	for i := 0; i < traceCount; i++ {
		trace, err := generateTrace(chartID, chart, wrapped, events, stepCount, i, rng, dl)
		if err != nil {
			return nil, fmt.Errorf("generate trace %d: %w", i, err)
		}
		traceJSON, err := marshalProtoJSON(trace, pretty)
		if err != nil {
			return nil, fmt.Errorf("marshal trace %d: %w", i, err)
		}
		traces = append(traces, json.RawMessage(traceJSON))
	}

	payload := traceDataset{
		StatechartID:        chartID,
		SourceFile:          sourcePath,
		Seed:                seed,
		Events:              events,
		GroundTruthTopology: buildTopology(chart),
		Traces:              traces,
	}

	if pretty {
		return json.MarshalIndent(payload, "", "  ")
	}
	return json.Marshal(payload)
}

func generateTrace(chartID string, chart *sc.Statechart, wrapped *semantics.Statechart, events []string, stepCount, index int, rng *rand.Rand, dl detailLevel) (*statechartspb.ExecutionTrace, error) {
	machineID := fmt.Sprintf("%s-%03d", chartID, index)
	machine, err := semantics.NewMachine(wrapped, machineID, emptyContext())
	if err != nil {
		return nil, fmt.Errorf("create machine: %w", err)
	}
	if err := machine.Start(); err != nil {
		return nil, fmt.Errorf("start machine: %w", err)
	}

	startedAt := time.Now()
	entries := make([]*statechartspb.TransitionLogEntry, 0, stepCount)
	var ignored uint64
	var firedTotal uint64

	for step := 0; step < stepCount; step++ {
		if !machine.IsRunning() {
			break
		}

		eventLabel := events[rng.Intn(len(events))]
		beforeConfig := cloneConfiguration(machine.GetCurrentConfiguration())
		beforeContext := cloneContext(machine.GetContext())

		// Evaluate guards at standard and debug levels.
		var guardResults []*statechartspb.GuardEvaluation
		if dl >= detailStandard {
			guardResults = evaluateGuards(wrapped, beforeConfig, beforeContext, eventLabel, dl)
		}

		stepStart := time.Now()
		transitioned, stepErr := machine.Step(eventLabel)
		stepDuration := time.Since(stepStart)

		afterConfig := cloneConfiguration(machine.GetCurrentConfiguration())
		afterContext := cloneContext(machine.GetContext())

		var fired []*statechartspb.TransitionRef
		var actions []*statechartspb.ActionExecution
		errorText := ""
		if stepErr != nil {
			errorText = stepErr.Error()
		}
		if transitioned {
			history := machine.GetStepHistory()
			if len(history) > 0 {
				last := history[len(history)-1]
				fired = transitionRefs(last.Transitions)
				actions = actionExecutions(last.Transitions)
			}
			firedTotal += uint64(len(fired))
		} else {
			ignored++
			// At minimal level, skip no-op steps entirely.
			if dl == detailMinimal {
				continue
			}
		}

		entry := &statechartspb.TransitionLogEntry{
			Id:               fmt.Sprintf("%s-step-%03d", machineID, step),
			Timestamp:        timestamppb.New(stepStart),
			Sequence:         uint64(step + 1),
			TriggerEvent:     &sc.Event{Label: eventLabel},
			SourceConfig:     beforeConfig,
			TargetConfig:     afterConfig,
			TransitionsFired: fired,
			GuardResults:     guardResults,
			ActionsExecuted:  actions,
			ProcessingTime:   durationpb.New(stepDuration),
			Error:            errorText,
			Metadata: map[string]string{
				"transitioned": fmt.Sprintf("%t", transitioned),
			},
		}

		// Context snapshots at standard+ levels.
		if dl >= detailStandard {
			entry.ContextBefore = beforeContext
			entry.ContextAfter = afterContext
		}

		// At standard+ detail, add inter-level and history metadata.
		if dl >= detailStandard && transitioned && len(fired) > 0 {
			tm := buildTransitionMetadata(wrapped, chart, fired, beforeConfig, afterConfig)
			if tm != nil {
				if b, err := json.Marshal(tm); err == nil {
					entry.Metadata["transition_metadata"] = string(b)
				}
			}
		}

		entries = append(entries, entry)
	}

	return &statechartspb.ExecutionTrace{
		TraceId:        fmt.Sprintf("%s-trace-%03d", chartID, index),
		MachineId:      machineID,
		ChartVersion:   buildChartVersion(chart),
		InitialConfig:  initialConfig(machine),
		InitialContext: emptyContext(),
		Entries:        entries,
		FinalConfig:    cloneConfiguration(machine.GetCurrentConfiguration()),
		FinalContext:   cloneContext(machine.GetContext()),
		Metadata: &statechartspb.TraceMetadata{
			StartedAt:            timestamppb.New(startedAt),
			EndedAt:              timestamppb.New(time.Now()),
			TotalTransitions:     firedTotal,
			TotalEventsProcessed: uint64(len(entries)),
			TotalEventsIgnored:   ignored,
		},
	}, nil
}

func initialConfig(machine *semantics.MachineWrapper) *sc.Configuration {
	history := machine.GetStepHistory()
	if len(history) == 0 {
		return cloneConfiguration(machine.GetCurrentConfiguration())
	}
	return cloneConfiguration(history[0].ResultingConfiguration)
}

func loadStatechart(path string) (*sc.Statechart, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if chart, err := parseProtoJSONChart(data); err == nil {
		return chart, nil
	}
	if chart, err := parseProtoTextChart(data); err == nil {
		return chart, nil
	}
	return nil, fmt.Errorf("could not parse %s as statechart JSON or textproto", path)
}

func parseProtoJSONChart(data []byte) (*sc.Statechart, error) {
	opts := protojson.UnmarshalOptions{DiscardUnknown: true}

	machine := &sc.Machine{}
	if err := opts.Unmarshal(data, machine); err == nil && machine.Statechart != nil {
		return machine.Statechart, nil
	}

	chart := &sc.Statechart{}
	if err := opts.Unmarshal(data, chart); err == nil && chart.RootState != nil {
		return chart, nil
	}
	return nil, fmt.Errorf("invalid chart JSON")
}

func parseProtoTextChart(data []byte) (*sc.Statechart, error) {
	opts := prototext.UnmarshalOptions{DiscardUnknown: true}

	machine := &sc.Machine{}
	if err := opts.Unmarshal(data, machine); err == nil && machine.Statechart != nil {
		return machine.Statechart, nil
	}

	chart := &sc.Statechart{}
	if err := opts.Unmarshal(data, chart); err == nil && chart.RootState != nil {
		return chart, nil
	}
	return nil, fmt.Errorf("invalid chart textproto")
}

func collectEvents(chart *sc.Statechart) []string {
	seen := make(map[string]bool)
	var events []string
	for _, event := range chart.Events {
		if event == nil || event.Label == "" || seen[event.Label] {
			continue
		}
		seen[event.Label] = true
		events = append(events, event.Label)
	}
	for _, transition := range chart.Transitions {
		if transition == nil || transition.Event == "" || seen[transition.Event] {
			continue
		}
		seen[transition.Event] = true
		events = append(events, transition.Event)
	}
	slices.Sort(events)
	return events
}

func parseCSV(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func buildTopology(chart *sc.Statechart) *groundTruthTopology {
	topology := &groundTruthTopology{
		States: make([]*topologyState, 0),
		Edges:  make([]*topologyEdge, 0, len(chart.Transitions)),
	}

	// Build depth and ancestry maps while walking.
	depthMap := map[string]int{}
	parentMap := map[string]string{}

	var walk func(state *sc.State, parent string, depth int)
	walk = func(state *sc.State, parent string, depth int) {
		if state == nil {
			return
		}
		depthMap[state.Label] = depth
		parentMap[state.Label] = parent
		topology.States = append(topology.States, &topologyState{
			Label:       state.Label,
			Type:        stateTypeString(state.Type),
			Parent:      parent,
			Depth:       depth,
			IsInitial:   state.IsInitial,
			IsHistory:   state.IsHistory,
			HistoryType: historyTypeString(state.HistoryType),
		})
		for _, child := range state.Children {
			walk(child, state.Label, depth+1)
		}
	}
	walk(chart.RootState, "", 0)

	// Helper: ancestors from state to root (inclusive).
	ancestors := func(label string) []string {
		var path []string
		cur := label
		for cur != "" {
			path = append(path, cur)
			cur = parentMap[cur]
		}
		return path
	}

	// Helper: LCA of two labels.
	lca := func(a, b string) string {
		aAnc := ancestors(a)
		bSet := map[string]bool{}
		for _, s := range ancestors(b) {
			bSet[s] = true
		}
		for _, s := range aAnc {
			if bSet[s] {
				return s
			}
		}
		return ""
	}

	for _, transition := range chart.Transitions {
		if transition == nil {
			continue
		}
		edge := &topologyEdge{
			Label:   transition.Label,
			From:    append([]string(nil), transition.From...),
			To:      append([]string(nil), transition.To...),
			Event:   transition.Event,
			Guard:   guardExpression(transition.Guard),
			Actions: actionLabels(transition.Actions),
		}

		// Compute inter-level metadata.
		if len(transition.From) > 0 && len(transition.To) > 0 {
			src := transition.From[0]
			tgt := transition.To[0]
			edge.SourceDepth = depthMap[src]
			edge.TargetDepth = depthMap[tgt]
			lcaLabel := lca(src, tgt)
			edge.LCA = lcaLabel
			edge.IsInterLevel = edge.SourceDepth != edge.TargetDepth

			// Compute exited/entered states (from source up to LCA, from LCA down to target).
			for _, s := range ancestors(src) {
				if s == lcaLabel {
					break
				}
				edge.ExitedStates = append(edge.ExitedStates, s)
			}
			for _, s := range ancestors(tgt) {
				if s == lcaLabel {
					break
				}
				edge.EnteredStates = append(edge.EnteredStates, s)
			}
			// EnteredStates should be shallowest-first (reverse the deepest-first order).
			slices.Reverse(edge.EnteredStates)
		}

		topology.Edges = append(topology.Edges, edge)
	}
	return topology
}

// buildTransitionMetadata computes inter-level and history metadata for fired transitions.
func buildTransitionMetadata(wrapped *semantics.Statechart, chart *sc.Statechart, fired []*statechartspb.TransitionRef, beforeConfig, afterConfig *sc.Configuration) *transitionMetadata {
	if len(fired) == 0 {
		return nil
	}

	// Build depth, parent, and state lookup maps from chart.
	depthMap := map[string]int{}
	parentMap := map[string]string{}
	stateMap := map[string]*sc.State{}
	var walkDepths func(state *sc.State, parent string, depth int)
	walkDepths = func(state *sc.State, parent string, depth int) {
		if state == nil {
			return
		}
		depthMap[state.Label] = depth
		parentMap[state.Label] = parent
		stateMap[state.Label] = state
		for _, child := range state.Children {
			walkDepths(child, state.Label, depth+1)
		}
	}
	walkDepths(chart.RootState, "", 0)

	ancestors := func(label string) []string {
		var path []string
		cur := label
		for cur != "" {
			path = append(path, cur)
			cur = parentMap[cur]
		}
		return path
	}

	ref := fired[0]
	tm := &transitionMetadata{}

	if len(ref.From) > 0 && len(ref.To) > 0 {
		src := ref.From[0]
		tgt := ref.To[0]
		tm.SourceDepth = depthMap[src]
		tm.TargetDepth = depthMap[tgt]
		tm.IsInterLevel = tm.SourceDepth != tm.TargetDepth

		lcaLabel, err := wrapped.LeastCommonAncestor(semantics.StateLabel(src), semantics.StateLabel(tgt))
		if err == nil {
			tm.LCA = string(lcaLabel)
		}

		lcaStr := tm.LCA
		for _, s := range ancestors(src) {
			if s == lcaStr {
				break
			}
			tm.ExitedStates = append(tm.ExitedStates, s)
		}
		entered := ancestors(tgt)
		for i := range entered {
			if entered[i] == lcaStr {
				entered = entered[:i]
				break
			}
		}
		slices.Reverse(entered)
		tm.EnteredStates = entered
	}

	// Detect history activation: check if any target was a history state
	// by comparing before/after configs — if target labels in the transition
	// differ from actual after-config states, history resolved.
	if len(ref.To) > 0 {
		for _, tgt := range ref.To {
			// Find the state in the chart to check is_history.
			state := stateMap[tgt]
			if state == nil || !state.IsHistory {
				continue
			}
			// This target was a history state — figure out what it resolved to.
			afterLabels := configLabels(afterConfig)
			beforeLabels := configLabels(beforeConfig)
			// Restored config = states in afterConfig that weren't in beforeConfig.
			restored := make([]string, 0)
			beforeSet := map[string]bool{}
			for _, l := range beforeLabels {
				beforeSet[l] = true
			}
			for _, l := range afterLabels {
				if !beforeSet[l] {
					restored = append(restored, l)
				}
			}

			histType := "shallow"
			if state.HistoryType == sc.HistoryType_HISTORY_TYPE_DEEP {
				histType = "deep"
			}

			tm.HistoryActivation = &historyActivation{
				HistoryState:    tgt,
				HistoryType:     histType,
				RestoredConfig:  restored,
				HadPriorHistory: len(restored) > 0,
			}
			break
		}
	}

	return tm
}

// configLabels extracts state labels from a configuration.
func configLabels(config *sc.Configuration) []string {
	if config == nil {
		return nil
	}
	labels := make([]string, 0, len(config.States))
	for _, s := range config.States {
		if s != nil {
			labels = append(labels, s.Label)
		}
	}
	return labels
}

func evaluateGuards(chart *semantics.Statechart, config *sc.Configuration, context *structpb.Struct, eventLabel string, dl detailLevel) []*statechartspb.GuardEvaluation {
	event := &sc.Event{
		Label:      eventLabel,
		Parameters: emptyContext(),
	}
	active := make(map[string]bool, len(config.States))
	for _, state := range config.States {
		if state != nil {
			active[state.Label] = true
		}
	}

	results := make([]*statechartspb.GuardEvaluation, 0)
	for _, transition := range chart.Transitions {
		if transition == nil {
			continue
		}
		if transition.Event != eventLabel && transition.Event != "" {
			continue
		}
		if !transitionTouchesActiveState(transition, active) {
			continue
		}
		passed, err := chart.EvaluateGuardWithTransition(transition.Guard, context, event, transition)
		guard := &statechartspb.GuardEvaluation{
			GuardExpression: guardExpression(transition.Guard),
			Result:          passed,
			Transition: &statechartspb.TransitionRef{
				Label: transition.Label,
				From:  append([]string(nil), transition.From...),
				To:    append([]string(nil), transition.To...),
				Event: transition.Event,
			},
		}
		// Include bound_values only at debug level.
		if dl >= detailDebug {
			guard.BoundValues = cloneContext(context)
		}
		if err != nil {
			guard.Error = err.Error()
		}
		results = append(results, guard)
	}
	return results
}

func transitionTouchesActiveState(transition *sc.Transition, active map[string]bool) bool {
	for _, source := range transition.From {
		if active[source] {
			return true
		}
	}
	return false
}

func transitionRefs(transitions []*sc.Transition) []*statechartspb.TransitionRef {
	refs := make([]*statechartspb.TransitionRef, 0, len(transitions))
	for _, transition := range transitions {
		if transition == nil {
			continue
		}
		refs = append(refs, &statechartspb.TransitionRef{
			Label: transition.Label,
			From:  append([]string(nil), transition.From...),
			To:    append([]string(nil), transition.To...),
			Event: transition.Event,
		})
	}
	return refs
}

func actionExecutions(transitions []*sc.Transition) []*statechartspb.ActionExecution {
	actions := make([]*statechartspb.ActionExecution, 0)
	for _, transition := range transitions {
		if transition == nil {
			continue
		}
		for _, action := range transition.Actions {
			if action == nil {
				continue
			}
			actions = append(actions, &statechartspb.ActionExecution{
				ActionName: action.Label,
				Parameters: emptyContext(),
			})
		}
	}
	return actions
}

func actionLabels(actions []*sc.Action) []string {
	labels := make([]string, 0, len(actions))
	for _, action := range actions {
		if action != nil && action.Label != "" {
			labels = append(labels, action.Label)
		}
	}
	return labels
}

func guardExpression(guard *sc.Guard) string {
	if guard == nil {
		return ""
	}
	if guard.Expression != "" {
		return guard.Expression
	}
	if guard.Condition != nil {
		return guard.Condition.Source
	}
	return ""
}

func stateTypeString(t sc.StateType) string {
	switch t {
	case sc.StateTypeBasic:
		return "basic"
	case sc.StateTypeOR:
		return "or"
	case sc.StateTypeAND:
		return "and"
	default:
		return "unspecified"
	}
}

func historyTypeString(t sc.HistoryType) string {
	switch t {
	case sc.HistoryType_HISTORY_TYPE_SHALLOW:
		return "shallow"
	case sc.HistoryType_HISTORY_TYPE_DEEP:
		return "deep"
	default:
		return ""
	}
}

func buildChartVersion(chart *sc.Statechart) *statechartspb.ChartVersion {
	b, err := protojson.MarshalOptions{
		UseProtoNames:   true,
		EmitUnpopulated: false,
	}.Marshal(chart)
	if err != nil {
		b = []byte(chart.Name)
	}
	sum := sha256.Sum256(b)
	return &statechartspb.ChartVersion{
		Version:     "trace-gen",
		ContentHash: hex.EncodeToString(sum[:]),
		CreatedAt:   timestamppb.New(time.Now()),
	}
}

func marshalProtoJSON(message proto.Message, pretty bool) ([]byte, error) {
	opts := protojson.MarshalOptions{
		UseProtoNames:   true,
		EmitUnpopulated: false,
	}
	if pretty {
		opts.Multiline = true
		opts.Indent = "  "
	}
	return opts.Marshal(message)
}

func cloneConfiguration(config *sc.Configuration) *sc.Configuration {
	if config == nil {
		return &sc.Configuration{}
	}
	return proto.Clone(config).(*sc.Configuration)
}

func cloneContext(context *structpb.Struct) *structpb.Struct {
	if context == nil {
		return emptyContext()
	}
	return proto.Clone(context).(*structpb.Struct)
}

func emptyContext() *structpb.Struct {
	return &structpb.Struct{Fields: make(map[string]*structpb.Value)}
}
