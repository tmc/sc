package main

import (
	"encoding/json"
	"math/rand"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	semantics "github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/encoding/protojson"
)

type coverageResult struct {
	TracesGenerated int     `json:"traces_generated"`
	HistoryRowCount int     `json:"history_row_count"`
	GuardEvalCount  int     `json:"guard_eval_count"`
	GuardCoverage   float64 `json:"guard_coverage"`
	EventCoverage   float64 `json:"event_coverage"`
	GoalsMet        bool    `json:"goals_met"`
}

// generateCoverageTraces generates traces until coverage goals are met or maxTraces is reached.
func generateCoverageTraces(chart *sc.Statechart, wrapped *semantics.Statechart, chartID string, events []string, maxTraces, maxSteps int, seed int64, dl detailLevel, pretty bool) ([]json.RawMessage, *coverageResult, error) {
	rng := rand.New(rand.NewSource(seed))

	needHistory := hasHistoryStates(chart)
	needGuards := hasGuards(chart)

	// Track guard coverage: guard expression -> {true: seen, false: seen}.
	guardSeen := make(map[string]map[bool]bool)
	historyFound := false
	eventsSeen := make(map[string]bool)

	var allTraces []json.RawMessage
	result := &coverageResult{}

	for i := 0; i < maxTraces; i++ {
		trace, err := generateTrace(chartID, chart, wrapped, events, maxSteps, i, rng, dl)
		if err != nil {
			return allTraces, result, err
		}
		traceJSON, err := marshalProtoJSON(trace, pretty)
		if err != nil {
			return allTraces, result, err
		}
		allTraces = append(allTraces, json.RawMessage(traceJSON))

		// Analyze trace for coverage.
		analyzeTraceCoverage(trace, chart, chartID, &historyFound, guardSeen, eventsSeen, result)

		// Check if goals are met.
		if goalsAreMet(needHistory, needGuards, historyFound, guardSeen, eventsSeen, events) {
			break
		}
	}

	// Compute final coverage stats.
	result.TracesGenerated = len(allTraces)
	result.EventCoverage = float64(len(eventsSeen)) / float64(max(len(events), 1))

	if len(guardSeen) > 0 {
		bothSides := 0
		for _, sides := range guardSeen {
			if sides[true] && sides[false] {
				bothSides++
			}
		}
		result.GuardCoverage = float64(bothSides) / float64(len(guardSeen))
	}

	result.GoalsMet = goalsAreMet(needHistory, needGuards, historyFound, guardSeen, eventsSeen, events)
	return allTraces, result, nil
}

func analyzeTraceCoverage(trace *statechartspb.ExecutionTrace, chart *sc.Statechart, chartID string, historyFound *bool, guardSeen map[string]map[bool]bool, eventsSeen map[string]bool, result *coverageResult) {
	// Extract history rows.
	hRows := extractHistoryRows(chartID, trace, chart)
	result.HistoryRowCount += len(hRows)
	if len(hRows) > 0 {
		*historyFound = true
	}

	// Extract guard eval rows.
	gRows := extractGuardEvalRows(chartID, trace)
	result.GuardEvalCount += len(gRows)
	for _, gr := range gRows {
		if gr.GuardExpression == "" {
			continue
		}
		if guardSeen[gr.GuardExpression] == nil {
			guardSeen[gr.GuardExpression] = make(map[bool]bool)
		}
		guardSeen[gr.GuardExpression][gr.Result] = true
	}

	// Track event coverage.
	for _, entry := range trace.Entries {
		if entry != nil && entry.TriggerEvent != nil {
			eventsSeen[entry.TriggerEvent.Label] = true
		}
	}
}

func goalsAreMet(needHistory, needGuards, historyFound bool, guardSeen map[string]map[bool]bool, eventsSeen map[string]bool, events []string) bool {
	if needHistory && !historyFound {
		return false
	}
	if needGuards {
		for _, sides := range guardSeen {
			if !sides[true] || !sides[false] {
				return false
			}
		}
	}
	// Check all events exercised.
	for _, e := range events {
		if !eventsSeen[e] {
			return false
		}
	}
	return true
}

// parseTraceFromJSON parses a raw JSON trace message back into an ExecutionTrace.
func parseTraceFromJSON(raw json.RawMessage) (*statechartspb.ExecutionTrace, error) {
	var trace statechartspb.ExecutionTrace
	if err := protojson.Unmarshal(raw, &trace); err != nil {
		return nil, err
	}
	return &trace, nil
}
