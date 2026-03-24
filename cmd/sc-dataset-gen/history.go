package main

import (
	"encoding/json"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
)

type historyRow struct {
	ChartID           string   `json:"chart_id"`
	TraceID           string   `json:"trace_id,omitempty"`
	StepSequence      int      `json:"step_sequence,omitempty"`
	HistoryType       string   `json:"history_type"`
	CompositeLabel    string   `json:"composite_label"`
	TriggerEvent      string   `json:"trigger_event"`
	PreExitLeaves     []string `json:"pre_exit_leaves"`
	RestoredLeaves    []string `json:"restored_leaves"`
	RestoredConfig    []string `json:"restored_config"`
	IsDefaultFallback bool     `json:"is_default_fallback"`
}

// extractHistoryRows scans trace entries for history-relevant steps.
func extractHistoryRows(chartID string, trace *statechartspb.ExecutionTrace, chart *sc.Statechart) []historyRow {
	if trace == nil {
		return nil
	}

	// Build parent map for composite label lookup.
	parentMap := map[string]string{}
	var buildParents func(s *sc.State, parent string)
	buildParents = func(s *sc.State, parent string) {
		if s == nil {
			return
		}
		parentMap[s.Label] = parent
		for _, child := range s.Children {
			buildParents(child, s.Label)
		}
	}
	buildParents(chart.RootState, "")

	var rows []historyRow
	for _, entry := range trace.Entries {
		if entry == nil {
			continue
		}
		raw, ok := entry.Metadata["transition_metadata"]
		if !ok {
			continue
		}
		var tm transitionMetadata
		if err := json.Unmarshal([]byte(raw), &tm); err != nil {
			continue
		}
		if tm.HistoryActivation == nil {
			continue
		}
		ha := tm.HistoryActivation

		// Determine composite (parent of the history state).
		composite := parentMap[ha.HistoryState]

		// Pre-exit leaves: leaf states from source config.
		preExit := configLabels(entry.SourceConfig)

		row := historyRow{
			ChartID:           chartID,
			TraceID:           trace.TraceId,
			StepSequence:      int(entry.Sequence),
			HistoryType:       ha.HistoryType,
			CompositeLabel:    composite,
			TriggerEvent:      entry.TriggerEvent.GetLabel(),
			PreExitLeaves:     preExit,
			RestoredLeaves:    ha.RestoredConfig,
			RestoredConfig:    configLabels(entry.TargetConfig),
			IsDefaultFallback: !ha.HadPriorHistory,
		}
		rows = append(rows, row)
	}
	return rows
}

// hasHistoryStates returns true if the chart contains any history pseudostates.
func hasHistoryStates(chart *sc.Statechart) bool {
	var walk func(s *sc.State) bool
	walk = func(s *sc.State) bool {
		if s == nil {
			return false
		}
		if s.IsHistory {
			return true
		}
		for _, child := range s.Children {
			if walk(child) {
				return true
			}
		}
		return false
	}
	return walk(chart.RootState)
}
