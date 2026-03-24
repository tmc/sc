package main

import (
	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

type guardEvalRow struct {
	ChartID         string         `json:"chart_id"`
	TraceID         string         `json:"trace_id,omitempty"`
	StepSequence    int            `json:"step_sequence,omitempty"`
	TransitionLabel string         `json:"transition_label"`
	GuardExpression string         `json:"guard_expression"`
	Event           string         `json:"event"`
	ActiveStates    []string       `json:"active_states"`
	BoundValues     map[string]any `json:"bound_values,omitempty"`
	Result          bool           `json:"result"`
	TransitionFired bool           `json:"transition_fired"`
	Error           string         `json:"error,omitempty"`
}

// extractGuardEvalRows extracts one row per guard evaluation from trace entries.
func extractGuardEvalRows(chartID string, trace *statechartspb.ExecutionTrace) []guardEvalRow {
	if trace == nil {
		return nil
	}

	var rows []guardEvalRow
	for _, entry := range trace.Entries {
		if entry == nil || len(entry.GuardResults) == 0 {
			continue
		}

		activeStates := configLabels(entry.SourceConfig)

		// Build set of fired transition labels.
		firedSet := make(map[string]bool, len(entry.TransitionsFired))
		for _, ref := range entry.TransitionsFired {
			if ref != nil && ref.Label != "" {
				firedSet[ref.Label] = true
			}
		}

		for _, gr := range entry.GuardResults {
			if gr == nil {
				continue
			}

			tLabel := ""
			event := ""
			if gr.Transition != nil {
				tLabel = gr.Transition.Label
				event = gr.Transition.Event
			}
			if event == "" && entry.TriggerEvent != nil {
				event = entry.TriggerEvent.Label
			}

			row := guardEvalRow{
				ChartID:         chartID,
				TraceID:         trace.TraceId,
				StepSequence:    int(entry.Sequence),
				TransitionLabel: tLabel,
				GuardExpression: gr.GuardExpression,
				Event:           event,
				ActiveStates:    activeStates,
				Result:          gr.Result,
				TransitionFired: firedSet[tLabel],
				Error:           gr.Error,
			}

			// Extract bound values if present.
			if gr.BoundValues != nil {
				row.BoundValues = structToMap(gr.BoundValues)
			}

			rows = append(rows, row)
		}
	}
	return rows
}

// structToMap converts a protobuf Struct to a plain map.
func structToMap(s *structpb.Struct) map[string]any {
	if s == nil {
		return nil
	}
	m := make(map[string]any, len(s.Fields))
	for k, v := range s.Fields {
		m[k] = v.AsInterface()
	}
	return m
}

// hasGuards returns true if the chart has any guarded transitions.
func hasGuards(chart *sc.Statechart) bool {
	for _, t := range chart.Transitions {
		if t != nil && t.Guard != nil {
			expr := t.Guard.Expression
			if t.Guard.Condition != nil && expr == "" {
				expr = t.Guard.Condition.Source
			}
			if expr != "" {
				return true
			}
		}
	}
	return false
}
