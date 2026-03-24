package main

import (
	"slices"

	"github.com/tmc/sc"
)

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
