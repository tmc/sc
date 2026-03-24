package main

import (
	"sort"

	sc "github.com/tmc/sc/gen/statecharts/v1"

	"github.com/tmc/sc/analysis"
)

// maxMatrixSize is the threshold above which adjacency and reachability
// matrices are omitted to avoid multi-megabyte output.
const maxMatrixSize = 200

type graphFeatures struct {
	AdjacencyMatrix    [][]int `json:"adjacency_matrix,omitempty"`
	ReachabilityMatrix [][]int `json:"reachability_matrix,omitempty"`
	StateCount         int     `json:"state_count"`
	TransitionCount    int     `json:"transition_count"`
	MaxDepth           int     `json:"max_depth"`
	MaxOutDegree       int     `json:"max_out_degree"`
	HasCycles          bool    `json:"has_cycles"`
	HasDeadEnds        bool    `json:"has_dead_ends"`
}

func buildGraphFeatures(chart *sc.Statechart) *graphFeatures {
	g := analysis.BuildGraph(chart)

	// Get sorted non-root states for matrix indices.
	states := g.AllStates()
	sort.Strings(states)
	idx := make(map[string]int, len(states))
	for i, s := range states {
		idx[s] = i
	}
	n := len(states)

	// Only build dense matrices for reasonably-sized charts.
	var adj, reach [][]int
	if n <= maxMatrixSize {
		adj = make([][]int, n)
		for i := range adj {
			adj[i] = make([]int, n)
		}
		for _, t := range g.Transitions {
			fi, fok := idx[t.From]
			ti, tok := idx[t.To]
			if fok && tok {
				adj[fi][ti] = 1
			}
		}

		reach = make([][]int, n)
		for i := range reach {
			reach[i] = make([]int, n)
			reachable := g.Reachable([]string{states[i]})
			for _, r := range reachable {
				if j, ok := idx[r]; ok {
					reach[i][j] = 1
				}
			}
		}
	}

	// Max depth: walk the state tree.
	maxDepth := 0
	var walkDepth func(s *sc.State, depth int)
	walkDepth = func(s *sc.State, depth int) {
		if s == nil {
			return
		}
		if depth > maxDepth {
			maxDepth = depth
		}
		for _, child := range s.Children {
			walkDepth(child, depth+1)
		}
	}
	walkDepth(chart.RootState, 0)

	// Max out-degree.
	maxOut := 0
	for _, s := range states {
		node := g.States[s]
		if node != nil && len(node.Outgoing) > maxOut {
			maxOut = len(node.Outgoing)
		}
	}

	// Has cycles: either from reachability diagonal or via DFS for large charts.
	hasCycles := false
	if reach != nil {
		for i := range n {
			if reach[i][i] == 1 {
				hasCycles = true
				break
			}
		}
	} else {
		// For large charts, detect cycles via DFS.
		hasCycles = detectCycles(g, states)
	}

	return &graphFeatures{
		AdjacencyMatrix:    adj,
		ReachabilityMatrix: reach,
		StateCount:         g.StateCount(),
		TransitionCount:    g.TransitionCount(),
		MaxDepth:           maxDepth,
		MaxOutDegree:       maxOut,
		HasCycles:          hasCycles,
		HasDeadEnds:        len(g.DeadEnds()) > 0,
	}
}

// detectCycles checks for cycles using DFS with a coloring scheme.
func detectCycles(g *analysis.StateGraph, states []string) bool {
	const (
		white = 0 // unvisited
		gray  = 1 // in current DFS path
		black = 2 // fully explored
	)
	color := make(map[string]int, len(states))
	for _, s := range states {
		color[s] = white
	}

	var dfs func(string) bool
	dfs = func(s string) bool {
		color[s] = gray
		node := g.States[s]
		if node != nil {
			for _, edge := range node.Outgoing {
				switch color[edge.To] {
				case gray:
					return true // back edge = cycle
				case white:
					if dfs(edge.To) {
						return true
					}
				}
			}
		}
		color[s] = black
		return false
	}

	for _, s := range states {
		if color[s] == white && dfs(s) {
			return true
		}
	}
	return false
}
