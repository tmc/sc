package main

import (
	"github.com/tmc/sc"
)

type classification struct {
	Families      []string `json:"families"`
	Depth         int      `json:"depth"`
	HasParallel   bool     `json:"has_parallel"`
	HasHistory    bool     `json:"has_history"`
	HasInterLevel bool     `json:"has_inter_level"`
}

func buildClassification(chart *sc.Statechart) *classification {
	c := &classification{}

	// Walk state tree to compute depth, parallel, and history flags.
	depthMap := map[string]int{}
	var walk func(s *sc.State, depth int)
	walk = func(s *sc.State, depth int) {
		if s == nil {
			return
		}
		depthMap[s.Label] = depth
		if depth > c.Depth {
			c.Depth = depth
		}
		if s.Type == sc.StateTypeAND {
			c.HasParallel = true
		}
		if s.IsHistory {
			c.HasHistory = true
		}
		for _, child := range s.Children {
			walk(child, depth+1)
		}
	}
	walk(chart.RootState, 0)

	// Check transitions for inter-level.
	for _, t := range chart.Transitions {
		if t == nil || len(t.From) == 0 || len(t.To) == 0 {
			continue
		}
		srcDepth := depthMap[t.From[0]]
		tgtDepth := depthMap[t.To[0]]
		if srcDepth != tgtDepth {
			c.HasInterLevel = true
			break
		}
	}

	// Assign families.
	if c.Depth <= 2 {
		c.Families = append(c.Families, "flat")
	}
	if c.Depth > 2 && !c.HasParallel {
		c.Families = append(c.Families, "hierarchical")
	}
	if c.HasParallel {
		c.Families = append(c.Families, "parallel")
	}
	if c.HasHistory {
		c.Families = append(c.Families, "history")
	}
	if c.HasInterLevel {
		c.Families = append(c.Families, "inter_level")
	}
	if len(c.Families) == 0 {
		c.Families = append(c.Families, "flat")
	}

	return c
}
