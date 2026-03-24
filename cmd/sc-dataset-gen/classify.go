package main

import (
	"github.com/tmc/sc"
)

type classification struct {
	Families      []string `json:"families"`
	Depth         int      `json:"depth"`
	HasParallel   bool     `json:"has_parallel"`
	HasHistory    bool     `json:"has_history"`
	HasGuards     bool     `json:"has_guards"`
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

	// Check transitions for inter-level and guards.
	for _, t := range chart.Transitions {
		if t == nil {
			continue
		}
		if !c.HasInterLevel && len(t.From) > 0 && len(t.To) > 0 {
			if depthMap[t.From[0]] != depthMap[t.To[0]] {
				c.HasInterLevel = true
			}
		}
		if !c.HasGuards && t.Guard != nil {
			if (t.Guard.Expression != "" || (t.Guard.Condition != nil && t.Guard.Condition.Source != "")) {
				c.HasGuards = true
			}
		}
	}

	// Assign families aligned with corpus naming.
	if c.Depth <= 2 {
		c.Families = append(c.Families, "flat")
	}
	if c.Depth == 3 {
		c.Families = append(c.Families, "shallow_nested")
	}
	if c.Depth >= 4 {
		c.Families = append(c.Families, "deep_nested")
	}
	if c.HasParallel {
		c.Families = append(c.Families, "orthogonal")
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
