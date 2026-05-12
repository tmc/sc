package semantics

import (
	"github.com/tmc/sc"
)

// Statechart wraps a statechart and provides a simple interface for evaluating semantics.
type Statechart struct {
	*sc.Statechart
}

// NewStatechart creates a new statechart from a statechart definition.
func NewStatechart(statechart *sc.Statechart) *Statechart {
	if statechart == nil {
		statechart = &sc.Statechart{}
	}
	s := &Statechart{
		Statechart: statechart,
	}
	// Ensures that the RootState is present if otherwise not.
	if s.RootState == nil {
		s.RootState = &sc.State{}
	}
	// Ensures the label of the root state is expected:
	s.RootState.Label = RootState.String()
	_ = normalizeStateTypes(s.Statechart)
	if s.RootState.Type == sc.StateTypeNormal && len(s.RootState.Children) == 1 && !hasInitialChild(s.RootState) {
		s.RootState.Children[0].IsInitial = true
	}
	return s
}

func hasInitialChild(state *sc.State) bool {
	for _, child := range state.Children {
		if child.IsInitial {
			return true
		}
	}
	return false
}
