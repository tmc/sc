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
	s := &Statechart{
		Statechart: statechart,
	}
	// Ensures that the RootState is present if otherwise not.
	if s.RootState == nil {
		s.RootState = &sc.State{}
	}
	// Ensures the label of the root state is expected:
	s.RootState.Label = RootState.String()
	return s
}
