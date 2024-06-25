package semantics

import (
	"github.com/tmc/sc"
)

type StateLabel string

// RootState represents the label for the root state of the statechart.
const RootState StateLabel = ""

// Statechart wraps a statechart and provides a simple interface for evaluating semantics.
type Statechart struct {
	*sc.Statechart
}

// NewStatechart creates a new statechart from a statechart definition.
func NewStatechart(sc *sc.Statechart) *Statechart {
	return &Statechart{sc}
}

// String returns the string representation of the StateLabel.
func (s StateLabel) String() string {
	if s == RootState {
		return "(root)"
	}
	return string(s)
}

// CreateStateLabels converts a variadic list of strings into a slice of StateLabel.
// It provides a convenient way to create multiple StateLabel instances at once.
func CreateStateLabels(labels ...string) []StateLabel {
	stateLabels := make([]StateLabel, len(labels))
	for i, label := range labels {
		stateLabels[i] = StateLabel(label)
	}
	return stateLabels
}
