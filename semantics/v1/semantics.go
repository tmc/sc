package semantics

import (
	"errors"

	"github.com/tmc/sc"
)

type StateLabel string

const RootState StateLabel = ""

var (
	ErrNotFound = errors.New("not found")
)

// Statechart wraps a statechart and provides a simple interface for evaluating semantics.
type Statechart struct {
	*sc.Statechart
}

// NewStatechart creates a new statechart from a statechart definition.
func NewStatechart(sc *sc.Statechart) *Statechart {
	return &Statechart{sc}
}

func (s StateLabel) String() string {
	if s == RootState {
		return "root"
	}
	return string(s)
}
