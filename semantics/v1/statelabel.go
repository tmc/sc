package semantics

// StateLabel represents a label for a state in the statechart.
type StateLabel string

// NewStateLabel creates a new StateLabel.
func NewStateLabel(label string) StateLabel {
	return StateLabel(label)
}

// String returns the string representation of the StateLabel.
func (sl StateLabel) String() string {
	return string(sl)
}

// RootState represents the root state of the statechart.
var RootState = NewStateLabel("__root__")

// CreateStateLabels converts a variadic list of strings into a slice of StateLabel.
// It provides a convenient way to create multiple StateLabel instances at once.
func CreateStateLabels(labels ...string) []StateLabel {
	stateLabels := make([]StateLabel, len(labels))
	for i, label := range labels {
		stateLabels[i] = NewStateLabel(label)
	}
	return stateLabels
}
