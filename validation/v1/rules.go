package validation

import (
	"fmt"

	"github.com/tmc/sc"
)

// validateUniqueStateLabels checks that all state labels are unique.
func validateUniqueStateLabels(statechart *sc.Statechart) error {
	return firstIssueError(collectUniqueStateLabelIssues(statechart))
}

// validateSingleDefaultChild ensures that XOR composite states have exactly one default child.
func validateSingleDefaultChild(statechart *sc.Statechart) error {
	return firstIssueError(collectSingleDefaultChildIssues(statechart))
}

// validateBasicHasNoChildren ensures that basic states have no children.
func validateBasicHasNoChildren(statechart *sc.Statechart) error {
	return firstIssueError(collectBasicHasNoChildrenIssues(statechart))
}

// validateCompoundHasChildren ensures that compound states have children.
func validateCompoundHasChildren(statechart *sc.Statechart) error {
	return firstIssueError(collectCompoundHasChildrenIssues(statechart))
}

// validateRootState ensures that the root state exists and has the correct label.
func validateRootState(statechart *sc.Statechart) error {
	if statechart.RootState == nil {
		return fmt.Errorf("root state is nil")
	}

	if statechart.RootState.Label != "__root__" {
		return fmt.Errorf("root state has an unexpected label of '%s' (expected '__root__')", statechart.RootState.Label)
	}

	return nil
}

// validateDeterministicTransitionSelection ensures that transitions are deterministic.
// This is a simplified implementation - a full implementation would need to analyze
// guards and potential conflicts.
func validateDeterministicTransitionSelection(statechart *sc.Statechart) error {
	return firstIssueError(collectDeterministicTransitionSelectionIssues(statechart))
}

// validateNoEventBroadcastCycles ensures there are no cycles in event broadcasts.
// This is a stub implementation. A complete implementation would need to analyze
// action-to-event relationships and detect cycles.
func validateNoEventBroadcastCycles(statechart *sc.Statechart) error {
	return firstIssueError(collectNoEventBroadcastCycleIssues(statechart))
}
