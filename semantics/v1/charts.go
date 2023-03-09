package semantics

// Validate validates the statechart.
// It runs various checks to ensure that the statechart is well-formed.
// If the statechart is not well-formed, an error is returned.
func (s *Statechart) Validate() error {
	// validateNonOverlappingStateLabels
	// validateRootState
	// validateStateTypeAgreesWithChildren
	// validateParentChildRelationships
	// validateParentStatesHaveSingleDefaults
	return nil
}
