package v1

import (
	"testing"

	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
)

func TestOrthogonalAlias(t *testing.T) {
	// Test that ORTHOGONAL is an alias for PARALLEL
	if pb.StateType_STATE_TYPE_ORTHOGONAL != pb.StateType_STATE_TYPE_PARALLEL {
		t.Errorf("STATE_TYPE_ORTHOGONAL should be equal to STATE_TYPE_PARALLEL")
	}

	// Test that the generated constants are available in the sc package
	if sc.StateTypeOrthogonal != sc.StateTypeParallel {
		t.Errorf("sc.StateTypeOrthogonal should be equal to sc.StateTypeParallel")
	}

	// Confirm the values
	if sc.StateTypeOrthogonal != 3 || sc.StateTypeParallel != 3 {
		t.Errorf("Orthogonal and Parallel state types should have value 3")
	}
}