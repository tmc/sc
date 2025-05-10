package v1

import (
	"fmt"
	"testing"

	"github.com/tmc/sc"
)

func Example_orthogonalStateType() {
	// Create a statechart with both parallel and orthogonal states
	// to demonstrate that they are equivalent
	
	// Create a state using PARALLEL terminology 
	parallelState := &sc.State{
		Label: "ParallelState",
		Type:  sc.StateTypeParallel,
		Children: []*sc.State{
			{
				Label: "Region1",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label:     "R1State1",
						Type:      sc.StateTypeBasic,
						IsInitial: true,
					},
				},
			},
			{
				Label: "Region2",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label:     "R2State1",
						Type:      sc.StateTypeBasic,
						IsInitial: true,
					},
				},
			},
		},
	}

	// Create an identical state using ORTHOGONAL terminology 
	orthogonalState := &sc.State{
		Label: "OrthogonalState",
		Type:  sc.StateTypeOrthogonal, // Using the ORTHOGONAL alias
		Children: []*sc.State{
			{
				Label: "Region1",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label:     "R1State1",
						Type:      sc.StateTypeBasic,
						IsInitial: true,
					},
				},
			},
			{
				Label: "Region2",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label:     "R2State1",
						Type:      sc.StateTypeBasic,
						IsInitial: true,
					},
				},
			},
		},
	}

	// Verify they have the same type
	fmt.Printf("Parallel state type: %d\n", parallelState.Type)
	fmt.Printf("Orthogonal state type: %d\n", orthogonalState.Type)
	fmt.Printf("Types are equal: %t\n", parallelState.Type == orthogonalState.Type)
	
	// Output:
	// Parallel state type: 3
	// Orthogonal state type: 3
	// Types are equal: true
}

func TestOrthogonalStateExample(t *testing.T) {
	// This is needed to run the example as a test
	Example_orthogonalStateType()
}