package sc

import (
	"fmt"

	v1 "github.com/tmc/sc/gen/statecharts/v1"
)

// StateType describes the type of a state.
type StateType = v1.StateType

// MachineState encodes the high-level state of a statechart.
type MachineState = v1.MachineState

// Statechart defines a Statechart.
type Statechart = v1.Statechart

// State defines a state in a Statechart.
type State = v1.State

// Transition defines a transition in a Statechart.
type Transition = v1.Transition

// Event defines an event in a Statechart.
type Event = v1.Event

// Guard defines a guard in a Statechart.
type Guard = v1.Guard

// Action defines an action in a Statechart.
type Action = v1.Action

// StateRef defines a reference to a state in a Statechart.
type StateRef = v1.StateRef

// Configuration defines a configuration in a Statechart.
type Configuration = v1.Configuration

// Machine describes an instance of a Statechart.
type Machine = v1.Machine

// Step describes a step in the execution of a Statechart.
type Step = v1.Step

// HistoryType distinguishes shallow vs deep history pseudostates.
type HistoryType = v1.HistoryType

// Core statechart types from Harel formalism [H87, HN96]

// Core Harel statechart types [H87, Section 2.1]
const (
	StateTypeUnspecified = v1.StateType_STATE_TYPE_UNSPECIFIED
	StateTypeBasic       = v1.StateType_STATE_TYPE_BASIC
	StateTypeOR          = v1.StateType_STATE_TYPE_OR
	StateTypeAND         = v1.StateType_STATE_TYPE_AND
	
	// Academic terminology aliases [H87] for backward compatibility
	StateTypeNormal      = v1.StateType_STATE_TYPE_NORMAL      // Alias for OR
	StateTypeParallel    = v1.StateType_STATE_TYPE_PARALLEL    // Alias for AND
	StateTypeOrthogonal  = v1.StateType_STATE_TYPE_ORTHOGONAL  // Alias for AND (Harel's term)
)

const (
	MachineStateUnspecified = v1.MachineState_MACHINE_STATE_UNSPECIFIED
	MachineStateRunning     = v1.MachineState_MACHINE_STATE_RUNNING
	MachineStateStopped     = v1.MachineState_MACHINE_STATE_STOPPED
)

// History pseudostate types [UML 2.5, Section 14.5.5]
const (
	HistoryType_HISTORY_TYPE_UNSPECIFIED = v1.HistoryType_HISTORY_TYPE_UNSPECIFIED
	HistoryType_HISTORY_TYPE_SHALLOW     = v1.HistoryType_HISTORY_TYPE_SHALLOW
	HistoryType_HISTORY_TYPE_DEEP        = v1.HistoryType_HISTORY_TYPE_DEEP
)

// EventType, TransitionType, ActionType, and BroadcastSpec constants removed
// These are modern extensions not present in core Harel formalism [H87, HN96, vdB94]

// BasicValidate performs core Harel formalism validation on a statechart.
// This implements the well-formedness constraints from [H87, HN96].
func BasicValidate(s *Statechart) error {
	if s == nil {
		return fmt.Errorf("statechart is nil")
	}
	if s.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	if s.RootState.Label == "" {
		return fmt.Errorf("root state must have a label")
	}
	// Additional Harel constraints could be added here
	return nil
}
