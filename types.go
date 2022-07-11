package sc

import v1 "github.com/tmc/sc/gen/statecharts/v1"

// StateType describes the type of a state.
type StateType = v1.StateType

// MachineState encodes the high-level state of a statechart.
type MachineState = v1.MachineState

// StateChart defines a Statechart.
type StateChart = v1.StateChart

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

// EventHistoryEntry describes an event that has occurred.
type EventHistoryEntry = v1.EventHistoryEntry

// TransitionHistoryEntry describes a transition that has been taken.
type TransitionHistoryEntry = v1.TransitionHistoryEntry
