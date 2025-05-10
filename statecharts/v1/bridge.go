// Package v1 provides bridging between the statecharts protobuf and Go implementations.
package v1

import (
	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
)

// Statechart is aliased from the generated protobuf package
type Statechart = pb.Statechart

// State is aliased from the generated protobuf package
type State = pb.State

// Event is aliased from the generated protobuf package
type Event = pb.Event

// Transition is aliased from the generated protobuf package
type Transition = pb.Transition

// Machine is aliased from the generated protobuf package
type Machine = pb.Machine

// FromNative converts a native sc.Statechart to a protobuf Statechart
func FromNative(statechart *sc.Statechart) *Statechart {
	if statechart == nil {
		return nil
	}

	result := &Statechart{
		RootState:   fromNativeState(statechart.RootState),
		Transitions: make([]*Transition, 0, len(statechart.Transitions)),
		Events:      make([]*Event, 0, len(statechart.Events)),
	}

	for _, transition := range statechart.Transitions {
		result.Transitions = append(result.Transitions, fromNativeTransition(transition))
	}

	for _, event := range statechart.Events {
		result.Events = append(result.Events, fromNativeEvent(event))
	}

	return result
}

// ToNative converts a protobuf Statechart to a native sc.Statechart
func ToNative(statechart *Statechart) *sc.Statechart {
	if statechart == nil {
		return nil
	}

	result := &sc.Statechart{
		RootState:   toNativeState(statechart.RootState),
		Transitions: make([]*sc.Transition, 0, len(statechart.Transitions)),
		Events:      make([]*sc.Event, 0, len(statechart.Events)),
	}

	for _, transition := range statechart.Transitions {
		result.Transitions = append(result.Transitions, toNativeTransition(transition))
	}

	for _, event := range statechart.Events {
		result.Events = append(result.Events, toNativeEvent(event))
	}

	return result
}

// Helper functions for conversion
func fromNativeState(state *sc.State) *State {
	if state == nil {
		return nil
	}

	result := &State{
		Label:     state.Label,
		Type:      pb.StateType(state.Type),
		IsInitial: state.IsInitial,
		IsFinal:   state.IsFinal,
		Children:  make([]*State, 0, len(state.Children)),
	}

	for _, child := range state.Children {
		result.Children = append(result.Children, fromNativeState(child))
	}

	return result
}

func toNativeState(state *State) *sc.State {
	if state == nil {
		return nil
	}

	result := &sc.State{
		Label:     state.Label,
		Type:      sc.StateType(state.Type),
		IsInitial: state.IsInitial,
		IsFinal:   state.IsFinal,
		Children:  make([]*sc.State, 0, len(state.Children)),
	}

	for _, child := range state.Children {
		result.Children = append(result.Children, toNativeState(child))
	}

	return result
}

func fromNativeTransition(transition *sc.Transition) *Transition {
	if transition == nil {
		return nil
	}

	result := &Transition{
		Label: transition.Label,
		From:  transition.From,
		To:    transition.To,
		Event: transition.Event,
	}

	if transition.Guard != nil {
		result.Guard = &pb.Guard{
			Expression: transition.Guard.Expression,
		}
	}

	if len(transition.Actions) > 0 {
		result.Actions = make([]*pb.Action, 0, len(transition.Actions))
		for _, action := range transition.Actions {
			result.Actions = append(result.Actions, &pb.Action{
				Label: action.Label,
			})
		}
	}

	return result
}

func toNativeTransition(transition *Transition) *sc.Transition {
	if transition == nil {
		return nil
	}

	result := &sc.Transition{
		Label: transition.Label,
		From:  transition.From,
		To:    transition.To,
		Event: transition.Event,
	}

	if transition.Guard != nil {
		result.Guard = &sc.Guard{
			Expression: transition.Guard.Expression,
		}
	}

	if len(transition.Actions) > 0 {
		result.Actions = make([]*sc.Action, 0, len(transition.Actions))
		for _, action := range transition.Actions {
			result.Actions = append(result.Actions, &sc.Action{
				Label: action.Label,
			})
		}
	}

	return result
}

func fromNativeEvent(event *sc.Event) *Event {
	if event == nil {
		return nil
	}

	return &Event{
		Label: event.Label,
	}
}

func toNativeEvent(event *Event) *sc.Event {
	if event == nil {
		return nil
	}

	return &sc.Event{
		Label: event.Label,
	}
}