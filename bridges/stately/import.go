// Package stately provides import/export functionality for XState/Stately editor machine JSON format.
package stately

import (
	"encoding/json"
	"fmt"

	sc "github.com/tmc/sc"
	scpb "github.com/tmc/sc/gen/statecharts/v1"
	xstatepb "github.com/tmc/sc/gen/xstate/v1"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/structpb"
)

// Import converts XState machine JSON to sc.Statechart.
// If machineName is empty, imports the first machine found.
func Import(data []byte, machineName string) (*sc.Statechart, error) {
	// Try parsing as array of machines first
	var machines []Machine
	if err := json.Unmarshal(data, &machines); err != nil {
		// Try single machine
		var machine Machine
		if err := json.Unmarshal(data, &machine); err != nil {
			return nil, fmt.Errorf("failed to parse XState JSON: %w", err)
		}
		machines = []Machine{machine}
	}

	if len(machines) == 0 {
		return nil, fmt.Errorf("no machines found in input")
	}

	// Find target machine
	var target *Machine
	for i := range machines {
		if machineName == "" || machines[i].Name == machineName {
			target = &machines[i]
			break
		}
	}
	if target == nil {
		return nil, fmt.Errorf("machine not found: %s", machineName)
	}

	return convertMachine(target)
}

// ImportAll converts all XState machines in the JSON to sc.Statecharts.
func ImportAll(data []byte) ([]*sc.Statechart, error) {
	var machines []Machine
	if err := json.Unmarshal(data, &machines); err != nil {
		// Try single machine
		var machine Machine
		if err := json.Unmarshal(data, &machine); err != nil {
			return nil, fmt.Errorf("failed to parse XState JSON: %w", err)
		}
		machines = []Machine{machine}
	}

	result := make([]*sc.Statechart, 0, len(machines))
	for i := range machines {
		chart, err := convertMachine(&machines[i])
		if err != nil {
			return nil, fmt.Errorf("failed to convert machine %q: %w", machines[i].Name, err)
		}
		result = append(result, chart)
	}
	return result, nil
}

// convertMachine converts a single XState Machine to sc.Statechart.
func convertMachine(m *Machine) (*sc.Statechart, error) {
	chart := &sc.Statechart{
		Name: m.Name,
	}

	// Build node ID to label map for resolving edge targets
	nodeIDToLabel := buildNodeIDMap(&m.Definition.RootNode)

	// Convert root state
	rootState, err := convertNode(&m.Definition.RootNode, nil, "")
	if err != nil {
		return nil, fmt.Errorf("failed to convert root state: %w", err)
	}
	chart.RootState = rootState

	// Convert edges to transitions
	for _, edge := range m.Definition.Edges {
		t, err := convertEdge(&edge, nodeIDToLabel)
		if err != nil {
			return nil, fmt.Errorf("failed to convert edge %s: %w", edge.ID, err)
		}
		chart.Transitions = append(chart.Transitions, t)
	}

	// Extract events from transitions
	eventSet := make(map[string]bool)
	for _, t := range chart.Transitions {
		if t.Event != "" {
			eventSet[t.Event] = true
		}
	}
	for eventLabel := range eventSet {
		chart.Events = append(chart.Events, &scpb.Event{Label: eventLabel})
	}

	// Convert context to variables
	if m.Definition.Context != nil {
		vars, err := structpb.NewStruct(m.Definition.Context)
		if err == nil {
			chart.Variables = vars
		}
	}

	return chart, nil
}

// buildNodeIDMap creates a mapping from XState node IDs to state labels.
func buildNodeIDMap(node *Node) map[string]string {
	result := make(map[string]string)
	buildNodeIDMapRecursive(node, result)
	return result
}

func buildNodeIDMapRecursive(node *Node, result map[string]string) {
	result[node.ID] = node.Data.Key
	for i := range node.Nodes {
		buildNodeIDMapRecursive(&node.Nodes[i], result)
	}
}

// convertNode converts an XState Node to sc.State.
func convertNode(node *Node, parent *Node, parentInitial string) (*sc.State, error) {
	s := &sc.State{
		Label:     node.Data.Key,
		IsInitial: parentInitial == node.Data.Key,
	}

	// Determine state type
	if node.Data.Type == "parallel" {
		s.Type = scpb.StateType_STATE_TYPE_PARALLEL
	} else if len(node.Nodes) > 0 {
		s.Type = scpb.StateType_STATE_TYPE_OR
	} else {
		s.Type = scpb.StateType_STATE_TYPE_BASIC
	}

	// Convert entry actions
	for _, ar := range node.Data.Entry {
		action := convertActionRef(&ar)
		s.EntryActions = append(s.EntryActions, action)
	}

	// Convert exit actions
	for _, ar := range node.Data.Exit {
		action := convertActionRef(&ar)
		s.ExitActions = append(s.ExitActions, action)
	}

	// Pack XState-specific layout data into extensions
	layout := &xstatepb.XStateLayout{
		Position: &xstatepb.Position{X: node.Position.X, Y: node.Position.Y},
		Size:     &xstatepb.Size{Width: node.Size.Width, Height: node.Size.Height},
		Color:    node.Data.Color,
		UniqueId: node.UniqueID,
	}
	if layoutAny, err := anypb.New(layout); err == nil {
		s.Extensions = append(s.Extensions, layoutAny)
	}

	// Pack XState-specific state data into extensions
	stateData := &xstatepb.XStateStateData{
		Description: node.Data.Description,
		Tags:        node.Data.Tags,
	}

	// Convert assets
	for _, asset := range node.Data.Assets {
		props, _ := structpb.NewStruct(asset.Properties)
		stateData.Assets = append(stateData.Assets, &xstatepb.XStateAsset{
			Name:       asset.Name,
			Template:   asset.Template,
			Properties: props,
		})
	}

	// Convert invokes
	for _, invoke := range node.Data.Invoke {
		input, _ := structpb.NewStruct(invoke.Input)
		settings, _ := structpb.NewStruct(invoke.Settings)
		stateData.Invokes = append(stateData.Invokes, &xstatepb.XStateInvoke{
			Id:       invoke.ID,
			Src:      invoke.Src,
			Kind:     invoke.Kind,
			Input:    input,
			Settings: settings,
		})
	}

	// Convert meta entries
	if len(node.Data.MetaEntries) > 0 {
		var entries [][]string
		if err := json.Unmarshal(node.Data.MetaEntries, &entries); err == nil {
			for _, entry := range entries {
				if len(entry) >= 2 {
					stateData.MetaEntries = append(stateData.MetaEntries, &xstatepb.MetaEntry{
						Key:   entry[0],
						Value: entry[1],
					})
				}
			}
		}
	}

	if stateDataAny, err := anypb.New(stateData); err == nil {
		s.Extensions = append(s.Extensions, stateDataAny)
	}

	// Convert children recursively
	for i := range node.Nodes {
		child, err := convertNode(&node.Nodes[i], node, node.Data.Initial)
		if err != nil {
			return nil, err
		}
		s.Children = append(s.Children, child)
	}

	return s, nil
}

// convertEdge converts an XState Edge to sc.Transition.
func convertEdge(edge *Edge, nodeIDToLabel map[string]string) (*sc.Transition, error) {
	t := &sc.Transition{
		Label: edge.ID,
	}

	// Resolve source and target state labels
	if srcLabel, ok := nodeIDToLabel[edge.Source]; ok {
		t.From = []string{srcLabel}
	} else {
		t.From = []string{edge.Source}
	}

	if edge.Target != "" {
		if tgtLabel, ok := nodeIDToLabel[edge.Target]; ok {
			t.To = []string{tgtLabel}
		} else {
			t.To = []string{edge.Target}
		}
	}

	// Set event based on event type
	switch edge.Data.EventTypeData.Type {
	case "named":
		t.Event = edge.Data.EventTypeData.EventType
	case "always":
		t.Event = "" // Completion transition
	case "after":
		t.Event = edge.Data.EventTypeData.EventType // Delay name
	case "invocation.done":
		t.Event = "done.invoke." + edge.Data.EventTypeData.EventType
	case "invocation.error":
		t.Event = "error.invoke." + edge.Data.EventTypeData.EventType
	case "state.done":
		t.Event = "done.state." + edge.Data.EventTypeData.EventType
	}

	// Convert guard
	if edge.Data.Guard != nil {
		t.Guard = &scpb.Guard{
			Expression: edge.Data.Guard.Name,
		}
		if edge.Data.Guard.Type != "" {
			t.Guard.Expression = edge.Data.Guard.Type
		}
	}

	// Convert transition actions
	for _, ar := range edge.Data.Actions {
		action := convertActionRef(&ar)
		t.Actions = append(t.Actions, action)
	}

	// Pack XState-specific layout data into extensions
	layout := &xstatepb.XStateLayout{
		Position: &xstatepb.Position{X: edge.Position.X, Y: edge.Position.Y},
		Size:     &xstatepb.Size{Width: edge.Size.Width, Height: edge.Size.Height},
		Color:    edge.Data.Color,
		UniqueId: edge.UniqueID,
	}
	if layoutAny, err := anypb.New(layout); err == nil {
		t.Extensions = append(t.Extensions, layoutAny)
	}

	// Pack XState-specific transition data into extensions
	transitionData := &xstatepb.XStateTransitionData{
		Internal:    edge.Data.Internal,
		TriggerType: mapTriggerType(edge.Data.EventTypeData.Type),
		Description: edge.Data.Description,
	}
	if transDataAny, err := anypb.New(transitionData); err == nil {
		t.Extensions = append(t.Extensions, transDataAny)
	}

	return t, nil
}

// convertActionRef converts an XState ActionRef to sc.Action.
func convertActionRef(ar *ActionRef) *sc.Action {
	action := &sc.Action{
		Label: ar.Action.Type,
	}
	if ar.Action.Params != nil {
		params, err := structpb.NewStruct(ar.Action.Params)
		if err == nil {
			action.Parameters = params
		}
	}
	return action
}

// mapTriggerType maps XState event type strings to proto enum values.
func mapTriggerType(eventType string) xstatepb.TransitionTriggerType {
	switch eventType {
	case "named":
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_EVENT
	case "always":
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_ALWAYS
	case "after":
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_AFTER
	case "invocation.done":
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_INVOKE_DONE
	case "invocation.error":
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_INVOKE_ERROR
	case "state.done":
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_STATE_DONE
	default:
		return xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_UNSPECIFIED
	}
}
