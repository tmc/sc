package stately

import (
	"encoding/json"
	"fmt"

	sc "github.com/tmc/sc"
	scpb "github.com/tmc/sc/gen/statecharts/v1"
	xstatepb "github.com/tmc/sc/gen/xstate/v1"
	"google.golang.org/protobuf/proto"
)

// Export converts an sc.Statechart to XState machine JSON format.
func Export(chart *sc.Statechart) (*Machine, error) {
	if chart == nil {
		return nil, fmt.Errorf("statechart is nil")
	}
	if chart.RootState == nil {
		return nil, fmt.Errorf("statechart root state is nil")
	}

	m := &Machine{
		Name: chart.Name,
		Definition: Definition{
			ID: chart.Name,
		},
	}

	// Build label to ID map for edge resolution
	labelToID := make(map[string]string)
	idCounter := 0

	// Convert root state
	rootNode, err := exportState(chart.RootState, labelToID, &idCounter)
	if err != nil {
		return nil, fmt.Errorf("failed to export root state: %w", err)
	}
	m.Definition.RootNode = *rootNode

	// Convert transitions to edges
	for _, t := range chart.Transitions {
		edge, err := exportTransition(t, labelToID)
		if err != nil {
			return nil, fmt.Errorf("failed to export transition %s: %w", t.Label, err)
		}
		m.Definition.Edges = append(m.Definition.Edges, *edge)
	}

	// Convert context variables
	if chart.Variables != nil {
		m.Definition.Context = chart.Variables.AsMap()
	}

	// Extract XState machine metadata from extensions if present
	extractMachineMetadata(chart, m)

	return m, nil
}

// ExportJSON converts an sc.Statechart to XState JSON bytes.
func ExportJSON(chart *sc.Statechart) ([]byte, error) {
	machine, err := Export(chart)
	if err != nil {
		return nil, err
	}
	return json.MarshalIndent(machine, "", "  ")
}

// exportState converts an sc.State to XState Node.
func exportState(s *sc.State, labelToID map[string]string, idCounter *int) (*Node, error) {
	*idCounter++
	nodeID := fmt.Sprintf("state-%d", *idCounter)
	labelToID[s.Label] = nodeID

	node := &Node{
		ID:       nodeID,
		UniqueID: nodeID,
		Data: NodeData{
			Key: s.Label,
		},
	}

	// Set type for parallel states
	if s.Type == scpb.StateType_STATE_TYPE_PARALLEL ||
		s.Type == scpb.StateType_STATE_TYPE_AND ||
		s.Type == scpb.StateType_STATE_TYPE_ORTHOGONAL {
		node.Data.Type = "parallel"
	}

	// Find initial child
	for _, child := range s.Children {
		if child.IsInitial {
			node.Data.Initial = child.Label
			break
		}
	}

	// Convert entry actions
	for _, action := range s.EntryActions {
		ar := exportAction(action)
		node.Data.Entry = append(node.Data.Entry, ar)
	}

	// Convert exit actions
	for _, action := range s.ExitActions {
		ar := exportAction(action)
		node.Data.Exit = append(node.Data.Exit, ar)
	}

	// Extract XState-specific data from extensions
	extractStateExtensions(s, node)

	// Convert children recursively
	for _, child := range s.Children {
		childNode, err := exportState(child, labelToID, idCounter)
		if err != nil {
			return nil, err
		}
		node.Nodes = append(node.Nodes, *childNode)
	}

	return node, nil
}

// exportTransition converts an sc.Transition to XState Edge.
func exportTransition(t *sc.Transition, labelToID map[string]string) (*Edge, error) {
	edge := &Edge{
		ID:       t.Label,
		UniqueID: t.Label,
	}

	// Resolve source state ID
	if len(t.From) > 0 {
		if id, ok := labelToID[t.From[0]]; ok {
			edge.Source = id
		} else {
			edge.Source = t.From[0]
		}
	}

	// Resolve target state ID
	if len(t.To) > 0 {
		if id, ok := labelToID[t.To[0]]; ok {
			edge.Target = id
		} else {
			edge.Target = t.To[0]
		}
	}

	// Set event type data
	edge.Data.EventTypeData = determineEventTypeData(t.Event)

	// Convert guard
	if t.Guard != nil && t.Guard.Expression != "" {
		edge.Data.Guard = &GuardRef{
			Kind: "named",
			Name: t.Guard.Expression,
			Type: t.Guard.Expression,
		}
	}

	// Convert actions
	for _, action := range t.Actions {
		ar := exportAction(action)
		edge.Data.Actions = append(edge.Data.Actions, ar)
	}

	// Extract XState-specific data from extensions
	extractTransitionExtensions(t, edge)

	return edge, nil
}

// exportAction converts an sc.Action to XState ActionRef.
func exportAction(action *sc.Action) ActionRef {
	ar := ActionRef{
		Kind: "named",
		Action: ActionConfig{
			Type: action.Label,
		},
	}
	if action.Parameters != nil {
		ar.Action.Params = action.Parameters.AsMap()
	}
	return ar
}

// determineEventTypeData creates EventTypeData from an event string.
func determineEventTypeData(event string) EventTypeData {
	if event == "" {
		return EventTypeData{Type: "always"}
	}

	// Check for special event prefixes
	if len(event) > 12 && event[:12] == "done.invoke." {
		return EventTypeData{
			Type:      "invocation.done",
			EventType: event[12:],
		}
	}
	if len(event) > 13 && event[:13] == "error.invoke." {
		return EventTypeData{
			Type:      "invocation.error",
			EventType: event[13:],
		}
	}
	if len(event) > 11 && event[:11] == "done.state." {
		return EventTypeData{
			Type:      "state.done",
			EventType: event[11:],
		}
	}

	// Regular named event
	return EventTypeData{
		Type:      "named",
		EventType: event,
	}
}

// extractMachineMetadata extracts XState machine metadata from chart extensions.
func extractMachineMetadata(chart *sc.Statechart, m *Machine) {
	// Look in root state extensions for machine-level metadata
	if chart.RootState == nil {
		return
	}

	for _, ext := range chart.RootState.Extensions {
		var machineData xstatepb.XStateMachineData
		if err := ext.UnmarshalTo(&machineData); err == nil {
			m.ID = machineData.Id
			m.ProjectVersionID = machineData.ProjectVersionId
			m.ForkParentID = machineData.ForkParentId
			m.LastEditedByID = machineData.LastEditedById
			m.OriginalCode = machineData.OriginalCode
			if machineData.CreatedAt != nil {
				m.CreatedAt = machineData.CreatedAt.AsTime().Format("2006-01-02T15:04:05.000Z")
			}
			if machineData.UpdatedAt != nil {
				m.UpdatedAt = machineData.UpdatedAt.AsTime().Format("2006-01-02T15:04:05.000Z")
			}

			// Extract schemas
			if machineData.Schemas != nil {
				m.Definition.Schemas = exportSchemas(machineData.Schemas)
			}

			// Extract implementations
			if machineData.Implementations != nil {
				m.Definition.Implementations = exportImplementations(machineData.Implementations)
			}
			break
		}
	}
}

// extractStateExtensions extracts XState-specific state data from extensions.
func extractStateExtensions(s *sc.State, node *Node) {
	for _, ext := range s.Extensions {
		// Try layout
		var layout xstatepb.XStateLayout
		if err := ext.UnmarshalTo(&layout); err == nil {
			if layout.Position != nil {
				node.Position = Position{X: layout.Position.X, Y: layout.Position.Y}
			}
			if layout.Size != nil {
				node.Size = Size{Width: layout.Size.Width, Height: layout.Size.Height}
			}
			node.Data.Color = layout.Color
			if layout.UniqueId != "" {
				node.UniqueID = layout.UniqueId
			}
			continue
		}

		// Try state data
		var stateData xstatepb.XStateStateData
		if err := ext.UnmarshalTo(&stateData); err == nil {
			node.Data.Description = stateData.Description
			node.Data.Tags = stateData.Tags

			// Export assets
			for _, asset := range stateData.Assets {
				a := Asset{
					Name:     asset.Name,
					Template: asset.Template,
				}
				if asset.Properties != nil {
					a.Properties = asset.Properties.AsMap()
				}
				node.Data.Assets = append(node.Data.Assets, a)
			}

			// Export invokes
			for _, invoke := range stateData.Invokes {
				ic := InvokeConfig{
					ID:   invoke.Id,
					Src:  invoke.Src,
					Kind: invoke.Kind,
				}
				if invoke.Input != nil {
					ic.Input = invoke.Input.AsMap()
				}
				if invoke.Settings != nil {
					ic.Settings = invoke.Settings.AsMap()
				}
				node.Data.Invoke = append(node.Data.Invoke, ic)
			}

			// Export meta entries
			if len(stateData.MetaEntries) > 0 {
				var entries [][]string
				for _, entry := range stateData.MetaEntries {
					entries = append(entries, []string{entry.Key, entry.Value})
				}
				if data, err := json.Marshal(entries); err == nil {
					node.Data.MetaEntries = data
				}
			}
			continue
		}
	}
}

// extractTransitionExtensions extracts XState-specific transition data from extensions.
func extractTransitionExtensions(t *sc.Transition, edge *Edge) {
	for _, ext := range t.Extensions {
		// Try layout
		var layout xstatepb.XStateLayout
		if err := ext.UnmarshalTo(&layout); err == nil {
			if layout.Position != nil {
				edge.Position = Position{X: layout.Position.X, Y: layout.Position.Y}
			}
			if layout.Size != nil {
				edge.Size = Size{Width: layout.Size.Width, Height: layout.Size.Height}
			}
			edge.Data.Color = layout.Color
			if layout.UniqueId != "" {
				edge.UniqueID = layout.UniqueId
			}
			continue
		}

		// Try transition data
		var transData xstatepb.XStateTransitionData
		if err := ext.UnmarshalTo(&transData); err == nil {
			edge.Data.Internal = transData.Internal
			edge.Data.Description = transData.Description

			// Override event type data from extension if present
			if transData.TriggerType != xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_UNSPECIFIED {
				edge.Data.EventTypeData.Type = mapTriggerTypeToString(transData.TriggerType)
			}
			continue
		}
	}
}

// mapTriggerTypeToString converts proto enum to XState string.
func mapTriggerTypeToString(tt xstatepb.TransitionTriggerType) string {
	switch tt {
	case xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_EVENT:
		return "named"
	case xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_ALWAYS:
		return "always"
	case xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_AFTER:
		return "after"
	case xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_INVOKE_DONE:
		return "invocation.done"
	case xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_INVOKE_ERROR:
		return "invocation.error"
	case xstatepb.TransitionTriggerType_TRANSITION_TRIGGER_TYPE_STATE_DONE:
		return "state.done"
	default:
		return "named"
	}
}

// exportSchemas converts XStateSchemas proto to JSON Schemas struct.
func exportSchemas(schemas *xstatepb.XStateSchemas) Schemas {
	s := Schemas{}
	if schemas.Tags != nil {
		s.Tags = schemas.Tags.AsMap()
	}
	if schemas.Input != nil {
		s.Input = schemas.Input.AsMap()
	}
	if schemas.Output != nil {
		s.Output = schemas.Output.AsMap()
	}
	if schemas.Actors != nil {
		s.Actors = schemas.Actors.AsMap()
	}
	if schemas.Delays != nil {
		s.Delays = schemas.Delays.AsMap()
	}
	if schemas.Events != nil {
		s.Events = schemas.Events.AsMap()
	}
	if schemas.Guards != nil {
		s.Guards = schemas.Guards.AsMap()
	}
	if schemas.Actions != nil {
		s.Actions = schemas.Actions.AsMap()
	}
	if schemas.Context != nil {
		s.Context = schemas.Context.AsMap()
	}
	return s
}

// exportImplementations converts XStateImplementations proto to JSON Implementations struct.
func exportImplementations(impls *xstatepb.XStateImplementations) Implementations {
	result := Implementations{
		Actions: make(map[string]ActionImpl),
		Guards:  make(map[string]GuardImpl),
		Actors:  make(map[string]ActorImpl),
	}

	for name, impl := range impls.Actions {
		ai := ActionImpl{
			ID:      impl.Id,
			Name:    impl.Name,
			Type:    "action",
			Code:    impl.Code,
			Imports: impl.Imports,
		}
		if impl.Schema != nil {
			ai.Schema = impl.Schema.AsMap()
		}
		result.Actions[name] = ai
	}

	for name, impl := range impls.Guards {
		gi := GuardImpl{
			ID:      impl.Id,
			Name:    impl.Name,
			Type:    "guard",
			Imports: impl.Imports,
		}
		if impl.Params != nil {
			gi.Params = impl.Params.AsMap()
		}
		result.Guards[name] = gi
	}

	for name, impl := range impls.Actors {
		ai := ActorImpl{
			ID:      impl.Id,
			Kind:    impl.Kind,
			Name:    impl.Name,
			Type:    "actor",
			Imports: impl.Imports,
		}
		if impl.Input != nil {
			ai.Input = impl.Input.AsMap()
		}
		if impl.Output != nil {
			ai.Output = impl.Output.AsMap()
		}
		result.Actors[name] = ai
	}

	return result
}

// Ensure proto is used (for type checking)
var _ = proto.Marshal
