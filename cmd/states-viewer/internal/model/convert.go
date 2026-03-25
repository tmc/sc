package model

import (
	"fmt"
	"image"

	sc "github.com/tmc/sc"
	v1 "github.com/tmc/sc/gen/statecharts/v1"
)

// DefaultNodeSize is the default size for nodes without explicit sizing.
var DefaultNodeSize = image.Point{X: 120, Y: 60}

// FromStatechart converts a protobuf Statechart to a Document.
func FromStatechart(chart *sc.Statechart) (*Document, error) {
	if chart == nil {
		return nil, fmt.Errorf("statechart is nil")
	}

	name := chart.GetName()
	if name == "" {
		name = "Untitled"
	}

	doc := NewDocument(name)

	// Convert root state and its children
	if chart.RootState != nil {
		if err := convertState(doc, chart.RootState, "", 0); err != nil {
			return nil, fmt.Errorf("convert root state: %w", err)
		}
	}

	// Convert transitions
	for i, t := range chart.GetTransitions() {
		edge := convertTransition(t, i)
		doc.AddEdge(edge)
	}

	return doc, nil
}

// convertState recursively converts a State and its children.
func convertState(doc *Document, state *v1.State, parentID string, depth int) error {
	if state == nil {
		return nil
	}

	label := state.GetLabel()
	if label == "" {
		return fmt.Errorf("state has no label")
	}

	// Determine node type
	nodeType := NodeTypeAtomic
	switch state.GetType() {
	case v1.StateType_STATE_TYPE_NORMAL:
		if len(state.GetChildren()) > 0 {
			nodeType = NodeTypeCompound
		}
	case v1.StateType_STATE_TYPE_PARALLEL:
		nodeType = NodeTypeParallel
	case v1.StateType_STATE_TYPE_BASIC:
		nodeType = NodeTypeAtomic
	}

	if state.GetIsFinal() {
		nodeType = NodeTypeFinal
	}
	if state.GetIsHistory() {
		nodeType = NodeTypeHistory
	}

	// Calculate size based on type and children
	size := DefaultNodeSize
	if nodeType == NodeTypeCompound || nodeType == NodeTypeParallel {
		// Compound/parallel nodes are larger to contain children
		children := state.GetChildren()
		if len(children) > 0 {
			// Base size plus space for children
			size = image.Point{
				X: DefaultNodeSize.X + 100,
				Y: DefaultNodeSize.Y * (len(children) + 1),
			}
		}
	}

	node := &FlowNode{
		ID:       label,
		Label:    label,
		Type:     nodeType,
		Size:     size,
		ParentID: parentID,
		Initial:  state.GetIsInitial(),
		Final:    state.GetIsFinal(),
		Scale:    1.0,
	}

	doc.AddNode(node)

	// Recursively convert children
	for _, child := range state.GetChildren() {
		if err := convertState(doc, child, label, depth+1); err != nil {
			return err
		}
	}

	return nil
}

// convertTransition converts a Transition to a FlowEdge.
func convertTransition(t *v1.Transition, index int) *FlowEdge {
	if t == nil {
		return nil
	}

	// Get source and target (handle multiple sources/targets by taking first)
	sourceID := ""
	if len(t.GetFrom()) > 0 {
		sourceID = t.GetFrom()[0]
	}

	targetID := ""
	if len(t.GetTo()) > 0 {
		targetID = t.GetTo()[0]
	}

	// Get guard expression
	guard := ""
	if t.GetGuard() != nil {
		if t.GetGuard().GetCondition() != nil {
			guard = t.GetGuard().GetCondition().GetSource()
		}
		if guard == "" {
			guard = t.GetGuard().GetExpression()
		}
	}

	// Get action
	action := ""
	if len(t.GetActions()) > 0 {
		action = t.GetActions()[0].GetLabel()
		if action == "" && t.GetActions()[0].GetBody() != nil {
			action = t.GetActions()[0].GetBody().GetSource()
		}
	}

	id := t.GetLabel()
	if id == "" {
		id = fmt.Sprintf("t%d", index)
	}

	return &FlowEdge{
		ID:       id,
		SourceID: sourceID,
		TargetID: targetID,
		Event:    t.GetEvent(),
		Guard:    guard,
		Action:   action,
	}
}

// SampleTrafficLight creates a sample traffic light statechart for testing.
func SampleTrafficLight() *Document {
	doc := NewDocument("Traffic Light")

	// Add states
	doc.AddNode(&FlowNode{
		ID:       "Red",
		Label:    "Red",
		Type:     NodeTypeAtomic,
		Position: image.Point{X: 200, Y: 100},
		Size:     DefaultNodeSize,
		Initial:  true,
		Scale:    1.0,
	})

	doc.AddNode(&FlowNode{
		ID:       "Yellow",
		Label:    "Yellow",
		Type:     NodeTypeAtomic,
		Position: image.Point{X: 400, Y: 100},
		Size:     DefaultNodeSize,
		Scale:    1.0,
	})

	doc.AddNode(&FlowNode{
		ID:       "Green",
		Label:    "Green",
		Type:     NodeTypeAtomic,
		Position: image.Point{X: 300, Y: 250},
		Size:     DefaultNodeSize,
		Scale:    1.0,
	})

	// Add transitions
	doc.AddEdge(&FlowEdge{
		ID:       "t1",
		SourceID: "Red",
		TargetID: "Green",
		Event:    "TIMER",
	})

	doc.AddEdge(&FlowEdge{
		ID:       "t2",
		SourceID: "Green",
		TargetID: "Yellow",
		Event:    "TIMER",
	})

	doc.AddEdge(&FlowEdge{
		ID:       "t3",
		SourceID: "Yellow",
		TargetID: "Red",
		Event:    "TIMER",
	})

	return doc
}

// SampleMediaPlayer creates a sample media player statechart for testing.
func SampleMediaPlayer() *Document {
	doc := NewDocument("Media Player")

	// Parent state
	doc.AddNode(&FlowNode{
		ID:       "Player",
		Label:    "Player",
		Type:     NodeTypeCompound,
		Position: image.Point{X: 300, Y: 200},
		Size:     image.Point{X: 400, Y: 300},
		Scale:    1.0,
	})

	// Child states
	doc.AddNode(&FlowNode{
		ID:       "Stopped",
		Label:    "Stopped",
		Type:     NodeTypeAtomic,
		Position: image.Point{X: 180, Y: 150},
		Size:     DefaultNodeSize,
		ParentID: "Player",
		Initial:  true,
		Scale:    1.0,
	})

	doc.AddNode(&FlowNode{
		ID:       "Playing",
		Label:    "Playing",
		Type:     NodeTypeAtomic,
		Position: image.Point{X: 420, Y: 150},
		Size:     DefaultNodeSize,
		ParentID: "Player",
		Scale:    1.0,
	})

	doc.AddNode(&FlowNode{
		ID:       "Paused",
		Label:    "Paused",
		Type:     NodeTypeAtomic,
		Position: image.Point{X: 420, Y: 280},
		Size:     DefaultNodeSize,
		ParentID: "Player",
		Scale:    1.0,
	})

	// Transitions
	doc.AddEdge(&FlowEdge{
		ID:       "t1",
		SourceID: "Stopped",
		TargetID: "Playing",
		Event:    "PLAY",
	})

	doc.AddEdge(&FlowEdge{
		ID:       "t2",
		SourceID: "Playing",
		TargetID: "Paused",
		Event:    "PAUSE",
	})

	doc.AddEdge(&FlowEdge{
		ID:       "t3",
		SourceID: "Paused",
		TargetID: "Playing",
		Event:    "PLAY",
	})

	doc.AddEdge(&FlowEdge{
		ID:       "t4",
		SourceID: "Playing",
		TargetID: "Stopped",
		Event:    "STOP",
	})

	doc.AddEdge(&FlowEdge{
		ID:       "t5",
		SourceID: "Paused",
		TargetID: "Stopped",
		Event:    "STOP",
	})

	return doc
}
