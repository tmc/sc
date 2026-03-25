// Package model provides data types for statechart visualization.
package model

import (
	"image"
)

// NodeType represents the type of a state node.
type NodeType int

const (
	NodeTypeAtomic   NodeType = iota // Leaf state, no children
	NodeTypeCompound                 // Has children, XOR semantics
	NodeTypeParallel                 // Has children, AND semantics
	NodeTypeFinal                    // Terminal state
	NodeTypeHistory                  // History pseudo-state
)

// String returns the string representation of a NodeType.
func (t NodeType) String() string {
	switch t {
	case NodeTypeAtomic:
		return "atomic"
	case NodeTypeCompound:
		return "compound"
	case NodeTypeParallel:
		return "parallel"
	case NodeTypeFinal:
		return "final"
	case NodeTypeHistory:
		return "history"
	default:
		return "unknown"
	}
}

// FlowNode represents a state in the visualization.
type FlowNode struct {
	ID       string      // Unique identifier (state label)
	Label    string      // Display label
	Type     NodeType    // State type
	Position image.Point // Canvas position (center)
	Size     image.Point // Width, Height
	ParentID string      // Parent state ID (empty if root-level)
	Initial  bool        // Is this the initial state
	Final    bool        // Is this a final state

	// Visual state (updated per frame)
	Hovered      bool    // Mouse is over this node
	Selected     bool    // Node is selected
	Active       bool    // State is currently active in simulation
	HoverT       float64 // Animation progress for hover (0-1)
	ActiveT      float64 // Animation progress for active glow (0-1)
	Scale        float64 // Current scale (for hover effect)
	BreathPhase  float64 // Phase for breathing animation
}

// Bounds returns the bounding rectangle of the node.
func (n *FlowNode) Bounds() image.Rectangle {
	halfW := n.Size.X / 2
	halfH := n.Size.Y / 2
	return image.Rect(
		n.Position.X-halfW,
		n.Position.Y-halfH,
		n.Position.X+halfW,
		n.Position.Y+halfH,
	)
}

// Contains returns true if the point is inside the node bounds.
func (n *FlowNode) Contains(p image.Point) bool {
	return p.In(n.Bounds())
}

// FlowEdge represents a transition in the visualization.
type FlowEdge struct {
	ID       string // Unique identifier
	SourceID string // Source state ID
	TargetID string // Target state ID
	Event    string // Triggering event
	Guard    string // Guard condition
	Action   string // Action to execute

	// Visual state
	Selected  bool    // Edge is selected
	Highlight bool    // Edge is highlighted (transition firing)
	HighlightT float64 // Animation progress for highlight
}

// Label returns the display label for the edge.
// Format: "event [guard] / action"
func (e *FlowEdge) Label() string {
	if e.Event == "" && e.Guard == "" && e.Action == "" {
		return ""
	}

	label := e.Event
	if e.Guard != "" {
		if label != "" {
			label += " "
		}
		label += "[" + e.Guard + "]"
	}
	if e.Action != "" {
		if label != "" {
			label += " / "
		} else {
			label = "/ "
		}
		label += e.Action
	}
	return label
}

// Port represents a connection point on a node.
type Port int

const (
	PortTop Port = iota
	PortRight
	PortBottom
	PortLeft
)

// PortPosition returns the position of a port on the given node.
func PortPosition(node *FlowNode, port Port) image.Point {
	bounds := node.Bounds()
	switch port {
	case PortTop:
		return image.Point{
			X: (bounds.Min.X + bounds.Max.X) / 2,
			Y: bounds.Min.Y,
		}
	case PortRight:
		return image.Point{
			X: bounds.Max.X,
			Y: (bounds.Min.Y + bounds.Max.Y) / 2,
		}
	case PortBottom:
		return image.Point{
			X: (bounds.Min.X + bounds.Max.X) / 2,
			Y: bounds.Max.Y,
		}
	case PortLeft:
		return image.Point{
			X: bounds.Min.X,
			Y: (bounds.Min.Y + bounds.Max.Y) / 2,
		}
	default:
		return node.Position
	}
}

// BestPorts returns the best source and target ports for connecting two nodes.
// Uses relative position to determine optimal connection points.
func BestPorts(source, target *FlowNode) (Port, Port) {
	sBounds := source.Bounds()
	tBounds := target.Bounds()

	// Determine primary axis
	dx := tBounds.Min.X - sBounds.Max.X // Gap on X
	dy := tBounds.Min.Y - sBounds.Max.Y // Gap on Y

	// Target is to the right
	if tBounds.Min.X >= sBounds.Max.X {
		// Target also below
		if tBounds.Min.Y >= sBounds.Max.Y {
			// Prefer horizontal if more horizontal gap
			if dx > dy {
				return PortRight, PortLeft
			}
			return PortBottom, PortTop
		}
		// Target also above
		if tBounds.Max.Y <= sBounds.Min.Y {
			if dx > -dy {
				return PortRight, PortLeft
			}
			return PortTop, PortBottom
		}
		// Target same vertical level
		return PortRight, PortLeft
	}

	// Target is to the left
	if tBounds.Max.X <= sBounds.Min.X {
		// Target also below
		if tBounds.Min.Y >= sBounds.Max.Y {
			if -dx > dy {
				return PortLeft, PortRight
			}
			return PortBottom, PortTop
		}
		// Target also above
		if tBounds.Max.Y <= sBounds.Min.Y {
			if -dx > -dy {
				return PortLeft, PortRight
			}
			return PortTop, PortBottom
		}
		// Target same vertical level
		return PortLeft, PortRight
	}

	// Target overlaps horizontally
	// Target below
	if tBounds.Min.Y >= sBounds.Max.Y {
		return PortBottom, PortTop
	}
	// Target above
	if tBounds.Max.Y <= sBounds.Min.Y {
		return PortTop, PortBottom
	}

	// Overlapping - use right/left as default
	return PortRight, PortLeft
}

// Document represents a loaded statechart document.
type Document struct {
	Name  string
	Nodes map[string]*FlowNode
	Edges []*FlowEdge

	// Ordered list for rendering (parents before children)
	NodeOrder []string

	// Reverse edge lookup (computed on demand)
	reverseEdges map[string]bool
}

// HasReverseEdge returns true if there's an edge going in the opposite direction.
// This is used to offset bidirectional edges so both are visible.
func (d *Document) HasReverseEdge(sourceID, targetID string) bool {
	if d.reverseEdges == nil {
		d.reverseEdges = make(map[string]bool)
		for _, e := range d.Edges {
			// Store both directions
			d.reverseEdges[e.SourceID+":"+e.TargetID] = true
		}
	}
	// Check if reverse exists
	return d.reverseEdges[targetID+":"+sourceID]
}

// EdgeIndex returns the index of this edge among all edges with the same source-target pair.
// Returns (index, total) - useful for offsetting multiple parallel edges.
func (d *Document) EdgeIndex(edge *FlowEdge) (int, int) {
	idx := 0
	total := 0
	for _, e := range d.Edges {
		if (e.SourceID == edge.SourceID && e.TargetID == edge.TargetID) ||
			(e.SourceID == edge.TargetID && e.TargetID == edge.SourceID) {
			if e == edge {
				idx = total
			}
			total++
		}
	}
	return idx, total
}

// NewDocument creates an empty document.
func NewDocument(name string) *Document {
	return &Document{
		Name:      name,
		Nodes:     make(map[string]*FlowNode),
		Edges:     make([]*FlowEdge, 0),
		NodeOrder: make([]string, 0),
	}
}

// AddNode adds a node to the document.
func (d *Document) AddNode(node *FlowNode) {
	d.Nodes[node.ID] = node
	d.NodeOrder = append(d.NodeOrder, node.ID)
}

// AddEdge adds an edge to the document.
func (d *Document) AddEdge(edge *FlowEdge) {
	d.Edges = append(d.Edges, edge)
}

// NodeAt returns the topmost node at the given point.
// Checks in reverse render order (children before parents).
func (d *Document) NodeAt(p image.Point) *FlowNode {
	// Check in reverse order (top to bottom in z-order)
	for i := len(d.NodeOrder) - 1; i >= 0; i-- {
		id := d.NodeOrder[i]
		if node, ok := d.Nodes[id]; ok {
			if node.Contains(p) {
				return node
			}
		}
	}
	return nil
}

// Children returns all direct children of a node.
func (d *Document) Children(parentID string) []*FlowNode {
	var children []*FlowNode
	for _, node := range d.Nodes {
		if node.ParentID == parentID {
			children = append(children, node)
		}
	}
	return children
}

// RootNodes returns all nodes without a parent.
func (d *Document) RootNodes() []*FlowNode {
	return d.Children("")
}
