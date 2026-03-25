// Package layout provides auto-layout algorithms for statechart visualization.
package layout

import (
	"image"

	"github.com/tmc/sc/cmd/states-viewer/internal/model"
)

// Layout positions nodes in a document.
type Layout interface {
	Apply(doc *model.Document)
}

// HierarchicalLayout arranges nodes in a hierarchical tree structure.
type HierarchicalLayout struct {
	LevelSpacing int // Vertical spacing between levels
	NodeSpacing  int // Horizontal spacing between siblings
	Padding      int // Padding around the layout
}

// NewHierarchicalLayout creates a new hierarchical layout with default settings.
func NewHierarchicalLayout() *HierarchicalLayout {
	return &HierarchicalLayout{
		LevelSpacing: 120,
		NodeSpacing:  40,
		Padding:      60,
	}
}

// Apply arranges nodes hierarchically.
func (l *HierarchicalLayout) Apply(doc *model.Document) {
	if doc == nil || len(doc.Nodes) == 0 {
		return
	}

	// Build depth map
	depths := make(map[string]int)
	l.computeDepths(doc, depths, "", 0)

	// Group nodes by depth
	levels := make(map[int][]*model.FlowNode)
	maxDepth := 0
	for id, depth := range depths {
		if node := doc.Nodes[id]; node != nil {
			levels[depth] = append(levels[depth], node)
			if depth > maxDepth {
				maxDepth = depth
			}
		}
	}

	// Position each level
	y := l.Padding
	for depth := 0; depth <= maxDepth; depth++ {
		nodes := levels[depth]
		if len(nodes) == 0 {
			continue
		}

		// Calculate total width needed
		totalWidth := 0
		for _, node := range nodes {
			totalWidth += node.Size.X + l.NodeSpacing
		}
		totalWidth -= l.NodeSpacing // Remove last spacing

		// Center the level
		x := l.Padding

		// Position nodes in this level
		for _, node := range nodes {
			// Center the node at this position
			node.Position = image.Point{
				X: x + node.Size.X/2,
				Y: y + node.Size.Y/2,
			}
			x += node.Size.X + l.NodeSpacing
		}

		// Move to next level
		y += l.getMaxHeight(nodes) + l.LevelSpacing
	}

	// Position children inside parents
	l.nestChildren(doc)

	// Rebuild node order for proper rendering
	doc.NodeOrder = l.buildRenderOrder(doc)
}

// computeDepths recursively computes the depth of each node.
func (l *HierarchicalLayout) computeDepths(doc *model.Document, depths map[string]int, parentID string, depth int) {
	children := doc.Children(parentID)
	for _, child := range children {
		depths[child.ID] = depth
		l.computeDepths(doc, depths, child.ID, depth+1)
	}
}

// getMaxHeight returns the maximum height of a set of nodes.
func (l *HierarchicalLayout) getMaxHeight(nodes []*model.FlowNode) int {
	maxH := 0
	for _, n := range nodes {
		if n.Size.Y > maxH {
			maxH = n.Size.Y
		}
	}
	return maxH
}

// nestChildren adjusts child node positions to be inside their parents.
func (l *HierarchicalLayout) nestChildren(doc *model.Document) {
	for _, node := range doc.Nodes {
		if node.ParentID == "" {
			continue
		}

		parent := doc.Nodes[node.ParentID]
		if parent == nil {
			continue
		}

		// Ensure parent is large enough
		children := doc.Children(parent.ID)
		if len(children) > 0 {
			// Calculate bounding box of children
			minX, minY := children[0].Position.X-children[0].Size.X/2, children[0].Position.Y-children[0].Size.Y/2
			maxX, maxY := children[0].Position.X+children[0].Size.X/2, children[0].Position.Y+children[0].Size.Y/2

			for _, child := range children[1:] {
				cx, cy := child.Position.X, child.Position.Y
				hw, hh := child.Size.X/2, child.Size.Y/2
				if cx-hw < minX {
					minX = cx - hw
				}
				if cy-hh < minY {
					minY = cy - hh
				}
				if cx+hw > maxX {
					maxX = cx + hw
				}
				if cy+hh > maxY {
					maxY = cy + hh
				}
			}

			// Add padding for header
			headerH := 32
			padding := 16

			// Resize parent to fit children
			parent.Size = image.Point{
				X: maxX - minX + padding*2,
				Y: maxY - minY + padding*2 + headerH,
			}

			// Reposition parent to center around children
			parent.Position = image.Point{
				X: (minX + maxX) / 2,
				Y: (minY + maxY) / 2 + headerH/2,
			}
		}
	}
}

// buildRenderOrder builds the node order for rendering (parents before children).
func (l *HierarchicalLayout) buildRenderOrder(doc *model.Document) []string {
	var order []string
	visited := make(map[string]bool)

	var visit func(id string)
	visit = func(id string) {
		if visited[id] {
			return
		}
		visited[id] = true

		// Add parent first if it exists
		node := doc.Nodes[id]
		if node != nil && node.ParentID != "" {
			visit(node.ParentID)
		}

		order = append(order, id)

		// Then add children
		for _, child := range doc.Children(id) {
			visit(child.ID)
		}
	}

	// Start with root nodes
	for _, root := range doc.RootNodes() {
		visit(root.ID)
	}

	// Add any remaining nodes
	for id := range doc.Nodes {
		visit(id)
	}

	return order
}

// GridLayout arranges nodes in a simple grid pattern.
type GridLayout struct {
	Columns     int
	CellWidth   int
	CellHeight  int
	Padding     int
}

// NewGridLayout creates a new grid layout.
func NewGridLayout(columns int) *GridLayout {
	return &GridLayout{
		Columns:    columns,
		CellWidth:  160,
		CellHeight: 100,
		Padding:    40,
	}
}

// Apply arranges nodes in a grid.
func (l *GridLayout) Apply(doc *model.Document) {
	if doc == nil {
		return
	}

	col := 0
	row := 0

	for _, id := range doc.NodeOrder {
		node := doc.Nodes[id]
		if node == nil {
			continue
		}

		// Skip nested nodes (they'll be positioned by parents)
		if node.ParentID != "" {
			continue
		}

		node.Position = image.Point{
			X: l.Padding + col*l.CellWidth + l.CellWidth/2,
			Y: l.Padding + row*l.CellHeight + l.CellHeight/2,
		}

		col++
		if col >= l.Columns {
			col = 0
			row++
		}
	}
}
