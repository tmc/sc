package canvas

import (
	"image"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"

	"github.com/tmc/sc/cmd/states-viewer/internal/model"
	"github.com/tmc/sc/cmd/states-viewer/internal/theme"
)

// InteractionMode represents the current interaction state.
type InteractionMode int

const (
	ModeNone InteractionMode = iota
	ModePan
	ModeDragNode
)

// Interaction manages user input and interaction state.
type Interaction struct {
	canvas      *Canvas
	doc         *model.Document
	mode        InteractionMode
	selectedID  string // Currently selected node ID
	dragNodeID  string // Node being dragged
	dragOffset  Point  // Offset from node center when drag started
	lastMouseX  int
	lastMouseY  int
	spaceHeld   bool // Space key held for pan mode
}

// NewInteraction creates a new interaction handler.
func NewInteraction(canvas *Canvas, doc *model.Document) *Interaction {
	return &Interaction{
		canvas: canvas,
		doc:    doc,
	}
}

// SetDocument updates the document being interacted with.
func (i *Interaction) SetDocument(doc *model.Document) {
	i.doc = doc
	i.selectedID = ""
	i.dragNodeID = ""
	i.mode = ModeNone
}

// SelectedNode returns the currently selected node.
func (i *Interaction) SelectedNode() *model.FlowNode {
	if i.selectedID == "" || i.doc == nil {
		return nil
	}
	return i.doc.Nodes[i.selectedID]
}

// Update processes input and updates interaction state.
// Returns true if the canvas needs to be redrawn.
func (i *Interaction) Update(dt float64) bool {
	if i.doc == nil {
		return false
	}

	needsRedraw := false

	// Get current mouse position
	mx, my := ebiten.CursorPosition()
	worldPos := i.canvas.ScreenToWorldPoint(mx, my)

	// Track space key for pan mode
	i.spaceHeld = ebiten.IsKeyPressed(ebiten.KeySpace)

	// Handle scroll wheel zoom
	_, wheelY := ebiten.Wheel()
	if wheelY != 0 {
		factor := 1.0 + wheelY*0.1
		i.canvas.ZoomAt(float64(mx), float64(my), factor)
		needsRedraw = true
	}

	// Handle mouse buttons
	leftPressed := ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)
	middlePressed := ebiten.IsMouseButtonPressed(ebiten.MouseButtonMiddle)
	leftJustPressed := inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft)
	_ = inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) // Reserved for future use

	// Update hover states
	needsRedraw = i.updateHoverStates(worldPos) || needsRedraw

	switch i.mode {
	case ModeNone:
		if leftJustPressed {
			// Check if clicking on a node
			node := i.doc.NodeAt(worldPos)
			if node != nil {
				if i.spaceHeld {
					// Start panning even when over node
					i.mode = ModePan
					i.lastMouseX = mx
					i.lastMouseY = my
				} else {
					// Select and start dragging
					i.selectNode(node.ID)
					i.mode = ModeDragNode
					i.dragNodeID = node.ID
					i.dragOffset = Point{
						X: float64(worldPos.X - node.Position.X),
						Y: float64(worldPos.Y - node.Position.Y),
					}
					needsRedraw = true
				}
			} else {
				// Clicked on background - deselect
				if i.selectedID != "" {
					i.selectNode("")
					needsRedraw = true
				}
				// Start panning
				i.mode = ModePan
				i.lastMouseX = mx
				i.lastMouseY = my
			}
		} else if middlePressed {
			// Middle click always starts pan
			i.mode = ModePan
			i.lastMouseX = mx
			i.lastMouseY = my
		}

	case ModePan:
		if leftPressed || middlePressed || i.spaceHeld {
			// Continue panning
			dx := float64(mx - i.lastMouseX)
			dy := float64(my - i.lastMouseY)
			if dx != 0 || dy != 0 {
				i.canvas.Pan(dx, dy)
				i.lastMouseX = mx
				i.lastMouseY = my
				needsRedraw = true
			}
		} else {
			// Stop panning
			i.mode = ModeNone
		}

	case ModeDragNode:
		if leftPressed && !i.spaceHeld {
			// Continue dragging node
			node := i.doc.Nodes[i.dragNodeID]
			if node != nil {
				newX := worldPos.X - int(i.dragOffset.X)
				newY := worldPos.Y - int(i.dragOffset.Y)

				// Snap to grid
				grid := theme.GridSpacing
				newX = ((newX + grid/2) / grid) * grid
				newY = ((newY + grid/2) / grid) * grid

				if newX != node.Position.X || newY != node.Position.Y {
					node.Position = image.Point{X: newX, Y: newY}
					needsRedraw = true
				}
			}
		} else {
			// Stop dragging
			i.mode = ModeNone
			i.dragNodeID = ""
		}
	}

	// Handle keyboard shortcuts
	needsRedraw = i.handleKeyboard() || needsRedraw

	return needsRedraw
}

// updateHoverStates updates node hover states based on mouse position.
func (i *Interaction) updateHoverStates(worldPos image.Point) bool {
	changed := false

	hoveredNode := i.doc.NodeAt(worldPos)

	for _, node := range i.doc.Nodes {
		wasHovered := node.Hovered
		node.Hovered = (hoveredNode == node)

		if node.Hovered != wasHovered {
			changed = true
		}

		// Update hover animation
		if node.Hovered {
			node.HoverT += 0.1
			if node.HoverT > 1 {
				node.HoverT = 1
			}
			node.Scale = theme.Lerp(1.0, 1.02, theme.EaseOut(node.HoverT))
		} else {
			node.HoverT -= 0.1
			if node.HoverT < 0 {
				node.HoverT = 0
			}
			node.Scale = theme.Lerp(1.0, 1.02, theme.EaseOut(node.HoverT))
		}
	}

	return changed
}

// selectNode selects a node by ID (or deselects if empty).
func (i *Interaction) selectNode(id string) {
	// Deselect previous
	if i.selectedID != "" {
		if node := i.doc.Nodes[i.selectedID]; node != nil {
			node.Selected = false
		}
	}

	i.selectedID = id

	// Select new
	if id != "" {
		if node := i.doc.Nodes[id]; node != nil {
			node.Selected = true
		}
	}
}

// handleKeyboard processes keyboard input.
func (i *Interaction) handleKeyboard() bool {
	needsRedraw := false

	// Escape to deselect
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
		if i.selectedID != "" {
			i.selectNode("")
			needsRedraw = true
		}
	}

	// F to fit to view
	if inpututil.IsKeyJustPressed(ebiten.KeyF) {
		i.zoomToFit()
		needsRedraw = true
	}

	// 0 to reset zoom
	if inpututil.IsKeyJustPressed(ebiten.Key0) {
		i.canvas.SetScale(1.0)
		needsRedraw = true
	}

	// + to zoom in
	if inpututil.IsKeyJustPressed(ebiten.KeyEqual) {
		i.canvas.ZoomAt(float64(i.lastMouseX), float64(i.lastMouseY), 1.2)
		needsRedraw = true
	}

	// - to zoom out
	if inpututil.IsKeyJustPressed(ebiten.KeyMinus) {
		i.canvas.ZoomAt(float64(i.lastMouseX), float64(i.lastMouseY), 0.8)
		needsRedraw = true
	}

	return needsRedraw
}

// zoomToFit zooms to fit all nodes in view.
func (i *Interaction) zoomToFit() {
	if i.doc == nil || len(i.doc.Nodes) == 0 {
		return
	}

	// Calculate bounding box of all nodes
	var bounds image.Rectangle
	first := true

	for _, node := range i.doc.Nodes {
		nodeBounds := node.Bounds()
		if first {
			bounds = nodeBounds
			first = false
		} else {
			bounds = bounds.Union(nodeBounds)
		}
	}

	// Add padding
	padding := 50
	i.canvas.ZoomToFit(bounds, padding)
}

// IsPanning returns true if currently panning.
func (i *Interaction) IsPanning() bool {
	return i.mode == ModePan
}

// IsDragging returns true if currently dragging a node.
func (i *Interaction) IsDragging() bool {
	return i.mode == ModeDragNode
}

// HandleTouch processes touch input for mobile.
// Returns true if touch was handled.
func (i *Interaction) HandleTouch() bool {
	touchIDs := inpututil.AppendJustPressedTouchIDs(nil)
	if len(touchIDs) == 0 {
		return false
	}

	// Handle single touch as click/drag
	if len(touchIDs) == 1 {
		x, y := ebiten.TouchPosition(touchIDs[0])
		worldPos := i.canvas.ScreenToWorldPoint(x, y)

		node := i.doc.NodeAt(worldPos)
		if node != nil {
			i.selectNode(node.ID)
			return true
		}
	}

	// TODO: Handle pinch zoom with two touches
	// TODO: Handle two-finger pan

	return false
}
