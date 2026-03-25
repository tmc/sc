package canvas

import (
	"image"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"

	"github.com/tmc/sc/cmd/states-viewer/internal/model"
	"github.com/tmc/sc/cmd/states-viewer/internal/theme"
)

// Renderer handles drawing of statechart elements.
type Renderer struct {
	canvas   *Canvas
	fontFace *text.GoTextFace
	time     float64 // Animation time in seconds
}

// NewRenderer creates a new renderer.
func NewRenderer(canvas *Canvas, fontFace *text.GoTextFace) *Renderer {
	return &Renderer{
		canvas:   canvas,
		fontFace: fontFace,
	}
}

// Update advances animation time.
func (r *Renderer) Update(dt float64) {
	r.time += dt
}

// DrawGrid draws the background dot grid.
func (r *Renderer) DrawGrid(dst *ebiten.Image) {
	w, h := r.canvas.Size()
	scale := r.canvas.Scale()
	offset := r.canvas.Offset()

	// Grid spacing in world coordinates
	spacing := float64(theme.GridSpacing)

	// Parallax factor
	parallax := 0.5

	// Calculate grid offset with parallax
	offsetX := math.Mod(offset.X*parallax, spacing)
	offsetY := math.Mod(offset.Y*parallax, spacing)

	// Dot size scales with zoom
	dotSize := float32(theme.Clamp(theme.GridDotSize*scale, 1.0, 4.0))

	// Draw dots
	startX := -offsetX * scale
	startY := -offsetY * scale
	stepX := spacing * scale
	stepY := spacing * scale

	for y := startY; y < float64(h); y += stepY {
		for x := startX; x < float64(w); x += stepX {
			vector.DrawFilledCircle(
				dst,
				float32(x),
				float32(y),
				dotSize/2,
				theme.GridDot,
				false,
			)
		}
	}
}

// DrawNode draws a single state node.
func (r *Renderer) DrawNode(dst *ebiten.Image, node *model.FlowNode) {
	bounds := node.Bounds()
	screenBounds := r.canvas.WorldBoundsToScreen(bounds)

	// Check if visible
	w, h := r.canvas.Size()
	if screenBounds.Max.X < 0 || screenBounds.Min.X > w ||
		screenBounds.Max.Y < 0 || screenBounds.Min.Y > h {
		return
	}

	scale := r.canvas.Scale()

	// Calculate visual properties
	x := float32(screenBounds.Min.X)
	y := float32(screenBounds.Min.Y)
	width := float32(screenBounds.Dx())
	height := float32(screenBounds.Dy())
	cornerRadius := float32(theme.NodeCornerRadius * scale)

	// Apply hover scale
	if node.Scale != 1.0 {
		centerX := x + width/2
		centerY := y + height/2
		width *= float32(node.Scale)
		height *= float32(node.Scale)
		x = centerX - width/2
		y = centerY - height/2
	}

	// Draw active glow
	if node.Active {
		r.drawActiveGlow(dst, x, y, width, height, cornerRadius)
	}

	// Draw shadow (if not active)
	if !node.Active && (node.Hovered || node.Selected) {
		r.drawShadow(dst, x, y, width, height, cornerRadius, theme.Shadow1)
	}

	// Determine fill color
	fillColor := theme.NodeBg
	if node.Active {
		// More visible green tint for active states
		fillColor = theme.LerpColor(theme.NodeBg, theme.Success, 0.15)
	}

	// Draw node based on type
	switch node.Type {
	case model.NodeTypeFinal:
		r.drawFinalNode(dst, x, y, width, height, fillColor, node)
	case model.NodeTypeParallel:
		r.drawParallelNode(dst, x, y, width, height, cornerRadius, fillColor, node)
	case model.NodeTypeCompound:
		r.drawCompoundNode(dst, x, y, width, height, cornerRadius, fillColor, node)
	case model.NodeTypeHistory:
		r.drawHistoryNode(dst, x, y, width, height, node)
	default:
		r.drawAtomicNode(dst, x, y, width, height, cornerRadius, fillColor, node)
	}
}

// drawActiveGlow draws the animated glow for active states.
func (r *Renderer) drawActiveGlow(dst *ebiten.Image, x, y, w, h, radius float32) {
	// Breathing animation - more pronounced
	alpha := theme.BreathingAlpha(r.time)
	baseAlpha := 0.4 + 0.3*alpha // Range: 0.4 to 0.7

	// Draw filled glow background (soft green tint)
	bgAlpha := uint8(40 * baseAlpha)
	bgColor := theme.WithAlpha(theme.Success, bgAlpha)
	vector.DrawFilledRect(dst, x-8, y-8, w+16, h+16, bgColor, false)

	// Draw expanding glow rings (more visible)
	for i := 0; i < 4; i++ {
		expand := float32(i*5 + 6)
		ringAlpha := uint8(float64(120*baseAlpha) * (1 - float64(i)*0.25))
		ringColor := theme.WithAlpha(theme.Success, ringAlpha)

		strokeWidth := float32(3 - float32(i)*0.5)
		if strokeWidth < 1 {
			strokeWidth = 1
		}

		vector.StrokeRect(
			dst,
			x-expand,
			y-expand,
			w+expand*2,
			h+expand*2,
			strokeWidth,
			ringColor,
			false,
		)
	}

	// Draw prominent inner border
	innerBorderColor := theme.WithAlpha(theme.Success, uint8(200*baseAlpha))
	vector.StrokeRect(dst, x-2, y-2, w+4, h+4, 3, innerBorderColor, false)
}

// drawShadow draws a shadow beneath a shape.
func (r *Renderer) drawShadow(dst *ebiten.Image, x, y, w, h, radius float32, shadow theme.Shadow) {
	// Simple shadow as offset filled rect with blur approximation
	shadowX := x + float32(shadow.OffsetX)
	shadowY := y + float32(shadow.OffsetY)

	// Draw multiple layers to approximate blur
	for i := 0; i < int(shadow.Blur); i += 2 {
		expand := float32(i)
		a := shadow.Color.A / uint8(1+i/2)
		c := color.RGBA{0, 0, 0, a}

		vector.DrawFilledRect(
			dst,
			shadowX-expand,
			shadowY-expand,
			w+expand*2,
			h+expand*2,
			c,
			false,
		)
	}
}

// drawAtomicNode draws a basic atomic state.
func (r *Renderer) drawAtomicNode(dst *ebiten.Image, x, y, w, h, radius float32, fill color.RGBA, node *model.FlowNode) {
	// Fill
	vector.DrawFilledRect(dst, x, y, w, h, fill, false)

	// Border
	borderColor := theme.BorderMedium
	borderWidth := float32(theme.NodeBorderWidth)
	if node.Active {
		borderColor = theme.Success
		borderWidth = float32(theme.NodeBorderSelect)
	} else if node.Selected {
		borderColor = theme.Accent
		borderWidth = float32(theme.NodeBorderSelect)
	} else if node.Hovered {
		borderColor = theme.AccentLight
		borderWidth = float32(theme.NodeBorderHover)
	}

	vector.StrokeRect(dst, x, y, w, h, borderWidth, borderColor, false)

	// Active state indicator (green dot in top-right corner)
	if node.Active {
		dotRadius := float32(6)
		dotX := x + w - dotRadius - 4
		dotY := y + dotRadius + 4
		// White background circle
		vector.DrawFilledCircle(dst, dotX, dotY, dotRadius+2, theme.NodeBg, false)
		// Green dot
		vector.DrawFilledCircle(dst, dotX, dotY, dotRadius, theme.Success, false)
	}

	// Initial state indicator (small arrow)
	if node.Initial && !node.Active {
		r.drawInitialIndicator(dst, x, y, h)
	}

	// Label
	r.drawNodeLabel(dst, node.Label, x, y, w, h)
}

// drawCompoundNode draws a compound state with header.
func (r *Renderer) drawCompoundNode(dst *ebiten.Image, x, y, w, h, radius float32, fill color.RGBA, node *model.FlowNode) {
	headerH := float32(28 * r.canvas.Scale())

	// Fill body
	vector.DrawFilledRect(dst, x, y, w, h, fill, false)

	// Header bar (with active tint if active)
	headerColor := theme.LerpColor(fill, theme.BorderLight, 0.5)
	if node.Active {
		headerColor = theme.LerpColor(headerColor, theme.Success, 0.2)
	}
	vector.DrawFilledRect(dst, x, y, w, headerH, headerColor, false)

	// Separator line
	vector.StrokeLine(dst, x, y+headerH, x+w, y+headerH, 1, theme.BorderMedium, false)

	// Border
	borderColor := theme.BorderMedium
	borderWidth := float32(theme.NodeBorderWidth)
	if node.Active {
		borderColor = theme.Success
		borderWidth = float32(theme.NodeBorderSelect)
	} else if node.Selected {
		borderColor = theme.Accent
		borderWidth = float32(theme.NodeBorderSelect)
	} else if node.Hovered {
		borderColor = theme.AccentLight
		borderWidth = float32(theme.NodeBorderHover)
	}

	vector.StrokeRect(dst, x, y, w, h, borderWidth, borderColor, false)

	// Active indicator in header
	if node.Active {
		dotRadius := float32(5)
		dotX := x + w - dotRadius - 6
		dotY := y + headerH/2
		vector.DrawFilledCircle(dst, dotX, dotY, dotRadius, theme.Success, false)
	}

	// Label in header
	r.drawNodeLabel(dst, node.Label, x, y, w, headerH)
}

// drawParallelNode draws a parallel state with dashed border.
func (r *Renderer) drawParallelNode(dst *ebiten.Image, x, y, w, h, radius float32, fill color.RGBA, node *model.FlowNode) {
	// Fill with tint
	tintedFill := theme.LerpColor(fill, theme.ParallelTint, 0.2)
	vector.DrawFilledRect(dst, x, y, w, h, tintedFill, false)

	// Dashed border
	borderColor := theme.Accent
	if node.Selected {
		borderColor = theme.Accent
	} else if node.Hovered {
		borderColor = theme.AccentLight
	}

	r.drawDashedRect(dst, x, y, w, h, 2, borderColor, 8, 4)

	// Label at top
	headerH := float32(24 * r.canvas.Scale())
	r.drawNodeLabel(dst, node.Label, x, y, w, headerH)
}

// drawFinalNode draws a final state as a filled circle.
func (r *Renderer) drawFinalNode(dst *ebiten.Image, x, y, w, h float32, fill color.RGBA, node *model.FlowNode) {
	centerX := x + w/2
	centerY := y + h/2
	radius := float32(math.Min(float64(w), float64(h))) / 2

	// Outer circle
	vector.DrawFilledCircle(dst, centerX, centerY, radius, theme.FinalFill, false)

	// Inner circle (target symbol)
	innerRadius := radius * 0.6
	vector.DrawFilledCircle(dst, centerX, centerY, innerRadius, theme.NodeBg, false)
	vector.DrawFilledCircle(dst, centerX, centerY, innerRadius*0.4, theme.FinalFill, false)

	// Border if selected/hovered
	if node.Selected {
		vector.StrokeCircle(dst, centerX, centerY, radius, 2, theme.Accent, false)
	} else if node.Hovered {
		vector.StrokeCircle(dst, centerX, centerY, radius, 2, theme.AccentLight, false)
	}
}

// drawHistoryNode draws a history pseudo-state.
func (r *Renderer) drawHistoryNode(dst *ebiten.Image, x, y, w, h float32, node *model.FlowNode) {
	centerX := x + w/2
	centerY := y + h/2
	radius := float32(math.Min(float64(w), float64(h))) / 2

	// Circle with amber fill
	vector.DrawFilledCircle(dst, centerX, centerY, radius, theme.Warning, false)
	vector.StrokeCircle(dst, centerX, centerY, radius, 1, theme.BorderMedium, false)

	// "H" label
	r.drawNodeLabel(dst, "H", x, y, w, h)
}

// drawInitialIndicator draws a small arrow pointing to the initial state.
func (r *Renderer) drawInitialIndicator(dst *ebiten.Image, x, y, h float32) {
	// Small filled circle and arrow on the left side
	indicatorX := x - 20
	indicatorY := y + h/2

	// Small filled circle (start point)
	vector.DrawFilledCircle(dst, indicatorX-8, indicatorY, 4, theme.TextSecondary, false)

	// Arrow line
	vector.StrokeLine(dst, indicatorX-4, indicatorY, indicatorX-2, indicatorY, 2, theme.TextSecondary, false)
}

// drawNodeLabel draws centered text on a node.
func (r *Renderer) drawNodeLabel(dst *ebiten.Image, label string, x, y, w, h float32) {
	if r.fontFace == nil || label == "" {
		return
	}

	// Scale font size with zoom
	fontSize := theme.FontSizeBody * r.canvas.Scale()
	fontSize = theme.Clamp(fontSize, 8, 24)

	// Measure text
	face := &text.GoTextFace{
		Source: r.fontFace.Source,
		Size:   fontSize,
	}

	textW, textH := text.Measure(label, face, 0)

	// Center position
	textX := float64(x) + float64(w)/2 - textW/2
	textY := float64(y) + float64(h)/2 - textH/2

	op := &text.DrawOptions{}
	op.GeoM.Translate(textX, textY)
	op.ColorScale.ScaleWithColor(theme.TextPrimary)

	text.Draw(dst, label, face, op)
}

// drawDashedRect draws a rectangle with a dashed stroke.
func (r *Renderer) drawDashedRect(dst *ebiten.Image, x, y, w, h, strokeWidth float32, c color.RGBA, dashLen, gapLen float32) {
	// Top edge
	r.drawDashedLine(dst, x, y, x+w, y, strokeWidth, c, dashLen, gapLen)
	// Right edge
	r.drawDashedLine(dst, x+w, y, x+w, y+h, strokeWidth, c, dashLen, gapLen)
	// Bottom edge
	r.drawDashedLine(dst, x+w, y+h, x, y+h, strokeWidth, c, dashLen, gapLen)
	// Left edge
	r.drawDashedLine(dst, x, y+h, x, y, strokeWidth, c, dashLen, gapLen)
}

// drawDashedLine draws a line with dashes.
func (r *Renderer) drawDashedLine(dst *ebiten.Image, x1, y1, x2, y2, strokeWidth float32, c color.RGBA, dashLen, gapLen float32) {
	dx := x2 - x1
	dy := y2 - y1
	length := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if length == 0 {
		return
	}

	// Normalize direction
	dx /= length
	dy /= length

	pos := float32(0)
	for pos < length {
		endPos := pos + dashLen
		if endPos > length {
			endPos = length
		}

		vector.StrokeLine(
			dst,
			x1+dx*pos, y1+dy*pos,
			x1+dx*endPos, y1+dy*endPos,
			strokeWidth,
			c,
			false,
		)

		pos += dashLen + gapLen
	}
}

// DrawEdge draws a transition edge.
func (r *Renderer) DrawEdge(dst *ebiten.Image, edge *model.FlowEdge, doc *model.Document) {
	nodes := doc.Nodes
	source, ok := nodes[edge.SourceID]
	if !ok {
		return
	}
	target, ok := nodes[edge.TargetID]
	if !ok {
		return
	}

	// Get best connection ports
	srcPort, tgtPort := model.BestPorts(source, target)
	srcPos := model.PortPosition(source, srcPort)
	tgtPos := model.PortPosition(target, tgtPort)

	// Convert to screen coordinates
	srcScreen := r.canvas.WorldToScreenPoint(srcPos)
	tgtScreen := r.canvas.WorldToScreenPoint(tgtPos)

	// DEBUG: Draw bright markers at source and target positions
	// Blue circle at source
	vector.DrawFilledCircle(dst, float32(srcScreen.X), float32(srcScreen.Y), 8, color.RGBA{0, 100, 255, 255}, false)
	// Green circle at target
	vector.DrawFilledCircle(dst, float32(tgtScreen.X), float32(tgtScreen.Y), 8, color.RGBA{0, 255, 100, 255}, false)

	// Draw edge line with very visible settings
	edgeColor := color.RGBA{255, 0, 255, 255} // Bright magenta for maximum visibility
	strokeWidth := float32(3.0)               // Thick line
	if edge.Selected {
		edgeColor = theme.Accent
		strokeWidth = 4.0
	}

	// Draw simple straight line between source and target
	vector.StrokeLine(dst, float32(srcScreen.X), float32(srcScreen.Y),
		float32(tgtScreen.X), float32(tgtScreen.Y), strokeWidth, edgeColor, false)

	// Draw arrow head
	r.drawArrowHead(dst, tgtScreen, tgtPort, edgeColor)

	// Draw label
	if label := edge.Label(); label != "" {
		r.drawEdgeLabel(dst, label, srcScreen, tgtScreen)
	}
}

// drawOrthogonalPath draws an L-shaped path between two points.
func (r *Renderer) drawOrthogonalPath(dst *ebiten.Image, src, tgt image.Point, srcPort, tgtPort model.Port, strokeWidth float32, c color.RGBA) {
	sx, sy := float32(src.X), float32(src.Y)
	tx, ty := float32(tgt.X), float32(tgt.Y)

	// DEBUG: Use simple straight line instead of orthogonal path
	vector.StrokeLine(dst, sx, sy, tx, ty, strokeWidth, c, false)

	// Original orthogonal path code (commented out for debugging):
	// var midX, midY float32
	// switch srcPort {
	// case model.PortTop, model.PortBottom:
	// 	midX = sx
	// 	midY = (sy + ty) / 2
	// 	vector.StrokeLine(dst, sx, sy, midX, midY, strokeWidth, c, false)
	// 	vector.StrokeLine(dst, midX, midY, tx, midY, strokeWidth, c, false)
	// 	vector.StrokeLine(dst, tx, midY, tx, ty, strokeWidth, c, false)
	// case model.PortLeft, model.PortRight:
	// 	midX = (sx + tx) / 2
	// 	midY = sy
	// 	vector.StrokeLine(dst, sx, sy, midX, midY, strokeWidth, c, false)
	// 	vector.StrokeLine(dst, midX, midY, midX, ty, strokeWidth, c, false)
	// 	vector.StrokeLine(dst, midX, ty, tx, ty, strokeWidth, c, false)
	// }
}

// drawArrowHead draws a triangle arrow at the target point.
func (r *Renderer) drawArrowHead(dst *ebiten.Image, target image.Point, port model.Port, c color.RGBA) {
	size := float32(8 * r.canvas.Scale())
	size = float32(theme.Clamp(float64(size), 6, 14))

	tx, ty := float32(target.X), float32(target.Y)

	// Arrow direction based on entry port
	var angle float64
	switch port {
	case model.PortTop:
		angle = math.Pi / 2 // pointing down
	case model.PortBottom:
		angle = -math.Pi / 2 // pointing up
	case model.PortLeft:
		angle = 0 // pointing right
	case model.PortRight:
		angle = math.Pi // pointing left
	}

	// Calculate arrow triangle points
	sin, cos := math.Sin(angle), math.Cos(angle)
	halfAngle := math.Pi / 6 // 30 degrees

	// Tip is at target, calculate base points
	baseLen := float64(size)
	p1x := tx - float32(baseLen*cos)
	p1y := ty - float32(baseLen*sin)

	// Side points
	sideLen := baseLen * math.Tan(halfAngle)
	p2x := p1x + float32(sideLen*sin)
	p2y := p1y - float32(sideLen*cos)
	p3x := p1x - float32(sideLen*sin)
	p3y := p1y + float32(sideLen*cos)

	// Draw filled triangle
	var path vector.Path
	path.MoveTo(tx, ty)
	path.LineTo(p2x, p2y)
	path.LineTo(p3x, p3y)
	path.Close()

	vs, is := path.AppendVerticesAndIndicesForFilling(nil, nil)
	for i := range vs {
		vs[i].ColorR = float32(c.R) / 255
		vs[i].ColorG = float32(c.G) / 255
		vs[i].ColorB = float32(c.B) / 255
		vs[i].ColorA = float32(c.A) / 255
	}

	op := &ebiten.DrawTrianglesOptions{}
	op.AntiAlias = true
	dst.DrawTriangles(vs, is, whitePixel(), op)
}

// drawEdgeLabel draws the event/guard/action label on an edge.
func (r *Renderer) drawEdgeLabel(dst *ebiten.Image, label string, src, tgt image.Point) {
	if r.fontFace == nil {
		return
	}

	// Position at midpoint
	midX := float64(src.X+tgt.X) / 2
	midY := float64(src.Y+tgt.Y) / 2

	fontSize := theme.FontSizeMono * r.canvas.Scale()
	fontSize = theme.Clamp(fontSize, 8, 16)

	face := &text.GoTextFace{
		Source: r.fontFace.Source,
		Size:   fontSize,
	}

	textW, textH := text.Measure(label, face, 0)

	// Draw background pill
	padding := float32(4)
	bgX := float32(midX) - float32(textW)/2 - padding
	bgY := float32(midY) - float32(textH)/2 - padding
	bgW := float32(textW) + padding*2
	bgH := float32(textH) + padding*2

	vector.DrawFilledRect(dst, bgX, bgY, bgW, bgH, theme.CanvasBg, false)

	// Draw text
	op := &text.DrawOptions{}
	op.GeoM.Translate(midX-textW/2, midY-textH/2)
	op.ColorScale.ScaleWithColor(theme.TextSecondary)

	text.Draw(dst, label, face, op)
}

// whitePixel returns a 1x1 white image for drawing filled shapes.
func whitePixel() *ebiten.Image {
	img := ebiten.NewImage(1, 1)
	img.Fill(color.White)
	return img
}
