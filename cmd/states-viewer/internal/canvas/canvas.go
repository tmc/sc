// Package canvas provides coordinate transforms and viewport management.
package canvas

import (
	"image"
	"math"

	"github.com/tmc/sc/cmd/states-viewer/internal/theme"
)

// Canvas manages viewport transforms and coordinate conversion.
type Canvas struct {
	// Viewport size in screen pixels
	width  int
	height int

	// Current transform state
	offset      Point  // Pan offset in world coordinates
	scale       float64 // Zoom level
	targetScale float64 // Target zoom for smooth animation
	targetOffset Point  // Target offset for smooth animation

	// Animation state
	animating     bool
	animProgress  float64
	startOffset   Point
	startScale    float64
}

// Point represents a floating-point 2D coordinate.
type Point struct {
	X, Y float64
}

// NewCanvas creates a new canvas with the given viewport size.
func NewCanvas(width, height int) *Canvas {
	return &Canvas{
		width:        width,
		height:       height,
		scale:        1.0,
		targetScale:  1.0,
		offset:       Point{},
		targetOffset: Point{},
	}
}

// SetSize updates the viewport size.
func (c *Canvas) SetSize(width, height int) {
	c.width = width
	c.height = height
}

// Size returns the viewport size.
func (c *Canvas) Size() (width, height int) {
	return c.width, c.height
}

// Scale returns the current zoom level.
func (c *Canvas) Scale() float64 {
	return c.scale
}

// Offset returns the current pan offset.
func (c *Canvas) Offset() Point {
	return c.offset
}

// SetScale sets the zoom level immediately.
func (c *Canvas) SetScale(scale float64) {
	c.scale = theme.Clamp(scale, theme.ZoomMin, theme.ZoomMax)
	c.targetScale = c.scale
}

// SetOffset sets the pan offset immediately.
func (c *Canvas) SetOffset(offset Point) {
	c.offset = offset
	c.targetOffset = offset
}

// Pan adjusts the offset by the given delta in screen pixels.
func (c *Canvas) Pan(dx, dy float64) {
	// Convert screen delta to world delta
	c.offset.X -= dx / c.scale
	c.offset.Y -= dy / c.scale
	c.targetOffset = c.offset
}

// ZoomAt zooms toward/away from a point (in screen coordinates).
func (c *Canvas) ZoomAt(screenX, screenY float64, factor float64) {
	// Get world position under cursor before zoom
	worldBefore := c.ScreenToWorld(screenX, screenY)

	// Apply zoom
	newScale := theme.Clamp(c.scale*factor, theme.ZoomMin, theme.ZoomMax)
	c.scale = newScale
	c.targetScale = newScale

	// Get world position under cursor after zoom
	worldAfter := c.ScreenToWorld(screenX, screenY)

	// Adjust offset to keep cursor position stable
	c.offset.X -= worldAfter.X - worldBefore.X
	c.offset.Y -= worldAfter.Y - worldBefore.Y
	c.targetOffset = c.offset
}

// ZoomToFit adjusts scale and offset to fit all nodes in view.
func (c *Canvas) ZoomToFit(bounds image.Rectangle, padding int) {
	if bounds.Empty() {
		return
	}

	// Calculate required scale
	boundsW := float64(bounds.Dx())
	boundsH := float64(bounds.Dy())
	viewW := float64(c.width - 2*padding)
	viewH := float64(c.height - 2*padding)

	scaleX := viewW / boundsW
	scaleY := viewH / boundsH
	newScale := math.Min(scaleX, scaleY)
	newScale = theme.Clamp(newScale, theme.ZoomMin, theme.ZoomMax)

	// Center the bounds
	centerX := float64(bounds.Min.X+bounds.Max.X) / 2
	centerY := float64(bounds.Min.Y+bounds.Max.Y) / 2

	// Animate to new position
	c.AnimateTo(Point{X: centerX, Y: centerY}, newScale)
}

// AnimateTo starts a smooth animation to the target offset and scale.
func (c *Canvas) AnimateTo(offset Point, scale float64) {
	c.animating = true
	c.animProgress = 0
	c.startOffset = c.offset
	c.startScale = c.scale
	c.targetOffset = offset
	c.targetScale = theme.Clamp(scale, theme.ZoomMin, theme.ZoomMax)
}

// Update advances animations. Call once per frame with delta time.
func (c *Canvas) Update(dt float64) {
	if !c.animating {
		return
	}

	// Advance animation
	c.animProgress += dt / theme.ZoomDuration
	if c.animProgress >= 1.0 {
		c.animProgress = 1.0
		c.animating = false
	}

	// Apply easing
	t := theme.EaseOut(c.animProgress)

	// Interpolate
	c.offset.X = theme.Lerp(c.startOffset.X, c.targetOffset.X, t)
	c.offset.Y = theme.Lerp(c.startOffset.Y, c.targetOffset.Y, t)
	c.scale = theme.Lerp(c.startScale, c.targetScale, t)
}

// WorldToScreen converts world coordinates to screen pixels.
func (c *Canvas) WorldToScreen(worldX, worldY float64) Point {
	// Screen center
	cx := float64(c.width) / 2
	cy := float64(c.height) / 2

	// Apply transform: translate to center, scale, then offset
	screenX := cx + (worldX-c.offset.X)*c.scale
	screenY := cy + (worldY-c.offset.Y)*c.scale

	return Point{X: screenX, Y: screenY}
}

// ScreenToWorld converts screen pixels to world coordinates.
func (c *Canvas) ScreenToWorld(screenX, screenY float64) Point {
	// Screen center
	cx := float64(c.width) / 2
	cy := float64(c.height) / 2

	// Inverse transform
	worldX := (screenX-cx)/c.scale + c.offset.X
	worldY := (screenY-cy)/c.scale + c.offset.Y

	return Point{X: worldX, Y: worldY}
}

// WorldToScreenPoint converts an image.Point in world coordinates.
func (c *Canvas) WorldToScreenPoint(p image.Point) image.Point {
	sp := c.WorldToScreen(float64(p.X), float64(p.Y))
	return image.Point{X: int(sp.X), Y: int(sp.Y)}
}

// ScreenToWorldPoint converts screen coordinates to world image.Point.
func (c *Canvas) ScreenToWorldPoint(x, y int) image.Point {
	wp := c.ScreenToWorld(float64(x), float64(y))
	return image.Point{X: int(wp.X), Y: int(wp.Y)}
}

// WorldBoundsToScreen converts a world rectangle to screen coordinates.
func (c *Canvas) WorldBoundsToScreen(bounds image.Rectangle) image.Rectangle {
	min := c.WorldToScreenPoint(bounds.Min)
	max := c.WorldToScreenPoint(bounds.Max)
	return image.Rect(min.X, min.Y, max.X, max.Y)
}

// VisibleWorldBounds returns the world coordinates visible in the viewport.
func (c *Canvas) VisibleWorldBounds() image.Rectangle {
	topLeft := c.ScreenToWorld(0, 0)
	bottomRight := c.ScreenToWorld(float64(c.width), float64(c.height))
	return image.Rect(
		int(topLeft.X),
		int(topLeft.Y),
		int(bottomRight.X),
		int(bottomRight.Y),
	)
}

// ScaledSize returns a size scaled by the current zoom level.
func (c *Canvas) ScaledSize(w, h int) (int, int) {
	return int(float64(w) * c.scale), int(float64(h) * c.scale)
}
