// Package ui provides user interface components.
package ui

import (
	"image"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"

	"github.com/tmc/sc/cmd/states-viewer/internal/model"
	"github.com/tmc/sc/cmd/states-viewer/internal/theme"
)

// Panel represents the floating control panel.
type Panel struct {
	bounds       image.Rectangle
	visible      bool
	slideT       float64 // Animation progress (0 = hidden, 1 = visible)
	fontFace     *text.GoTextFace
	eventInput   *TextInput
	sendButton   *Button
	resetButton  *Button
	backButton   *Button
	fwdButton    *Button
	simulation   *model.SimulationState

	// Callbacks
	OnSendEvent func(event string)
	OnReset     func()
	OnStepBack  func()
	OnStepFwd   func()
}

// NewPanel creates a new control panel.
func NewPanel(fontFace *text.GoTextFace) *Panel {
	p := &Panel{
		fontFace: fontFace,
		visible:  true,
		slideT:   1.0,
	}

	// Create event input
	p.eventInput = &TextInput{
		Placeholder: "Event name...",
		FontFace:    fontFace,
	}

	// Create buttons
	p.sendButton = &Button{
		Label:    "Send Event",
		FontFace: fontFace,
	}
	p.sendButton.OnClick = func() {
		if p.OnSendEvent != nil && p.eventInput.Text != "" {
			p.OnSendEvent(p.eventInput.Text)
		}
	}

	p.resetButton = &Button{
		Label:    "Reset",
		FontFace: fontFace,
		Style:    ButtonStyleSecondary,
	}
	p.resetButton.OnClick = func() {
		if p.OnReset != nil {
			p.OnReset()
		}
	}

	p.backButton = &Button{
		Label:    "←",
		FontFace: fontFace,
		Style:    ButtonStyleSecondary,
	}
	p.backButton.OnClick = func() {
		if p.OnStepBack != nil {
			p.OnStepBack()
		}
	}

	p.fwdButton = &Button{
		Label:    "→",
		FontFace: fontFace,
		Style:    ButtonStyleSecondary,
	}
	p.fwdButton.OnClick = func() {
		if p.OnStepFwd != nil {
			p.OnStepFwd()
		}
	}

	return p
}

// SetSimulation sets the simulation state to display.
func (p *Panel) SetSimulation(sim *model.SimulationState) {
	p.simulation = sim
}

// SetBounds sets the panel bounds (typically right side of screen).
func (p *Panel) SetBounds(screenW, screenH int) {
	panelW := theme.PanelWidth
	p.bounds = image.Rect(
		screenW-panelW-16,
		16,
		screenW-16,
		screenH-16,
	)

	// Layout child components
	padding := theme.PanelPadding
	y := p.bounds.Min.Y + padding + 40 // Leave room for title
	w := p.bounds.Dx() - padding*2

	// Event input
	p.eventInput.Bounds = image.Rect(
		p.bounds.Min.X+padding,
		y,
		p.bounds.Max.X-padding,
		y+theme.ButtonHeight,
	)
	y += theme.ButtonHeight + 8

	// Send button
	p.sendButton.Bounds = image.Rect(
		p.bounds.Min.X+padding,
		y,
		p.bounds.Max.X-padding,
		y+theme.ButtonHeight,
	)
	y += theme.ButtonHeight + 16

	// History navigation row
	btnW := (w - 8) / 2
	p.backButton.Bounds = image.Rect(
		p.bounds.Min.X+padding,
		y,
		p.bounds.Min.X+padding+btnW,
		y+theme.ButtonHeight,
	)
	p.fwdButton.Bounds = image.Rect(
		p.bounds.Max.X-padding-btnW,
		y,
		p.bounds.Max.X-padding,
		y+theme.ButtonHeight,
	)
	y += theme.ButtonHeight + 8

	// Reset button
	p.resetButton.Bounds = image.Rect(
		p.bounds.Min.X+padding,
		y,
		p.bounds.Max.X-padding,
		y+theme.ButtonHeight,
	)
}

// Show shows the panel with animation.
func (p *Panel) Show() {
	p.visible = true
}

// Hide hides the panel with animation.
func (p *Panel) Hide() {
	p.visible = false
}

// Toggle toggles panel visibility.
func (p *Panel) Toggle() {
	p.visible = !p.visible
}

// IsVisible returns true if the panel is visible.
func (p *Panel) IsVisible() bool {
	return p.visible || p.slideT > 0
}

// Update updates the panel state.
func (p *Panel) Update(dt float64) bool {
	needsRedraw := false

	// Animate slide
	targetT := 0.0
	if p.visible {
		targetT = 1.0
	}

	if p.slideT != targetT {
		speed := 1.0 / theme.PanelDuration
		if p.slideT < targetT {
			p.slideT += dt * speed
			if p.slideT > targetT {
				p.slideT = targetT
			}
		} else {
			p.slideT -= dt * speed
			if p.slideT < targetT {
				p.slideT = targetT
			}
		}
		needsRedraw = true
	}

	if !p.IsVisible() {
		return needsRedraw
	}

	// Update child components
	if p.eventInput.Update() {
		needsRedraw = true
	}
	if p.sendButton.Update() {
		needsRedraw = true
	}
	if p.resetButton.Update() {
		needsRedraw = true
	}
	if p.backButton.Update() {
		needsRedraw = true
	}
	if p.fwdButton.Update() {
		needsRedraw = true
	}

	return needsRedraw
}

// Draw draws the panel.
func (p *Panel) Draw(dst *ebiten.Image) {
	if !p.IsVisible() {
		return
	}

	// Apply slide animation
	offsetX := float32((1 - theme.EaseOut(p.slideT)) * float64(theme.PanelWidth+32))

	// Draw panel background with glass effect
	x := float32(p.bounds.Min.X) + offsetX
	y := float32(p.bounds.Min.Y)
	w := float32(p.bounds.Dx())
	h := float32(p.bounds.Dy())

	// Background
	vector.DrawFilledRect(dst, x, y, w, h, theme.PanelBg, false)

	// Border
	vector.StrokeRect(dst, x, y, w, h, 1, theme.BorderLight, false)

	// Title
	if p.fontFace != nil {
		face := &text.GoTextFace{
			Source: p.fontFace.Source,
			Size:   theme.FontSizeTitle,
		}
		op := &text.DrawOptions{}
		op.GeoM.Translate(float64(x)+theme.PanelPadding, float64(y)+theme.PanelPadding)
		op.ColorScale.ScaleWithColor(theme.TextPrimary)
		text.Draw(dst, "Simulation", face, op)
	}

	// Draw child components with offset
	p.drawWithOffset(dst, offsetX)
}

// drawWithOffset draws child components with horizontal offset.
func (p *Panel) drawWithOffset(dst *ebiten.Image, offsetX float32) {
	// Save original bounds
	origEventBounds := p.eventInput.Bounds
	origSendBounds := p.sendButton.Bounds
	origResetBounds := p.resetButton.Bounds
	origBackBounds := p.backButton.Bounds
	origFwdBounds := p.fwdButton.Bounds

	// Apply offset
	off := int(offsetX)
	p.eventInput.Bounds = p.eventInput.Bounds.Add(image.Point{X: off})
	p.sendButton.Bounds = p.sendButton.Bounds.Add(image.Point{X: off})
	p.resetButton.Bounds = p.resetButton.Bounds.Add(image.Point{X: off})
	p.backButton.Bounds = p.backButton.Bounds.Add(image.Point{X: off})
	p.fwdButton.Bounds = p.fwdButton.Bounds.Add(image.Point{X: off})

	// Draw
	p.eventInput.Draw(dst)
	p.sendButton.Draw(dst)
	p.resetButton.Draw(dst)
	p.backButton.Draw(dst)
	p.fwdButton.Draw(dst)

	// Draw step counter
	if p.simulation != nil && p.fontFace != nil {
		current, total := p.simulation.StepCount()
		stepText := "Step 0"
		if current > 0 {
			stepText = formatStep(current, total)
		}

		face := &text.GoTextFace{
			Source: p.fontFace.Source,
			Size:   theme.FontSizeCaption,
		}

		// Position between back/fwd buttons
		textW, _ := text.Measure(stepText, face, 0)
		centerX := float64(p.backButton.Bounds.Max.X+p.fwdButton.Bounds.Min.X) / 2 - textW/2
		centerY := float64(p.backButton.Bounds.Min.Y + p.backButton.Bounds.Dy()/2 - 6)

		op := &text.DrawOptions{}
		op.GeoM.Translate(centerX, centerY)
		op.ColorScale.ScaleWithColor(theme.TextSecondary)
		text.Draw(dst, stepText, face, op)
	}

	// Draw active states section (hierarchical tree view)
	if p.simulation != nil && p.fontFace != nil {
		y := float64(p.resetButton.Bounds.Max.Y) + 24

		// Section title
		face := &text.GoTextFace{
			Source: p.fontFace.Source,
			Size:   theme.FontSizeCaption,
		}
		op := &text.DrawOptions{}
		op.GeoM.Translate(float64(p.bounds.Min.X+off)+theme.PanelPadding, y)
		op.ColorScale.ScaleWithColor(theme.TextMuted)
		text.Draw(dst, "Configuration", face, op)
		y += 20

		// List active states with hierarchy
		bodyFace := &text.GoTextFace{
			Source: p.fontFace.Source,
			Size:   theme.FontSizeBody,
		}

		for _, state := range p.simulation.ActiveStatesHierarchy() {
			// Indentation based on level
			indent := float64(state.Level) * 12
			baseX := float64(p.bounds.Min.X+off) + theme.PanelPadding + indent

			// Tree connector lines for children
			if state.Level > 0 {
				// Draw "└─" style connector
				connX := float32(baseX) - 6
				connY := float32(y) + 6
				vector.StrokeLine(dst, connX-4, connY-10, connX-4, connY, 1, theme.BorderMedium, false)
				vector.StrokeLine(dst, connX-4, connY, connX+2, connY, 1, theme.BorderMedium, false)
			}

			// Green dot
			vector.DrawFilledCircle(
				dst,
				float32(baseX)+4,
				float32(y)+6,
				3,
				theme.Success,
				false,
			)

			// State name
			op := &text.DrawOptions{}
			op.GeoM.Translate(baseX+12, y)
			op.ColorScale.ScaleWithColor(theme.TextPrimary)
			text.Draw(dst, state.ID, bodyFace, op)
			y += 22
		}
	}

	// Restore bounds
	p.eventInput.Bounds = origEventBounds
	p.sendButton.Bounds = origSendBounds
	p.resetButton.Bounds = origResetBounds
	p.backButton.Bounds = origBackBounds
	p.fwdButton.Bounds = origFwdBounds
}

// Contains returns true if the point is inside the panel.
func (p *Panel) Contains(x, y int) bool {
	if !p.IsVisible() {
		return false
	}
	return image.Point{X: x, Y: y}.In(p.bounds)
}

func formatStep(current, total int) string {
	return string(rune('0'+current%10)) + "/" + string(rune('0'+total%10))
}

// ButtonStyle represents button visual style.
type ButtonStyle int

const (
	ButtonStylePrimary ButtonStyle = iota
	ButtonStyleSecondary
)

// Button represents a clickable button.
type Button struct {
	Label    string
	Bounds   image.Rectangle
	FontFace *text.GoTextFace
	Style    ButtonStyle
	OnClick  func()

	hovered bool
	pressed bool
	hoverT  float64
	pressT  float64
}

// Update updates button state.
func (b *Button) Update() bool {
	mx, my := ebiten.CursorPosition()
	wasHovered := b.hovered
	b.hovered = image.Point{X: mx, Y: my}.In(b.Bounds)

	// Update hover animation
	if b.hovered {
		b.hoverT += 0.15
		if b.hoverT > 1 {
			b.hoverT = 1
		}
	} else {
		b.hoverT -= 0.15
		if b.hoverT < 0 {
			b.hoverT = 0
		}
	}

	// Check for click
	if b.hovered && ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		if !b.pressed {
			b.pressed = true
			b.pressT = 1
		}
	} else {
		if b.pressed && b.hovered {
			// Released while hovered = click
			if b.OnClick != nil {
				b.OnClick()
			}
		}
		b.pressed = false
		b.pressT *= 0.8
	}

	return b.hovered != wasHovered || b.pressT > 0.01
}

// Draw draws the button.
func (b *Button) Draw(dst *ebiten.Image) {
	x := float32(b.Bounds.Min.X)
	y := float32(b.Bounds.Min.Y)
	w := float32(b.Bounds.Dx())
	h := float32(b.Bounds.Dy())

	// Apply press scale
	if b.pressT > 0 {
		scale := float32(1 - 0.05*b.pressT)
		centerX := x + w/2
		centerY := y + h/2
		w *= scale
		h *= scale
		x = centerX - w/2
		y = centerY - h/2
	}

	// Colors based on style
	var bgColor, textColor color.RGBA
	switch b.Style {
	case ButtonStylePrimary:
		bgColor = theme.Accent
		if b.hovered {
			bgColor = theme.AccentLight
		}
		textColor = theme.TextLight
	case ButtonStyleSecondary:
		bgColor = theme.BorderLight
		if b.hovered {
			bgColor = theme.BorderMedium
		}
		textColor = theme.TextPrimary
	}

	// Background
	vector.DrawFilledRect(dst, x, y, w, h, bgColor, false)

	// Border for secondary
	if b.Style == ButtonStyleSecondary {
		vector.StrokeRect(dst, x, y, w, h, 1, theme.BorderMedium, false)
	}

	// Label
	if b.FontFace != nil {
		face := &text.GoTextFace{
			Source: b.FontFace.Source,
			Size:   theme.FontSizeBody,
		}
		textW, textH := text.Measure(b.Label, face, 0)

		op := &text.DrawOptions{}
		op.GeoM.Translate(
			float64(x)+float64(w)/2-textW/2,
			float64(y)+float64(h)/2-textH/2,
		)
		op.ColorScale.ScaleWithColor(textColor)
		text.Draw(dst, b.Label, face, op)
	}
}

// TextInput represents a text input field.
type TextInput struct {
	Text        string
	Placeholder string
	Bounds      image.Rectangle
	FontFace    *text.GoTextFace
	Focused     bool

	hovered    bool
	cursorT    float64
	cursorBlink bool
}

// Update updates input state.
func (t *TextInput) Update() bool {
	mx, my := ebiten.CursorPosition()
	wasHovered := t.hovered
	t.hovered = image.Point{X: mx, Y: my}.In(t.Bounds)

	// Focus on click
	if t.hovered && ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		t.Focused = true
	} else if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) && !t.hovered {
		t.Focused = false
	}

	// Handle text input when focused
	if t.Focused {
		// Get input characters
		chars := ebiten.AppendInputChars(nil)
		t.Text += string(chars)

		// Handle backspace
		if ebiten.IsKeyPressed(ebiten.KeyBackspace) {
			if len(t.Text) > 0 {
				t.Text = t.Text[:len(t.Text)-1]
			}
		}

		// Cursor blink
		t.cursorT += 0.05
		if t.cursorT >= 1 {
			t.cursorT = 0
			t.cursorBlink = !t.cursorBlink
		}
	}

	return t.hovered != wasHovered || t.Focused
}

// Draw draws the text input.
func (t *TextInput) Draw(dst *ebiten.Image) {
	x := float32(t.Bounds.Min.X)
	y := float32(t.Bounds.Min.Y)
	w := float32(t.Bounds.Dx())
	h := float32(t.Bounds.Dy())

	// Background
	vector.DrawFilledRect(dst, x, y, w, h, theme.NodeBg, false)

	// Border
	borderColor := theme.BorderMedium
	if t.Focused {
		borderColor = theme.Accent
	} else if t.hovered {
		borderColor = theme.AccentLight
	}
	vector.StrokeRect(dst, x, y, w, h, 1, borderColor, false)

	// Text or placeholder
	if t.FontFace != nil {
		face := &text.GoTextFace{
			Source: t.FontFace.Source,
			Size:   theme.FontSizeBody,
		}

		displayText := t.Text
		textColor := theme.TextPrimary
		if displayText == "" {
			displayText = t.Placeholder
			textColor = theme.TextMuted
		}

		_, textH := text.Measure(displayText, face, 0)

		op := &text.DrawOptions{}
		op.GeoM.Translate(
			float64(x)+8,
			float64(y)+float64(h)/2-textH/2,
		)
		op.ColorScale.ScaleWithColor(textColor)
		text.Draw(dst, displayText, face, op)

		// Cursor
		if t.Focused && t.cursorBlink {
			textW, _ := text.Measure(t.Text, face, 0)
			cursorX := float32(x) + 8 + float32(textW)
			cursorY := float32(y) + 8
			cursorH := float32(h) - 16
			vector.StrokeLine(dst, cursorX, cursorY, cursorX, cursorY+cursorH, 2, theme.Accent, false)
		}
	}
}
