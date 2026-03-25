// Package app provides the main application structure for the statechart viewer.
package app

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"os"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/examples/resources/fonts"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"google.golang.org/protobuf/encoding/protojson"

	sc "github.com/tmc/sc"
	"github.com/tmc/sc/cmd/states-viewer/internal/canvas"
	"github.com/tmc/sc/cmd/states-viewer/internal/layout"
	"github.com/tmc/sc/cmd/states-viewer/internal/model"
	"github.com/tmc/sc/cmd/states-viewer/internal/theme"
	"github.com/tmc/sc/cmd/states-viewer/internal/ui"
)

// App represents the main application.
type App struct {
	width      int
	height     int
	title      string
	fontSource *text.GoTextFaceSource
	fontFace   *text.GoTextFace

	// Core components
	canvas      *canvas.Canvas
	renderer    *canvas.Renderer
	interaction *canvas.Interaction
	panel       *ui.Panel

	// Document state
	doc        *model.Document
	simulation *model.SimulationState

	// Animation
	lastTime float64
	time     float64

	// UI state
	uiMode       theme.UIMode
	showPanel    bool
	needsRedraw  bool
}

// New creates a new application instance.
func New(width, height int, title string) *App {
	app := &App{
		width:       width,
		height:      height,
		title:       title,
		showPanel:   true,
		needsRedraw: true,
	}

	// Initialize font
	if err := app.initFont(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to load font: %v\n", err)
	}

	// Initialize canvas
	app.canvas = canvas.NewCanvas(width, height)

	// Initialize renderer
	app.renderer = canvas.NewRenderer(app.canvas, app.fontFace)

	// Load sample document
	app.loadSampleDocument()

	// Initialize interaction
	app.interaction = canvas.NewInteraction(app.canvas, app.doc)

	// Initialize panel
	app.panel = ui.NewPanel(app.fontFace)
	app.panel.SetSimulation(app.simulation)
	app.panel.SetBounds(width, height)
	app.setupPanelCallbacks()

	// Zoom to fit initially
	app.zoomToFit()

	return app
}

// initFont initializes the font face.
func (a *App) initFont() error {
	source, err := text.NewGoTextFaceSource(bytes.NewReader(fonts.MPlus1pRegular_ttf))
	if err != nil {
		return fmt.Errorf("parse font: %w", err)
	}

	a.fontSource = source
	a.fontFace = &text.GoTextFace{
		Source: source,
		Size:   theme.FontSizeBody,
	}

	return nil
}

// loadSampleDocument loads the sample traffic light document.
func (a *App) loadSampleDocument() {
	// Use the media player sample for a more interesting demo
	a.doc = model.SampleMediaPlayer()

	// Apply layout
	layouter := layout.NewHierarchicalLayout()
	layouter.Apply(a.doc)

	// Initialize simulation
	a.simulation = model.NewSimulationState(a.doc)
}

// LoadFile loads a statechart from a JSON file.
func (a *App) LoadFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	var chart sc.Statechart
	if err := protojson.Unmarshal(data, &chart); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}

	doc, err := model.FromStatechart(&chart)
	if err != nil {
		return fmt.Errorf("convert: %w", err)
	}

	// Apply layout
	layouter := layout.NewHierarchicalLayout()
	layouter.Apply(doc)

	// Set as current document
	a.doc = doc
	a.simulation = model.NewSimulationState(doc)
	a.interaction.SetDocument(doc)
	a.panel.SetSimulation(a.simulation)

	// Zoom to fit
	a.zoomToFit()

	a.needsRedraw = true
	return nil
}

// setupPanelCallbacks sets up callbacks for panel buttons.
func (a *App) setupPanelCallbacks() {
	a.panel.OnSendEvent = func(event string) {
		if a.simulation != nil {
			a.simulation.SendEvent(event)
			a.needsRedraw = true
		}
	}

	a.panel.OnReset = func() {
		if a.simulation != nil {
			a.simulation.Reset()
			a.needsRedraw = true
		}
	}

	a.panel.OnStepBack = func() {
		if a.simulation != nil {
			a.simulation.StepBack()
			a.needsRedraw = true
		}
	}

	a.panel.OnStepFwd = func() {
		if a.simulation != nil {
			a.simulation.StepForward()
			a.needsRedraw = true
		}
	}
}

// zoomToFit zooms the canvas to fit all nodes.
func (a *App) zoomToFit() {
	if a.doc == nil || len(a.doc.Nodes) == 0 {
		return
	}

	// Calculate bounding box
	var bounds image.Rectangle
	first := true
	for _, node := range a.doc.Nodes {
		nodeBounds := node.Bounds()
		if first {
			bounds = nodeBounds
			first = false
		} else {
			bounds = bounds.Union(nodeBounds)
		}
	}

	a.canvas.ZoomToFit(bounds, 80)
}

// Update implements ebiten.Game.
func (a *App) Update() error {
	// Calculate delta time
	const targetFPS = 60.0
	dt := 1.0 / targetFPS
	a.time += dt

	// Update canvas animations
	a.canvas.Update(dt)

	// Update renderer
	a.renderer.Update(dt)

	// Update interactions
	if a.interaction.Update(dt) {
		a.needsRedraw = true
	}

	// Update panel
	if a.panel.Update(dt) {
		a.needsRedraw = true
	}

	// Update node animations
	a.updateNodeAnimations(dt)

	// Handle keyboard shortcuts
	if ebiten.IsKeyPressed(ebiten.KeyP) {
		a.panel.Toggle()
		a.needsRedraw = true
	}

	return nil
}

// updateNodeAnimations updates animation states for all nodes.
func (a *App) updateNodeAnimations(dt float64) {
	if a.doc == nil {
		return
	}

	for _, node := range a.doc.Nodes {
		// Update active animation
		if node.Active {
			node.ActiveT += dt * 2
			if node.ActiveT > 1 {
				node.ActiveT = 1
			}
			node.BreathPhase = a.time
		} else {
			node.ActiveT -= dt * 2
			if node.ActiveT < 0 {
				node.ActiveT = 0
			}
		}
	}
}

// Draw implements ebiten.Game.
func (a *App) Draw(screen *ebiten.Image) {
	// Clear background
	screen.Fill(theme.CanvasBg)

	// Draw grid
	a.renderer.DrawGrid(screen)

	// DEBUG: Draw a bright red test line to verify vector drawing works
	vector.StrokeLine(screen, 50, 50, 400, 200, 5, color.RGBA{255, 0, 0, 255}, false)

	// Draw edges
	// DEBUG: Print status
	if a.time < 0.05 {
		if a.doc == nil {
			fmt.Fprintf(os.Stderr, "DEBUG: doc is nil!\n")
		} else {
			fmt.Fprintf(os.Stderr, "DEBUG: doc has %d edges, %d nodes\n", len(a.doc.Edges), len(a.doc.Nodes))
		}
	}
	if a.doc != nil {
		for _, edge := range a.doc.Edges {
			a.renderer.DrawEdge(screen, edge, a.doc)
		}
	}

	// Draw nodes (in order for proper z-layering)
	if a.doc != nil {
		for _, id := range a.doc.NodeOrder {
			if node := a.doc.Nodes[id]; node != nil {
				a.renderer.DrawNode(screen, node)
			}
		}
	}

	// Draw panel
	a.panel.Draw(screen)

	// Draw help text
	a.drawHelpText(screen)
}

// drawHelpText draws keyboard shortcuts at bottom left.
func (a *App) drawHelpText(screen *ebiten.Image) {
	if a.fontFace == nil {
		return
	}

	help := "F: Fit to view | P: Toggle panel | Scroll: Zoom | Drag: Pan"

	face := &text.GoTextFace{
		Source: a.fontSource,
		Size:   theme.FontSizeCaption,
	}

	op := &text.DrawOptions{}
	op.GeoM.Translate(16, float64(a.height)-24)
	op.ColorScale.ScaleWithColor(theme.TextMuted)
	text.Draw(screen, help, face, op)
}

// Layout implements ebiten.Game.
func (a *App) Layout(outsideWidth, outsideHeight int) (int, int) {
	// Handle resize
	if outsideWidth != a.width || outsideHeight != a.height {
		a.width = outsideWidth
		a.height = outsideHeight
		a.canvas.SetSize(outsideWidth, outsideHeight)
		a.panel.SetBounds(outsideWidth, outsideHeight)
		a.uiMode = theme.GetUIMode(outsideWidth)
		a.needsRedraw = true
	}

	return outsideWidth, outsideHeight
}

// Run starts the application.
func (a *App) Run() error {
	ebiten.SetWindowSize(a.width, a.height)
	ebiten.SetWindowTitle(a.title)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	return ebiten.RunGame(a)
}
