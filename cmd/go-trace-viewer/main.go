package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

//go:embed fonts/Inter.ttf
var interFontData []byte

var (
	interFaceSource *text.GoTextFaceSource
	interFace10     *text.GoTextFace
	interFace11     *text.GoTextFace
	interFace13     *text.GoTextFace
	interFace15     *text.GoTextFace
	interFace18     *text.GoTextFace
	interFace24     *text.GoTextFace
)

func init() {
	var err error
	interFaceSource, err = text.NewGoTextFaceSource(bytes.NewReader(interFontData))
	if err != nil {
		log.Fatal(err)
	}

	interFace10 = &text.GoTextFace{Source: interFaceSource, Size: 10}
	interFace11 = &text.GoTextFace{Source: interFaceSource, Size: 11}
	interFace13 = &text.GoTextFace{Source: interFaceSource, Size: 13}
	interFace15 = &text.GoTextFace{Source: interFaceSource, Size: 15}
	interFace18 = &text.GoTextFace{Source: interFaceSource, Size: 18}
	interFace24 = &text.GoTextFace{Source: interFaceSource, Size: 24}
}

// ============================================================================
// DESIGN SYSTEM
// ============================================================================

var (
	colorBg0 = color.RGBA{0x09, 0x09, 0x0b, 0xff}
	colorBg1 = color.RGBA{0x0f, 0x0f, 0x13, 0xff}
	colorBg2 = color.RGBA{0x17, 0x17, 0x1d, 0xff}
	colorBg3 = color.RGBA{0x1f, 0x1f, 0x27, 0xff}

	colorBoard    = color.RGBA{0xdc, 0xb4, 0x68, 0xff}
	colorGridLine = color.RGBA{0x6b, 0x56, 0x3c, 0xff}
	colorHoshi    = color.RGBA{0x4a, 0x3c, 0x2a, 0xff}

	colorBlackStone  = color.RGBA{0x1a, 0x1a, 0x1c, 0xff}
	colorBlackShine  = color.RGBA{0x44, 0x44, 0x48, 0xff}
	colorWhiteStone  = color.RGBA{0xf8, 0xf8, 0xfa, 0xff}
	colorWhiteShadow = color.RGBA{0xc8, 0xc8, 0xcc, 0xff}

	colorAccent      = color.RGBA{0x63, 0x5b, 0xff, 0xff}
	colorAccentHover = color.RGBA{0x7a, 0x73, 0xff, 0xff}
	colorAccentMuted = color.RGBA{0x63, 0x5b, 0xff, 0x30}

	colorText100 = color.RGBA{0xff, 0xff, 0xff, 0xff}
	colorText80  = color.RGBA{0xcc, 0xcc, 0xd4, 0xff}
	colorText60  = color.RGBA{0x8a, 0x8a, 0x96, 0xff}
	colorText40  = color.RGBA{0x5a, 0x5a, 0x66, 0xff}

	colorSuccess = color.RGBA{0x22, 0xc5, 0x5e, 0xff}
	colorWarning = color.RGBA{0xf5, 0xa6, 0x23, 0xff}
	colorError   = color.RGBA{0xef, 0x44, 0x44, 0xff}

	// Statechart colors
	colorStateNormal   = color.RGBA{0x2d, 0x3a, 0x4f, 0xff}
	colorStateActive   = color.RGBA{0x3b, 0x82, 0xf6, 0xff}
	colorStateInitial  = color.RGBA{0x22, 0xc5, 0x5e, 0xff}
	colorStateFinal    = color.RGBA{0xef, 0x44, 0x44, 0xff}
	colorTransition    = color.RGBA{0x64, 0x74, 0x8b, 0xff}
	colorTransitionAct = color.RGBA{0xfb, 0xbf, 0x24, 0xff}
)

const (
	sp1  = 4
	sp2  = 8
	sp3  = 12
	sp4  = 16
	sp5  = 20
	sp6  = 24
	sp8  = 32
	sp10 = 40
)

const boardSize = 9

// ============================================================================
// STATECHART TYPES
// ============================================================================

type StateType int

const (
	StateBasic StateType = iota
	StateComposite
	StateParallel
	StateInitial
	StateFinal
)

type SCState struct {
	ID       string
	Label    string
	Type     StateType
	X, Y     float64 // Position for rendering
	W, H     float64 // Size
	Children []*SCState
	Parent   *SCState
	IsActive bool
}

type SCTransition struct {
	From     *SCState
	To       *SCState
	Event    string
	Guard    string
	IsActive bool
}

type SCMachine struct {
	Name        string
	States      []*SCState
	Transitions []*SCTransition
	Current     *SCState
}

// GoGameMachine creates a statechart representing Go game rules
func GoGameMachine(pos Position) *SCMachine {
	// Create states
	stateIdle := &SCState{ID: "idle", Label: "Idle", Type: StateBasic, X: 50, Y: 150, W: 80, H: 40}
	stateBlackTurn := &SCState{ID: "black_turn", Label: "Black", Type: StateBasic, X: 180, Y: 80, W: 80, H: 40}
	stateWhiteTurn := &SCState{ID: "white_turn", Label: "White", Type: StateBasic, X: 180, Y: 220, W: 80, H: 40}
	stateCapture := &SCState{ID: "capture", Label: "Capture", Type: StateBasic, X: 320, Y: 150, W: 80, H: 40}
	stateKoCheck := &SCState{ID: "ko_check", Label: "Ko Check", Type: StateBasic, X: 450, Y: 150, W: 90, H: 40}
	stateGameOver := &SCState{ID: "game_over", Label: "Game Over", Type: StateFinal, X: 580, Y: 150, W: 100, H: 40}

	// Determine active state based on position
	var current *SCState
	if pos.MoveNum == 0 {
		current = stateIdle
	} else if pos.MoveNum%2 == 1 {
		// After odd move, white's turn (or capture check)
		if pos.LastMove != nil && len(pos.LastMove.Captures) > 0 {
			current = stateCapture
		} else {
			current = stateWhiteTurn
		}
	} else {
		// After even move, black's turn
		if pos.LastMove != nil && len(pos.LastMove.Captures) > 0 {
			current = stateCapture
		} else {
			current = stateBlackTurn
		}
	}
	current.IsActive = true

	states := []*SCState{stateIdle, stateBlackTurn, stateWhiteTurn, stateCapture, stateKoCheck, stateGameOver}

	// Create transitions
	transitions := []*SCTransition{
		{From: stateIdle, To: stateBlackTurn, Event: "START"},
		{From: stateBlackTurn, To: stateCapture, Event: "PLACE", Guard: "captures > 0"},
		{From: stateBlackTurn, To: stateWhiteTurn, Event: "PLACE", Guard: "captures == 0"},
		{From: stateWhiteTurn, To: stateCapture, Event: "PLACE", Guard: "captures > 0"},
		{From: stateWhiteTurn, To: stateBlackTurn, Event: "PLACE", Guard: "captures == 0"},
		{From: stateCapture, To: stateKoCheck, Event: "REMOVE"},
		{From: stateKoCheck, To: stateBlackTurn, Event: "OK", Guard: "turn == white"},
		{From: stateKoCheck, To: stateWhiteTurn, Event: "OK", Guard: "turn == black"},
		{From: stateBlackTurn, To: stateGameOver, Event: "PASS", Guard: "prev_pass"},
		{From: stateWhiteTurn, To: stateGameOver, Event: "PASS", Guard: "prev_pass"},
	}

	// Mark active transition
	if pos.LastMove != nil && !pos.LastMove.Pass {
		for _, t := range transitions {
			if t.From.IsActive || t.To.IsActive {
				if t.Event == "PLACE" {
					t.IsActive = true
					break
				}
			}
		}
	}

	return &SCMachine{
		Name:        "Go Game",
		States:      states,
		Transitions: transitions,
		Current:     current,
	}
}

// ============================================================================
// RESPONSIVE LAYOUT
// ============================================================================

type LayoutMode int

const (
	LayoutDesktop LayoutMode = iota
	LayoutTablet
	LayoutMobile
)

type Layout struct {
	Mode         LayoutMode
	CellSize     float64
	BoardPadding float64
	GridCols     int
	GridRows     int
	BoardGap     float64
	HeaderHeight float64
	FooterHeight float64
}

func computeLayout(w, h int) Layout {
	aspect := float64(h) / float64(w)
	if w >= 1200 {
		return Layout{LayoutDesktop, 32, 24, 4, 4, sp4, 64, 88}
	} else if w >= 768 || (w >= 600 && aspect < 1.4) {
		return Layout{LayoutTablet, 36, 28, 3, 3, sp3, 56, 80}
	}
	return Layout{LayoutMobile, 44, 32, 2, 2, sp2, 52, 96}
}

func (l Layout) BoardSize() float64 {
	return l.CellSize*float64(boardSize-1) + l.BoardPadding*2
}

// ============================================================================
// GAME TYPES
// ============================================================================

type Stone int

const (
	Empty Stone = iota
	Black
	White
)

type Point struct{ X, Y int }

type Move struct {
	X, Y     int
	Color    Stone
	Pass     bool
	Captures []Point
}

type Position struct {
	Board    [boardSize][boardSize]Stone
	MoveNum  int
	LastMove *Move
}

type Trace struct {
	Name      string
	Moves     []Move
	Positions []Position
}

type JSONMove struct {
	MoveNum  int     `json:"move_num"`
	Player   string  `json:"player"`
	X        int     `json:"x"`
	Y        int     `json:"y"`
	IsPass   bool    `json:"is_pass"`
	Captures [][]int `json:"captures,omitempty"`
}

type JSONTrace struct {
	GameID string     `json:"game_id"`
	Moves  []JSONMove `json:"moves"`
}

// ============================================================================
// ANIMATION
// ============================================================================

type Anim struct {
	Value, Target, Velocity float64
}

func (a *Anim) Update(dt, stiffness, damping float64) {
	force := (a.Target - a.Value) * stiffness
	a.Velocity = a.Velocity*damping + force*dt
	a.Value += a.Velocity * dt
}

func (a *Anim) Set(v float64) {
	a.Value, a.Target, a.Velocity = v, v, 0
}

// ============================================================================
// GAME STATE
// ============================================================================

type Game struct {
	traces       []Trace
	currentTrace int
	startPos     int
	paused       bool
	tick         int

	layout  Layout
	screenW int
	screenH int

	hoveredBoard int
	focusedBoard int
	showHelp     bool

	pulsePhase  float64
	hoverAnims  [16]Anim
	markerPulse float64
	splitAnim   Anim // Animation for split view
}

func NewGame(traces []Trace) *Game {
	g := &Game{
		traces:       traces,
		paused:       true,
		hoveredBoard: -1,
		focusedBoard: -1,
	}
	for i := range g.hoverAnims {
		g.hoverAnims[i].Set(0)
	}
	g.splitAnim.Set(0)
	return g
}

func (g *Game) Update() error {
	g.tick++
	g.pulsePhase += 0.03
	g.markerPulse += 0.08

	dt := 1.0 / 60.0
	for i := range g.hoverAnims {
		g.hoverAnims[i].Target = 0
		if i == g.hoveredBoard {
			g.hoverAnims[i].Target = 1
		}
		g.hoverAnims[i].Update(dt, 200, 0.8)
	}

	// Split view animation
	if g.focusedBoard >= 0 {
		g.splitAnim.Target = 1
	} else {
		g.splitAnim.Target = 0
	}
	g.splitAnim.Update(dt, 150, 0.85)

	w, h := ebiten.WindowSize()
	if w != g.screenW || h != g.screenH {
		g.screenW, g.screenH = w, h
		g.layout = computeLayout(w, h)
	}

	g.handleInput()

	if !g.paused && g.focusedBoard < 0 {
		speed := 50
		if g.tick%speed == 0 {
			g.advancePosition(1)
		}
	}

	return nil
}

func (g *Game) handleInput() {
	mx, my := ebiten.CursorPosition()
	g.updateHover(mx, my)

	if inpututil.IsKeyJustPressed(ebiten.KeySlash) || inpututil.IsKeyJustPressed(ebiten.KeyF1) {
		g.showHelp = !g.showHelp
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
		if g.showHelp {
			g.showHelp = false
		} else if g.focusedBoard >= 0 {
			g.focusedBoard = -1
		}
	}
	if inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		g.paused = !g.paused
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyRight) || inpututil.IsKeyJustPressed(ebiten.KeyL) {
		g.advancePosition(1)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyLeft) || inpututil.IsKeyJustPressed(ebiten.KeyH) {
		g.advancePosition(-1)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyDown) || inpututil.IsKeyJustPressed(ebiten.KeyJ) {
		g.advancePosition(g.layout.GridCols)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyUp) || inpututil.IsKeyJustPressed(ebiten.KeyK) {
		g.advancePosition(-g.layout.GridCols)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyTab) {
		g.currentTrace = (g.currentTrace + 1) % len(g.traces)
		g.startPos = 0
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyBackquote) {
		g.currentTrace = (g.currentTrace - 1 + len(g.traces)) % len(g.traces)
		g.startPos = 0
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyHome) || inpututil.IsKeyJustPressed(ebiten.KeyG) {
		g.startPos = 0
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEnd) {
		trace := g.traces[g.currentTrace]
		g.startPos = max(0, len(trace.Positions)-g.layout.GridCols*g.layout.GridRows)
	}

	for i := ebiten.Key0; i <= ebiten.Key9; i++ {
		if inpututil.IsKeyJustPressed(i) {
			g.startPos = int(i-ebiten.Key0) * 10
			g.clampPosition()
		}
	}

	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		if g.hoveredBoard >= 0 && g.focusedBoard < 0 {
			g.focusedBoard = g.startPos + g.hoveredBoard
		} else if g.focusedBoard >= 0 {
			g.focusedBoard = -1
		}
	}

	_, dy := ebiten.Wheel()
	if dy != 0 {
		if dy > 0 {
			g.advancePosition(-1)
		} else {
			g.advancePosition(1)
		}
	}
}

func (g *Game) updateHover(mx, my int) {
	g.hoveredBoard = -1
	if g.focusedBoard >= 0 {
		return
	}

	bs := g.layout.BoardSize()
	gap := g.layout.BoardGap
	totalW := float64(g.layout.GridCols)*bs + float64(g.layout.GridCols-1)*gap
	totalH := float64(g.layout.GridRows)*bs + float64(g.layout.GridRows-1)*gap
	availH := float64(g.screenH) - g.layout.HeaderHeight - g.layout.FooterHeight
	startX := (float64(g.screenW) - totalW) / 2
	startY := g.layout.HeaderHeight + (availH-totalH)/2

	for row := 0; row < g.layout.GridRows; row++ {
		for col := 0; col < g.layout.GridCols; col++ {
			x := startX + float64(col)*(bs+gap)
			y := startY + float64(row)*(bs+gap)
			if float64(mx) >= x && float64(mx) < x+bs && float64(my) >= y && float64(my) < y+bs {
				g.hoveredBoard = row*g.layout.GridCols + col
				return
			}
		}
	}
}

func (g *Game) advancePosition(delta int) {
	g.startPos += delta
	g.clampPosition()
}

func (g *Game) clampPosition() {
	trace := g.traces[g.currentTrace]
	maxPos := len(trace.Positions) - 1
	if g.startPos < 0 {
		g.startPos = 0
	}
	if g.startPos > maxPos {
		g.startPos = maxPos
	}
}

func (g *Game) Layout(w, h int) (int, int) {
	return w, h
}

// ============================================================================
// RENDERING
// ============================================================================

func (g *Game) Draw(screen *ebiten.Image) {
	g.drawBackground(screen)
	g.drawHeader(screen)

	if g.focusedBoard >= 0 || g.splitAnim.Value > 0.01 {
		g.drawSplitView(screen)
	} else {
		g.drawBoardGrid(screen)
	}

	g.drawFooter(screen)

	if g.showHelp {
		g.drawHelpOverlay(screen)
	}
}

func (g *Game) drawBackground(screen *ebiten.Image) {
	screen.Fill(colorBg1)
	w, h := float32(g.screenW), float32(g.screenH)
	corners := []struct{ x, y float32 }{{0, 0}, {w, 0}, {0, h}, {w, h}}
	for _, c := range corners {
		for r := float32(200); r > 0; r -= 20 {
			alpha := uint8(float32(15) * (r / 200))
			vector.DrawFilledCircle(screen, c.x, c.y, r, color.RGBA{0, 0, 0, alpha}, false)
		}
	}
}

func (g *Game) drawHeader(screen *ebiten.Image) {
	w := float32(g.screenW)
	h := float32(g.layout.HeaderHeight)
	trace := g.traces[g.currentTrace]

	for y := float32(0); y < h; y++ {
		t := y / h
		alpha := uint8(lerp(0xf8, 0xe0, float64(t)))
		vector.DrawFilledRect(screen, 0, y, w, 1,
			color.RGBA{colorBg2.R, colorBg2.G, colorBg2.B, alpha}, false)
	}
	vector.DrawFilledRect(screen, 0, h-1, w, 1, color.RGBA{0x30, 0x30, 0x3c, 0xff}, false)

	title := trace.Name
	if len(title) > 32 {
		title = title[:29] + "..."
	}
	g.drawTextAt(screen, title, sp6, int(h/2-8), interFace18, colorText100)

	statusX := int(w) - sp6 - 90
	statusY := int(h/2) - 14
	if g.paused {
		g.drawPill(screen, statusX, statusY, "PAUSED", colorText60, colorBg3)
	} else {
		pulse := 0.7 + 0.3*math.Sin(g.pulsePhase*2)
		c := color.RGBA{uint8(float64(colorSuccess.R) * pulse), uint8(float64(colorSuccess.G) * pulse), uint8(float64(colorSuccess.B) * pulse), 0xff}
		g.drawPill(screen, statusX, statusY, "PLAYING", c, colorBg3)
	}

	traceInfo := fmt.Sprintf("%d / %d", g.currentTrace+1, len(g.traces))
	tw, _ := text.Measure(traceInfo, interFace13, 0)
	g.drawTextAt(screen, traceInfo, int(w/2-float32(tw)/2), int(h/2-6), interFace13, colorText60)
}

func (g *Game) drawBoardGrid(screen *ebiten.Image) {
	trace := g.traces[g.currentTrace]
	bs := g.layout.BoardSize()
	gap := g.layout.BoardGap

	totalW := float64(g.layout.GridCols)*bs + float64(g.layout.GridCols-1)*gap
	totalH := float64(g.layout.GridRows)*bs + float64(g.layout.GridRows-1)*gap
	availH := float64(g.screenH) - g.layout.HeaderHeight - g.layout.FooterHeight
	startX := (float64(g.screenW) - totalW) / 2
	startY := g.layout.HeaderHeight + (availH-totalH)/2

	for row := 0; row < g.layout.GridRows; row++ {
		for col := 0; col < g.layout.GridCols; col++ {
			idx := row*g.layout.GridCols + col
			posIdx := g.startPos + idx
			if posIdx >= len(trace.Positions) {
				continue
			}

			x := startX + float64(col)*(bs+gap)
			y := startY + float64(row)*(bs+gap)
			hover := g.hoverAnims[idx].Value

			g.drawBoard(screen, x, y, trace.Positions[posIdx], hover, idx == 0, 1.0)
		}
	}
}

func (g *Game) drawSplitView(screen *ebiten.Image) {
	trace := g.traces[g.currentTrace]
	posIdx := g.focusedBoard
	if posIdx < 0 {
		posIdx = g.startPos
	}
	if posIdx >= len(trace.Positions) {
		return
	}

	pos := trace.Positions[posIdx]
	split := g.splitAnim.Value

	// Calculate split dimensions
	headerH := g.layout.HeaderHeight
	footerH := g.layout.FooterHeight
	availW := float64(g.screenW)
	availH := float64(g.screenH) - headerH - footerH

	// Left side: Board
	boardScale := 1.4
	if g.layout.Mode == LayoutMobile {
		boardScale = 1.0
	}

	bs := g.layout.BoardSize() * boardScale
	leftW := availW * (0.5 * split)
	if split < 0.01 {
		leftW = availW
	}

	// Board position with animation
	boardX := (leftW - bs) / 2
	if split < 0.5 {
		boardX = (availW - bs) / 2
	}
	boardY := headerH + (availH-bs)/2

	// Draw divider
	if split > 0.01 {
		divX := float32(availW * 0.5)
		vector.DrawFilledRect(screen, divX-1, float32(headerH), 2, float32(availH),
			color.RGBA{0x30, 0x30, 0x3c, uint8(255 * split)}, false)
	}

	// Draw board
	g.drawBoard(screen, boardX, boardY, pos, 0, true, boardScale)

	// Right side: Statechart
	if split > 0.1 {
		rightX := availW * 0.5
		rightW := availW * 0.5
		rightY := headerH
		rightH := availH

		// Statechart panel background
		alpha := uint8(255 * split)
		vector.DrawFilledRect(screen, float32(rightX), float32(rightY),
			float32(rightW), float32(rightH),
			color.RGBA{colorBg0.R, colorBg0.G, colorBg0.B, alpha}, false)

		// Generate and draw statechart
		machine := GoGameMachine(pos)
		g.drawStatechart(screen, rightX+sp6, rightY+sp6, rightW-sp6*2, rightH-sp6*2, machine, split)
	}

	// Exit hint
	if split > 0.5 {
		hint := "Click to exit"
		tw, _ := text.Measure(hint, interFace11, 0)
		hintX := int(leftW/2 - tw/2)
		hintY := int(boardY + bs + sp4)
		g.drawTextAt(screen, hint, hintX, hintY, interFace11, colorText40)
	}
}

func (g *Game) drawStatechart(screen *ebiten.Image, x, y, w, h float64, machine *SCMachine, alpha float64) {
	// Title
	titleAlpha := uint8(255 * alpha)
	g.drawTextAt(screen, "State Machine: "+machine.Name, int(x), int(y), interFace18,
		color.RGBA{colorText100.R, colorText100.G, colorText100.B, titleAlpha})

	// Scale and offset for statechart
	scaleX := w / 700
	scaleY := (h - 60) / 320
	scale := math.Min(scaleX, scaleY)
	offsetX := x + (w-700*scale)/2
	offsetY := y + 50

	// Draw transitions first (behind states)
	for _, t := range machine.Transitions {
		g.drawTransition(screen, offsetX, offsetY, scale, t, alpha)
	}

	// Draw states
	for _, s := range machine.States {
		g.drawState(screen, offsetX, offsetY, scale, s, alpha)
	}

	// Legend
	legendY := y + h - 80
	g.drawLegend(screen, x, legendY, alpha)
}

func (g *Game) drawState(screen *ebiten.Image, offsetX, offsetY, scale float64, s *SCState, alpha float64) {
	x := offsetX + s.X*scale
	y := offsetY + s.Y*scale
	w := s.W * scale
	h := s.H * scale

	// State color based on type and active status
	var fillColor, borderColor color.RGBA
	if s.IsActive {
		fillColor = colorStateActive
		borderColor = color.RGBA{0x60, 0xa5, 0xfa, 0xff}
	} else {
		switch s.Type {
		case StateInitial:
			fillColor = color.RGBA{colorStateInitial.R, colorStateInitial.G, colorStateInitial.B, 0x40}
			borderColor = colorStateInitial
		case StateFinal:
			fillColor = color.RGBA{colorStateFinal.R, colorStateFinal.G, colorStateFinal.B, 0x40}
			borderColor = colorStateFinal
		default:
			fillColor = colorStateNormal
			borderColor = color.RGBA{0x47, 0x55, 0x69, 0xff}
		}
	}

	// Apply alpha
	fillColor.A = uint8(float64(fillColor.A) * alpha)
	borderColor.A = uint8(float64(borderColor.A) * alpha)

	// Draw rounded rect (approximated)
	radius := float32(6 * scale)

	// Shadow
	if s.IsActive {
		for i := 0; i < 3; i++ {
			off := float32(2 + i*2)
			a := uint8(30 * alpha / float64(i+1))
			vector.DrawFilledRect(screen, float32(x)+off, float32(y)+off, float32(w), float32(h),
				color.RGBA{colorStateActive.R, colorStateActive.G, colorStateActive.B, a}, false)
		}
	}

	// Fill
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h), fillColor, false)

	// Border
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), float32(2*scale), borderColor, false)

	// Rounded corners overlay (draw small circles at corners)
	_ = radius

	// Label
	labelAlpha := uint8(255 * alpha)
	lw, lh := text.Measure(s.Label, interFace13, 0)
	lx := x + (w-lw)/2
	ly := y + (h-lh)/2
	g.drawTextAt(screen, s.Label, int(lx), int(ly), interFace13,
		color.RGBA{colorText100.R, colorText100.G, colorText100.B, labelAlpha})

	// Active indicator (pulsing glow)
	if s.IsActive {
		pulse := 0.5 + 0.5*math.Sin(g.pulsePhase*3)
		glowAlpha := uint8(60 * pulse * alpha)
		for i := 1; i <= 3; i++ {
			expand := float64(i * 3)
			vector.StrokeRect(screen, float32(x-expand), float32(y-expand),
				float32(w+expand*2), float32(h+expand*2), 1,
				color.RGBA{colorStateActive.R, colorStateActive.G, colorStateActive.B, glowAlpha / uint8(i)}, false)
		}
	}
}

func (g *Game) drawTransition(screen *ebiten.Image, offsetX, offsetY, scale float64, t *SCTransition, alpha float64) {
	// Calculate start and end points
	fromX := offsetX + (t.From.X+t.From.W)*scale
	fromY := offsetY + (t.From.Y+t.From.H/2)*scale
	toX := offsetX + t.To.X*scale
	toY := offsetY + (t.To.Y+t.To.H/2)*scale

	// For self-loops or complex routing, we'd need bezier curves
	// For now, draw straight lines with arrow

	var lineColor color.RGBA
	var lineWidth float32
	if t.IsActive {
		lineColor = colorTransitionAct
		lineWidth = float32(3 * scale)
	} else {
		lineColor = colorTransition
		lineWidth = float32(1.5 * scale)
	}
	lineColor.A = uint8(float64(lineColor.A) * alpha)

	// Draw line
	vector.StrokeLine(screen, float32(fromX), float32(fromY), float32(toX), float32(toY), lineWidth, lineColor, false)

	// Draw arrowhead
	angle := math.Atan2(toY-fromY, toX-fromX)
	arrowLen := 10 * scale
	arrowAngle := 0.5

	ax1 := toX - arrowLen*math.Cos(angle-arrowAngle)
	ay1 := toY - arrowLen*math.Sin(angle-arrowAngle)
	ax2 := toX - arrowLen*math.Cos(angle+arrowAngle)
	ay2 := toY - arrowLen*math.Sin(angle+arrowAngle)

	vector.StrokeLine(screen, float32(toX), float32(toY), float32(ax1), float32(ay1), lineWidth, lineColor, false)
	vector.StrokeLine(screen, float32(toX), float32(toY), float32(ax2), float32(ay2), lineWidth, lineColor, false)

	// Draw event label
	if t.Event != "" {
		midX := (fromX + toX) / 2
		midY := (fromY+toY)/2 - 10*scale

		labelAlpha := uint8(200 * alpha)
		g.drawTextAt(screen, t.Event, int(midX), int(midY), interFace10,
			color.RGBA{colorText80.R, colorText80.G, colorText80.B, labelAlpha})

		// Guard condition
		if t.Guard != "" {
			guardY := midY + 12*scale
			g.drawTextAt(screen, "["+t.Guard+"]", int(midX), int(guardY), interFace10,
				color.RGBA{colorText60.R, colorText60.G, colorText60.B, uint8(150 * alpha)})
		}
	}
}

func (g *Game) drawLegend(screen *ebiten.Image, x, y, alpha float64) {
	items := []struct {
		color color.RGBA
		label string
	}{
		{colorStateActive, "Active State"},
		{colorStateNormal, "Inactive State"},
		{colorTransitionAct, "Active Transition"},
		{colorTransition, "Transition"},
	}

	legendX := x
	for _, item := range items {
		c := item.color
		c.A = uint8(float64(c.A) * alpha)

		// Color box
		vector.DrawFilledRect(screen, float32(legendX), float32(y), 12, 12, c, false)

		// Label
		labelAlpha := uint8(180 * alpha)
		g.drawTextAt(screen, item.label, int(legendX)+16, int(y), interFace10,
			color.RGBA{colorText60.R, colorText60.G, colorText60.B, labelAlpha})

		legendX += 120
	}
}

func (g *Game) drawBoard(screen *ebiten.Image, x, y float64, pos Position, hover float64, showCoords bool, scale float64) {
	cs := g.layout.CellSize * scale
	pad := g.layout.BoardPadding * scale
	bs := cs*float64(boardSize-1) + pad*2

	// Shadow
	for i := 0; i < 3; i++ {
		off := float64(4-i) * scale
		alpha := uint8(40 - i*10)
		vector.DrawFilledRect(screen, float32(x+off), float32(y+off), float32(bs), float32(bs),
			color.RGBA{0, 0, 0, alpha}, false)
	}

	// Hover glow
	if hover > 0.01 {
		glowSize := 3.0 * hover * scale
		glowAlpha := uint8(40 * hover)
		vector.DrawFilledRect(screen, float32(x-glowSize), float32(y-glowSize),
			float32(bs+glowSize*2), float32(bs+glowSize*2),
			color.RGBA{colorAccent.R, colorAccent.G, colorAccent.B, glowAlpha}, false)
	}

	// Board surface
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(bs), float32(bs), colorBoard, false)

	// Wood grain
	for i := 0; i < int(bs); i += 4 {
		noise := math.Sin(float64(i)*0.1) * 0.5
		alpha := uint8(8 + 6*noise)
		vector.DrawFilledRect(screen, float32(x), float32(y+float64(i)), float32(bs), 2,
			color.RGBA{0x90, 0x70, 0x40, alpha}, false)
	}

	// Grid
	lineW := float32(1.2 * scale)
	for i := 0; i < boardSize; i++ {
		lx := float32(x + pad + float64(i)*cs)
		ly := float32(y + pad + float64(i)*cs)
		vector.StrokeLine(screen, lx, float32(y+pad), lx, float32(y+pad+float64(boardSize-1)*cs), lineW, colorGridLine, false)
		vector.StrokeLine(screen, float32(x+pad), ly, float32(x+pad+float64(boardSize-1)*cs), ly, lineW, colorGridLine, false)
	}

	// Star points
	hoshiR := float32(3.5 * scale)
	for _, h := range []Point{{2, 2}, {6, 2}, {4, 4}, {2, 6}, {6, 6}} {
		hx := float32(x + pad + float64(h.X)*cs)
		hy := float32(y + pad + float64(h.Y)*cs)
		vector.DrawFilledCircle(screen, hx, hy, hoshiR, colorHoshi, true)
	}

	// Stones
	stoneR := float32((cs/2 - 2) * scale)
	for sy := 0; sy < boardSize; sy++ {
		for sx := 0; sx < boardSize; sx++ {
			stone := pos.Board[sy][sx]
			if stone == Empty {
				continue
			}

			cx := float32(x + pad + float64(sx)*cs)
			cy := float32(y + pad + float64(sy)*cs)

			vector.DrawFilledCircle(screen, cx+1.5, cy+2, stoneR, color.RGBA{0, 0, 0, 0x40}, true)

			if stone == Black {
				vector.DrawFilledCircle(screen, cx, cy, stoneR, colorBlackStone, true)
				vector.DrawFilledCircle(screen, cx-stoneR*0.35, cy-stoneR*0.35, stoneR*0.22, colorBlackShine, true)
			} else {
				vector.DrawFilledCircle(screen, cx, cy, stoneR, colorWhiteStone, true)
				vector.StrokeCircle(screen, cx, cy, stoneR-0.5, 1, colorWhiteShadow, true)
			}

			// Last move marker
			if pos.LastMove != nil && pos.LastMove.X == sx && pos.LastMove.Y == sy {
				pulse := 0.7 + 0.3*math.Sin(g.markerPulse)
				markerR := stoneR * 0.32 * float32(pulse)
				var markerC color.RGBA
				if stone == Black {
					markerC = colorText100
				} else {
					markerC = colorBlackStone
				}
				vector.DrawFilledCircle(screen, cx, cy, markerR, markerC, true)
			}
		}
	}

	// Move number badge
	if pos.MoveNum > 0 {
		label := fmt.Sprintf("#%d", pos.MoveNum)
		lw, lh := text.Measure(label, interFace13, 0)
		badgeW := lw + sp3*2
		badgeH := lh + sp1*2 + 2
		badgeX := x + sp2
		badgeY := y + sp2

		vector.DrawFilledRect(screen, float32(badgeX), float32(badgeY), float32(badgeW), float32(badgeH),
			color.RGBA{0, 0, 0, 0xdd}, false)
		vector.StrokeRect(screen, float32(badgeX), float32(badgeY), float32(badgeW), float32(badgeH), 1,
			color.RGBA{0x40, 0x40, 0x50, 0xff}, false)
		g.drawTextAt(screen, label, int(badgeX)+sp3, int(badgeY)+sp1+2, interFace13, colorText100)
	}

	// Coordinates
	if showCoords {
		coords := "ABCDEFGHJ"
		for i := 0; i < boardSize; i++ {
			cx := int(x + pad + float64(i)*cs - 3)
			cy := int(y + bs - 12)
			g.drawTextAt(screen, string(coords[i]), cx, cy, interFace11, colorGridLine)

			nx := int(x + 3)
			ny := int(y + pad + float64(boardSize-1-i)*cs - 5)
			g.drawTextAt(screen, fmt.Sprintf("%d", i+1), nx, ny, interFace11, colorGridLine)
		}
	}
}

func (g *Game) drawFooter(screen *ebiten.Image) {
	w := float32(g.screenW)
	h := float32(g.layout.FooterHeight)
	y := float32(g.screenH) - h
	trace := g.traces[g.currentTrace]

	for fy := float32(0); fy < h; fy++ {
		t := 1 - fy/h
		alpha := uint8(lerp(0xe0, 0xf8, float64(t)))
		vector.DrawFilledRect(screen, 0, y+fy, w, 1,
			color.RGBA{colorBg2.R, colorBg2.G, colorBg2.B, alpha}, false)
	}
	vector.DrawFilledRect(screen, 0, y, w, 1, color.RGBA{0x30, 0x30, 0x3c, 0xff}, false)

	timelineY := y + sp5
	timelineX := float32(sp8)
	timelineW := w - float32(sp8*2)
	trackH := float32(6)

	vector.DrawFilledRect(screen, timelineX, timelineY, timelineW, trackH, colorBg0, false)

	total := len(trace.Positions)
	if total > 1 {
		progress := float32(g.startPos) / float32(total-1)
		progressW := timelineW * progress

		vector.DrawFilledRect(screen, timelineX, timelineY, progressW, trackH, colorAccent, false)

		handleX := timelineX + progressW
		handleY := timelineY + trackH/2

		vector.DrawFilledCircle(screen, handleX, handleY, 12, colorAccentMuted, true)
		vector.DrawFilledCircle(screen, handleX, handleY, 8, colorAccent, true)
		vector.DrawFilledCircle(screen, handleX, handleY, 4, colorText100, true)
	}

	posText := fmt.Sprintf("Move %d of %d", g.startPos+1, total)
	tw, _ := text.Measure(posText, interFace15, 0)
	textY := int(y + sp5 + trackH + sp4)
	g.drawTextAt(screen, posText, int(w/2-float32(tw)/2), textY, interFace15, colorText80)

	if g.layout.Mode == LayoutDesktop {
		hints := "← → Navigate   Space Play/Pause   ? Help"
		tw, _ := text.Measure(hints, interFace11, 0)
		hintsY := int(y + h - sp4 - 12)
		g.drawTextAt(screen, hints, int(w/2-float32(tw)/2), hintsY, interFace11, colorText40)
	}
}

func (g *Game) drawHelpOverlay(screen *ebiten.Image) {
	w, h := float32(g.screenW), float32(g.screenH)
	vector.DrawFilledRect(screen, 0, 0, w, h, color.RGBA{0, 0, 0, 0xcc}, false)

	panelW := float32(420)
	panelH := float32(380)
	if g.layout.Mode == LayoutMobile {
		panelW = w - sp8*2
		panelH = 440
	}
	panelX := (w - panelW) / 2
	panelY := (h - panelH) / 2

	vector.DrawFilledRect(screen, panelX, panelY, panelW, panelH, colorBg3, false)
	vector.StrokeRect(screen, panelX, panelY, panelW, panelH, 1, color.RGBA{0x40, 0x40, 0x4c, 0xff}, false)

	g.drawTextAt(screen, "Keyboard Shortcuts", int(panelX)+sp6, int(panelY)+sp6, interFace24, colorText100)

	shortcuts := []struct{ key, desc string }{
		{"← → H L", "Navigate moves"},
		{"↑ ↓ J K", "Jump by row"},
		{"Space", "Play / Pause"},
		{"Tab / `", "Switch traces"},
		{"0-9", "Jump to position"},
		{"G / Home", "Go to start"},
		{"End", "Go to end"},
		{"Click", "Focus board + Show statechart"},
		{"Scroll", "Navigate moves"},
		{"Esc", "Close overlay"},
	}

	startY := int(panelY) + sp6 + 48
	for i, s := range shortcuts {
		row := startY + i*28
		g.drawTextAt(screen, s.key, int(panelX)+sp6, row, interFace15, colorAccentHover)
		g.drawTextAt(screen, s.desc, int(panelX)+140, row, interFace15, colorText60)
	}

	g.drawTextAt(screen, "Press ? or Esc to close", int(panelX)+sp6, int(panelY+panelH)-sp6-14, interFace13, colorText40)
}

// ============================================================================
// HELPERS
// ============================================================================

func (g *Game) drawTextAt(screen *ebiten.Image, s string, x, y int, face *text.GoTextFace, c color.Color) {
	op := &text.DrawOptions{}
	op.GeoM.Translate(float64(x), float64(y))
	op.ColorScale.ScaleWithColor(c)
	text.Draw(screen, s, face, op)
}

func (g *Game) drawPill(screen *ebiten.Image, x, y int, label string, fg, bg color.Color) {
	lw, lh := text.Measure(label, interFace13, 0)
	w := lw + sp4*2
	h := lh + sp2*2

	bgC := bg.(color.RGBA)
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h),
		color.RGBA{bgC.R, bgC.G, bgC.B, 0xf0}, false)

	fgC := fg.(color.RGBA)
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), 1,
		color.RGBA{fgC.R, fgC.G, fgC.B, 0x50}, false)

	g.drawTextAt(screen, label, x+sp4, y+sp2, interFace13, fg)
}

func lerp(a, b, t float64) float64 {
	return a + (b-a)*t
}

// ============================================================================
// DATA LOADING
// ============================================================================

func loadTracesFromDir(dir string) ([]Trace, error) {
	var traces []Trace
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		path := filepath.Join(dir, e.Name())
		ext := strings.ToLower(filepath.Ext(e.Name()))
		var trace *Trace
		if ext == ".sgf" {
			trace, _ = loadSGF(path)
		} else if ext == ".json" {
			trace, _ = loadJSON(path)
		}
		if trace != nil {
			traces = append(traces, *trace)
		}
	}
	return traces, nil
}

func loadSGF(path string) (*Trace, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	trace := &Trace{Name: filepath.Base(path)}
	re := regexp.MustCompile(`([BW])\[([a-z]{0,2})\]`)

	for _, m := range re.FindAllStringSubmatch(string(data), -1) {
		clr := Black
		if m[1] == "W" {
			clr = White
		}
		if len(m[2]) < 2 {
			trace.Moves = append(trace.Moves, Move{Color: clr, Pass: true})
			continue
		}
		x, y := int(m[2][0]-'a'), int(m[2][1]-'a')
		if x >= 0 && x < boardSize && y >= 0 && y < boardSize {
			trace.Moves = append(trace.Moves, Move{X: x, Y: y, Color: clr})
		}
	}

	if len(trace.Moves) == 0 {
		return nil, fmt.Errorf("no moves")
	}
	trace.Positions = buildPositions(trace.Moves)
	return trace, nil
}

func loadJSON(path string) (*Trace, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var jt JSONTrace
	if err := json.Unmarshal(data, &jt); err != nil {
		return nil, err
	}

	trace := &Trace{Name: jt.GameID}
	if trace.Name == "" {
		trace.Name = filepath.Base(path)
	}

	for _, jm := range jt.Moves {
		clr := Black
		if jm.Player == "W" {
			clr = White
		}
		move := Move{X: jm.X, Y: jm.Y, Color: clr, Pass: jm.IsPass}
		for _, c := range jm.Captures {
			if len(c) >= 2 {
				move.Captures = append(move.Captures, Point{X: c[0], Y: c[1]})
			}
		}
		trace.Moves = append(trace.Moves, move)
	}

	if len(trace.Moves) == 0 {
		return nil, fmt.Errorf("no moves")
	}
	trace.Positions = buildPositionsWithCaptures(trace.Moves)
	return trace, nil
}

func buildPositions(moves []Move) []Position {
	positions := make([]Position, len(moves)+1)
	positions[0] = Position{MoveNum: 0}

	var board [boardSize][boardSize]Stone
	for i, m := range moves {
		if !m.Pass {
			board[m.Y][m.X] = m.Color
		}
		positions[i+1] = Position{Board: board, MoveNum: i + 1, LastMove: &moves[i]}
	}
	return positions
}

func buildPositionsWithCaptures(moves []Move) []Position {
	positions := make([]Position, len(moves)+1)
	positions[0] = Position{MoveNum: 0}

	var board [boardSize][boardSize]Stone
	for i, m := range moves {
		if !m.Pass {
			board[m.Y][m.X] = m.Color
			for _, c := range m.Captures {
				if c.X >= 0 && c.X < boardSize && c.Y >= 0 && c.Y < boardSize {
					board[c.Y][c.X] = Empty
				}
			}
		}
		positions[i+1] = Position{Board: board, MoveNum: i + 1, LastMove: &moves[i]}
	}
	return positions
}

func loadSampleTraces() []Trace {
	moves := []Move{
		{X: 2, Y: 2, Color: Black}, {X: 6, Y: 2, Color: White},
		{X: 2, Y: 6, Color: Black}, {X: 6, Y: 6, Color: White},
		{X: 4, Y: 4, Color: Black}, {X: 3, Y: 3, Color: White},
	}
	trace := Trace{Name: "Sample Game", Moves: moves}
	trace.Positions = buildPositions(moves)
	return []Trace{trace}
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	dir := flag.String("dir", "", "Directory with SGF/JSON files")
	flag.Parse()

	var traces []Trace
	if *dir != "" {
		traces, _ = loadTracesFromDir(*dir)
	} else if flag.NArg() > 0 {
		for _, p := range flag.Args() {
			ext := strings.ToLower(filepath.Ext(p))
			var t *Trace
			if ext == ".sgf" {
				t, _ = loadSGF(p)
			} else if ext == ".json" {
				t, _ = loadJSON(p)
			}
			if t != nil {
				traces = append(traces, *t)
			}
		}
	}

	if len(traces) == 0 {
		traces = loadSampleTraces()
	}

	ebiten.SetWindowSize(1280, 800)
	ebiten.SetWindowTitle("Go Trace Viewer + Statechart")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	fmt.Printf("Go Trace Viewer • %d traces\n", len(traces))
	fmt.Println("Click a board to see statechart • Press ? for help")

	if err := ebiten.RunGame(NewGame(traces)); err != nil {
		log.Fatal(err)
	}
}
