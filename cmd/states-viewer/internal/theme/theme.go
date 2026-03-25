// Package theme provides the visual design system for the statechart viewer.
// Colors, typography, shadows, and layout constants following Stripe-inspired design.
package theme

import (
	"image/color"
	"math"
)

// UIMode represents the current interface mode.
type UIMode int

const (
	UIDesktop UIMode = iota
	UIMobile
)

// Breakpoints for responsive design.
const (
	BreakpointMobile  = 480
	BreakpointTablet  = 768
	BreakpointDesktop = 1024
)

// GetUIMode returns the appropriate UI mode for the given width.
func GetUIMode(width int) UIMode {
	if width < BreakpointTablet {
		return UIMobile
	}
	return UIDesktop
}

// Color palette - Light mode (Stripe-inspired).
var (
	// Canvas & Backgrounds
	CanvasBg   = color.RGBA{250, 250, 250, 255} // #FAFAFA - warm off-white
	PanelBg    = color.RGBA{255, 255, 255, 255} // Fully opaque white (was 85% transparent)
	NodeBg     = color.RGBA{255, 255, 255, 255} // Pure white
	NodeBgDark = color.RGBA{38, 38, 38, 255}    // #262626

	// Borders & Lines
	BorderLight  = color.RGBA{229, 231, 235, 255} // #E5E7EB - Gray-200
	BorderMedium = color.RGBA{209, 213, 219, 255} // #D1D5DB - Gray-300
	EdgeLine     = color.RGBA{50, 50, 50, 255}    // Much darker gray for visibility
	BorderDark   = color.RGBA{51, 51, 51, 255}    // #333333

	// Text
	TextPrimary   = color.RGBA{17, 24, 39, 255}   // #111827 - Gray-900
	TextSecondary = color.RGBA{107, 114, 128, 255} // #6B7280 - Gray-500
	TextMuted     = color.RGBA{156, 163, 175, 255} // #9CA3AF - Gray-400
	TextLight     = color.RGBA{249, 250, 251, 255} // #F9FAFB

	// Accent (Indigo)
	Accent       = color.RGBA{99, 102, 241, 255}  // #6366F1 - Indigo-500
	AccentLight  = color.RGBA{129, 140, 248, 255} // #818CF8 - Indigo-400
	AccentSubtle = color.RGBA{238, 242, 255, 255} // #EEF2FF - Indigo-50

	// Semantic
	Success     = color.RGBA{34, 197, 94, 255}  // #22C55E - Green-500
	SuccessGlow = color.RGBA{34, 197, 94, 64}   // Green with 25% alpha
	Warning     = color.RGBA{245, 158, 11, 255} // #F59E0B - Amber-500
	Error       = color.RGBA{239, 68, 68, 255}  // #EF4444 - Red-500

	// State type colors
	ParallelTint = color.RGBA{99, 102, 241, 40}   // Indigo tint
	FinalFill    = color.RGBA{107, 114, 128, 255} // Gray
	HistoryTint  = color.RGBA{251, 191, 36, 255}  // Amber

	// Grid
	GridDot = color.RGBA{200, 200, 200, 255}
)

// Dark mode colors.
var (
	CanvasBgDark  = color.RGBA{15, 15, 15, 255}   // #0F0F0F
	PanelBgDark   = color.RGBA{30, 30, 30, 230}   // rgba(30,30,30,0.9)
	TextPrimaryDk = color.RGBA{249, 250, 251, 255} // #F9FAFB
)

// Shadow represents a box shadow configuration.
type Shadow struct {
	OffsetX float64
	OffsetY float64
	Blur    float64
	Color   color.RGBA
}

// Elevation levels for shadows.
var (
	// Level 0: At rest (no shadow)
	Shadow0 = Shadow{}

	// Level 1: Subtle hover
	Shadow1 = Shadow{
		OffsetX: 0,
		OffsetY: 1,
		Blur:    3,
		Color:   color.RGBA{0, 0, 0, 13}, // 5% opacity
	}

	// Level 2: Selected / Floating panels
	Shadow2 = Shadow{
		OffsetX: 0,
		OffsetY: 4,
		Blur:    12,
		Color:   color.RGBA{0, 0, 0, 20}, // 8% opacity
	}

	// Level 3: Modal / Bottom sheet
	Shadow3 = Shadow{
		OffsetX: 0,
		OffsetY: 8,
		Blur:    24,
		Color:   color.RGBA{0, 0, 0, 31}, // 12% opacity
	}
)

// Layout constants.
const (
	// Node sizing
	DefaultNodeWidth  = 120
	DefaultNodeHeight = 60
	NodeCornerRadius  = 8.0
	NodePadding       = 12
	NodeBorderWidth   = 1.0
	NodeBorderHover   = 2.0
	NodeBorderSelect  = 2.5

	// Grid
	GridSpacing = 20
	GridDotSize = 2.0

	// Panel
	PanelWidth      = 220
	PanelPadding    = 16
	PanelCorner     = 12.0
	TimelineHeight  = 48
	TimelineKnobR   = 8

	// Buttons
	ButtonHeight    = 40
	ButtonPadding   = 12
	ButtonCorner    = 8.0
	TouchTargetMin  = 44

	// Animation durations (in seconds)
	HoverDuration   = 0.15
	PressDuration   = 0.1
	GlowDuration    = 0.4
	PanelDuration   = 0.25
	ZoomDuration    = 0.2
	BreathingPeriod = 2.0
)

// Typography sizes.
const (
	FontSizeTitle   = 18
	FontSizeBody    = 14
	FontSizeCaption = 12
	FontSizeMono    = 13
)

// Zoom limits.
const (
	ZoomMin = 0.1
	ZoomMax = 5.0
)

// EaseOut provides an ease-out interpolation.
func EaseOut(t float64) float64 {
	return 1 - math.Pow(1-t, 3)
}

// EaseIn provides an ease-in interpolation.
func EaseIn(t float64) float64 {
	return math.Pow(t, 3)
}

// EaseInOut provides an ease-in-out interpolation.
func EaseInOut(t float64) float64 {
	if t < 0.5 {
		return 4 * t * t * t
	}
	return 1 - math.Pow(-2*t+2, 3)/2
}

// Spring provides a spring animation interpolation with slight overshoot.
func Spring(t float64) float64 {
	const overshoot = 1.1
	if t < 0.5 {
		return 2 * t * t * overshoot
	}
	return 1 - math.Pow(-2*t+2, 2)/2*overshoot + (overshoot-1)*(1-t)
}

// Lerp linearly interpolates between a and b.
func Lerp(a, b, t float64) float64 {
	return a + (b-a)*t
}

// LerpColor interpolates between two colors.
func LerpColor(a, b color.RGBA, t float64) color.RGBA {
	return color.RGBA{
		R: uint8(Lerp(float64(a.R), float64(b.R), t)),
		G: uint8(Lerp(float64(a.G), float64(b.G), t)),
		B: uint8(Lerp(float64(a.B), float64(b.B), t)),
		A: uint8(Lerp(float64(a.A), float64(b.A), t)),
	}
}

// Clamp restricts a value to a range.
func Clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// BreathingAlpha returns an alpha value for the breathing glow animation.
// phase should be time in seconds.
func BreathingAlpha(phase float64) float64 {
	// Oscillate between 0.7 and 1.0
	return 0.7 + 0.3*math.Sin(phase*2*math.Pi/BreathingPeriod)
}

// WithAlpha returns a copy of the color with the specified alpha.
func WithAlpha(c color.RGBA, alpha uint8) color.RGBA {
	return color.RGBA{R: c.R, G: c.G, B: c.B, A: alpha}
}
