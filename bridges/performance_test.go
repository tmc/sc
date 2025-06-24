package bridges

import (
	"fmt"
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/scxml"
	"github.com/tmc/sc/bridges/xstate"
)

// BenchmarkConversionPerformance benchmarks the performance of format conversions.
func BenchmarkConversionPerformance(b *testing.B) {
	sizes := []struct {
		Name   string
		States int
		Depth  int
	}{
		{"Small", 5, 2},
		{"Medium", 20, 3},
		{"Large", 100, 4},
		{"XLarge", 500, 5},
	}

	for _, size := range sizes {
		statechart := generateStatechartOfSize(size.States, size.Depth)

		b.Run(fmt.Sprintf("XState_Export_%s", size.Name), func(b *testing.B) {
			converter := xstate.NewConverter()
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := converter.Export(statechart)
				if err != nil {
					b.Fatalf("Export failed: %v", err)
				}
			}
		})

		b.Run(fmt.Sprintf("SCXML_Export_%s", size.Name), func(b *testing.B) {
			converter := scxml.NewConverter()
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := converter.Export(statechart)
				if err != nil {
					b.Fatalf("Export failed: %v", err)
				}
			}
		})

		// Test import performance with pre-converted data
		xstateConverter := xstate.NewConverter()
		xstateMachine, _ := xstateConverter.Export(statechart)

		b.Run(fmt.Sprintf("XState_Import_%s", size.Name), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := xstateConverter.Import(xstateMachine)
				if err != nil {
					b.Fatalf("Import failed: %v", err)
				}
			}
		})

		scxmlConverter := scxml.NewConverter()
		scxmlDoc, _ := scxmlConverter.Export(statechart)

		b.Run(fmt.Sprintf("SCXML_Import_%s", size.Name), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_, err := scxmlConverter.Import(scxmlDoc)
				if err != nil {
					b.Fatalf("Import failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkRoundTripPerformance benchmarks complete round-trip conversions.
func BenchmarkRoundTripPerformance(b *testing.B) {
	statechart := generateStatechartOfSize(50, 3)

	b.Run("XState_RoundTrip", func(b *testing.B) {
		converter := xstate.NewConverter()
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			machine, err := converter.Export(statechart)
			if err != nil {
				b.Fatalf("Export failed: %v", err)
			}
			
			_, err = converter.Import(machine)
			if err != nil {
				b.Fatalf("Import failed: %v", err)
			}
		}
	})

	b.Run("SCXML_RoundTrip", func(b *testing.B) {
		converter := scxml.NewConverter()
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			doc, err := converter.Export(statechart)
			if err != nil {
				b.Fatalf("Export failed: %v", err)
			}
			
			_, err = converter.Import(doc)
			if err != nil {
				b.Fatalf("Import failed: %v", err)
			}
		}
	})

	b.Run("Cross_Format_Conversion", func(b *testing.B) {
		xstateConverter := xstate.NewConverter()
		scxmlConverter := scxml.NewConverter()
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			// Harel → XState → Harel → SCXML → Harel
			machine, err := xstateConverter.Export(statechart)
			if err != nil {
				b.Fatalf("XState export failed: %v", err)
			}
			
			harel1, err := xstateConverter.Import(machine)
			if err != nil {
				b.Fatalf("XState import failed: %v", err)
			}
			
			doc, err := scxmlConverter.Export(harel1)
			if err != nil {
				b.Fatalf("SCXML export failed: %v", err)
			}
			
			_, err = scxmlConverter.Import(doc)
			if err != nil {
				b.Fatalf("SCXML import failed: %v", err)
			}
		}
	})
}

// BenchmarkMemoryUsage tests memory allocation patterns during conversion.
func BenchmarkMemoryUsage(b *testing.B) {
	statechart := generateStatechartOfSize(100, 3)

	b.Run("XState_Memory", func(b *testing.B) {
		converter := xstate.NewConverter()
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			machine, _ := converter.Export(statechart)
			_ = machine // Prevent optimization
		}
	})

	b.Run("SCXML_Memory", func(b *testing.B) {
		converter := scxml.NewConverter()
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			doc, _ := converter.Export(statechart)
			_ = doc // Prevent optimization
		}
	})
}

// TestConversionScalability tests how conversions scale with complexity.
func TestConversionScalability(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scalability test in short mode")
	}

	scalabilityTests := []struct {
		Name        string
		StateCount  int
		MaxDuration string // Expected reasonable max duration
	}{
		{"Tiny", 10, "1ms"},
		{"Small", 50, "10ms"},
		{"Medium", 200, "50ms"},
		{"Large", 1000, "500ms"},
	}

	for _, test := range scalabilityTests {
		t.Run(test.Name, func(t *testing.T) {
			statechart := generateStatechartOfSize(test.StateCount, 3)

			// Test XState conversion scalability
			t.Run("XState", func(t *testing.T) {
				converter := xstate.NewConverter()
				
				machine, err := converter.Export(statechart)
				if err != nil {
					t.Fatalf("Export failed for %d states: %v", test.StateCount, err)
				}

				_, err = converter.Import(machine)
				if err != nil {
					t.Fatalf("Import failed for %d states: %v", test.StateCount, err)
				}
			})

			// Test SCXML conversion scalability
			t.Run("SCXML", func(t *testing.T) {
				converter := scxml.NewConverter()
				
				doc, err := converter.Export(statechart)
				if err != nil {
					t.Fatalf("Export failed for %d states: %v", test.StateCount, err)
				}

				_, err = converter.Import(doc)
				if err != nil {
					t.Fatalf("Import failed for %d states: %v", test.StateCount, err)
				}
			})
		})
	}
}

// Helper function to generate statecharts of specific sizes for testing.
func generateStatechartOfSize(stateCount, maxDepth int) *sc.Statechart {
	root := &sc.State{
		Label: "__root__",
		Type:  sc.StateTypeNormal,
	}

	var allStates []*sc.State
	var transitions []*sc.Transition
	var events []*sc.Event

	// Create states with hierarchical structure
	statesCreated := 0
	currentDepth := 0
	
	// Create a balanced hierarchy
	statesByDepth := map[int][]*sc.State{0: {root}}

	for currentDepth < maxDepth && statesCreated < stateCount {
		currentLevelStates := statesByDepth[currentDepth]
		nextLevelStates := []*sc.State{}

		for _, parent := range currentLevelStates {
			if statesCreated >= stateCount {
				break
			}

			// Add 2-4 children per parent
			childCount := 2 + (statesCreated % 3)
			for i := 0; i < childCount && statesCreated < stateCount; i++ {
				child := &sc.State{
					Label:     fmt.Sprintf("state_%d_%d_%d", currentDepth+1, len(parent.Children), i),
					Type:      sc.StateTypeBasic,
					IsInitial: i == 0, // First child is initial
				}

				// Sometimes make compound states
				if currentDepth < maxDepth-1 && statesCreated%7 == 0 {
					child.Type = sc.StateTypeNormal
				}

				parent.Children = append(parent.Children, child)
				if parent != root {
					allStates = append(allStates, child)
				}
				nextLevelStates = append(nextLevelStates, child)
				statesCreated++
			}

			// Make root state normal if it has children
			if parent == root && len(parent.Children) > 0 {
				parent.Type = sc.StateTypeNormal
			}
		}

		currentDepth++
		if len(nextLevelStates) > 0 {
			statesByDepth[currentDepth] = nextLevelStates
		}
	}

	// Create some transitions between states
	eventLabels := []string{"NEXT", "PREV", "RESET", "ERROR", "SUCCESS"}
	for i, event := range eventLabels {
		events = append(events, &sc.Event{Label: event})
		
		// Create transitions using this event
		transitionCount := min(statesCreated/5, 10) // Reasonable number of transitions
		for j := 0; j < transitionCount; j++ {
			sourceIdx := (i*transitionCount + j) % len(allStates)
			targetIdx := (sourceIdx + 1 + j) % len(allStates)
			
			if sourceIdx < len(allStates) && targetIdx < len(allStates) {
				transition := &sc.Transition{
					Label: fmt.Sprintf("t_%d_%s", j, event),
					From:  []string{allStates[sourceIdx].Label},
					To:    []string{allStates[targetIdx].Label},
					Event: event,
				}
				transitions = append(transitions, transition)
			}
		}
	}

	return &sc.Statechart{
		RootState:   root,
		Transitions: transitions,
		Events:      events,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}