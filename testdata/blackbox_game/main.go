// Package main implements a black box game for ML statechart inference.
//
// The game is a combination lock with hidden state:
// - Hidden: 3-digit combination (e.g., 7-2-5)
// - Hidden: Current position in sequence (0, 1, 2, or 3=solved)
// - Hidden: Whether each attempted digit was correct
//
// Events: DIAL_0 through DIAL_9, TRY_OPEN, RESET
// Outputs: CLICK (correct digit), BUZZ (wrong digit), OPEN (lock opens), LOCKED (try failed)
//
// The ML model must infer the hidden state machine from (event, output) traces.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"
)

// Event represents player input
type Event string

const (
	EventDial0   Event = "DIAL_0"
	EventDial1   Event = "DIAL_1"
	EventDial2   Event = "DIAL_2"
	EventDial3   Event = "DIAL_3"
	EventDial4   Event = "DIAL_4"
	EventDial5   Event = "DIAL_5"
	EventDial6   Event = "DIAL_6"
	EventDial7   Event = "DIAL_7"
	EventDial8   Event = "DIAL_8"
	EventDial9   Event = "DIAL_9"
	EventTryOpen Event = "TRY_OPEN"
	EventReset   Event = "RESET"
)

// AllDialEvents for convenience
var AllDialEvents = []Event{
	EventDial0, EventDial1, EventDial2, EventDial3, EventDial4,
	EventDial5, EventDial6, EventDial7, EventDial8, EventDial9,
}

// Output represents observable feedback
type Output string

const (
	OutputClick  Output = "CLICK"  // Correct digit entered
	OutputBuzz   Output = "BUZZ"   // Wrong digit entered
	OutputOpen   Output = "OPEN"   // Lock successfully opened
	OutputLocked Output = "LOCKED" // Try failed, lock remains closed
	OutputReset  Output = "RESET"  // Lock reset to initial state
)

// Step records one event-output pair
type Step struct {
	Event  Event  `json:"event"`
	Output Output `json:"output"`
}

// Trace records a complete game session
type Trace struct {
	Combination []int  `json:"combination"` // Ground truth (hidden from ML)
	Steps       []Step `json:"steps"`
	Solved      bool   `json:"solved"`
}

// Lock implements the hidden statechart
type Lock struct {
	combination []int // Hidden 3-digit code
	position    int   // Hidden: how many correct digits entered (0-3)
	attempts    []int // Hidden: digits entered so far
}

// NewLock creates a lock with random combination
func NewLock() *Lock {
	combo := make([]int, 3)
	for i := range combo {
		combo[i] = rand.Intn(10)
	}
	return &Lock{
		combination: combo,
		position:    0,
		attempts:    make([]int, 0, 3),
	}
}

// NewLockWithCombo creates a lock with specific combination
func NewLockWithCombo(combo []int) *Lock {
	return &Lock{
		combination: combo,
		position:    0,
		attempts:    make([]int, 0, 3),
	}
}

// Process handles an event and returns the output
func (l *Lock) Process(event Event) Output {
	switch event {
	case EventReset:
		l.position = 0
		l.attempts = l.attempts[:0]
		return OutputReset

	case EventTryOpen:
		if l.position == 3 {
			return OutputOpen
		}
		return OutputLocked

	default:
		// Dial events
		digit := eventToDigit(event)
		if digit < 0 {
			return OutputBuzz
		}

		// Check if this digit matches the expected position
		if l.position < 3 && digit == l.combination[l.position] {
			l.position++
			l.attempts = append(l.attempts, digit)
			return OutputClick
		}

		// Wrong digit - reset progress
		l.position = 0
		l.attempts = l.attempts[:0]
		return OutputBuzz
	}
}

// IsSolved returns whether the lock is open
func (l *Lock) IsSolved() bool {
	return l.position == 3
}

// Combination returns the hidden combination (for trace output)
func (l *Lock) Combination() []int {
	return l.combination
}

func eventToDigit(e Event) int {
	switch e {
	case EventDial0:
		return 0
	case EventDial1:
		return 1
	case EventDial2:
		return 2
	case EventDial3:
		return 3
	case EventDial4:
		return 4
	case EventDial5:
		return 5
	case EventDial6:
		return 6
	case EventDial7:
		return 7
	case EventDial8:
		return 8
	case EventDial9:
		return 9
	default:
		return -1
	}
}

func digitToEvent(d int) Event {
	return AllDialEvents[d]
}

// generateRandomTrace generates a trace with random exploration
func generateRandomTrace(lock *Lock, maxSteps int) Trace {
	trace := Trace{
		Combination: lock.Combination(),
		Steps:       make([]Step, 0, maxSteps),
	}

	for i := 0; i < maxSteps && !lock.IsSolved(); i++ {
		// Random event selection
		var event Event
		r := rand.Float32()
		if r < 0.8 {
			// 80% dial events
			event = AllDialEvents[rand.Intn(10)]
		} else if r < 0.95 {
			// 15% try open
			event = EventTryOpen
		} else {
			// 5% reset
			event = EventReset
		}

		output := lock.Process(event)
		trace.Steps = append(trace.Steps, Step{Event: event, Output: output})

		if output == OutputOpen {
			trace.Solved = true
			break
		}
	}

	return trace
}

// generateSmartTrace generates a trace that actually solves the lock
func generateSmartTrace(lock *Lock) Trace {
	trace := Trace{
		Combination: lock.Combination(),
		Steps:       make([]Step, 0, 10),
	}

	// Some random exploration first
	numExplore := rand.Intn(5)
	for i := 0; i < numExplore; i++ {
		event := AllDialEvents[rand.Intn(10)]
		output := lock.Process(event)
		trace.Steps = append(trace.Steps, Step{Event: event, Output: output})
	}

	// Reset and solve correctly
	output := lock.Process(EventReset)
	trace.Steps = append(trace.Steps, Step{Event: EventReset, Output: output})

	// Enter correct combination
	for _, digit := range lock.Combination() {
		event := digitToEvent(digit)
		output := lock.Process(event)
		trace.Steps = append(trace.Steps, Step{Event: event, Output: output})
	}

	// Open the lock
	output = lock.Process(EventTryOpen)
	trace.Steps = append(trace.Steps, Step{Event: EventTryOpen, Output: output})
	trace.Solved = output == OutputOpen

	return trace
}

// generateExplorationTrace generates trace showing systematic exploration
func generateExplorationTrace(lock *Lock) Trace {
	trace := Trace{
		Combination: lock.Combination(),
		Steps:       make([]Step, 0, 50),
	}

	// Try to discover the combination through trial and error
	for pos := 0; pos < 3; pos++ {
		// Try each digit until we get a CLICK
		for digit := 0; digit < 10; digit++ {
			if len(trace.Steps) > 100 {
				break
			}

			event := digitToEvent(digit)
			output := lock.Process(event)
			trace.Steps = append(trace.Steps, Step{Event: event, Output: output})

			if output == OutputClick {
				// Found correct digit for this position
				break
			}
			// Wrong digit resets, so we need to re-enter previous correct digits
			for i := 0; i < pos; i++ {
				event := digitToEvent(lock.Combination()[i])
				output := lock.Process(event)
				trace.Steps = append(trace.Steps, Step{Event: event, Output: output})
			}
		}
	}

	// Try to open
	output := lock.Process(EventTryOpen)
	trace.Steps = append(trace.Steps, Step{Event: EventTryOpen, Output: output})
	trace.Solved = output == OutputOpen

	return trace
}

func main() {
	var (
		numTraces    = flag.Int("traces", 100, "Number of traces to generate")
		maxSteps     = flag.Int("max-steps", 50, "Maximum steps per random trace")
		outputFile   = flag.String("output", "traces.json", "Output JSON file")
		traceType    = flag.String("type", "mixed", "Trace type: random, smart, explore, mixed")
		seed         = flag.Int64("seed", 0, "Random seed (0 for time-based)")
		interactive  = flag.Bool("interactive", false, "Run interactive mode")
		showSolution = flag.Bool("show-solution", false, "Show the lock combination")
	)
	flag.Parse()

	// Seed random
	if *seed == 0 {
		rand.Seed(time.Now().UnixNano())
	} else {
		rand.Seed(*seed)
	}

	if *interactive {
		runInteractive(*showSolution)
		return
	}

	// Generate traces
	traces := make([]Trace, 0, *numTraces)

	for i := 0; i < *numTraces; i++ {
		lock := NewLock()

		var trace Trace
		switch *traceType {
		case "random":
			trace = generateRandomTrace(lock, *maxSteps)
		case "smart":
			trace = generateSmartTrace(lock)
		case "explore":
			trace = generateExplorationTrace(lock)
		case "mixed":
			r := rand.Float32()
			if r < 0.4 {
				trace = generateRandomTrace(lock, *maxSteps)
			} else if r < 0.7 {
				trace = generateSmartTrace(lock)
			} else {
				trace = generateExplorationTrace(lock)
			}
		default:
			fmt.Fprintf(os.Stderr, "Unknown trace type: %s\n", *traceType)
			os.Exit(1)
		}

		traces = append(traces, trace)
	}

	// Output statistics
	solved := 0
	totalSteps := 0
	for _, t := range traces {
		if t.Solved {
			solved++
		}
		totalSteps += len(t.Steps)
	}

	fmt.Printf("Generated %d traces\n", len(traces))
	fmt.Printf("  Solved: %d (%.1f%%)\n", solved, float64(solved)/float64(len(traces))*100)
	fmt.Printf("  Avg steps: %.1f\n", float64(totalSteps)/float64(len(traces)))

	// Write to file
	data, err := json.MarshalIndent(traces, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*outputFile, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Wrote traces to %s\n", *outputFile)
}

func runInteractive(showSolution bool) {
	lock := NewLock()

	if showSolution {
		fmt.Printf("(Hidden combination: %v)\n", lock.Combination())
	}

	fmt.Println("Combination Lock Game")
	fmt.Println("Commands: 0-9 (dial), o (try open), r (reset), q (quit)")
	fmt.Println()

	var input string
	for {
		fmt.Print("> ")
		fmt.Scanln(&input)

		var event Event
		switch input {
		case "0", "1", "2", "3", "4", "5", "6", "7", "8", "9":
			digit := int(input[0] - '0')
			event = digitToEvent(digit)
		case "o":
			event = EventTryOpen
		case "r":
			event = EventReset
		case "q":
			fmt.Println("Goodbye!")
			return
		default:
			fmt.Println("Unknown command. Use 0-9, o, r, or q.")
			continue
		}

		output := lock.Process(event)
		fmt.Printf("  -> %s\n", output)

		if output == OutputOpen {
			fmt.Println("\nCongratulations! You opened the lock!")
			return
		}
	}
}
