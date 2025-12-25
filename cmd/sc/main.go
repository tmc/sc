// Command sc provides a CLI tool for operating on statechart definition files.
//
// It supports reading from stdin (for Unix pipelines), files, and txtar archives.
//
// Usage:
//
//	sc <command> [flags] [file]
//
// Commands:
//
//	validate  - Check if statechart is well-formed
//	info      - Show summary information about the statechart
//	states    - List all states (outputs one per line)
//	events    - List all events (outputs one per line)
//	mermaid   - Generate Mermaid diagram
//	step      - Send event(s) to machine and output resulting state
//	dot       - Generate Graphviz DOT format
//
// Examples:
//
//	# Validate a statechart
//	sc validate chart.json
//
//	# Pipeline: generate mermaid from stdin
//	cat chart.json | sc mermaid
//
//	# List states, filter with grep
//	sc states chart.json | grep -i error
//
//	# Send events and trace state changes
//	sc step -e POWER_ON -e ARM chart.json
//
//	# Use txtar for complex definitions
//	sc validate definition.txtar
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/sc"
	"github.com/tmc/sc/internal/xstate"
	"github.com/tmc/sc/semantics/v1"
	"golang.org/x/tools/txtar"
	"google.golang.org/protobuf/encoding/protojson"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "sc: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return usage(stderr)
	}

	cmd := args[0]
	args = args[1:]

	switch cmd {
	case "validate":
		return cmdValidate(args, stdin, stdout, stderr)
	case "info":
		return cmdInfo(args, stdin, stdout, stderr)
	case "states":
		return cmdStates(args, stdin, stdout, stderr)
	case "events":
		return cmdEvents(args, stdin, stdout, stderr)
	case "mermaid":
		return cmdMermaid(args, stdin, stdout, stderr)
	case "step":
		return cmdStep(args, stdin, stdout, stderr)
	case "dot":
		return cmdDot(args, stdin, stdout, stderr)
	case "export":
		return cmdExport(args, stdin, stdout, stderr)
	case "import":
		return cmdImport(args, stdin, stdout, stderr)
	case "help", "-h", "--help":
		return usage(stderr)
	default:
		return fmt.Errorf("unknown command: %s", cmd)
	}
}

func usage(w io.Writer) error {
	fmt.Fprintln(w, `sc - statechart operations tool

Usage: sc <command> [flags] [file]

Commands:
  validate  Check if statechart is well-formed
  info      Show summary information
  states    List all states (one per line)
  events    List all events (one per line)
  mermaid   Generate Mermaid diagram
  step      Send events and output resulting configuration
  dot       Generate Graphviz DOT format
  export    Export in various formats (json, ct, xstate)
  import    Import from external formats (xstate)

Input:
  - If no file specified, reads from stdin
  - Supports JSON, and txtar formats
  - txtar files allow multi-file definitions with overlays

Examples:
  sc validate chart.json
  cat chart.json | sc mermaid
  sc states chart.json | grep Error
  sc step -e POWER_ON -e ARM chart.json
  sc info definition.txtar
  sc import -format xstate machines.json
  sc export -format xstate chart.json`)
	return nil
}

// loadInput reads a statechart from file or stdin.
// Supports JSON and txtar formats.
func loadInput(args []string, stdin io.Reader) (*sc.Machine, *sc.Statechart, error) {
	var data []byte
	var filename string
	var err error

	if len(args) > 0 && args[0] != "-" {
		filename = args[0]
		data, err = os.ReadFile(filename)
		if err != nil {
			return nil, nil, fmt.Errorf("read file: %w", err)
		}
	} else {
		filename = "<stdin>"
		data, err = io.ReadAll(stdin)
		if err != nil {
			return nil, nil, fmt.Errorf("read stdin: %w", err)
		}
	}

	// Try txtar first if it looks like txtar (by extension or content)
	if isTxtarFile(filename) || isTxtar(data) {
		return loadTxtar(data, filename)
	}

	// Try JSON
	return loadJSON(data)
}

// isTxtar checks if data looks like a txtar archive
func isTxtar(data []byte) bool {
	// txtar files have lines starting with "-- filename --"
	// Check if file extension is .txtar
	// Or check content for file markers
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if strings.HasPrefix(line, "-- ") && strings.HasSuffix(line, " --") {
			return true
		}
		// Only check first 50 lines for the marker
		if lineNum > 50 {
			break
		}
	}
	return false
}

// isTxtarFile checks if a filename looks like a txtar file
func isTxtarFile(filename string) bool {
	return strings.HasSuffix(filename, ".txtar")
}

// loadTxtar loads a statechart from a txtar archive.
// The archive should contain a base chart.json, with optional overlays.
func loadTxtar(data []byte, filename string) (*sc.Machine, *sc.Statechart, error) {
	ar := txtar.Parse(data)

	// Find the main chart file
	var chartData []byte
	for _, f := range ar.Files {
		name := filepath.Base(f.Name)
		if name == "chart.json" || name == "statechart.json" || name == "machine.json" {
			chartData = f.Data
			break
		}
		// Also accept first .json file
		if strings.HasSuffix(f.Name, ".json") && chartData == nil {
			chartData = f.Data
		}
	}

	if chartData == nil {
		return nil, nil, fmt.Errorf("txtar archive has no chart.json or .json file")
	}

	machine, chart, err := loadJSON(chartData)
	if err != nil {
		return nil, nil, fmt.Errorf("parse chart in txtar: %w", err)
	}

	// Apply overlays (additional transition/state files)
	for _, f := range ar.Files {
		if strings.HasSuffix(f.Name, ".overlay.json") {
			if err := applyOverlay(chart, f.Data); err != nil {
				return nil, nil, fmt.Errorf("apply overlay %s: %w", f.Name, err)
			}
		}
	}

	return machine, chart, nil
}

// applyOverlay merges overlay data into the statechart
func applyOverlay(chart *sc.Statechart, data []byte) error {
	var overlay struct {
		Transitions []*sc.Transition `json:"transitions"`
		Events      []*sc.Event      `json:"events"`
	}
	if err := json.Unmarshal(data, &overlay); err != nil {
		return err
	}

	chart.Transitions = append(chart.Transitions, overlay.Transitions...)
	chart.Events = append(chart.Events, overlay.Events...)
	return nil
}

// loadJSON loads a statechart from JSON data.
// Handles both Machine format (with configuration) and raw Statechart format.
func loadJSON(data []byte) (*sc.Machine, *sc.Statechart, error) {
	// Try Machine format first
	machine := &sc.Machine{}
	opts := protojson.UnmarshalOptions{DiscardUnknown: true}
	if err := opts.Unmarshal(data, machine); err == nil && machine.Statechart != nil {
		return machine, machine.Statechart, nil
	}

	// Try raw Statechart format
	chart := &sc.Statechart{}
	if err := opts.Unmarshal(data, chart); err == nil && chart.RootState != nil {
		return nil, chart, nil
	}

	return nil, nil, fmt.Errorf("invalid JSON: not a Machine or Statechart")
}

// cmdValidate validates a statechart
func cmdValidate(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	_, chart, err := loadInput(args, stdin)
	if err != nil {
		return err
	}

	wrapper := semantics.NewStatechart(chart)
	if err := wrapper.Validate(); err != nil {
		fmt.Fprintf(stdout, "INVALID: %v\n", err)
		return nil
	}

	fmt.Fprintln(stdout, "VALID")
	return nil
}

// cmdInfo shows summary information
func cmdInfo(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	machine, chart, err := loadInput(args, stdin)
	if err != nil {
		return err
	}

	wrapper := semantics.NewStatechart(chart)

	// Count states
	stateCount := 0
	leafCount := 0
	parallelCount := 0
	var countStates func(*sc.State)
	countStates = func(s *sc.State) {
		if s == nil {
			return
		}
		stateCount++
		if len(s.Children) == 0 {
			leafCount++
		}
		if s.Type == sc.StateTypeParallel {
			parallelCount++
		}
		for _, child := range s.Children {
			countStates(child)
		}
	}
	countStates(chart.RootState)

	fmt.Fprintf(stdout, "Name:        %s\n", chart.Name)
	fmt.Fprintf(stdout, "States:      %d total, %d leaf, %d parallel\n", stateCount, leafCount, parallelCount)
	fmt.Fprintf(stdout, "Transitions: %d\n", len(chart.Transitions))
	fmt.Fprintf(stdout, "Events:      %d\n", len(chart.Events))

	// Validation status
	if err := wrapper.Validate(); err != nil {
		fmt.Fprintf(stdout, "Valid:       no (%v)\n", err)
	} else {
		fmt.Fprintf(stdout, "Valid:       yes\n")
	}

	// Machine state if present
	if machine != nil {
		fmt.Fprintf(stdout, "Machine ID:  %s\n", machine.Id)
		fmt.Fprintf(stdout, "State:       %s\n", machine.State.String())
		if machine.Configuration != nil {
			var labels []string
			for _, s := range machine.Configuration.States {
				labels = append(labels, s.Label)
			}
			fmt.Fprintf(stdout, "Config:      %s\n", strings.Join(labels, ", "))
		}
	}

	return nil
}

// cmdStates lists all states (one per line for easy piping)
func cmdStates(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("states", flag.ContinueOnError)
	showType := fs.Bool("t", false, "show state type")
	leafOnly := fs.Bool("leaf", false, "only show leaf states")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	var printState func(*sc.State, int)
	printState = func(s *sc.State, depth int) {
		if s == nil {
			return
		}

		isLeaf := len(s.Children) == 0
		if *leafOnly && !isLeaf {
			// Skip non-leaf, but still recurse
			for _, child := range s.Children {
				printState(child, depth+1)
			}
			return
		}

		if *showType {
			typeStr := stateTypeString(s.Type)
			fmt.Fprintf(stdout, "%s\t%s\n", s.Label, typeStr)
		} else {
			fmt.Fprintln(stdout, s.Label)
		}

		for _, child := range s.Children {
			printState(child, depth+1)
		}
	}

	printState(chart.RootState, 0)
	return nil
}

func stateTypeString(t sc.StateType) string {
	switch t {
	case sc.StateTypeBasic:
		return "basic"
	case sc.StateTypeOR:
		return "or"
	case sc.StateTypeAND:
		return "parallel"
	default:
		return "unknown"
	}
}

// cmdEvents lists all events (one per line)
func cmdEvents(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	_, chart, err := loadInput(args, stdin)
	if err != nil {
		return err
	}

	seen := make(map[string]bool)

	// From events list
	for _, e := range chart.Events {
		if !seen[e.Label] {
			fmt.Fprintln(stdout, e.Label)
			seen[e.Label] = true
		}
	}

	// From transitions (in case events list is incomplete)
	for _, t := range chart.Transitions {
		if t.Event != "" && !seen[t.Event] {
			fmt.Fprintln(stdout, t.Event)
			seen[t.Event] = true
		}
	}

	return nil
}

// cmdMermaid generates a Mermaid statechart diagram
func cmdMermaid(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("mermaid", flag.ContinueOnError)
	direction := fs.String("dir", "TB", "diagram direction: TB, LR, BT, RL")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	fmt.Fprintf(stdout, "stateDiagram-v2\n")
	fmt.Fprintf(stdout, "  direction %s\n", *direction)
	fmt.Fprintln(stdout)

	// Generate states
	generateMermaidStates(stdout, chart.RootState, "  ")

	// Generate transitions
	for _, t := range chart.Transitions {
		for _, from := range t.From {
			for _, to := range t.To {
				label := t.Event
				if t.Guard != nil && t.Guard.Expression != "" {
					label = fmt.Sprintf("%s [%s]", label, sanitizeLabel(t.Guard.Expression))
				}
				if label != "" {
					fmt.Fprintf(stdout, "  %s --> %s : %s\n", sanitizeID(from), sanitizeID(to), sanitizeLabel(label))
				} else {
					fmt.Fprintf(stdout, "  %s --> %s\n", sanitizeID(from), sanitizeID(to))
				}
			}
		}
	}

	return nil
}

func generateMermaidStates(w io.Writer, s *sc.State, indent string) {
	if s == nil || s.Label == "__root__" {
		// Process children of root directly
		for _, child := range s.Children {
			generateMermaidStates(w, child, indent)
		}
		return
	}

	id := sanitizeID(s.Label)

	if len(s.Children) > 0 {
		// Composite state
		fmt.Fprintf(w, "%sstate %s {\n", indent, id)
		for _, child := range s.Children {
			generateMermaidStates(w, child, indent+"  ")
		}
		// Mark initial state
		for _, child := range s.Children {
			if child.IsInitial {
				fmt.Fprintf(w, "%s  [*] --> %s\n", indent, sanitizeID(child.Label))
				break
			}
		}
		fmt.Fprintf(w, "%s}\n", indent)
	} else {
		// Leaf state - just declare if needed for clarity
		if s.IsFinal {
			fmt.Fprintf(w, "%s%s --> [*]\n", indent, id)
		}
	}

	if s.IsInitial && s.Label != "__root__" {
		// This is handled by parent
	}
}

func sanitizeID(s string) string {
	// Mermaid doesn't like spaces or special chars in IDs
	return strings.ReplaceAll(strings.ReplaceAll(s, " ", "_"), "-", "_")
}

func sanitizeLabel(s string) string {
	// Mermaid transition labels can't contain certain characters
	// Replace problematic chars and truncate long expressions
	s = strings.ReplaceAll(s, "&&", " AND ")
	s = strings.ReplaceAll(s, "||", " OR ")
	s = strings.ReplaceAll(s, "&", " and ")
	s = strings.ReplaceAll(s, "|", " or ")
	s = strings.ReplaceAll(s, "<", " lt ")
	s = strings.ReplaceAll(s, ">", " gt ")
	s = strings.ReplaceAll(s, "\"", "'")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")
	s = strings.ReplaceAll(s, "#", "")
	s = strings.ReplaceAll(s, ";", "")
	// Truncate very long expressions
	if len(s) > 80 {
		s = s[:77] + "..."
	}
	return s
}

// cmdStep sends events to the machine and outputs resulting configuration
func cmdStep(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("step", flag.ContinueOnError)
	var events eventSlice
	fs.Var(&events, "e", "event to send (can be repeated)")
	verbose := fs.Bool("v", false, "verbose output showing each step")
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	machine, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	// Create a machine wrapper
	wrapper := semantics.NewStatechart(chart)

	var mw *semantics.MachineWrapper
	if machine != nil && machine.Id != "" {
		mw, err = semantics.NewMachine(wrapper, machine.Id, machine.Context)
	} else {
		mw, err = semantics.NewMachine(wrapper, "cli-machine", nil)
	}
	if err != nil {
		return fmt.Errorf("create machine: %w", err)
	}

	if err := mw.Start(); err != nil {
		return fmt.Errorf("start machine: %w", err)
	}

	// Process each event
	for _, event := range events {
		if *verbose {
			fmt.Fprintf(stderr, ">>> %s\n", event)
		}

		transitioned, err := mw.Step(event)
		if err != nil {
			return fmt.Errorf("step %s: %w", event, err)
		}

		if *verbose {
			config := mw.GetCurrentConfiguration()
			var labels []string
			for _, s := range config.States {
				labels = append(labels, s.Label)
			}
			if transitioned {
				fmt.Fprintf(stderr, "    -> %s\n", strings.Join(labels, ", "))
			} else {
				fmt.Fprintf(stderr, "    (no transition)\n")
			}
		}
	}

	// Output final configuration
	config := mw.GetCurrentConfiguration()
	if *outputJSON {
		out := struct {
			Configuration []string `json:"configuration"`
		}{}
		for _, s := range config.States {
			out.Configuration = append(out.Configuration, s.Label)
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	// Plain text: one state per line
	for _, s := range config.States {
		fmt.Fprintln(stdout, s.Label)
	}

	return nil
}

// eventSlice is a flag type for collecting multiple -e flags
type eventSlice []string

func (e *eventSlice) String() string {
	return strings.Join(*e, ",")
}

func (e *eventSlice) Set(value string) error {
	*e = append(*e, value)
	return nil
}

// cmdDot generates Graphviz DOT format
func cmdDot(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("dot", flag.ContinueOnError)
	rankdir := fs.String("rankdir", "TB", "graph direction: TB, LR, BT, RL")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	fmt.Fprintln(stdout, "digraph statechart {")
	fmt.Fprintf(stdout, "  rankdir=%s;\n", *rankdir)
	fmt.Fprintln(stdout, "  node [shape=box, style=rounded];")
	fmt.Fprintln(stdout)

	// Generate nodes
	var genNodes func(*sc.State, string)
	genNodes = func(s *sc.State, parent string) {
		if s == nil {
			return
		}
		id := sanitizeID(s.Label)
		if s.Label != "__root__" {
			attrs := ""
			if s.IsInitial {
				attrs = ", style=\"rounded,bold\""
			}
			if s.Type == sc.StateTypeParallel {
				attrs = ", shape=box, style=\"rounded,dashed\""
			}
			fmt.Fprintf(stdout, "  %s [label=\"%s\"%s];\n", id, s.Label, attrs)
		}
		for _, child := range s.Children {
			genNodes(child, id)
		}
	}
	genNodes(chart.RootState, "")

	fmt.Fprintln(stdout)

	// Generate edges
	for _, t := range chart.Transitions {
		for _, from := range t.From {
			for _, to := range t.To {
				label := t.Event
				if t.Guard != nil && t.Guard.Expression != "" {
					label = fmt.Sprintf("%s\\n[%s]", label, t.Guard.Expression)
				}
				fmt.Fprintf(stdout, "  %s -> %s [label=\"%s\"];\n", sanitizeID(from), sanitizeID(to), label)
			}
		}
	}

	fmt.Fprintln(stdout, "}")
	return nil
}

// cmdExport exports the statechart in various formats
func cmdExport(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	format := fs.String("format", "json", "output format: json, ct, xstate")
	version := fs.String("version", "", "version label (for ct format)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	switch *format {
	case "json":
		opts := protojson.MarshalOptions{
			Multiline: true,
			Indent:    "  ",
		}
		data, err := opts.Marshal(chart)
		if err != nil {
			return fmt.Errorf("marshal json: %w", err)
		}
		fmt.Fprintln(stdout, string(data))
		return nil

	case "ct":
		return exportCT(chart, *version, stdout)

	case "xstate":
		data, err := xstate.ExportJSON(chart)
		if err != nil {
			return fmt.Errorf("export xstate: %w", err)
		}
		fmt.Fprintln(stdout, string(data))
		return nil

	default:
		return fmt.Errorf("unknown format: %s (supported: json, ct, xstate)", *format)
	}
}

// cmdImport imports statecharts from external formats
func cmdImport(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("import", flag.ContinueOnError)
	format := fs.String("format", "xstate", "input format: xstate")
	machine := fs.String("machine", "", "machine name to extract (for multi-machine files)")
	all := fs.Bool("all", false, "import all machines (outputs JSON array)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	// Read input data
	var data []byte
	var err error
	if len(fs.Args()) > 0 && fs.Args()[0] != "-" {
		data, err = os.ReadFile(fs.Args()[0])
		if err != nil {
			return fmt.Errorf("read file: %w", err)
		}
	} else {
		data, err = io.ReadAll(stdin)
		if err != nil {
			return fmt.Errorf("read stdin: %w", err)
		}
	}

	switch *format {
	case "xstate":
		if *all {
			charts, err := xstate.ImportAll(data)
			if err != nil {
				return fmt.Errorf("import xstate: %w", err)
			}
			// Output as JSON array
			opts := protojson.MarshalOptions{
				Multiline: true,
				Indent:    "  ",
			}
			fmt.Fprintln(stdout, "[")
			for i, chart := range charts {
				out, err := opts.Marshal(chart)
				if err != nil {
					return fmt.Errorf("marshal chart %d: %w", i, err)
				}
				if i > 0 {
					fmt.Fprintln(stdout, ",")
				}
				fmt.Fprint(stdout, string(out))
			}
			fmt.Fprintln(stdout, "\n]")
			return nil
		}

		chart, err := xstate.Import(data, *machine)
		if err != nil {
			return fmt.Errorf("import xstate: %w", err)
		}
		opts := protojson.MarshalOptions{
			Multiline: true,
			Indent:    "  ",
		}
		out, err := opts.Marshal(chart)
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}
		fmt.Fprintln(stdout, string(out))
		return nil

	default:
		return fmt.Errorf("unknown format: %s (supported: xstate)", *format)
	}
}

// CTExport represents the category theory export format
type CTExport struct {
	Category       string            `json:"category"`
	Version        *CTVersion        `json:"version,omitempty"`
	Objects        []CTObject        `json:"objects"`
	Morphisms      []CTMorphism      `json:"morphisms"`
	Products       []CTProduct       `json:"products,omitempty"`
	Coproducts     []CTCoproduct     `json:"coproducts,omitempty"`
	Comonads       []CTComonad       `json:"comonads,omitempty"`
	Initial        string            `json:"initial,omitempty"`         // Simple initial (non-parallel root)
	Initials       map[string]string `json:"initials,omitempty"`        // Per-region initials (parallel)
	Final          []string          `json:"final,omitempty"`
	TraceCoalgebra *CTTraceCoalgebra `json:"trace_coalgebra,omitempty"`
}

type CTVersion struct {
	Hash   string `json:"hash"`
	Semver string `json:"semver,omitempty"`
}

type CTObject struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Children []string `json:"children,omitempty"`
	Parent   string   `json:"parent,omitempty"`
}

type CTMorphism struct {
	ID      string            `json:"id"`
	Dom     string            `json:"dom"`
	Cod     string            `json:"cod"`
	Label   CTMorphismLabel   `json:"label"`
	Actions []string          `json:"actions,omitempty"`
}

type CTMorphismLabel struct {
	Event string `json:"event,omitempty"`
	Guard string `json:"guard,omitempty"`
}

type CTProduct struct {
	ID      string   `json:"id"`
	Factors []string `json:"factors"`
}

type CTCoproduct struct {
	ID       string   `json:"id"`
	Summands []string `json:"summands"`
}

type CTComonad struct {
	ID    string `json:"id"`
	Type  string `json:"type"` // shallow, deep
	Scope string `json:"scope"`
}

type CTTraceCoalgebra struct {
	StepFunctor string `json:"step_functor"`
}

func exportCT(chart *sc.Statechart, version string, w io.Writer) error {
	export := &CTExport{
		Category: chart.Name,
		TraceCoalgebra: &CTTraceCoalgebra{
			StepFunctor: "Event -> (Config × Trace) + 1",
		},
	}

	// Compute content hash
	opts := protojson.MarshalOptions{Multiline: false}
	data, _ := opts.Marshal(chart)
	hash := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	export.Version = &CTVersion{
		Hash:   hash,
		Semver: version,
	}

	// Collect objects (states)
	var initial string
	initials := make(map[string]string) // region -> initial state
	var finals []string
	var products []CTProduct
	var coproducts []CTCoproduct
	var comonads []CTComonad

	var collectStates func(*sc.State, string, bool)
	collectStates = func(s *sc.State, parent string, inParallel bool) {
		if s == nil {
			return
		}

		// Skip root, but process children
		if s.Label == "__root__" {
			for _, child := range s.Children {
				collectStates(child, "", false)
			}
			return
		}

		// Track if we're entering a parallel state
		isParallel := s.Type == sc.StateTypeAND

		// Determine type
		stateType := "basic"
		if s.IsHistory {
			// Handle history pseudostates
			switch s.HistoryType {
			case sc.HistoryType_HISTORY_TYPE_SHALLOW:
				stateType = "history_shallow"
				comonads = append(comonads, CTComonad{
					ID:    s.Label,
					Type:  "shallow",
					Scope: parent,
				})
			case sc.HistoryType_HISTORY_TYPE_DEEP:
				stateType = "history_deep"
				comonads = append(comonads, CTComonad{
					ID:    s.Label,
					Type:  "deep",
					Scope: parent,
				})
			}
		} else {
			switch s.Type {
			case sc.StateTypeOR:
				stateType = "or"
			case sc.StateTypeAND:
				stateType = "parallel"
			}
		}

		// Collect children IDs
		var childIDs []string
		for _, child := range s.Children {
			childIDs = append(childIDs, child.Label)
		}

		obj := CTObject{
			ID:       s.Label,
			Type:     stateType,
			Children: childIDs,
			Parent:   parent,
		}
		export.Objects = append(export.Objects, obj)

		// Track initial/final
		if s.IsInitial {
			if inParallel && parent != "" {
				// In a parallel region, track per-region initial
				initials[parent] = s.Label
			} else {
				initial = s.Label
			}
		}
		if s.IsFinal {
			finals = append(finals, s.Label)
		}

		// Parallel states are products
		if s.Type == sc.StateTypeAND && len(s.Children) > 0 {
			products = append(products, CTProduct{
				ID:      s.Label,
				Factors: childIDs,
			})
		}

		// OR states are coproducts
		if s.Type == sc.StateTypeOR && len(s.Children) > 0 {
			coproducts = append(coproducts, CTCoproduct{
				ID:       s.Label,
				Summands: childIDs,
			})
		}

		// Recurse - children of parallel states are "in parallel"
		for _, child := range s.Children {
			collectStates(child, s.Label, isParallel || inParallel)
		}
	}
	collectStates(chart.RootState, "", false)

	export.Initial = initial
	if len(initials) > 0 {
		export.Initials = initials
	}
	export.Final = finals
	export.Products = products
	export.Coproducts = coproducts
	export.Comonads = comonads

	// Collect morphisms (transitions)
	for i, t := range chart.Transitions {
		// Handle multiple from/to as separate morphisms
		for _, from := range t.From {
			for _, to := range t.To {
				label := CTMorphismLabel{
					Event: t.Event,
				}
				if t.Guard != nil && t.Guard.Expression != "" {
					label.Guard = t.Guard.Expression
				}

				var actions []string
				for _, a := range t.Actions {
					actions = append(actions, a.Label)
				}

				morph := CTMorphism{
					ID:      fmt.Sprintf("t%d", i),
					Dom:     from,
					Cod:     to,
					Label:   label,
					Actions: actions,
				}
				export.Morphisms = append(export.Morphisms, morph)
			}
		}
	}

	// Output JSON
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(export)
}
