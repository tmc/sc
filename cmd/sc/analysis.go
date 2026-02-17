package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	sc "github.com/tmc/sc/gen/statecharts/v1"
	"github.com/tmc/sc/analysis"
)

// cmdTransitions lists transitions with optional filtering.
func cmdTransitions(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("transitions", flag.ContinueOnError)
	from := fs.String("from", "", "filter by source state")
	to := fs.String("to", "", "filter by target state")
	event := fs.String("event", "", "filter by event")
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)
	transitions := g.FilterTransitions(*from, *to, *event)

	if *outputJSON {
		type jsonTransition struct {
			From  string `json:"from"`
			To    string `json:"to"`
			Event string `json:"event,omitempty"`
			Guard string `json:"guard,omitempty"`
		}
		out := struct {
			Transitions []jsonTransition `json:"transitions"`
		}{}
		for _, t := range transitions {
			out.Transitions = append(out.Transitions, jsonTransition{
				From:  t.From,
				To:    t.To,
				Event: t.Event,
				Guard: t.Guard,
			})
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	for _, t := range transitions {
		if t.Guard != "" {
			fmt.Fprintf(stdout, "%s -> %s [%s] {%s}\n", t.From, t.To, t.Event, t.Guard)
		} else if t.Event != "" {
			fmt.Fprintf(stdout, "%s -> %s [%s]\n", t.From, t.To, t.Event)
		} else {
			fmt.Fprintf(stdout, "%s -> %s\n", t.From, t.To)
		}
	}
	return nil
}

// cmdOrphans lists states with no incoming transitions.
func cmdOrphans(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("orphans", flag.ContinueOnError)
	includeInitial := fs.Bool("include-initial", false, "include initial states")
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)
	orphans := g.Orphans(*includeInitial)

	if *outputJSON {
		out := struct {
			Orphans []string `json:"orphans"`
		}{Orphans: orphans}
		if out.Orphans == nil {
			out.Orphans = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	for _, s := range orphans {
		fmt.Fprintln(stdout, s)
	}
	return nil
}

// cmdSinks lists states with no outgoing transitions.
func cmdSinks(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("sinks", flag.ContinueOnError)
	includeFinal := fs.Bool("include-final", false, "include final states")
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)
	sinks := g.Sinks(*includeFinal)

	if *outputJSON {
		out := struct {
			Sinks []string `json:"sinks"`
		}{Sinks: sinks}
		if out.Sinks == nil {
			out.Sinks = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	for _, s := range sinks {
		fmt.Fprintln(stdout, s)
	}
	return nil
}

// cmdDisconnected lists states with neither incoming nor outgoing transitions.
func cmdDisconnected(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("disconnected", flag.ContinueOnError)
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)
	disconnected := g.Disconnected()

	if *outputJSON {
		out := struct {
			Disconnected []string `json:"disconnected"`
		}{Disconnected: disconnected}
		if out.Disconnected == nil {
			out.Disconnected = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	for _, s := range disconnected {
		fmt.Fprintln(stdout, s)
	}
	return nil
}

// cmdReachable lists states reachable from the initial configuration.
func cmdReachable(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("reachable", flag.ContinueOnError)
	from := fs.String("from", "", "override start state")
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)

	var reachable []string
	if *from != "" {
		reachable = g.Reachable([]string{*from})
	} else {
		reachable = g.ReachableFromInitial()
	}

	if *outputJSON {
		out := struct {
			Reachable []string `json:"reachable"`
		}{Reachable: reachable}
		if out.Reachable == nil {
			out.Reachable = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	for _, s := range reachable {
		fmt.Fprintln(stdout, s)
	}
	return nil
}

// cmdUnreachable lists states not reachable from the initial configuration.
func cmdUnreachable(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("unreachable", flag.ContinueOnError)
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)
	unreachable := g.Unreachable()

	if *outputJSON {
		out := struct {
			Unreachable []string `json:"unreachable"`
		}{Unreachable: unreachable}
		if out.Unreachable == nil {
			out.Unreachable = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	for _, s := range unreachable {
		fmt.Fprintln(stdout, s)
	}
	return nil
}

// cmdPath finds a path between two states.
func cmdPath(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("path", flag.ContinueOnError)
	all := fs.Bool("all", false, "show all paths (up to limit)")
	maxDepth := fs.Int("max-depth", 10, "maximum path depth for --all")
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	remaining := fs.Args()
	if len(remaining) < 2 {
		return fmt.Errorf("usage: sc path <from> <to> [file]")
	}

	from := remaining[0]
	to := remaining[1]
	fileArgs := remaining[2:]

	_, chart, err := loadInput(fileArgs, stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)

	if *all {
		paths := g.FindAllPaths(from, to, *maxDepth)
		if *outputJSON {
			type jsonPath struct {
				States []string `json:"states"`
				Length int      `json:"length"`
			}
			out := struct {
				Paths []jsonPath `json:"paths"`
			}{}
			for _, p := range paths {
				out.Paths = append(out.Paths, jsonPath{
					States: p.States,
					Length: p.Length(),
				})
			}
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(out)
		}

		for i, p := range paths {
			fmt.Fprintf(stdout, "Path %d: %s\n", i+1, formatPath(p))
		}
		if len(paths) == 0 {
			fmt.Fprintf(stderr, "No path from %s to %s\n", from, to)
		}
		return nil
	}

	path := g.FindPath(from, to)
	if path == nil {
		fmt.Fprintf(stderr, "No path from %s to %s\n", from, to)
		return nil
	}

	if *outputJSON {
		out := struct {
			States []string `json:"states"`
			Length int      `json:"length"`
		}{
			States: path.States,
			Length: path.Length(),
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	fmt.Fprintln(stdout, formatPath(path))
	return nil
}

func formatPath(p *analysis.Path) string {
	if len(p.States) == 0 {
		return ""
	}
	if len(p.Transitions) == 0 {
		return p.States[0]
	}

	var parts []string
	parts = append(parts, p.States[0])
	for i, t := range p.Transitions {
		if t.Event != "" {
			parts = append(parts, fmt.Sprintf("-[%s]->", t.Event))
		} else {
			parts = append(parts, "->")
		}
		parts = append(parts, p.States[i+1])
	}
	return strings.Join(parts, " ")
}

// cmdAnalyze runs comprehensive analysis and outputs a report.
func cmdAnalyze(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	outputJSON := fs.Bool("json", false, "output as JSON")
	quiet := fs.Bool("quiet", false, "only show problems")
	if err := fs.Parse(args); err != nil {
		return err
	}

	_, chart, err := loadInput(fs.Args(), stdin)
	if err != nil {
		return err
	}

	g := analysis.BuildGraph(chart)

	// Collect analysis
	orphans := g.Orphans(false)
	sinks := g.Sinks(false)
	disconnected := g.Disconnected()
	reachable := g.ReachableFromInitial()
	unreachable := g.Unreachable()
	deadEnds := g.DeadEnds()

	allStates := g.AllStates()
	leafStates := g.LeafStates()

	if *outputJSON {
		out := struct {
			Name         string   `json:"name"`
			States       int      `json:"states"`
			LeafStates   int      `json:"leaf_states"`
			Transitions  int      `json:"transitions"`
			Orphans      []string `json:"orphans"`
			Sinks        []string `json:"sinks"`
			Disconnected []string `json:"disconnected"`
			Reachable    []string `json:"reachable"`
			Unreachable  []string `json:"unreachable"`
			DeadEnds     []string `json:"dead_ends"`
		}{
			Name:         chart.Name,
			States:       len(allStates),
			LeafStates:   len(leafStates),
			Transitions:  g.TransitionCount(),
			Orphans:      orphans,
			Sinks:        sinks,
			Disconnected: disconnected,
			Reachable:    reachable,
			Unreachable:  unreachable,
			DeadEnds:     deadEnds,
		}
		// Ensure empty arrays instead of null
		if out.Orphans == nil {
			out.Orphans = []string{}
		}
		if out.Sinks == nil {
			out.Sinks = []string{}
		}
		if out.Disconnected == nil {
			out.Disconnected = []string{}
		}
		if out.Reachable == nil {
			out.Reachable = []string{}
		}
		if out.Unreachable == nil {
			out.Unreachable = []string{}
		}
		if out.DeadEnds == nil {
			out.DeadEnds = []string{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	// Text report
	name := chart.Name
	if name == "" {
		name = "Statechart"
	}

	if !*quiet {
		fmt.Fprintf(stdout, "Analysis: %s\n", name)
		fmt.Fprintln(stdout, strings.Repeat("=", len("Analysis: ")+len(name)))
		fmt.Fprintln(stdout)
		fmt.Fprintf(stdout, "States:      %d total (%d leaf)\n", len(allStates), len(leafStates))
		fmt.Fprintf(stdout, "Transitions: %d\n", g.TransitionCount())
		fmt.Fprintln(stdout)
	}

	// Issues section
	hasIssues := len(orphans) > 0 || len(sinks) > 0 || len(disconnected) > 0 || len(unreachable) > 0 || len(deadEnds) > 0

	if hasIssues || !*quiet {
		fmt.Fprintln(stdout, "Issues:")
	}

	if len(orphans) > 0 {
		fmt.Fprintf(stdout, "  Orphans (%d):      %s\n", len(orphans), strings.Join(orphans, ", "))
	} else if !*quiet {
		fmt.Fprintln(stdout, "  Orphans:          none")
	}

	if len(sinks) > 0 {
		fmt.Fprintf(stdout, "  Sinks (%d):        %s\n", len(sinks), strings.Join(sinks, ", "))
	} else if !*quiet {
		fmt.Fprintln(stdout, "  Sinks:            none")
	}

	if len(disconnected) > 0 {
		fmt.Fprintf(stdout, "  Disconnected (%d): %s\n", len(disconnected), strings.Join(disconnected, ", "))
	} else if !*quiet {
		fmt.Fprintln(stdout, "  Disconnected:     none")
	}

	if len(unreachable) > 0 {
		fmt.Fprintf(stdout, "  Unreachable (%d):  %s\n", len(unreachable), strings.Join(unreachable, ", "))
	} else if !*quiet {
		fmt.Fprintln(stdout, "  Unreachable:      none")
	}

	if len(deadEnds) > 0 {
		fmt.Fprintf(stdout, "  Dead ends (%d):   %s\n", len(deadEnds), strings.Join(deadEnds, ", "))
	} else if !*quiet {
		fmt.Fprintln(stdout, "  Dead ends:        none")
	}

	if !*quiet {
		fmt.Fprintln(stdout)
		reachPct := 0.0
		if len(allStates) > 0 {
			reachPct = float64(len(reachable)) / float64(len(allStates)) * 100
		}
		fmt.Fprintf(stdout, "Reachability: %d/%d states (%.1f%%)\n", len(reachable), len(allStates), reachPct)
	}

	return nil
}

// cmdDiff compares two statecharts and shows differences.
func cmdDiff(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("diff", flag.ContinueOnError)
	outputJSON := fs.Bool("json", false, "output as JSON")
	quiet := fs.Bool("quiet", false, "only show summary")
	if err := fs.Parse(args); err != nil {
		return err
	}

	remaining := fs.Args()
	if len(remaining) < 2 {
		return fmt.Errorf("usage: sc diff <chart1> <chart2>")
	}

	// Load both charts
	_, chart1, err := loadInput([]string{remaining[0]}, nil)
	if err != nil {
		return fmt.Errorf("load %s: %w", remaining[0], err)
	}
	_, chart2, err := loadInput([]string{remaining[1]}, nil)
	if err != nil {
		return fmt.Errorf("load %s: %w", remaining[1], err)
	}

	diff := analysis.Diff(chart1, chart2)
	diff.FromName = remaining[0]
	diff.ToName = remaining[1]

	if *outputJSON {
		type jsonStateMod struct {
			Label   string   `json:"label"`
			Changes []string `json:"changes"`
		}
		type jsonTransMod struct {
			From     string   `json:"from"`
			To       string   `json:"to"`
			Event    string   `json:"event"`
			Changes  []string `json:"changes"`
		}
		type jsonTrans struct {
			From  string `json:"from"`
			To    string `json:"to"`
			Event string `json:"event"`
		}
		out := struct {
			From                string         `json:"from"`
			To                  string         `json:"to"`
			Compatible          bool           `json:"compatible"`
			Summary             string         `json:"summary"`
			StatesAdded         []string       `json:"states_added"`
			StatesRemoved       []string       `json:"states_removed"`
			StatesModified      []jsonStateMod `json:"states_modified"`
			TransitionsAdded    []jsonTrans    `json:"transitions_added"`
			TransitionsRemoved  []jsonTrans    `json:"transitions_removed"`
			TransitionsModified []jsonTransMod `json:"transitions_modified"`
			EventsAdded         []string       `json:"events_added"`
			EventsRemoved       []string       `json:"events_removed"`
			BreakingChanges     []string       `json:"breaking_changes"`
		}{
			From:            diff.FromName,
			To:              diff.ToName,
			Compatible:      diff.IsCompatible,
			Summary:         diff.Summary(),
			StatesAdded:     diff.StatesAdded,
			StatesRemoved:   diff.StatesRemoved,
			EventsAdded:     diff.EventsAdded,
			EventsRemoved:   diff.EventsRemoved,
			BreakingChanges: diff.BreakingChanges,
		}
		// Ensure non-null arrays
		if out.StatesAdded == nil {
			out.StatesAdded = []string{}
		}
		if out.StatesRemoved == nil {
			out.StatesRemoved = []string{}
		}
		if out.EventsAdded == nil {
			out.EventsAdded = []string{}
		}
		if out.EventsRemoved == nil {
			out.EventsRemoved = []string{}
		}
		if out.BreakingChanges == nil {
			out.BreakingChanges = []string{}
		}
		for _, m := range diff.StatesModified {
			out.StatesModified = append(out.StatesModified, jsonStateMod{
				Label:   m.Label,
				Changes: m.Changes,
			})
		}
		for _, t := range diff.TransitionsAdded {
			out.TransitionsAdded = append(out.TransitionsAdded, jsonTrans{
				From: t.From, To: t.To, Event: t.Event,
			})
		}
		for _, t := range diff.TransitionsRemoved {
			out.TransitionsRemoved = append(out.TransitionsRemoved, jsonTrans{
				From: t.From, To: t.To, Event: t.Event,
			})
		}
		for _, m := range diff.TransitionsModified {
			out.TransitionsModified = append(out.TransitionsModified, jsonTransMod{
				From: m.From, To: m.To, Event: m.Event, Changes: m.Changes,
			})
		}
		if out.StatesModified == nil {
			out.StatesModified = []jsonStateMod{}
		}
		if out.TransitionsAdded == nil {
			out.TransitionsAdded = []jsonTrans{}
		}
		if out.TransitionsRemoved == nil {
			out.TransitionsRemoved = []jsonTrans{}
		}
		if out.TransitionsModified == nil {
			out.TransitionsModified = []jsonTransMod{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	// Text output
	fmt.Fprintf(stdout, "Diff: %s -> %s\n", diff.FromName, diff.ToName)
	fmt.Fprintln(stdout, strings.Repeat("=", 40))

	if !diff.HasChanges() {
		fmt.Fprintln(stdout, "No changes")
		return nil
	}

	fmt.Fprintf(stdout, "Summary: %s\n", diff.Summary())
	if diff.IsCompatible {
		fmt.Fprintln(stdout, "Compatibility: COMPATIBLE")
	} else {
		fmt.Fprintln(stdout, "Compatibility: BREAKING")
	}
	fmt.Fprintln(stdout)

	if *quiet {
		return nil
	}

	// States
	if len(diff.StatesAdded) > 0 {
		fmt.Fprintln(stdout, "States Added:")
		for _, s := range diff.StatesAdded {
			fmt.Fprintf(stdout, "  + %s\n", s)
		}
	}
	if len(diff.StatesRemoved) > 0 {
		fmt.Fprintln(stdout, "States Removed:")
		for _, s := range diff.StatesRemoved {
			fmt.Fprintf(stdout, "  - %s\n", s)
		}
	}
	if len(diff.StatesModified) > 0 {
		fmt.Fprintln(stdout, "States Modified:")
		for _, m := range diff.StatesModified {
			fmt.Fprintf(stdout, "  ~ %s: %s\n", m.Label, strings.Join(m.Changes, ", "))
		}
	}

	// Transitions
	if len(diff.TransitionsAdded) > 0 {
		fmt.Fprintln(stdout, "Transitions Added:")
		for _, t := range diff.TransitionsAdded {
			fmt.Fprintf(stdout, "  + %s -> %s [%s]\n", t.From, t.To, t.Event)
		}
	}
	if len(diff.TransitionsRemoved) > 0 {
		fmt.Fprintln(stdout, "Transitions Removed:")
		for _, t := range diff.TransitionsRemoved {
			fmt.Fprintf(stdout, "  - %s -> %s [%s]\n", t.From, t.To, t.Event)
		}
	}
	if len(diff.TransitionsModified) > 0 {
		fmt.Fprintln(stdout, "Transitions Modified:")
		for _, m := range diff.TransitionsModified {
			fmt.Fprintf(stdout, "  ~ %s -> %s [%s]: %s\n", m.From, m.To, m.Event, strings.Join(m.Changes, ", "))
		}
	}

	// Events
	if len(diff.EventsAdded) > 0 {
		fmt.Fprintln(stdout, "Events Added:")
		for _, e := range diff.EventsAdded {
			fmt.Fprintf(stdout, "  + %s\n", e)
		}
	}
	if len(diff.EventsRemoved) > 0 {
		fmt.Fprintln(stdout, "Events Removed:")
		for _, e := range diff.EventsRemoved {
			fmt.Fprintf(stdout, "  - %s\n", e)
		}
	}

	// Breaking changes
	if len(diff.BreakingChanges) > 0 {
		fmt.Fprintln(stdout)
		fmt.Fprintln(stdout, "Breaking Changes:")
		for _, b := range diff.BreakingChanges {
			fmt.Fprintf(stdout, "  ! %s\n", b)
		}
	}

	return nil
}

// cmdEvolution derives evolution from a series of statechart files.
func cmdEvolution(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("evolution", flag.ContinueOnError)
	outputJSON := fs.Bool("json", false, "output as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	remaining := fs.Args()
	if len(remaining) < 2 {
		return fmt.Errorf("usage: sc evolution <chart1> <chart2> [chart3...]")
	}

	// Load all charts
	var charts []*sc.Statechart
	var names []string
	for _, file := range remaining {
		_, chart, err := loadInput([]string{file}, nil)
		if err != nil {
			return fmt.Errorf("load %s: %w", file, err)
		}
		charts = append(charts, chart)
		names = append(names, file)
	}

	ev := analysis.DeriveEvolution(charts, names)

	if *outputJSON {
		type jsonVersion struct {
			Name       string `json:"name"`
			States     int    `json:"states"`
			Transitions int   `json:"transitions"`
		}
		type jsonStep struct {
			From        string `json:"from"`
			To          string `json:"to"`
			Summary     string `json:"summary"`
			Compatible  bool   `json:"compatible"`
		}
		out := struct {
			Versions []jsonVersion `json:"versions"`
			Timeline []jsonStep    `json:"timeline"`
		}{}
		for _, v := range ev.Versions {
			out.Versions = append(out.Versions, jsonVersion{
				Name:        v.Name,
				States:      v.StateCount,
				Transitions: v.TransCount,
			})
		}
		for _, s := range ev.Timeline {
			out.Timeline = append(out.Timeline, jsonStep{
				From:       s.FromVersion,
				To:         s.ToVersion,
				Summary:    s.Diff.Summary(),
				Compatible: s.Diff.IsCompatible,
			})
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	// Text output
	fmt.Fprintln(stdout, "Evolution Timeline")
	fmt.Fprintln(stdout, "==================")
	fmt.Fprintln(stdout)

	// Version summary
	fmt.Fprintln(stdout, "Versions:")
	for i, v := range ev.Versions {
		fmt.Fprintf(stdout, "  %d. %s (%d states, %d transitions)\n",
			i+1, v.Name, v.StateCount, v.TransCount)
	}
	fmt.Fprintln(stdout)

	// Timeline
	fmt.Fprintln(stdout, "Changes:")
	for _, step := range ev.Timeline {
		compat := "COMPATIBLE"
		if !step.Diff.IsCompatible {
			compat = "BREAKING"
		}
		fmt.Fprintf(stdout, "  %s -> %s: %s [%s]\n",
			step.FromVersion, step.ToVersion, step.Diff.Summary(), compat)
	}

	return nil
}
