// Command sc applies validation, analysis, visualization, conversion, and
// simulation operations to statechart definitions.
//
// sc reads statecharts from files, standard input, or txtar archives. It is
// designed to compose well with pipes: commands either print structured text,
// emit machine-readable formats, or write diagrams that can be passed to other
// tools.
//
// Usage:
//
//	sc <command> [flags] [file]
//
// If file is omitted, sc reads from standard input.
//
// Core commands:
//
//	validate     Check that a chart is well formed.
//	info         Print summary information.
//	states       List states.
//	events       List events.
//	mermaid      Emit Mermaid state-diagram text.
//	step         Execute a sequence of events through a machine instance.
//	dot          Emit Graphviz DOT.
//	export       Convert a chart to json, ct, or xstate.
//	import       Read external formats such as XState JSON.
//
// Analysis commands:
//
//	transitions  Inspect transitions, optionally with filters.
//	orphans      List states with no incoming transitions.
//	sinks        List states with no outgoing transitions.
//	disconnected List states with no transitions at all.
//	reachable    List states reachable from the initial configuration.
//	unreachable  List states not reachable from the initial configuration.
//	path         Find a path between two states.
//	analyze      Print a combined report.
//	diff         Compare two charts.
//	evolution    Derive an evolution summary from a chart series.
//
// Generation commands:
//
//	simulate     Generate execution traces for testing or ML pipelines.
//
// Examples:
//
//	sc validate chart.json
//	cat chart.json | sc mermaid
//	sc step -e POWER_ON -e START chart.json
//	sc analyze -quiet chart.json
//	sc export -format xstate chart.json
package main
