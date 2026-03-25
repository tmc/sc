// Command go-trace-viewer opens an interactive viewer for Go traces and their
// associated rule-state visualizations.
//
// The viewer accepts SGF and JSON traces, renders a 9x9 board, and shows a
// companion statechart for the current position. If no input is provided, it
// falls back to bundled sample traces.
//
// Usage:
//
//	go-trace-viewer [flags] [trace.sgf|trace.json ...]
//
// Flags:
//
//	-dir   Directory containing SGF and JSON traces to load.
//
// Examples:
//
//	go-trace-viewer
//	go-trace-viewer -dir ./traces
//	go-trace-viewer ./traces/game_01.sgf ./traces/game_01.json
package main
