// Deprecated: use sc-dataset-gen instead, which produces traces plus additional
// data products (graph features, classification, vocabulary, history rows,
// guard evaluation rows, and mutation pairs). When invoked as sc-trace-gen
// (via symlink), sc-dataset-gen defaults to traces+topology output.
//
// Command sc-trace-gen generates execution traces from the Go semantics engine.
//
// The command loads a statechart, samples random events, runs them through
// semantics/v1, and writes a JSON dataset containing two pieces of information:
// a ground-truth topology summary and one or more execution traces in the
// ExecutionTrace schema.
//
// This command is intended for benchmarking, replay, and ML data preparation.
// It records source and target configurations, fired transitions, guard
// evaluations, context snapshots, and basic processing metadata for each step.
//
// Usage:
//
//	sc-trace-gen -chart file.json [flags]
//
// Flags:
//
//	-chart    Path to a statechart in proto JSON or textproto form.
//	-out      Output file. If omitted, write JSON to standard output.
//	-traces   Number of traces to generate. Default 1.
//	-steps    Maximum random events per trace. Default 20.
//	-seed     PRNG seed used for reproducible event selection.
//	-events   Comma-separated event override. By default events are collected
//	          from the chart.
//	-pretty   Pretty-print JSON output. Default true.
//
// Examples:
//
//	sc-trace-gen -chart ./testdata/traffic_light.json
//	sc-trace-gen -chart ./testdata/history_example.json -traces 50 -steps 100 -seed 7 -out traces.json
package main
