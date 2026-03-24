// Command sc-starlark-gen generates a small prompt-to-Starlark-statechart training set.
//
// The dataset is synthetic. It currently covers simple linear machines,
// cycles, and traffic-light charts and writes a JSON file with train and test
// splits for downstream ML experiments.
//
// Usage:
//
//	sc-starlark-gen [flags]
//
// Flags:
//
//	-out     Output directory. Default "ml/data".
//	-count   Total number of examples to generate. Default 1000.
//
// The output file is written as arc_data.json in the output directory. The
// split is 90/10 train/test.
//
// Examples:
//
//	sc-starlark-gen
//	sc-starlark-gen -count 250 -out ./tmp/data
package main
