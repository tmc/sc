// Command arc-solver-gen generates a small synthetic ARC-style solver corpus.
//
// The command writes a JSON file containing train and test splits for simple
// grid-transformation tasks. The current generator mixes recoloring and
// gravity-like examples and formats the targets as worked solution sketches
// suitable for downstream language-model experiments.
//
// Usage:
//
//	arc-solver-gen [flags]
//
// Flags:
//
//	-out     Output directory. Default "ml/data/solver".
//	-count   Total number of examples to generate. Default 2000.
//
// The output file is written as recolor_data.json in the output directory.
// The split is 90/10 train/test.
//
// Examples:
//
//	arc-solver-gen
//	arc-solver-gen -count 500 -out ./tmp/solver
package main
