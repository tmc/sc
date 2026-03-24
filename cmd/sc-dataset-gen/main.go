// Command sc-dataset-gen generates multi-product ML datasets from statecharts.
//
// It produces execution traces, topology, graph features, classification,
// vocabulary mappings, history rows, guard evaluation rows, and mutation
// pairs. In corpus mode it processes a manifest of charts in parallel and
// writes JSONL with optional train/eval/test splits.
//
// When invoked as sc-trace-gen (via symlink or binary name), the default
// products are limited to traces and topology for backward compatibility.
//
// Usage:
//
//	sc-dataset-gen -chart file.json                          # all products
//	sc-dataset-gen -chart file.json -products traces         # just traces
//	sc-dataset-gen -corpus manifest.json -out dir/ -workers 8  # batch mode
//
// Flags:
//
//	-chart       Path to a statechart JSON or textproto file.
//	-corpus      Path to corpus manifest JSON for batch mode.
//	-out         Output file (single chart) or directory (corpus).
//	-pretty      Pretty-print JSON output (default true).
//	-products    Comma-separated products: traces,topology,graph,classification,
//	             vocabulary,history,guard_eval,mutation,all (default "all").
//	-traces      Number of traces per chart (default 1).
//	-steps       Max steps per trace (default 20).
//	-detail      Trace detail: minimal, standard, debug (default "debug").
//	-events      Comma-separated event override.
//	-seed        PRNG seed (default 1).
//	-coverage    Enable coverage-driven trace generation.
//	-mutations   Number of mutation variants per chart (default 3).
//	-split       Enable train/eval/test split assignment in corpus mode.
//	-split-ratio Split percentages as train,eval,test (default "80,10,10").
//	-families    Comma-separated family filter for corpus mode.
//	-limit       Limit charts processed in corpus mode.
//	-workers     Parallel workers for corpus mode (default 1).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	semantics "github.com/tmc/sc/semantics/v1"
)

type datasetRecord struct {
	StatechartID        string               `json:"statechart_id"`
	SourceFile          string               `json:"source_file"`
	Seed                int64                `json:"seed"`
	Events              []string             `json:"events"`
	GroundTruthTopology *groundTruthTopology `json:"ground_truth_topology,omitempty"`
	Traces              []json.RawMessage    `json:"traces,omitempty"`
	GraphFeatures       *graphFeatures       `json:"graph_features,omitempty"`
	Classification      *classification      `json:"classification,omitempty"`
	Vocabulary          *vocabulary           `json:"vocabulary,omitempty"`
	HistoryRows         []historyRow          `json:"history_rows,omitempty"`
	GuardEvalRows       []guardEvalRow        `json:"guard_eval_rows,omitempty"`
	MutationRows        []mutationRow         `json:"mutation_rows,omitempty"`
	CoverageResult      *coverageResult       `json:"coverage_result,omitempty"`
}

type productSet map[string]bool

func parseProducts(s string) productSet {
	ps := make(productSet)
	for _, p := range parseCSV(s) {
		ps[strings.ToLower(p)] = true
	}
	return ps
}

func (ps productSet) has(name string) bool {
	return ps[name] || ps["all"]
}

func (ps productSet) list() []string {
	var out []string
	for k := range ps {
		out = append(out, k)
	}
	return out
}

func main() {
	if err := run(os.Args, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	// Detect backward-compat mode via binary name.
	binName := filepath.Base(args[0])
	isTraceGen := strings.Contains(binName, "sc-trace-gen")

	defaultProducts := "all"
	defaultDetail := "debug"
	if isTraceGen {
		defaultProducts = "traces,topology"
		defaultDetail = "standard"
	}

	fs := flag.NewFlagSet("sc-dataset-gen", flag.ContinueOnError)
	fs.SetOutput(stderr)

	chartPath := fs.String("chart", "", "path to a statechart JSON or textproto file")
	corpusPath := fs.String("corpus", "", "path to corpus manifest JSON for batch mode")
	outPath := fs.String("out", "", "output file or directory")
	pretty := fs.Bool("pretty", true, "pretty-print JSON output")
	productsFlag := fs.String("products", defaultProducts, "comma-separated product list")
	traceCount := fs.Int("traces", 1, "number of traces per chart")
	stepCount := fs.Int("steps", 20, "max steps per trace")
	detail := fs.String("detail", defaultDetail, "trace detail: minimal, standard, debug")
	eventsOverride := fs.String("events", "", "comma-separated event override")
	seed := fs.Int64("seed", 1, "PRNG seed")
	coverageFlag := fs.Bool("coverage", false, "enable coverage-driven trace generation")
	mutationCount := fs.Int("mutations", 3, "number of mutations per chart")
	split := fs.Bool("split", false, "enable train/eval/test splits in corpus mode")
	splitRatio := fs.String("split-ratio", "80,10,10", "train,eval,test percentages")
	families := fs.String("families", "", "comma-separated family filter")
	limit := fs.Int("limit", 0, "limit charts in corpus mode")
	workers := fs.Int("workers", 1, "parallel workers for corpus mode")

	if err := fs.Parse(args[1:]); err != nil {
		return err
	}

	dl, err := parseDetailLevel(*detail)
	if err != nil {
		return err
	}
	products := parseProducts(*productsFlag)

	if *corpusPath != "" {
		ratio, err := parseSplitRatio(*splitRatio)
		if err != nil {
			return err
		}
		return corpusMode(corpusConfig{
			manifestPath: *corpusPath,
			outDir:       *outPath,
			products:     products,
			traceCount:   *traceCount,
			stepCount:    *stepCount,
			detail:       dl,
			seed:         *seed,
			split:        *split,
			splitRatio:   ratio,
			families:     parseCSV(*families),
			limit:        *limit,
			workers:      *workers,
			pretty:       *pretty,
			events:       *eventsOverride,
			coverage:     *coverageFlag,
			mutations:    *mutationCount,
		}, stdout, stderr)
	}

	if *chartPath == "" {
		return fmt.Errorf("either -chart or -corpus is required")
	}

	return singleChartMode(*chartPath, products, *traceCount, *stepCount, dl, *eventsOverride, *seed, *pretty, *coverageFlag, *mutationCount, *outPath, stdout)
}

func singleChartMode(chartPath string, products productSet, traceCount, stepCount int, dl detailLevel, eventsOverride string, seed int64, pretty, coverage bool, mutationCount int, outPath string, stdout io.Writer) error {
	chart, err := loadStatechart(chartPath)
	if err != nil {
		return fmt.Errorf("load statechart: %w", err)
	}
	wrapped := semantics.NewStatechart(chart)
	if err := wrapped.Validate(); err != nil {
		return fmt.Errorf("validate statechart: %w", err)
	}

	events := collectEvents(chart)
	if eventsOverride != "" {
		events = parseCSV(eventsOverride)
	}

	chartID := chart.Name
	if chartID == "" {
		base := filepath.Base(chartPath)
		chartID = strings.TrimSuffix(base, filepath.Ext(base))
	}

	record := &datasetRecord{
		StatechartID: chartID,
		SourceFile:   chartPath,
		Seed:         seed,
		Events:       events,
	}

	if products.has("topology") {
		record.GroundTruthTopology = buildTopology(chart)
	}
	if products.has("graph") {
		record.GraphFeatures = buildGraphFeatures(chart)
	}
	if products.has("classification") {
		record.Classification = buildClassification(chart)
	}
	if products.has("vocabulary") {
		record.Vocabulary = buildVocabulary(chart)
	}

	// Generate traces if needed by any trace-derived product.
	needTraces := products.has("traces") || products.has("history") || products.has("guard_eval")
	if needTraces && len(events) > 0 {
		if coverage {
			traces, covResult, err := generateCoverageTraces(chart, wrapped, chartID, events, max(traceCount, 10), stepCount, seed, dl, pretty)
			if err != nil {
				return fmt.Errorf("coverage traces: %w", err)
			}
			record.Traces = traces
			record.CoverageResult = covResult
		} else {
			rng := rand.New(rand.NewSource(seed))
			traces, err := generateTraces(chart, wrapped, chartPath, events, traceCount, stepCount, seed, rng, pretty, dl)
			if err != nil {
				return fmt.Errorf("generate traces: %w", err)
			}
			record.Traces = traces
		}

		// Derive history rows.
		if products.has("history") {
			for _, raw := range record.Traces {
				trace, err := parseTraceFromJSON(raw)
				if err != nil {
					continue
				}
				record.HistoryRows = append(record.HistoryRows, extractHistoryRows(chartID, trace, chart)...)
			}
		}

		// Derive guard eval rows.
		if products.has("guard_eval") {
			for _, raw := range record.Traces {
				trace, err := parseTraceFromJSON(raw)
				if err != nil {
					continue
				}
				record.GuardEvalRows = append(record.GuardEvalRows, extractGuardEvalRows(chartID, trace)...)
			}
		}

		// Strip traces from output if not explicitly requested.
		if !products.has("traces") {
			record.Traces = nil
		}
	}

	if products.has("mutation") && len(chart.Transitions) > 0 {
		record.MutationRows = generateMutations(chart, chartID, seed, mutationCount)
	}

	var out []byte
	if pretty {
		out, err = json.MarshalIndent(record, "", "  ")
	} else {
		out, err = json.Marshal(record)
	}
	if err != nil {
		return fmt.Errorf("marshal output: %w", err)
	}

	if outPath != "" {
		return os.WriteFile(outPath, out, 0o644)
	}
	stdout.Write(out)
	stdout.Write([]byte("\n"))
	return nil
}

func parseSplitRatio(s string) ([3]int, error) {
	parts := strings.Split(s, ",")
	if len(parts) != 3 {
		return [3]int{}, fmt.Errorf("split-ratio must have 3 values (e.g. 80,10,10)")
	}
	var ratio [3]int
	for i, p := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return [3]int{}, fmt.Errorf("invalid split-ratio value %q: %w", p, err)
		}
		ratio[i] = v
	}
	if ratio[0]+ratio[1]+ratio[2] != 100 {
		return [3]int{}, fmt.Errorf("split-ratio values must sum to 100")
	}
	return ratio, nil
}

