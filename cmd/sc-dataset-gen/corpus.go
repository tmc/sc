package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	semantics "github.com/tmc/sc/semantics/v1"
)

type corpusManifest struct {
	Generated  string          `json:"generated"`
	CorpusDir  string          `json:"corpus_dir"`
	TotalValid int             `json:"total_valid"`
	Charts     []manifestEntry `json:"charts"`
}

type manifestEntry struct {
	Path     string   `json:"path"`
	Name     string   `json:"name"`
	Families []string `json:"families,omitempty"`
}

type corpusConfig struct {
	manifestPath string
	outDir       string
	products     productSet
	traceCount   int
	stepCount    int
	detail       detailLevel
	seed         int64
	split        bool
	splitRatio   [3]int // train, eval, test percentages
	families     []string
	limit        int
	workers      int
	pretty       bool
	events       string
	coverage     bool
	mutations    int
}

type corpusOutputManifest struct {
	Generated      string         `json:"generated"`
	SourceManifest string         `json:"source_manifest"`
	TotalProcessed int            `json:"total_processed"`
	TotalErrors    int            `json:"total_errors"`
	Splits         map[string]int `json:"splits,omitempty"`
	Products       []string       `json:"products"`
}

func corpusMode(cfg corpusConfig, stdout, stderr io.Writer) error {
	data, err := os.ReadFile(cfg.manifestPath)
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	var manifest corpusManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}

	entries := manifest.Charts
	if cfg.limit > 0 && cfg.limit < len(entries) {
		entries = entries[:cfg.limit]
	}

	if cfg.outDir == "" {
		cfg.outDir = "."
	}
	if err := os.MkdirAll(cfg.outDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Shared output file handles.
	type outputFiles struct {
		mu    sync.Mutex
		files map[string]*os.File
	}
	out := &outputFiles{files: make(map[string]*os.File)}

	getFile := func(name string) (*os.File, error) {
		out.mu.Lock()
		defer out.mu.Unlock()
		if f, ok := out.files[name]; ok {
			return f, nil
		}
		f, err := os.Create(filepath.Join(cfg.outDir, name))
		if err != nil {
			return nil, err
		}
		out.files[name] = f
		return f, nil
	}

	defer func() {
		out.mu.Lock()
		defer out.mu.Unlock()
		for _, f := range out.files {
			f.Close()
		}
	}()

	// Deterministic split assignment based on path hash.
	splitFor := func(path string) string {
		if !cfg.split {
			return "dataset"
		}
		h := sha256.New()
		binary.Write(h, binary.BigEndian, cfg.seed)
		h.Write([]byte(path))
		sum := h.Sum(nil)
		bucket := int(binary.BigEndian.Uint32(sum[:4])) % 100
		if bucket < cfg.splitRatio[0] {
			return "train"
		}
		if bucket < cfg.splitRatio[0]+cfg.splitRatio[1] {
			return "eval"
		}
		return "test"
	}

	var totalProcessed atomic.Int64
	var totalErrors atomic.Int64
	splitCounts := &sync.Map{}

	work := make(chan manifestEntry, len(entries))
	for _, e := range entries {
		work <- e
	}
	close(work)

	workers := cfg.workers
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for entry := range work {
				chartPath := entry.Path
				if !filepath.IsAbs(chartPath) && manifest.CorpusDir != "" {
					chartPath = filepath.Join(manifest.CorpusDir, chartPath)
				}

				record, err := processChartForCorpus(chartPath, cfg)
				if err != nil {
					fmt.Fprintf(stderr, "error processing %s: %v\n", chartPath, err)
					totalErrors.Add(1)
					continue
				}

				split := splitFor(entry.Path)

				// Filter by families if requested.
				if len(cfg.families) > 0 && record.Classification != nil {
					matched := false
					for _, want := range cfg.families {
						for _, have := range record.Classification.Families {
							if want == have {
								matched = true
								break
							}
						}
						if matched {
							break
						}
					}
					if !matched {
						continue
					}
				}

				line, err := json.Marshal(record)
				if err != nil {
					fmt.Fprintf(stderr, "marshal error for %s: %v\n", chartPath, err)
					totalErrors.Add(1)
					continue
				}

				fileName := split + ".jsonl"
				f, err := getFile(fileName)
				if err != nil {
					fmt.Fprintf(stderr, "open output %s: %v\n", fileName, err)
					totalErrors.Add(1)
					continue
				}

				out.mu.Lock()
				f.Write(line)
				f.Write([]byte("\n"))
				out.mu.Unlock()

				totalProcessed.Add(1)
				if v, ok := splitCounts.LoadOrStore(split, new(atomic.Int64)); ok {
					v.(*atomic.Int64).Add(1)
				} else {
					v.(*atomic.Int64).Add(1)
				}
			}
		}()
	}
	wg.Wait()

	splits := make(map[string]int)
	splitCounts.Range(func(key, value any) bool {
		splits[key.(string)] = int(value.(*atomic.Int64).Load())
		return true
	})

	outManifest := corpusOutputManifest{
		Generated:      time.Now().Format(time.RFC3339),
		SourceManifest: cfg.manifestPath,
		TotalProcessed: int(totalProcessed.Load()),
		TotalErrors:    int(totalErrors.Load()),
		Splits:         splits,
		Products:       cfg.products.list(),
	}
	manifestJSON, err := json.MarshalIndent(outManifest, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal output manifest: %w", err)
	}
	return os.WriteFile(filepath.Join(cfg.outDir, "corpus-manifest.json"), manifestJSON, 0o644)
}

func processChartForCorpus(chartPath string, cfg corpusConfig) (*datasetRecord, error) {
	chart, err := loadStatechart(chartPath)
	if err != nil {
		return nil, fmt.Errorf("load: %w", err)
	}
	wrapped := semantics.NewStatechart(chart)
	if err := wrapped.Validate(); err != nil {
		return nil, fmt.Errorf("validate: %w", err)
	}

	events := collectEvents(chart)
	if cfg.events != "" {
		events = parseCSV(cfg.events)
	}

	chartID := chart.Name
	if chartID == "" {
		base := filepath.Base(chartPath)
		chartID = base[:len(base)-len(filepath.Ext(base))]
	}

	record := &datasetRecord{
		StatechartID: chartID,
		SourceFile:   chartPath,
		Seed:         cfg.seed,
		Events:       events,
	}

	// Classification is needed for family filtering and validation metadata.
	if cfg.products.has("classification") || cfg.products.has("validation") || cfg.products.has("all") || len(cfg.families) > 0 {
		record.Classification = buildClassification(chart)
	}

	if cfg.products.has("topology") || cfg.products.has("all") {
		record.GroundTruthTopology = buildTopology(chart)
	}

	if cfg.products.has("graph") || cfg.products.has("all") {
		record.GraphFeatures = buildGraphFeatures(chart)
	}

	if cfg.products.has("vocabulary") || cfg.products.has("all") {
		record.Vocabulary = buildVocabulary(chart)
	}

	needTraces := cfg.products.has("traces") || cfg.products.has("history") || cfg.products.has("guard_eval") || cfg.products.has("all")
	if needTraces && len(events) > 0 {
		rng := rand.New(rand.NewSource(cfg.seed))
		if cfg.coverage {
			traces, covResult, err := generateCoverageTraces(chart, wrapped, chartID, events, cfg.traceCount, cfg.stepCount, cfg.seed, cfg.detail, false)
			if err != nil {
				return nil, fmt.Errorf("coverage traces: %w", err)
			}
			record.Traces = traces
			record.CoverageResult = covResult
		} else {
			traces, err := generateTraces(chart, wrapped, chartPath, events, cfg.traceCount, cfg.stepCount, cfg.seed, rng, false, cfg.detail)
			if err != nil {
				return nil, fmt.Errorf("traces: %w", err)
			}
			record.Traces = traces
		}

		if (cfg.products.has("history") || cfg.products.has("all")) && len(record.Traces) > 0 {
			for _, raw := range record.Traces {
				trace, err := parseTraceFromJSON(raw)
				if err != nil {
					continue
				}
				record.HistoryRows = append(record.HistoryRows, extractHistoryRows(chartID, trace, chart)...)
			}
		}
		if (cfg.products.has("guard_eval") || cfg.products.has("all")) && len(record.Traces) > 0 {
			for _, raw := range record.Traces {
				trace, err := parseTraceFromJSON(raw)
				if err != nil {
					continue
				}
				record.GuardEvalRows = append(record.GuardEvalRows, extractGuardEvalRows(chartID, trace)...)
			}
		}
	}

	if (cfg.products.has("mutation") || cfg.products.has("all")) && len(chart.Transitions) > 0 {
		record.MutationRows = generateMutations(chart, chartID, cfg.seed, cfg.mutations)
	}

	if cfg.products.has("validation") || cfg.products.has("all") {
		var fams []string
		if record.Classification != nil {
			fams = record.Classification.Families
		}
		record.ValidationRows = buildValidationRows(chart, chartID, fams, cfg.seed, cfg.mutations)
	}

	// Remove traces from output if not explicitly requested.
	if !cfg.products.has("traces") && !cfg.products.has("all") {
		record.Traces = nil
	}

	return record, nil
}
