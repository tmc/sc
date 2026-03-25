package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/encoding/protojson"
)

const defaultProgressEvery = 10

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("sc-infer", flag.ContinueOnError)
	fs.SetOutput(stderr)

	format := fs.String("format", "auto", "input format: auto, text, jsonl, trace")
	emit := fs.String("emit", "final", "when to emit chart: final, periodic, onChange")
	period := fs.Int("period", 100, "emit chart every N observations when -emit=periodic")
	minConfidence := fs.Float64("min-confidence", 0.0, "minimum confidence threshold for including edges")
	inferHierarchy := fs.Bool("infer-hierarchy", true, "attempt to infer state hierarchy from transition patterns")
	inferParallel := fs.Bool("infer-parallel", true, "attempt to detect parallel regions from co-active states")
	quiet := fs.Bool("quiet", false, "suppress stderr progress")
	pretty := fs.Bool("pretty", true, "pretty-print JSON output")
	outPath := fs.String("o", "", "write final chart to file instead of stdout")
	if err := fs.Parse(args); err != nil {
		return err
	}

	switch *emit {
	case "final", "periodic", "onChange":
	default:
		return fmt.Errorf("invalid -emit value %q: must be final, periodic, or onChange", *emit)
	}
	if *period <= 0 {
		return fmt.Errorf("period must be > 0")
	}
	if *minConfidence < 0 {
		return fmt.Errorf("min-confidence must be >= 0")
	}

	data, err := io.ReadAll(stdin)
	if err != nil {
		return fmt.Errorf("read stdin: %w", err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return fmt.Errorf("input is empty")
	}

	resolvedFormat := *format
	if resolvedFormat == "auto" {
		resolvedFormat, err = detectFormat(data)
		if err != nil {
			return err
		}
	}

	inferer := newInferer()
	lastEmitObservation := -1

	emitChart := func(final bool) error {
		chart, err := inferer.BuildChart(buildOptions{
			MinConfidence:  *minConfidence,
			InferHierarchy: *inferHierarchy,
			InferParallel:  *inferParallel,
		})
		if err != nil {
			return err
		}

		payload, err := marshalChart(chart, *pretty)
		if err != nil {
			return fmt.Errorf("marshal chart: %w", err)
		}

		if final && *outPath != "" {
			if err := os.WriteFile(*outPath, payload, 0o644); err != nil {
				return fmt.Errorf("write output: %w", err)
			}
			lastEmitObservation = inferer.ObservationCount()
			return nil
		}

		if _, err := stdout.Write(payload); err != nil {
			return fmt.Errorf("write stdout: %w", err)
		}
		if len(payload) == 0 || payload[len(payload)-1] != '\n' {
			if _, err := stdout.Write([]byte("\n")); err != nil {
				return fmt.Errorf("write trailing newline: %w", err)
			}
		}
		lastEmitObservation = inferer.ObservationCount()
		return nil
	}

	onObservation := func(structuralChange bool) error {
		if !*quiet && (structuralChange || inferer.ObservationCount()%defaultProgressEvery == 0) {
			if _, err := fmt.Fprintf(stderr, "\r%s", inferer.ProgressLine()); err != nil {
				return fmt.Errorf("write progress: %w", err)
			}
		}

		switch *emit {
		case "periodic":
			if inferer.ObservationCount()%*period == 0 {
				return emitChart(false)
			}
		case "onChange":
			if structuralChange {
				return emitChart(false)
			}
		}
		return nil
	}

	switch resolvedFormat {
	case "text":
		err = parseTextInput(data, inferer, onObservation)
	case "jsonl":
		err = parseJSONLInput(data, inferer, onObservation)
	case "trace":
		err = parseTraceInput(data, inferer, onObservation)
	default:
		err = fmt.Errorf("unsupported format %q", resolvedFormat)
	}
	if err != nil {
		return err
	}
	if inferer.ObservationCount() == 0 {
		return fmt.Errorf("no observations found")
	}

	if !*quiet {
		if _, err := fmt.Fprintf(stderr, "\r%s\n", inferer.ProgressLine()); err != nil {
			return fmt.Errorf("write progress: %w", err)
		}
	}

	if lastEmitObservation != inferer.ObservationCount() || *emit == "final" {
		if err := emitChart(true); err != nil {
			return err
		}
	}
	return nil
}

func marshalChart(chart *sc.Statechart, pretty bool) ([]byte, error) {
	opts := protojson.MarshalOptions{
		UseProtoNames:   true,
		EmitUnpopulated: false,
	}
	if pretty {
		opts.Multiline = true
		opts.Indent = "  "
	}
	return opts.Marshal(chart)
}
