package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/sc/runtime"
	"go.starlark.net/starlark"
	"golang.org/x/tools/txtar"
)

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: sc-run <script.star|archive.txt>\n")
		os.Exit(1)
	}

	filename := args[0]
	data, err := os.ReadFile(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	var scriptName string
	var loader func(*starlark.Thread, string) (starlark.StringDict, error)

	// Check for txtar
	if strings.HasSuffix(filename, ".txt") || strings.HasSuffix(filename, ".txtar") || strings.Contains(string(data), "-- ") {
		archive := txtar.Parse(data)
		if len(archive.Files) == 0 {
			// Treat as plain starlark if no files found in recognized txtar
			scriptName = filename
			loader = makeLoader(nil)
		} else {
			// Found txtar archive
			tempDir, err := os.MkdirTemp("", "sc-run-*")
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error creating temp dir: %v\n", err)
				os.Exit(1)
			}
			defer os.RemoveAll(tempDir) // cleanup

			// Write files
			for _, f := range archive.Files {
				path := filepath.Join(tempDir, f.Name)
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					fmt.Fprintf(os.Stderr, "Error creating dirs: %v\n", err)
					os.Exit(1)
				}
				if err := os.WriteFile(path, f.Data, 0644); err != nil {
					fmt.Fprintf(os.Stderr, "Error writing file %s: %v\n", f.Name, err)
					os.Exit(1)
				}
			}

			// Determine entry point (first .star file or main.star)
			for _, f := range archive.Files {
				if strings.HasSuffix(f.Name, ".star") {
					if scriptName == "" || f.Name == "main.star" {
						scriptName = filepath.Join(tempDir, f.Name)
					}
				}
			}
			if scriptName == "" {
				fmt.Fprintf(os.Stderr, "Error: No .star file found in archive\n")
				os.Exit(1)
			}
			loader = makeLoader(tempDir)
		}
	} else {
		// Plain file
		scriptName = filename
		loader = makeLoader(filepath.Dir(filename))
	}

	// Create Starlark thread
	thread := &starlark.Thread{Name: "main", Load: loader}

	// Execute the script
	globals, err := starlark.ExecFile(thread, scriptName, nil, nil)
	if err != nil {
		if evalErr, ok := err.(*starlark.EvalError); ok {
			fmt.Fprintf(os.Stderr, "%s\n", evalErr.Backtrace())
		}
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Check if the script defines a main() function and call it
	if mainFn, ok := globals["main"]; ok {
		_, err := starlark.Call(thread, mainFn, nil, nil)
		if err != nil {
			if evalErr, ok := err.(*starlark.EvalError); ok {
				fmt.Fprintf(os.Stderr, "%s\n", evalErr.Backtrace())
			}
			fmt.Fprintf(os.Stderr, "Error calling main: %v\n", err)
			os.Exit(1)
		}
	}
}

func makeLoader(rootDir interface{}) func(*starlark.Thread, string) (starlark.StringDict, error) {
	var root string
	if r, ok := rootDir.(string); ok {
		root = r
	}

	return func(thread *starlark.Thread, module string) (starlark.StringDict, error) {
		if module == "sc" {
			mod := runtime.NewModule()
			return starlark.StringDict{"sc": mod}, nil
		}

		// Load local files if root is set
		if root != "" {
			path := filepath.Join(root, module)
			if !strings.HasSuffix(path, ".star") {
				path += ".star"
			}
			return starlark.ExecFile(thread, path, nil, nil)
		}

		return nil, fmt.Errorf("module %s not found", module)
	}
}
