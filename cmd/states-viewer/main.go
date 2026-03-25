package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/tmc/sc/cmd/states-viewer/internal/app"
)

func main() {
	flag.Parse()

	viewer := app.New(1200, 800, "States Viewer")

	// Load file if provided
	if flag.NArg() > 0 {
		path := flag.Arg(0)
		if err := viewer.LoadFile(path); err != nil {
			fmt.Fprintf(os.Stderr, "error loading %s: %v\n", path, err)
			os.Exit(1)
		}
	}

	if err := viewer.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
