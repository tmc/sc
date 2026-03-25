// Command web-visualizer serves an interactive web UI and HTTP API for working
// with statechart machines.
//
// The server embeds its frontend assets, persists machine state locally, and
// exposes endpoints for machine management, event processing, validation,
// history inspection, export, and AI-assisted editing. When -charts-dir is
// provided, the server also exposes chart file operations for bidirectional
// editing against files on disk.
//
// Usage:
//
//	web-visualizer [flags]
//
// Flags:
//
//	-version     Print version information and exit.
//	-charts-dir  Directory of chart files to expose for editing.
//
// The API is rooted at /api/v1. The command is intended for local interactive
// use while developing, debugging, or demonstrating charts.
//
// Examples:
//
//	web-visualizer
//	web-visualizer -charts-dir ./testdata
package main
