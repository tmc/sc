// Command sc-mcp serves statechart tools over the Model Context Protocol.
//
// The server communicates on standard input and standard output and exposes a
// small tool surface for validating, inspecting, visualizing, and stepping
// statechart definitions. When -dir is set, it also publishes file-backed MCP
// resources for charts on disk.
//
// Usage:
//
//	sc-mcp [flags]
//
// Flags:
//
//	-dir   Directory of statechart files to expose as MCP resources.
//
// Exposed tools include validation, state and event listing, Mermaid and DOT
// generation, transition listing, and event stepping. The command is intended
// to be embedded in editor, agent, or automation workflows rather than run
// directly by hand.
//
// Examples:
//
//	sc-mcp
//	sc-mcp -dir ./testdata
package main
