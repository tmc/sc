// Command sc-run executes Starlark programs against the sc runtime module.
//
// The input may be a single .star file or a txtar archive containing one or
// more Starlark files. Archives are unpacked to a temporary directory; the
// first .star file, or main.star when present, is used as the entry point.
//
// Scripts may load the built-in "sc" module provided by the runtime package.
// If a script defines main(), sc-run calls it after file execution.
//
// Usage:
//
//	sc-run <script.star|archive.txt|archive.txtar>
//
// Examples:
//
//	sc-run hello.star
//	sc-run machine.txtar
package main
