// Command event_demo runs a small interactive demonstration of statechart
// event processing.
//
// The program constructs a turnstile machine, feeds it a sequence of events
// through the semantics event processor, enables tracing, and prints the
// resulting machine state and trace information. It is intended as a runnable
// example of prioritized events, filters, and asynchronous processing.
//
// Usage:
//
//	event_demo [flags]
//
// Flags:
//
//	-version   Print version information and exit.
//
// Examples:
//
//	event_demo
//	event_demo -version
package main
