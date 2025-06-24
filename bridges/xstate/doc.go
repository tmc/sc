// Package xstate provides bidirectional interoperability with the XState JavaScript
// statechart library and Stately.ai visual editor.
//
// XState is a popular JavaScript library for creating, interpreting, and executing
// finite state machines and statecharts. It follows SCXML semantics and provides
// extensive features for modern web applications.
//
// # Conversion Features
//
//   - Import XState machine definitions to Harel core semantics
//   - Export Harel statecharts to XState-compatible JSON
//   - Handle XState actors, services, and context mapping
//   - Support for XState's extended SCXML features
//
// # Semantic Mapping
//
// XState extends SCXML with additional features that are mapped as follows:
//   - XState actors → Harel event processors
//   - XState services → Harel action definitions  
//   - XState context → Harel machine context
//   - XState guards → Harel transition guards
//
// # Usage Example
//
//	// Import XState machine definition
//	converter := xstate.NewConverter()
//	statechart, err := converter.Import(xstateMachine)
//	
//	// Export to XState format
//	xstateMachine, err := converter.Export(statechart)
//
// # Limitations
//
// Some XState features don't have direct Harel equivalents:
//   - Invoke/spawn semantics are simplified
//   - Advanced actor patterns are flattened
//   - Custom XState interpreters aren't supported
package xstate
