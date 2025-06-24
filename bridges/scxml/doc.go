// Package scxml provides bidirectional interoperability with the W3C State Chart XML
// (SCXML) standard format.
//
// SCXML is a W3C recommendation that provides a generic state-machine-based
// execution environment based on Harel statecharts. It uses XML syntax to
// describe statechart semantics in a standardized, interoperable format.
//
// # Conversion Features
//
//   - Import SCXML documents to Harel core semantics
//   - Export Harel statecharts to SCXML-compliant XML
//   - Handle SCXML datamodel and executable content
//   - Support for SCXML transitions, guards, and actions
//
// # Semantic Mapping
//
// SCXML features are mapped to Harel core as follows:
//   - SCXML <state> → Harel State
//   - SCXML <parallel> → Harel Parallel State  
//   - SCXML <transition> → Harel Transition
//   - SCXML <datamodel> → Harel Machine Context
//   - SCXML executable content → Harel Actions
//
// # Usage Example
//
//	// Import SCXML document
//	converter := scxml.NewConverter()
//	statechart, err := converter.ImportFromFile("machine.scxml")
//	
//	// Export to SCXML format
//	scxmlDoc, err := converter.Export(statechart)
//
// # Standards Compliance
//
// This implementation aims for compliance with:
//   - W3C SCXML 1.0 Recommendation
//   - SCXML execution semantics
//   - SCXML datamodel patterns
//
// # Limitations
//
// Some SCXML features are simplified or not supported:
//   - Complex datamodel expressions are limited
//   - External communication is simplified
//   - Platform-specific extensions are ignored
package scxml