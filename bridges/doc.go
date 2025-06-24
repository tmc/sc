// Package bridges provides bidirectional conversion utilities between
// the core Harel statechart formalism and external state machine formats.
//
// The bridges package maintains the academic purity of the core Harel
// formalism while enabling interoperability with popular state machine
// libraries and standards including XState, SCXML, UML State Machines,
// Qt State Charts, and others.
//
// # Architecture
//
// Each bridge package provides:
//   - Import: Convert external format → Harel core semantics
//   - Export: Convert Harel core semantics → external format
//   - Validation: Ensure semantic consistency during conversion
//   - Mapping: Handle differences between formalisms gracefully
//
// # Supported Bridges
//
//   - xstate: XState/Stately.ai JavaScript statechart library
//   - scxml: W3C State Chart XML standard
//   - uml: UML State Machine diagrams
//
// # Design Principles
//
//   - Preserve core Harel formalism semantics
//   - Provide bidirectional conversion capabilities  
//   - Handle semantic differences through explicit mapping
//   - Maintain type safety throughout conversion process
//   - Document conversion limitations and assumptions
package bridges