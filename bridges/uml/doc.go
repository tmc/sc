// Package uml provides bidirectional interoperability with UML State Machine
// diagrams and XMI format.
//
// UML State Machines are an object-based variant of Harel statecharts,
// adapted and extended by the Unified Modeling Language specification.
// They introduce hierarchically nested states and are commonly used in
// software engineering and system design.
//
// # Conversion Features
//
//   - Import UML State Machine models from XMI format
//   - Export Harel statecharts to UML-compatible models
//   - Handle UML regions, pseudostates, and stereotypes
//   - Support for UML state machine behavioral semantics
//
// # Semantic Mapping
//
// UML State Machine features are mapped to Harel core as follows:
//   - UML State → Harel State
//   - UML CompositeState → Harel Normal State
//   - UML Region → Harel Orthogonal decomposition
//   - UML Transition → Harel Transition
//   - UML Guard → Harel Guard condition
//   - UML Effect → Harel Action
//
// # Usage Example
//
//	// Import UML model from XMI
//	converter := uml.NewConverter()
//	statechart, err := converter.ImportFromXMI("model.xmi")
//	
//	// Export to UML format
//	umlModel, err := converter.Export(statechart)
//
// # Standards Compliance
//
// This implementation supports:
//   - UML 2.x State Machine semantics
//   - XMI 2.x serialization format
//   - UML behavioral features subset
//
// # Limitations
//
// Some UML features are simplified:
//   - Behavioral State Machines only (no Protocol State Machines)
//   - Limited support for UML Actions and Activities
//   - Pseudostates are simplified to basic semantics
//   - UML profiles and stereotypes are preserved as metadata
package uml