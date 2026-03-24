// Package semantics provides the core semantic operations for statecharts.
//
// This package implements Harel's statechart semantics including:
//
// # Reconciled Paper Semantics
//
// The package also provides an explicit implementation of the operational
// semantics formalized in Rik Eshuis, "Reconciling statechart semantics":
//
//   - fixpoint
//   - Statemate
//   - single-event Statemate
//   - UML
//
// Use [Statechart.Reactions] or [Statechart.ReactionsWithOptions] when you
// need the paper's execution models. The older [MachineWrapper] and
// [EventProcessor] types remain useful runtime helpers, but they are not by
// themselves the paper-faithful execution model.
//
// Internal event generation for the paper semantics engine is represented by
// transition actions whose labels start with `raise:`, `emit:`, or `send:`.
//
// # Transition Execution
//
// The transition execution system follows the formal semantics defined by David Harel
// in his original statechart formulation. The main components include:
//
//  1. Finding enabled transitions for a given event and configuration
//  2. Resolving conflicts between competing transitions using priority rules
//  3. Executing selected transitions with proper entry/exit semantics
//  4. Supporting complex transition patterns (compound, cross-region)
//
// # Conflict Resolution
//
// When multiple transitions are enabled for the same event, conflicts are resolved using
// these priority rules:
//
//  1. Transitions from deeper states have higher priority (inner-to-outer precedence)
//  2. Among transitions at the same hierarchical level, lexicographic ordering is used
//  3. Transitions that share source states or ancestrally related source states conflict
//
// # Entry/Exit Actions
//
// The system supports entry and exit actions through a convention-based approach:
//
//   - Entry actions: Named with prefix "entry_" + state label
//   - Exit actions: Named with prefix "exit_" + state label
//   - Actions are executed via a pluggable ActionRegistry system
//
// # Complex Transitions
//
// Advanced transition types are supported:
//
//   - Compound Transitions: Multiple atomic transitions executed atomically
//   - Cross-Region Transitions: Transitions that affect multiple orthogonal regions
//   - Hierarchical Transitions: Transitions crossing multiple hierarchy levels
//
// # Usage Examples
//
// Basic transition execution:
//
//	statechart := semantics.NewStatechart(protoStatechart)
//	result, err := statechart.ExecuteTransitions(config, context, "event_name")
//
// Register custom actions:
//
//	semantics.RegisterGlobalAction("my_action", func(ctx *structpb.Struct) error {
//		// Custom action logic
//		return nil
//	})
//
// Execute compound transitions:
//
//	compound := &semantics.CompoundTransition{
//		Label:       "compound_transition",
//		Transitions: []*sc.Transition{t1, t2},
//		Event:       "COMPOUND_EVENT",
//		Actions:     []*sc.Action{{Label: "compound_action"}},
//	}
//	result, err := statechart.ExecuteCompoundTransition(compound, config, context)
//
// # State Management
//
// The package provides comprehensive state hierarchy management including:
//
//   - Finding least common ancestors
//   - Determining ancestral relationships
//   - Checking orthogonality between states
//   - Configuration validation and default completion
//
// All operations are designed to be deterministic and follow the formal semantics
// to ensure correct statechart behavior across different execution contexts.
package semantics
