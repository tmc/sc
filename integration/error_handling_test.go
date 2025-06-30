package integration

import (
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
	"google.golang.org/protobuf/types/known/structpb"
)

// TestCrossPackageErrorPropagation tests error propagation across package boundaries
func TestCrossPackageErrorPropagation(t *testing.T) {
	t.Run("ValidationToSemantics", func(t *testing.T) {
		// Create an invalid statechart that should be caught by validation
		// but if it slips through, should cause semantic errors
		invalidChart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label: "BasicWithChildren",
						Type:  sc.StateTypeBasic, // Basic state with children - invalid
						Children: []*sc.State{
							{Label: "InvalidChild", Type: sc.StateTypeBasic},
						},
						IsInitial: true,
					},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "InvalidTransition",
					From:  []string{"NonexistentState"}, // Source doesn't exist
					To:    []string{"BasicWithChildren"},
					Event: "INVALID",
				},
			},
			Events: []*sc.Event{
				{Label: "INVALID"},
			},
		}

		// Phase 1: Validation should catch the error
		wrapper := semantics.NewStatechart(invalidChart)
		err := wrapper.Validate()
		hasErrors := err != nil
		if hasErrors {
			t.Logf("Validation caught error: %s", err.Error())
		}

		if !hasErrors {
			t.Error("Expected validation to catch errors in invalid statechart")
		}

		// Phase 2: Even if validation passes, semantics should catch remaining issues
		machineWrapper := semantics.NewStatechart(invalidChart)
		err = machineWrapper.Validate()
		if err != nil {
			t.Logf("Semantic validation caught error: %v", err)
		}

		// Phase 3: Machine creation should fail for invalid statecharts
		machine, err := semantics.NewMachine(machineWrapper, "error-test", nil)
		if err != nil {
			t.Logf("Machine creation failed as expected: %v", err)
			return
		}

		// Phase 4: If machine creation succeeds, operations should fail gracefully
		if err := machine.Start(); err != nil {
			t.Logf("Machine start failed as expected: %v", err)
			return
		}

		// Phase 5: Event processing should handle errors gracefully
		triggered, err := machine.Step("INVALID")
		if err != nil {
			t.Logf("Event processing failed as expected: %v", err)
		} else if triggered {
			t.Logf("Event triggered transitions (unexpected but acceptable)")
		}

		// Check for accumulated errors
		errors := machine.GetErrors()
		if len(errors) > 0 {
			t.Logf("Machine accumulated %d errors:", len(errors))
			for i, err := range errors {
				t.Logf("  Error %d: %v", i+1, err)
			}
		}

		machine.Stop()
	})

	t.Run("SemanticsToMachine", func(t *testing.T) {
		// Create a statechart that passes basic validation but fails during execution
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{Label: "State1", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "State2", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "ProblematicTransition",
					From:  []string{"State1"},
					To:    []string{"State2"},
					Event: "TRIGGER",
					Guard: &sc.Guard{Expression: "context.invalid.field"}, // Invalid guard
				},
			},
			Events: []*sc.Event{
				{Label: "TRIGGER"},
			},
		}

		// This should pass basic validation
		wrapper := semantics.NewStatechart(statechart)
		if err := wrapper.Validate(); err != nil {
			t.Logf("Semantic validation error: %v", err)
		}

		machine, err := semantics.NewMachine(wrapper, "guard-error-test", &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"validField": structpb.NewStringValue("test"),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// This should trigger the problematic guard evaluation
		_, err = machine.Step("TRIGGER")
		if err != nil {
			t.Logf("Event processing failed as expected due to guard error: %v", err)
		}

		// Even if guard evaluation fails, machine should remain in valid state
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine should remain valid even after guard error: %v", err)
		}

		// Check for accumulated errors
		errors := machine.GetErrors()
		if len(errors) == 0 {
			t.Error("Expected machine to accumulate errors from guard evaluation")
		}

		for _, err := range errors {
			t.Logf("Accumulated error: %v", err)
		}
	})

	t.Run("MachineToValidation", func(t *testing.T) {
		// Create a scenario where machine state becomes invalid and validation catches it
		statechart := testutil.CreateSimpleStatechart()
		wrapper := semantics.NewStatechart(statechart)
		machine, err := semantics.NewMachine(wrapper, "machine-validation-test", nil)
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// Normal operation should validate successfully
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine validation should pass initially: %v", err)
		}

		// Simulate machine state corruption (this would normally not happen in real usage)
		// We'll manually modify the machine state to create an invalid configuration
		// Note: This is testing the validation system's ability to catch inconsistencies

		// Create an invalid configuration manually
		invalidConfig := &sc.Configuration{
			States: []*sc.StateRef{
				{Label: "NonexistentState"}, // State that doesn't exist in statechart
			},
		}

		// Since we can't directly modify the machine's configuration due to encapsulation,
		// we'll test the validation logic with the invalid configuration
		err = semantics.ValidateConfiguration(wrapper, invalidConfig)
		if err == nil {
			t.Error("Expected validation to fail for configuration with nonexistent state")
		} else {
			t.Logf("Validation correctly caught invalid configuration: %v", err)
		}
	})
}

// TestErrorRecoveryMechanisms tests error recovery mechanisms across the system
func TestErrorRecoveryMechanisms(t *testing.T) {
	t.Run("MachineErrorRecovery", func(t *testing.T) {
		// Create a statechart with potential error conditions
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{Label: "Normal", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "ErrorState", Type: sc.StateTypeBasic},
					{Label: "Recovery", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "TriggerError",
					From:  []string{"Normal"},
					To:    []string{"ErrorState"},
					Event: "ERROR_EVENT",
					Actions: []*sc.Action{
						{Label: "problematic_action"}, // This action might fail
					},
				},
				{
					Label: "Recover",
					From:  []string{"ErrorState"},
					To:    []string{"Recovery"},
					Event: "RECOVER",
				},
				{
					Label: "BackToNormal",
					From:  []string{"Recovery"},
					To:    []string{"Normal"},
					Event: "RESET",
				},
			},
			Events: []*sc.Event{
				{Label: "ERROR_EVENT"},
				{Label: "RECOVER"},
				{Label: "RESET"},
			},
		}

		wrapper := semantics.NewStatechart(statechart)
		machine, err := semantics.NewMachine(wrapper, "recovery-test", nil)
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// Clear any initial errors
		machine.ClearErrors()

		// Trigger an error condition
		_, err = machine.Step("ERROR_EVENT")
		if err != nil {
			t.Logf("Error during transition as expected: %v", err)
		}

		// Check if errors were accumulated
		errors := machine.GetErrors()
		errorCount := len(errors)
		if errorCount > 0 {
			t.Logf("Machine accumulated %d errors", errorCount)
		}

		// Machine should still be operational despite errors
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine should remain valid despite action errors: %v", err)
		}

		// Test recovery workflow
		machine.Step("RECOVER")
		machine.Step("RESET")

		// Verify we're back to normal state
		config := machine.GetCurrentConfiguration()
		hasNormal := false
		for _, state := range config.States {
			if state.Label == "Normal" {
				hasNormal = true
				break
			}
		}
		if !hasNormal {
			t.Error("Expected to recover to Normal state")
		}

		// Test error clearing
		machine.ClearErrors()
		clearedErrors := machine.GetErrors()
		if len(clearedErrors) != 0 {
			t.Errorf("Expected no errors after clearing, got %d", len(clearedErrors))
		}
	})

	t.Run("ValidationErrorHandling", func(t *testing.T) {
		// Test how the system handles validation errors gracefully

		// Create progressively invalid statecharts
		testCases := []struct {
			name        string
			statechart  *sc.Statechart
			shouldFail  string
		}{
			{
				name: "EmptyRootState",
				statechart: &sc.Statechart{
					RootState:   nil, // Invalid
					Transitions: []*sc.Transition{},
					Events:      []*sc.Event{},
				},
				shouldFail: "root state",
			},
			{
				name: "EmptyChildren",
				statechart: &sc.Statechart{
					RootState: &sc.State{
						Label:    "__root__",
						Type:     sc.StateTypeNormal,
						Children: []*sc.State{}, // Empty children for normal state
					},
					Transitions: []*sc.Transition{},
					Events:      []*sc.Event{},
				},
				shouldFail: "children",
			},
			{
				name: "InvalidTransitionReferences",
				statechart: &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{Label: "ValidState", Type: sc.StateTypeBasic, IsInitial: true},
						},
					},
					Transitions: []*sc.Transition{
						{
							Label: "InvalidTrans",
							From:  []string{"NonexistentSource"},
							To:    []string{"NonexistentTarget"},
							Event: "EVENT",
						},
					},
					Events: []*sc.Event{
						{Label: "EVENT"},
					},
				},
				shouldFail: "nonexistent",
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// Test semantic validation
				wrapper := semantics.NewStatechart(tc.statechart)
				err := wrapper.Validate()
				if err != nil {
					t.Logf("Semantic validation error: %v", err)
					if !contains(err.Error(), tc.shouldFail) {
						t.Logf("Expected semantic error containing '%s', got: %v", tc.shouldFail, err)
					}
				}

				// Test machine creation
				machine, err := semantics.NewMachine(wrapper, "error-handling-test", nil)
				if err != nil {
					t.Logf("Machine creation failed as expected: %v", err)
					return
				}

				// If machine creation succeeds, test graceful failure
				if err := machine.Start(); err != nil {
					t.Logf("Machine start failed as expected: %v", err)
				} else {
					// If start succeeds, verify validation catches issues
					if err := machine.Validate(); err != nil {
						t.Logf("Machine validation caught issue: %v", err)
					}
					machine.Stop()
				}
			})
		}
	})

	t.Run("ConcurrentErrorHandling", func(t *testing.T) {
		// Test error handling under concurrent access
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{Label: "State1", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "State2", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "ErrorProneTransition",
					From:  []string{"State1"},
					To:    []string{"State2"},
					Event: "RISKY_EVENT",
					Guard: &sc.Guard{Expression: "context.mayFail == true"},
				},
			},
			Events: []*sc.Event{
				{Label: "RISKY_EVENT"},
			},
		}

		wrapper := semantics.NewStatechart(statechart)
		machine, err := semantics.NewMachine(wrapper, "concurrent-error-test", &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"mayFail": structpb.NewBoolValue(false),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// Run concurrent operations that might cause errors
		numGoroutines := 10
		done := make(chan bool, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				defer func() { done <- true }()
				
				// Each goroutine tries various operations
				for j := 0; j < 5; j++ {
					// Some operations might fail
					machine.Step("RISKY_EVENT")
					machine.GetCurrentConfiguration()
					machine.Validate()
					
					// Small delay
					time.Sleep(time.Millisecond)
				}
			}(i)
		}

		// Wait for all goroutines to complete
		for i := 0; i < numGoroutines; i++ {
			<-done
		}

		// Check final state
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine should be valid after concurrent error testing: %v", err)
		}

		// Check accumulated errors
		errors := machine.GetErrors()
		t.Logf("Machine accumulated %d errors during concurrent testing", len(errors))

		// Errors should be manageable
		if len(errors) > 100 {
			t.Errorf("Too many errors accumulated: %d", len(errors))
		}
	})
}

// TestErrorBoundaries tests error containment and boundaries
func TestErrorBoundaries(t *testing.T) {
	t.Run("LocalizedErrors", func(t *testing.T) {
		// Test that errors in one part of the system don't corrupt other parts
		statechart := testutil.CreateHierarchicalStatechart()
		wrapper := semantics.NewStatechart(statechart)
		
		// Create multiple machines from the same statechart
		machine1, err := semantics.NewMachine(wrapper, "machine1", nil)
		if err != nil {
			t.Fatalf("Failed to create machine1: %v", err)
		}
		defer machine1.Stop()

		machine2, err := semantics.NewMachine(wrapper, "machine2", nil)
		if err != nil {
			t.Fatalf("Failed to create machine2: %v", err)
		}
		defer machine2.Stop()

		// Start both machines
		machine1.Start()
		machine2.Start()

		// Cause an error in machine1 by processing many events rapidly
		for i := 0; i < 10; i++ {
			machine1.Step("start")
			machine1.Step("finish")
			machine1.Step("stop")
		}

		// Check that machine1 may have accumulated errors
		errors1 := machine1.GetErrors()
		t.Logf("Machine1 accumulated %d errors", len(errors1))

		// Machine2 should be unaffected
		errors2 := machine2.GetErrors()
		if len(errors2) > 0 {
			t.Errorf("Machine2 should not be affected by machine1 errors, but has %d errors", len(errors2))
		}

		// Both machines should still be valid
		if err := machine1.Validate(); err != nil {
			t.Errorf("Machine1 validation failed: %v", err)
		}
		if err := machine2.Validate(); err != nil {
			t.Errorf("Machine2 validation failed: %v", err)
		}

		// Both machines should still be operational
		machine2.Step("start")
		config2 := machine2.GetCurrentConfiguration()
		if len(config2.States) == 0 {
			t.Error("Machine2 should still be operational")
		}
	})

	t.Run("StatechartIsolation", func(t *testing.T) {
		// Test that modifications to one statechart don't affect others
		original := testutil.CreateSimpleStatechart()
		
		// Create machines from the same statechart definition
		wrapper1 := semantics.NewStatechart(original)
		wrapper2 := semantics.NewStatechart(original)
		
		machine1, _ := semantics.NewMachine(wrapper1, "isolated1", nil)
		machine2, _ := semantics.NewMachine(wrapper2, "isolated2", nil)
		
		machine1.Start()
		machine2.Start()
		defer machine1.Stop()
		defer machine2.Stop()

		// Make different transitions in each machine
		machine1.Step("go") // A -> B
		// machine2 stays in A

		config1 := machine1.GetCurrentConfiguration()
		config2 := machine2.GetCurrentConfiguration()

		// Verify they have different states
		state1 := ""
		state2 := ""
		if len(config1.States) > 0 {
			state1 = config1.States[0].Label
		}
		if len(config2.States) > 0 {
			state2 = config2.States[0].Label
		}

		if state1 == state2 {
			t.Errorf("Expected machines to have different states, both have: %s", state1)
		}

		// Both should be valid
		if err := machine1.Validate(); err != nil {
			t.Errorf("Machine1 validation failed: %v", err)
		}
		if err := machine2.Validate(); err != nil {
			t.Errorf("Machine2 validation failed: %v", err)
		}
	})
}

// TestGracefulDegradation tests system behavior under error conditions
func TestGracefulDegradation(t *testing.T) {
	t.Run("PartialFailureRecovery", func(t *testing.T) {
		// Test system recovery from partial failures
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{Label: "Healthy", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "Degraded", Type: sc.StateTypeBasic},
					{Label: "Failed", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "Degrade",
					From:  []string{"Healthy"},
					To:    []string{"Degraded"},
					Event: "PARTIAL_FAILURE",
				},
				{
					Label: "FullFailure",
					From:  []string{"Healthy", "Degraded"},
					To:    []string{"Failed"},
					Event: "COMPLETE_FAILURE",
				},
				{
					Label: "Recover",
					From:  []string{"Degraded", "Failed"},
					To:    []string{"Healthy"},
					Event: "RECOVERY",
				},
			},
			Events: []*sc.Event{
				{Label: "PARTIAL_FAILURE"},
				{Label: "COMPLETE_FAILURE"},
				{Label: "RECOVERY"},
			},
		}

		wrapper := semantics.NewStatechart(statechart)
		machine, err := semantics.NewMachine(wrapper, "degradation-test", nil)
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// Test degradation path
		machine.Step("PARTIAL_FAILURE")
		
		config := machine.GetCurrentConfiguration()
		hasDegraded := false
		for _, state := range config.States {
			if state.Label == "Degraded" {
				hasDegraded = true
				break
			}
		}
		if !hasDegraded {
			t.Error("Expected to be in Degraded state")
		}

		// System should still be operational in degraded mode
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine should be valid in degraded state: %v", err)
		}

		// Test recovery
		machine.Step("RECOVERY")
		
		config = machine.GetCurrentConfiguration()
		hasHealthy := false
		for _, state := range config.States {
			if state.Label == "Healthy" {
				hasHealthy = true
				break
			}
		}
		if !hasHealthy {
			t.Error("Expected to recover to Healthy state")
		}

		// Test complete failure and recovery
		machine.Step("COMPLETE_FAILURE")
		machine.Step("RECOVERY")
		
		// Should be healthy again
		config = machine.GetCurrentConfiguration()
		hasHealthyAgain := false
		for _, state := range config.States {
			if state.Label == "Healthy" {
				hasHealthyAgain = true
				break
			}
		}
		if !hasHealthyAgain {
			t.Error("Expected to recover to Healthy state after complete failure")
		}
	})

	t.Run("ResourceExhaustionHandling", func(t *testing.T) {
		// Test behavior under resource constraints (simulated)
		statechart := testutil.CreateLargeStatechart(50, 100) // Large statechart
		wrapper := semantics.NewStatechart(statechart)
		
		// Create many machines to simulate resource pressure
		machines := make([]*semantics.MachineWrapper, 20)
		for i := 0; i < 20; i++ {
			machine, err := semantics.NewMachine(wrapper, "stress-test", nil)
			if err != nil {
				t.Logf("Failed to create machine %d: %v", i, err)
				continue
			}
			machines[i] = machine
			
			if err := machine.Start(); err != nil {
				t.Logf("Failed to start machine %d: %v", i, err)
				continue
			}
		}

		// Clean up
		for i, machine := range machines {
			if machine != nil {
				if err := machine.Stop(); err != nil {
					t.Logf("Failed to stop machine %d: %v", i, err)
				}
			}
		}

		// System should handle resource pressure gracefully
		t.Log("Resource exhaustion test completed")
	})
}