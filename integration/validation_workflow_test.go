package integration

import (
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
)

// TestValidationToExecutionWorkflow tests the complete workflow from validation to execution
func TestValidationToExecutionWorkflow(t *testing.T) {
	testCases := []struct {
		name            string
		description     string
		createChart     func() *sc.Statechart
		expectedValid   bool
		expectedErrors  []string
		testExecution   bool
		executionEvents []string
	}{
		{
			name:        "ValidSimpleChart",
			description: "Valid simple statechart should pass validation and execute correctly",
			createChart: func() *sc.Statechart {
				return testutil.CreateSimpleStatechart()
			},
			expectedValid:   true,
			expectedErrors:  nil,
			testExecution:   true,
			executionEvents: []string{"go"},
		},
		{
			name:        "ValidHierarchicalChart",
			description: "Valid hierarchical statechart should pass validation and execute correctly",
			createChart: func() *sc.Statechart {
				return testutil.CreateHierarchicalStatechart()
			},
			expectedValid:   true,
			expectedErrors:  nil,
			testExecution:   true,
			executionEvents: []string{"start", "finish"},
		},
		{
			name:        "ValidOrthogonalChart",
			description: "Valid orthogonal statechart should pass validation and execute correctly",
			createChart: func() *sc.Statechart {
				return testutil.CreateOrthogonalStatechart()
			},
			expectedValid:   true,
			expectedErrors:  nil,
			testExecution:   true,
			executionEvents: []string{"switch_a", "switch_b"},
		},
		{
			name:        "InvalidDuplicateStates",
			description: "Statechart with duplicate state labels should fail validation",
			createChart: func() *sc.Statechart {
				return &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{Label: "DuplicateLabel", Type: sc.StateTypeBasic, IsInitial: true},
							{Label: "DuplicateLabel", Type: sc.StateTypeBasic}, // Duplicate!
						},
					},
					Transitions: []*sc.Transition{},
					Events:      []*sc.Event{},
				}
			},
			expectedValid:   false,
			expectedErrors:  []string{"duplicate", "label"},
			testExecution:   false,
			executionEvents: nil,
		},
		{
			name:        "InvalidBasicWithChildren",
			description: "Basic state with children should fail validation",
			createChart: func() *sc.Statechart {
				return &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{
								Label: "BasicWithChildren",
								Type:  sc.StateTypeBasic, // Basic type but has children
								Children: []*sc.State{
									{Label: "Child", Type: sc.StateTypeBasic},
								},
								IsInitial: true,
							},
						},
					},
					Transitions: []*sc.Transition{},
					Events:      []*sc.Event{},
				}
			},
			expectedValid:   false,
			expectedErrors:  []string{"basic", "children"},
			testExecution:   false,
			executionEvents: nil,
		},
		{
			name:        "InvalidCompoundWithoutChildren",
			description: "Compound state without children should fail validation",
			createChart: func() *sc.Statechart {
				return &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{
								Label:     "CompoundWithoutChildren",
								Type:      sc.StateTypeNormal, // Compound type but no children
								Children:  []*sc.State{},      // Empty children
								IsInitial: true,
							},
						},
					},
					Transitions: []*sc.Transition{},
					Events:      []*sc.Event{},
				}
			},
			expectedValid:   false,
			expectedErrors:  []string{"compound", "children"},
			testExecution:   false,
			executionEvents: nil,
		},
		{
			name:        "InvalidMultipleDefaultChildren",
			description: "State with multiple initial children should fail validation",
			createChart: func() *sc.Statechart {
				return &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{
								Label: "ParentState",
								Type:  sc.StateTypeNormal,
								Children: []*sc.State{
									{Label: "Child1", Type: sc.StateTypeBasic, IsInitial: true},  // Multiple
									{Label: "Child2", Type: sc.StateTypeBasic, IsInitial: true},  // initial
									{Label: "Child3", Type: sc.StateTypeBasic, IsInitial: false}, // children
								},
								IsInitial: true,
							},
						},
					},
					Transitions: []*sc.Transition{},
					Events:      []*sc.Event{},
				}
			},
			expectedValid:   false,
			expectedErrors:  []string{"single", "default"},
			testExecution:   false,
			executionEvents: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Phase 1: Create the statechart
			statechart := tc.createChart()
			if statechart == nil {
				t.Fatal("Failed to create statechart")
			}

			// Phase 2: Perform validation using semantic validation
			wrapper := semantics.NewStatechart(statechart)
			err := wrapper.Validate()
			hasErrors := err != nil
			if hasErrors {
				t.Logf("Validation error: %s", err.Error())
			}

			if tc.expectedValid && hasErrors {
				t.Fatalf("Expected statechart to be valid, but validation found errors")
			}

			if !tc.expectedValid && !hasErrors {
				t.Fatalf("Expected statechart to be invalid, but validation passed")
			}

			// Phase 4: Check for expected error messages
			if !tc.expectedValid && len(tc.expectedErrors) > 0 && err != nil {
				errorMessage := err.Error()
				for _, expectedError := range tc.expectedErrors {
					if !contains(errorMessage, expectedError) {
						t.Errorf("Expected validation error containing '%s', but not found in: %s", 
							expectedError, errorMessage)
					}
				}
			}

			// Phase 5: Test execution workflow if validation passed
			if tc.testExecution && tc.expectedValid {
				t.Run("ExecutionWorkflow", func(t *testing.T) {
					// wrapper is already created and validated above

					// Create machine
					machine, err := semantics.NewMachine(wrapper, "workflow-test", nil)
					if err != nil {
						t.Fatalf("Failed to create machine: %v", err)
					}

					// Start machine
					if err := machine.Start(); err != nil {
						t.Fatalf("Failed to start machine: %v", err)
					}

					// Get initial configuration
					initialConfig := machine.GetCurrentConfiguration()
					if len(initialConfig.States) == 0 {
						t.Error("Expected initial configuration to have at least one state")
					}

					// Process execution events
					for i, event := range tc.executionEvents {
						triggered, err := machine.Step(event)
						if err != nil {
							t.Fatalf("Failed to process event '%s' (step %d): %v", event, i+1, err)
						}
						t.Logf("Event '%s' triggered transitions: %t", event, triggered)

						// Validate machine state after each step
						if err := machine.Validate(); err != nil {
							t.Errorf("Machine validation failed after event '%s': %v", event, err)
						}
					}

					// Get final configuration
					finalConfig := machine.GetCurrentConfiguration()
					if len(finalConfig.States) == 0 {
						t.Error("Expected final configuration to have at least one state")
					}

					// Verify step history
					history := machine.GetStepHistory()
					expectedSteps := 1 + len(tc.executionEvents) // start + events
					if len(history) < expectedSteps {
						t.Errorf("Expected at least %d steps in history, got %d", expectedSteps, len(history))
					}

					// Stop machine
					if err := machine.Stop(); err != nil {
						t.Fatalf("Failed to stop machine: %v", err)
					}

					// Final validation
					if err := machine.Validate(); err != nil {
						t.Errorf("Final machine validation failed: %v", err)
					}

					// Check for accumulated errors
					errors := machine.GetErrors()
					if len(errors) > 0 {
						t.Logf("Machine accumulated %d errors during execution:", len(errors))
						for i, err := range errors {
							t.Logf("  Error %d: %v", i+1, err)
						}
					}
				})
			}
		})
	}
}

// TestValidationRuleIntegration tests integration of different validation rules
func TestValidationRuleIntegration(t *testing.T) {
	t.Run("SelectiveRuleApplication", func(t *testing.T) {
		// Create a statechart that violates multiple rules
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeNormal,
				Children: []*sc.State{
					{
						Label: "BasicWithChildren",
						Type:  sc.StateTypeBasic, // Violates BASIC_HAS_NO_CHILDREN
						Children: []*sc.State{
							{Label: "Child", Type: sc.StateTypeBasic},
						},
						IsInitial: true,
					},
					{
						Label: "BasicWithChildren", // Violates UNIQUE_STATE_LABELS
						Type:  sc.StateTypeBasic,
					},
				},
			},
			Transitions: []*sc.Transition{},
			Events:      []*sc.Event{},
		}

		// Test with semantic validation - should catch multiple errors
		t.Run("AllRulesEnabled", func(t *testing.T) {
			wrapper := semantics.NewStatechart(statechart)
			err := wrapper.Validate()
			
			if err == nil {
				t.Error("Expected validation to fail due to multiple rule violations")
			} else {
				t.Logf("Validation failed as expected: %v", err)
				// Check that error contains expected violations
				errorMsg := err.Error()
				hasBasicError := contains(errorMsg, "basic") || contains(errorMsg, "children")
				hasDuplicateError := contains(errorMsg, "duplicate") || contains(errorMsg, "unique")
				
				if !hasBasicError && !hasDuplicateError {
					t.Errorf("Expected validation error to mention basic/children or duplicate/unique violations, got: %v", err)
				}
			}
		})

		// Test semantic validation behavior (note: semantic validation doesn't have rule ignoring)
		t.Run("SemanticValidationBehavior", func(t *testing.T) {
			wrapper := semantics.NewStatechart(statechart)
			err := wrapper.Validate()
			
			if err == nil {
				t.Error("Expected validation to fail")
			} else {
				t.Logf("Semantic validation error: %v", err)
				// Semantic validation will catch structural issues
				errorMsg := err.Error()
				if !contains(errorMsg, "basic") && !contains(errorMsg, "duplicate") && !contains(errorMsg, "children") {
					t.Logf("Note: semantic validation may catch different errors than rule-based validation")
				}
			}
		})
	})
}

// TestValidationWithSemanticExecution tests that validation catches errors that would cause semantic execution to fail
func TestValidationWithSemanticExecution(t *testing.T) {
	testCases := []struct {
		name         string
		createChart  func() *sc.Statechart
		shouldPass   bool
		executionErr string
	}{
		{
			name: "TransitionToNonexistentState",
			createChart: func() *sc.Statechart {
				return &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
							{Label: "B", Type: sc.StateTypeBasic},
						},
					},
					Transitions: []*sc.Transition{
						{
							Label: "AtoNonexistent",
							From:  []string{"A"},
							To:    []string{"NonexistentState"}, // Invalid target
							Event: "go",
						},
					},
					Events: []*sc.Event{
						{Label: "go"},
					},
				}
			},
			shouldPass:   false,
			executionErr: "nonexistent",
		},
		{
			name: "TransitionFromNonexistentState",
			createChart: func() *sc.Statechart {
				return &sc.Statechart{
					RootState: &sc.State{
						Label: "__root__",
						Type:  sc.StateTypeNormal,
						Children: []*sc.State{
							{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
							{Label: "B", Type: sc.StateTypeBasic},
						},
					},
					Transitions: []*sc.Transition{
						{
							Label: "NonexistentToB",
							From:  []string{"NonexistentState"}, // Invalid source
							To:    []string{"B"},
							Event: "go",
						},
					},
					Events: []*sc.Event{
						{Label: "go"},
					},
				}
			},
			shouldPass:   false,
			executionErr: "nonexistent",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			statechart := tc.createChart()

			// Validate first using semantic validation
			wrapper := semantics.NewStatechart(statechart)
			err := wrapper.Validate()
			hasErrors := err != nil
			
			if hasErrors {
				t.Logf("Validation error: %v", err)
			}

			if tc.shouldPass && hasErrors {
				t.Fatalf("Expected validation to pass, but found errors")
			}

			if !tc.shouldPass && !hasErrors {
				// If validation didn't catch the error, execution should fail
				t.Log("Validation didn't catch error, testing execution failure")
				
				wrapper := semantics.NewStatechart(statechart)
				machine, err := semantics.NewMachine(wrapper, "error-test", nil)
				if err != nil {
					t.Logf("Machine creation failed as expected: %v", err)
					return
				}

				if err := machine.Start(); err != nil {
					t.Logf("Machine start failed as expected: %v", err)
					return
				}

				// Try to execute the problematic transition
				_, err = machine.Step("go")
				if err == nil {
					t.Error("Expected execution to fail, but it succeeded")
				} else {
					t.Logf("Execution failed as expected: %v", err)
					if tc.executionErr != "" && !contains(err.Error(), tc.executionErr) {
						t.Errorf("Expected error to contain '%s', got: %v", tc.executionErr, err)
					}
				}
			}
		})
	}
}

// contains checks if string s contains substring
func contains(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}