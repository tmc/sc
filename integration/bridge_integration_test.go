package integration

import (
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/xstate"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
)

// TestBridgeToExecutionWorkflow tests the complete workflow from external format to execution
func TestBridgeToExecutionWorkflow(t *testing.T) {
	testCases := []struct {
		name          string
		description   string
		setupXState   func() xstate.Machine
		executionTest func(t *testing.T, statechart *sc.Statechart)
	}{
		{
			name:        "SimpleXStateImport",
			description: "Import simple XState machine and execute",
			setupXState: func() xstate.Machine {
				return xstate.Machine{
					ID:      "simple-machine",
					Initial: "idle",
					States: map[string]*xstate.State{
						"idle": {
							On: map[string]*xstate.Transition{
								"START": {Target: "active"},
							},
						},
						"active": {
							On: map[string]*xstate.Transition{
								"STOP": {Target: "idle"},
							},
						},
					},
				}
			},
			executionTest: func(t *testing.T, statechart *sc.Statechart) {
				wrapper := semantics.NewStatechart(statechart)
				machine, err := semantics.NewMachine(wrapper, "xstate-import-test", nil)
				if err != nil {
					t.Fatalf("Failed to create machine: %v", err)
				}

				if err := machine.Start(); err != nil {
					t.Fatalf("Failed to start machine: %v", err)
				}

				// Should start in idle state
				config := machine.GetCurrentConfiguration()
				found := false
				for _, state := range config.States {
					if state.Label == "idle" {
						found = true
						break
					}
				}
				if !found {
					t.Error("Expected machine to start in 'idle' state")
				}

				// Trigger START event
				triggered, err := machine.Step("START")
				if err != nil {
					t.Fatalf("Failed to process START event: %v", err)
				}
				if !triggered {
					t.Error("Expected START event to trigger transition")
				}

				// Should now be in active state
				config = machine.GetCurrentConfiguration()
				found = false
				for _, state := range config.States {
					if state.Label == "active" {
						found = true
						break
					}
				}
				if !found {
					t.Error("Expected machine to be in 'active' state after START")
				}

				// Trigger STOP event
				triggered, err = machine.Step("STOP")
				if err != nil {
					t.Fatalf("Failed to process STOP event: %v", err)
				}
				if !triggered {
					t.Error("Expected STOP event to trigger transition")
				}

				// Should be back in idle state
				config = machine.GetCurrentConfiguration()
				found = false
				for _, state := range config.States {
					if state.Label == "idle" {
						found = true
						break
					}
				}
				if !found {
					t.Error("Expected machine to be back in 'idle' state after STOP")
				}

				if err := machine.Stop(); err != nil {
					t.Fatalf("Failed to stop machine: %v", err)
				}
			},
		},
		{
			name:        "HierarchicalXStateImport",
			description: "Import hierarchical XState machine and execute",
			setupXState: func() xstate.Machine {
				return xstate.Machine{
					ID:      "hierarchical-machine",
					Initial: "offline",
					States: map[string]*xstate.State{
						"offline": {
							On: map[string]*xstate.Transition{
								"CONNECT": {Target: "online"},
							},
						},
						"online": {
							Type:    "compound",
							Initial: "idle",
							States: map[string]*xstate.State{
								"idle": {
									On: map[string]*xstate.Transition{
										"FETCH": {Target: "loading"},
									},
								},
								"loading": {
									On: map[string]*xstate.Transition{
										"SUCCESS": {Target: "idle"},
										"ERROR":   {Target: "error"},
									},
								},
								"error": {
									On: map[string]*xstate.Transition{
										"RETRY": {Target: "loading"},
									},
								},
							},
							On: map[string]*xstate.Transition{
								"DISCONNECT": {Target: "offline"},
							},
						},
					},
				}
			},
			executionTest: func(t *testing.T, statechart *sc.Statechart) {
				wrapper := semantics.NewStatechart(statechart)
				machine, err := semantics.NewMachine(wrapper, "hierarchical-test", nil)
				if err != nil {
					t.Fatalf("Failed to create machine: %v", err)
				}

				if err := machine.Start(); err != nil {
					t.Fatalf("Failed to start machine: %v", err)
				}

				// Should start in offline state
				config := machine.GetCurrentConfiguration()
				hasOffline := false
				for _, state := range config.States {
					if state.Label == "offline" {
						hasOffline = true
						break
					}
				}
				if !hasOffline {
					t.Error("Expected machine to start in 'offline' state")
				}

				// Connect to online
				machine.Step("CONNECT")
				
				// Should now be in online.idle
				config = machine.GetCurrentConfiguration()
				hasOnline := false
				hasIdle := false
				for _, state := range config.States {
					if state.Label == "online" {
						hasOnline = true
					}
					if state.Label == "idle" {
						hasIdle = true
					}
				}
				if !hasOnline || !hasIdle {
					t.Errorf("Expected to be in online.idle, got states: %v", config.States)
				}

				// Start loading
				machine.Step("FETCH")
				
				// Should be in online.loading
				config = machine.GetCurrentConfiguration()
				hasLoading := false
				for _, state := range config.States {
					if state.Label == "loading" {
						hasLoading = true
						break
					}
				}
				if !hasLoading {
					t.Error("Expected to be in loading state")
				}

				// Simulate error
				machine.Step("ERROR")
				
				// Should be in online.error
				config = machine.GetCurrentConfiguration()
				hasError := false
				for _, state := range config.States {
					if state.Label == "error" {
						hasError = true
						break
					}
				}
				if !hasError {
					t.Error("Expected to be in error state")
				}

				// Disconnect (should exit entire online state)
				machine.Step("DISCONNECT")
				
				// Should be back in offline
				config = machine.GetCurrentConfiguration()
				hasOffline = false
				for _, state := range config.States {
					if state.Label == "offline" {
						hasOffline = true
						break
					}
				}
				if !hasOffline {
					t.Error("Expected to be back in offline state")
				}

				if err := machine.Stop(); err != nil {
					t.Fatalf("Failed to stop machine: %v", err)
				}
			},
		},
		{
			name:        "ParallelXStateImport",
			description: "Import parallel XState machine and execute",
			setupXState: func() xstate.Machine {
				return xstate.Machine{
					ID: "parallel-machine",
					States: map[string]*xstate.State{
						"networkConnection": {
							Type:    "compound",
							Initial: "disconnected",
							States: map[string]*xstate.State{
								"disconnected": {
									On: map[string]*xstate.Transition{
										"CONNECT": {Target: "connected"},
									},
								},
								"connected": {
									On: map[string]*xstate.Transition{
										"DISCONNECT": {Target: "disconnected"},
									},
								},
							},
						},
						"dataSync": {
							Type:    "compound",
							Initial: "idle",
							States: map[string]*xstate.State{
								"idle": {
									On: map[string]*xstate.Transition{
										"SYNC": {Target: "syncing"},
									},
								},
								"syncing": {
									On: map[string]*xstate.Transition{
										"COMPLETE": {Target: "idle"},
									},
								},
							},
						},
					},
				}
			},
			executionTest: func(t *testing.T, statechart *sc.Statechart) {
				wrapper := semantics.NewStatechart(statechart)
				machine, err := semantics.NewMachine(wrapper, "parallel-test", nil)
				if err != nil {
					t.Fatalf("Failed to create machine: %v", err)
				}

				if err := machine.Start(); err != nil {
					t.Fatalf("Failed to start machine: %v", err)
				}

				// Should start in both disconnected and idle states
				config := machine.GetCurrentConfiguration()
				hasDisconnected := false
				hasIdle := false
				for _, state := range config.States {
					if state.Label == "disconnected" {
						hasDisconnected = true
					}
					if state.Label == "idle" {
						hasIdle = true
					}
				}
				if !hasDisconnected || !hasIdle {
					t.Errorf("Expected both disconnected and idle states, got: %v", config.States)
				}

				// Connect network (should only affect network region)
				machine.Step("CONNECT")
				
				config = machine.GetCurrentConfiguration()
				hasConnected := false
				hasIdleStill := false
				for _, state := range config.States {
					if state.Label == "connected" {
						hasConnected = true
					}
					if state.Label == "idle" {
						hasIdleStill = true
					}
				}
				if !hasConnected || !hasIdleStill {
					t.Errorf("Expected connected and idle states, got: %v", config.States)
				}

				// Start sync (should only affect sync region)
				machine.Step("SYNC")
				
				config = machine.GetCurrentConfiguration()
				hasConnectedStill := false
				hasSyncing := false
				for _, state := range config.States {
					if state.Label == "connected" {
						hasConnectedStill = true
					}
					if state.Label == "syncing" {
						hasSyncing = true
					}
				}
				if !hasConnectedStill || !hasSyncing {
					t.Errorf("Expected connected and syncing states, got: %v", config.States)
				}

				if err := machine.Stop(); err != nil {
					t.Fatalf("Failed to stop machine: %v", err)
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Phase 1: Setup XState machine
			xstateMachine := tc.setupXState()

			// Phase 2: Validate XState machine
			converter := xstate.NewConverter()
			if err := converter.Validate(xstateMachine); err != nil {
				t.Fatalf("XState validation failed: %v", err)
			}

			// Phase 3: Convert to Harel format
			statechart, err := converter.Import(xstateMachine)
			if err != nil {
				t.Fatalf("Failed to convert XState to Harel: %v", err)
			}

			// Phase 4: Validate converted statechart
			wrapper := semantics.NewStatechart(statechart)
			if err := wrapper.Validate(); err != nil {
				t.Fatalf("Converted statechart validation failed: %v", err)
			}

			// Phase 5: Test execution
			tc.executionTest(t, statechart)
		})
	}
}

// TestBridgeRoundtripIntegration tests roundtrip conversion (XState -> Harel -> XState)
func TestBridgeRoundtripIntegration(t *testing.T) {
	testCases := []struct {
		name        string
		description string
		createXState func() xstate.Machine
	}{
		{
			name:        "SimpleRoundtrip",
			description: "Simple statechart roundtrip conversion",
			createXState: func() xstate.Machine {
				return xstate.Machine{
					ID:      "roundtrip-simple",
					Initial: "start",
					States: map[string]*xstate.State{
						"start": {
							On: map[string]*xstate.Transition{
								"NEXT": {Target: "end"},
							},
						},
						"end": {
							Type: "final",
						},
					},
				}
			},
		},
		{
			name:        "HierarchicalRoundtrip",
			description: "Hierarchical statechart roundtrip conversion",
			createXState: func() xstate.Machine {
				return xstate.Machine{
					ID:      "roundtrip-hierarchical",
					Initial: "parent",
					States: map[string]*xstate.State{
						"parent": {
							Type:    "compound",
							Initial: "child1",
							States: map[string]*xstate.State{
								"child1": {
									On: map[string]*xstate.Transition{
										"SWITCH": {Target: "child2"},
									},
								},
								"child2": {},
							},
						},
					},
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Phase 1: Original XState machine
			original := tc.createXState()
			converter := xstate.NewConverter()

			// Phase 2: Convert to Harel
			statechart, err := converter.Import(original)
			if err != nil {
				t.Fatalf("Failed to import XState: %v", err)
			}

			// Phase 3: Test execution of converted statechart
			wrapper := semantics.NewStatechart(statechart)
			machine, err := semantics.NewMachine(wrapper, "roundtrip-test", nil)
			if err != nil {
				t.Fatalf("Failed to create machine from converted statechart: %v", err)
			}

			if err := machine.Start(); err != nil {
				t.Fatalf("Failed to start converted machine: %v", err)
			}

			// Get initial configuration
			initialConfig := machine.GetCurrentConfiguration()
			if len(initialConfig.States) == 0 {
				t.Error("Expected initial configuration to have states")
			}

			if err := machine.Stop(); err != nil {
				t.Fatalf("Failed to stop converted machine: %v", err)
			}

			// Phase 4: Convert back to XState
			exported, err := converter.Export(statechart)
			if err != nil {
				t.Fatalf("Failed to export to XState: %v", err)
			}

			// Phase 5: Validate exported machine
			if err := converter.Validate(exported); err != nil {
				t.Fatalf("Exported XState validation failed: %v", err)
			}

			// Phase 6: Basic structure verification
			if exported.ID == "" {
				t.Error("Exported machine should have an ID")
			}

			if len(exported.States) == 0 {
				t.Error("Exported machine should have states")
			}

			// Verify initial state is preserved (if any)
			if original.Initial != "" && exported.Initial == "" {
				t.Error("Initial state should be preserved in export")
			}
		})
	}
}

// TestBridgeErrorHandling tests error handling in bridge conversions
func TestBridgeErrorHandling(t *testing.T) {
	converter := xstate.NewConverter()

	testCases := []struct {
		name           string
		createXState   func() xstate.Machine
		expectedError  string
		shouldFailConversion bool
	}{
		{
			name: "MissingMachineID",
			createXState: func() xstate.Machine {
				return xstate.Machine{
					// ID is missing
					States: map[string]*xstate.State{
						"state1": {},
					},
				}
			},
			expectedError:        "id",
			shouldFailConversion: true,
		},
		{
			name: "EmptyStates",
			createXState: func() xstate.Machine {
				return xstate.Machine{
					ID:     "empty-machine",
					States: map[string]*xstate.State{}, // No states
				}
			},
			expectedError:        "states",
			shouldFailConversion: false, // Validation should catch this
		},
		{
			name: "InvalidStateType",
			createXState: func() xstate.Machine {
				return xstate.Machine{
					ID: "invalid-type-machine",
					States: map[string]*xstate.State{
						"state1": {
							Type: "invalid_type", // Unknown state type
						},
					},
				}
			},
			expectedError:        "type",
			shouldFailConversion: false, // Should use default fallback
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			xstateMachine := tc.createXState()
			
			// Test validation first
			validationErr := converter.Validate(xstateMachine)
			
			// Test conversion
			statechart, conversionErr := converter.Import(xstateMachine)
			
			if tc.shouldFailConversion {
				if conversionErr == nil {
					t.Fatalf("Expected conversion to fail, but it succeeded")
				}
				if tc.expectedError != "" && !contains(conversionErr.Error(), tc.expectedError) {
					t.Errorf("Expected error to contain '%s', got: %v", tc.expectedError, conversionErr)
				}
				return
			}

			// If conversion succeeds, test that the result is usable
			if conversionErr == nil {
				wrapper := semantics.NewStatechart(statechart)
				if err := wrapper.Validate(); err != nil {
					t.Logf("Converted statechart validation failed (expected for some cases): %v", err)
				}
				
				// Try to create a machine (might fail for invalid statecharts)
				machine, err := semantics.NewMachine(wrapper, "error-test", nil)
				if err != nil {
					t.Logf("Machine creation failed (might be expected): %v", err)
				} else {
					// If machine creation succeeds, try basic operations
					if err := machine.Start(); err != nil {
						t.Logf("Machine start failed (might be expected): %v", err)
					} else {
						machine.Stop()
					}
				}
			}
			
			// Log validation result for comparison
			if validationErr != nil {
				t.Logf("XState validation error: %v", validationErr)
			}
		})
	}
}

// TestBridgeWithNativeStatecharts tests using native statecharts with bridge export
func TestBridgeWithNativeStatecharts(t *testing.T) {
	testCases := []struct {
		name           string
		createChart    func() *sc.Statechart
		expectedStates []string
	}{
		{
			name: "SimpleStatechartExport",
			createChart: func() *sc.Statechart {
				return testutil.CreateSimpleStatechart()
			},
			expectedStates: []string{"A", "B"},
		},
		{
			name: "HierarchicalStatechartExport",
			createChart: func() *sc.Statechart {
				return testutil.CreateHierarchicalStatechart()
			},
			expectedStates: []string{"Active", "Inactive", "Idle", "Processing"},
		},
		{
			name: "OrthogonalStatechartExport",
			createChart: func() *sc.Statechart {
				return testutil.CreateOrthogonalStatechart()
			},
			expectedStates: []string{"RegionA", "RegionB", "A1", "A2", "B1", "B2"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Phase 1: Create native statechart
			statechart := tc.createChart()

			// Phase 2: Validate native statechart works with semantics
			wrapper := semantics.NewStatechart(statechart)
			machine, err := semantics.NewMachine(wrapper, "export-test", nil)
			if err != nil {
				t.Fatalf("Failed to create machine from native statechart: %v", err)
			}

			if err := machine.Start(); err != nil {
				t.Fatalf("Failed to start native machine: %v", err)
			}

			initialConfig := machine.GetCurrentConfiguration()
			if len(initialConfig.States) == 0 {
				t.Error("Expected initial configuration to have states")
			}

			if err := machine.Stop(); err != nil {
				t.Fatalf("Failed to stop native machine: %v", err)
			}

			// Phase 3: Export to XState
			converter := xstate.NewConverter()
			xstateMachine, err := converter.Export(statechart)
			if err != nil {
				t.Fatalf("Failed to export to XState: %v", err)
			}

			// Phase 4: Validate exported XState machine
			if err := converter.Validate(xstateMachine); err != nil {
				t.Fatalf("Exported XState validation failed: %v", err)
			}

			// Phase 5: Verify structure
			if len(xstateMachine.States) == 0 {
				t.Error("Exported machine should have states")
			}

			// Check that expected states are present (at least at top level)
			foundStates := make(map[string]bool)
			for stateName := range xstateMachine.States {
				foundStates[stateName] = true
			}

			// Count how many expected states we find
			foundCount := 0
			for _, expectedState := range tc.expectedStates {
				if foundStates[expectedState] {
					foundCount++
				}
			}

			if foundCount == 0 {
				t.Errorf("Expected to find some of these states %v in exported machine, but found none", tc.expectedStates)
			}

			// Phase 6: Test that exported machine can be re-imported
			reimported, err := converter.Import(xstateMachine)
			if err != nil {
				t.Fatalf("Failed to re-import exported machine: %v", err)
			}

			// Phase 7: Test execution of re-imported machine
			reimportedWrapper := semantics.NewStatechart(reimported)
			reimportedMachine, err := semantics.NewMachine(reimportedWrapper, "reimport-test", nil)
			if err != nil {
				t.Fatalf("Failed to create machine from re-imported statechart: %v", err)
			}

			if err := reimportedMachine.Start(); err != nil {
				t.Fatalf("Failed to start re-imported machine: %v", err)
			}

			reimportedConfig := reimportedMachine.GetCurrentConfiguration()
			if len(reimportedConfig.States) == 0 {
				t.Error("Expected re-imported machine to have initial configuration")
			}

			if err := reimportedMachine.Stop(); err != nil {
				t.Fatalf("Failed to stop re-imported machine: %v", err)
			}
		})
	}
}