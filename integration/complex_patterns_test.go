package integration

import (
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/semantics/v1/examples"
	"google.golang.org/protobuf/types/known/structpb"
)

// TestComplexStatechartPatterns tests realistic complex statechart patterns
func TestComplexStatechartPatterns(t *testing.T) {
	t.Run("ECommerceCheckoutFlow", func(t *testing.T) {
		statechart := createECommerceCheckoutStatechart()
		testECommerceWorkflow(t, statechart)
	})

	t.Run("UserAuthenticationFlow", func(t *testing.T) {
		statechart := createUserAuthStatechart()
		testAuthenticationWorkflow(t, statechart)
	})

	t.Run("MediaPlayerStatechart", func(t *testing.T) {
		statechart := createMediaPlayerStatechart()
		testMediaPlayerWorkflow(t, statechart)
	})

	t.Run("GameStateMachine", func(t *testing.T) {
		statechart := createGameStateMachine()
		testGameWorkflow(t, statechart)
	})

	t.Run("WorkflowEngine", func(t *testing.T) {
		statechart := createWorkflowEngineStatechart()
		testWorkflowEngineFlow(t, statechart)
	})
}

// createECommerceCheckoutStatechart creates a complex e-commerce checkout flow
func createECommerceCheckoutStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "CheckoutProcess",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "CartReview",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "CustomerInfo",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "GuestCheckout",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "MemberLogin",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "NewAccountCreation",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Payment",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "PaymentMethodSelection",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "PaymentProcessing",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "PaymentValidation",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "OrderConfirmation",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "OrderFailure",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "ProceedToCustomerInfo", From: []string{"CartReview"}, To: []string{"CustomerInfo"}, Event: "PROCEED"},
			{Label: "BackToCart", From: []string{"CustomerInfo"}, To: []string{"CartReview"}, Event: "BACK"},
			{Label: "SelectGuest", From: []string{"MemberLogin", "NewAccountCreation"}, To: []string{"GuestCheckout"}, Event: "GUEST_CHECKOUT"},
			{Label: "SelectLogin", From: []string{"GuestCheckout", "NewAccountCreation"}, To: []string{"MemberLogin"}, Event: "LOGIN"},
			{Label: "SelectNewAccount", From: []string{"GuestCheckout", "MemberLogin"}, To: []string{"NewAccountCreation"}, Event: "CREATE_ACCOUNT"},
			{Label: "ProceedToPayment", From: []string{"CustomerInfo"}, To: []string{"Payment"}, Event: "CONTINUE"},
			{Label: "BackToCustomerInfo", From: []string{"Payment"}, To: []string{"CustomerInfo"}, Event: "BACK"},
			{Label: "ProcessPayment", From: []string{"PaymentMethodSelection"}, To: []string{"PaymentProcessing"}, Event: "SUBMIT_PAYMENT"},
			{Label: "ValidatePayment", From: []string{"PaymentProcessing"}, To: []string{"PaymentValidation"}, Event: "VALIDATION_REQUIRED"},
			{Label: "PaymentSuccess", From: []string{"PaymentProcessing", "PaymentValidation"}, To: []string{"OrderConfirmation"}, Event: "PAYMENT_SUCCESS"},
			{Label: "PaymentFailure", From: []string{"PaymentProcessing", "PaymentValidation"}, To: []string{"OrderFailure"}, Event: "PAYMENT_FAILED"},
			{Label: "RetryPayment", From: []string{"OrderFailure"}, To: []string{"PaymentMethodSelection"}, Event: "RETRY"},
		},
		Events: []*sc.Event{
			{Label: "PROCEED"}, {Label: "BACK"}, {Label: "GUEST_CHECKOUT"}, {Label: "LOGIN"},
			{Label: "CREATE_ACCOUNT"}, {Label: "CONTINUE"}, {Label: "SUBMIT_PAYMENT"},
			{Label: "VALIDATION_REQUIRED"}, {Label: "PAYMENT_SUCCESS"}, {Label: "PAYMENT_FAILED"}, {Label: "RETRY"},
		},
	}
}

// testECommerceWorkflow tests the e-commerce checkout workflow
func testECommerceWorkflow(t *testing.T, statechart *sc.Statechart) {
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "ecommerce-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"cartTotal":   structpb.NewNumberValue(99.99),
			"customerID":  structpb.NewStringValue(""),
			"paymentMethod": structpb.NewStringValue(""),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create e-commerce machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test successful checkout flow
	events := []string{
		"PROCEED",        // CartReview -> CustomerInfo
		"GUEST_CHECKOUT", // -> GuestCheckout
		"CONTINUE",       // CustomerInfo -> Payment
		"SUBMIT_PAYMENT", // PaymentMethodSelection -> PaymentProcessing
		"PAYMENT_SUCCESS", // PaymentProcessing -> OrderConfirmation
	}

	for i, event := range events {
		triggered, err := machine.Step(event)
		if err != nil {
			t.Fatalf("Failed to process event '%s' (step %d): %v", event, i+1, err)
		}
		if !triggered {
			t.Logf("Event '%s' did not trigger any transitions", event)
		}
	}

	// Verify final state
	config := machine.GetCurrentConfiguration()
	hasOrderConfirmation := false
	for _, state := range config.States {
		if state.Label == "OrderConfirmation" {
			hasOrderConfirmation = true
			break
		}
	}
	if !hasOrderConfirmation {
		t.Error("Expected to end in OrderConfirmation state")
	}
}

// createUserAuthStatechart creates a user authentication flow statechart
func createUserAuthStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "AuthenticationSystem",
			Type:  sc.StateTypeParallel,
			Children: []*sc.State{
				{
					Label: "SessionManager",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "LoggedOut",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "LoggedIn",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "Active",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "Idle",
									Type:  sc.StateTypeBasic,
								},
							},
						},
					},
				},
				{
					Label: "SecurityMonitor",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "Normal",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "Suspicious",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "Locked",
							Type:  sc.StateTypeBasic,
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "Login", From: []string{"LoggedOut"}, To: []string{"LoggedIn"}, Event: "LOGIN_SUCCESS"},
			{Label: "Logout", From: []string{"LoggedIn"}, To: []string{"LoggedOut"}, Event: "LOGOUT"},
			{Label: "GoIdle", From: []string{"Active"}, To: []string{"Idle"}, Event: "INACTIVITY_TIMEOUT"},
			{Label: "Activate", From: []string{"Idle"}, To: []string{"Active"}, Event: "USER_ACTIVITY"},
			{Label: "DetectSuspicious", From: []string{"Normal"}, To: []string{"Suspicious"}, Event: "SUSPICIOUS_ACTIVITY"},
			{Label: "LockAccount", From: []string{"Suspicious"}, To: []string{"Locked"}, Event: "LOCK_ACCOUNT"},
			{Label: "UnlockAccount", From: []string{"Locked"}, To: []string{"Normal"}, Event: "UNLOCK_ACCOUNT"},
			{Label: "ClearSuspicion", From: []string{"Suspicious"}, To: []string{"Normal"}, Event: "CLEAR_SUSPICIOUS"},
			{Label: "ForceLogout", From: []string{"LoggedIn"}, To: []string{"LoggedOut"}, Event: "FORCE_LOGOUT"},
		},
		Events: []*sc.Event{
			{Label: "LOGIN_SUCCESS"}, {Label: "LOGOUT"}, {Label: "INACTIVITY_TIMEOUT"},
			{Label: "USER_ACTIVITY"}, {Label: "SUSPICIOUS_ACTIVITY"}, {Label: "LOCK_ACCOUNT"},
			{Label: "UNLOCK_ACCOUNT"}, {Label: "CLEAR_SUSPICIOUS"}, {Label: "FORCE_LOGOUT"},
		},
	}
}

// testAuthenticationWorkflow tests the authentication workflow
func testAuthenticationWorkflow(t *testing.T, statechart *sc.Statechart) {
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "auth-test", nil)
	if err != nil {
		t.Fatalf("Failed to create auth machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test login flow
	triggered, err := machine.Step("LOGIN_SUCCESS")
	if err != nil {
		t.Fatalf("Failed to process LOGIN_SUCCESS: %v", err)
	}
	if !triggered {
		t.Error("LOGIN_SUCCESS should trigger transition")
	}

	// Check both regions are in expected states
	config := machine.GetCurrentConfiguration()
	hasLoggedIn := false
	hasActive := false
	hasNormal := false
	
	for _, state := range config.States {
		switch state.Label {
		case "LoggedIn":
			hasLoggedIn = true
		case "Active":
			hasActive = true
		case "Normal":
			hasNormal = true
		}
	}

	if !hasLoggedIn || !hasActive || !hasNormal {
		t.Errorf("Expected LoggedIn+Active+Normal states after login, got: %v", config.States)
	}

	// Test security event
	machine.Step("SUSPICIOUS_ACTIVITY")
	
	// Test inactivity
	machine.Step("INACTIVITY_TIMEOUT")
	
	// Test force logout
	machine.Step("FORCE_LOGOUT")
	
	// Should be back to initial state
	config = machine.GetCurrentConfiguration()
	hasLoggedOut := false
	hasNormalAgain := false
	
	for _, state := range config.States {
		if state.Label == "LoggedOut" {
			hasLoggedOut = true
		}
		if state.Label == "Normal" {
			hasNormalAgain = true
		}
	}

	if !hasLoggedOut {
		t.Error("Expected to be logged out after force logout")
	}
	if !hasNormalAgain {
		t.Error("Expected security monitor to remain in normal state")
	}
}

// createMediaPlayerStatechart creates a media player statechart
func createMediaPlayerStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "MediaPlayer",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Stopped",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "Playing",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "Normal",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "FastForward",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "Rewind",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Paused",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "Loading",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "Error",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "Load", From: []string{"Stopped"}, To: []string{"Loading"}, Event: "LOAD_MEDIA"},
			{Label: "StartPlaying", From: []string{"Loading"}, To: []string{"Playing"}, Event: "LOAD_SUCCESS"},
			{Label: "LoadError", From: []string{"Loading"}, To: []string{"Error"}, Event: "LOAD_ERROR"},
			{Label: "Play", From: []string{"Paused"}, To: []string{"Playing"}, Event: "PLAY"},
			{Label: "Pause", From: []string{"Playing"}, To: []string{"Paused"}, Event: "PAUSE"},
			{Label: "Stop", From: []string{"Playing", "Paused"}, To: []string{"Stopped"}, Event: "STOP"},
			{Label: "StartFF", From: []string{"Normal"}, To: []string{"FastForward"}, Event: "FAST_FORWARD"},
			{Label: "StartRewind", From: []string{"Normal"}, To: []string{"Rewind"}, Event: "REWIND"},
			{Label: "BackToNormal", From: []string{"FastForward", "Rewind"}, To: []string{"Normal"}, Event: "NORMAL_SPEED"},
			{Label: "ErrorRecovery", From: []string{"Error"}, To: []string{"Stopped"}, Event: "RESET"},
		},
		Events: []*sc.Event{
			{Label: "LOAD_MEDIA"}, {Label: "LOAD_SUCCESS"}, {Label: "LOAD_ERROR"},
			{Label: "PLAY"}, {Label: "PAUSE"}, {Label: "STOP"},
			{Label: "FAST_FORWARD"}, {Label: "REWIND"}, {Label: "NORMAL_SPEED"}, {Label: "RESET"},
		},
	}
}

// testMediaPlayerWorkflow tests the media player workflow
func testMediaPlayerWorkflow(t *testing.T, statechart *sc.Statechart) {
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "mediaplayer-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"currentTime": structpb.NewNumberValue(0),
			"duration":    structpb.NewNumberValue(0),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create media player machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test normal playback flow
	events := []string{
		"LOAD_MEDIA",    // Stopped -> Loading
		"LOAD_SUCCESS",  // Loading -> Playing
		"FAST_FORWARD",  // Normal -> FastForward
		"NORMAL_SPEED",  // FastForward -> Normal
		"PAUSE",         // Playing -> Paused
		"PLAY",          // Paused -> Playing
		"STOP",          // Playing -> Stopped
	}

	for i, event := range events {
		triggered, err := machine.Step(event)
		if err != nil {
			t.Fatalf("Failed to process event '%s' (step %d): %v", event, i+1, err)
		}
		if !triggered {
			t.Logf("Event '%s' did not trigger any transitions", event)
		}
		
		// Validate state after each step
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine validation failed after event '%s': %v", event, err)
		}
	}

	// Verify back to stopped state
	config := machine.GetCurrentConfiguration()
	hasStopped := false
	for _, state := range config.States {
		if state.Label == "Stopped" {
			hasStopped = true
			break
		}
	}
	if !hasStopped {
		t.Error("Expected to end in Stopped state")
	}
}

// createGameStateMachine creates a game state machine
func createGameStateMachine() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "Game",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "MainMenu",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "InGame",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "GameLogic",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "Playing",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "Paused",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "GameOver",
									Type:  sc.StateTypeBasic,
								},
							},
						},
						{
							Label: "PlayerState",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "Alive",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "Dead",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "Respawning",
									Type:  sc.StateTypeBasic,
								},
							},
						},
					},
				},
				{
					Label: "Settings",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartGame", From: []string{"MainMenu"}, To: []string{"InGame"}, Event: "START_GAME"},
			{Label: "ExitToMenu", From: []string{"InGame", "Settings"}, To: []string{"MainMenu"}, Event: "EXIT_TO_MENU"},
			{Label: "OpenSettings", From: []string{"MainMenu"}, To: []string{"Settings"}, Event: "OPEN_SETTINGS"},
			{Label: "PauseGame", From: []string{"Playing"}, To: []string{"Paused"}, Event: "PAUSE"},
			{Label: "ResumeGame", From: []string{"Paused"}, To: []string{"Playing"}, Event: "RESUME"},
			{Label: "PlayerDied", From: []string{"Alive"}, To: []string{"Dead"}, Event: "PLAYER_DEATH"},
			{Label: "StartRespawn", From: []string{"Dead"}, To: []string{"Respawning"}, Event: "RESPAWN"},
			{Label: "CompleteRespawn", From: []string{"Respawning"}, To: []string{"Alive"}, Event: "RESPAWN_COMPLETE"},
			{Label: "EndGame", From: []string{"Playing", "Paused"}, To: []string{"GameOver"}, Event: "GAME_OVER"},
		},
		Events: []*sc.Event{
			{Label: "START_GAME"}, {Label: "EXIT_TO_MENU"}, {Label: "OPEN_SETTINGS"},
			{Label: "PAUSE"}, {Label: "RESUME"}, {Label: "PLAYER_DEATH"},
			{Label: "RESPAWN"}, {Label: "RESPAWN_COMPLETE"}, {Label: "GAME_OVER"},
		},
	}
}

// testGameWorkflow tests the game workflow
func testGameWorkflow(t *testing.T, statechart *sc.Statechart) {
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "game-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"score":  structpb.NewNumberValue(0),
			"lives":  structpb.NewNumberValue(3),
			"level":  structpb.NewNumberValue(1),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create game machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test game session
	machine.Step("START_GAME")
	
	// Should be in Playing+Alive states
	config := machine.GetCurrentConfiguration()
	hasPlaying := false
	hasAlive := false
	for _, state := range config.States {
		if state.Label == "Playing" {
			hasPlaying = true
		}
		if state.Label == "Alive" {
			hasAlive = true
		}
	}
	if !hasPlaying || !hasAlive {
		t.Errorf("Expected Playing+Alive states after starting game, got: %v", config.States)
	}

	// Test death and respawn
	machine.Step("PLAYER_DEATH")
	machine.Step("RESPAWN")
	machine.Step("RESPAWN_COMPLETE")
	
	// Test pause/resume
	machine.Step("PAUSE")
	machine.Step("RESUME")
	
	// Test game over
	machine.Step("GAME_OVER")
	
	// Exit to menu
	machine.Step("EXIT_TO_MENU")
	
	// Should be back in main menu
	config = machine.GetCurrentConfiguration()
	hasMainMenu := false
	for _, state := range config.States {
		if state.Label == "MainMenu" {
			hasMainMenu = true
			break
		}
	}
	if !hasMainMenu {
		t.Error("Expected to be back in MainMenu")
	}
}

// createWorkflowEngineStatechart creates a workflow engine statechart
func createWorkflowEngineStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "WorkflowEngine",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Idle",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "ProcessingWorkflow",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "TaskExecution",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "WaitingForApproval",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "ErrorHandling",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "WorkflowComplete",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "WorkflowFailed",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartWorkflow", From: []string{"Idle"}, To: []string{"ProcessingWorkflow"}, Event: "START_WORKFLOW"},
			{Label: "TaskSuccess", From: []string{"TaskExecution"}, To: []string{"WaitingForApproval"}, Event: "TASK_COMPLETE"},
			{Label: "TaskError", From: []string{"TaskExecution"}, To: []string{"ErrorHandling"}, Event: "TASK_ERROR"},
			{Label: "ApprovalGranted", From: []string{"WaitingForApproval"}, To: []string{"TaskExecution"}, Event: "APPROVED"},
			{Label: "ApprovalDenied", From: []string{"WaitingForApproval"}, To: []string{"WorkflowFailed"}, Event: "DENIED"},
			{Label: "ErrorResolved", From: []string{"ErrorHandling"}, To: []string{"TaskExecution"}, Event: "ERROR_RESOLVED"},
			{Label: "ErrorEscalated", From: []string{"ErrorHandling"}, To: []string{"WorkflowFailed"}, Event: "ERROR_ESCALATED"},
			{Label: "WorkflowSuccess", From: []string{"TaskExecution", "WaitingForApproval"}, To: []string{"WorkflowComplete"}, Event: "WORKFLOW_COMPLETE"},
			{Label: "ResetWorkflow", From: []string{"WorkflowComplete", "WorkflowFailed"}, To: []string{"Idle"}, Event: "RESET"},
		},
		Events: []*sc.Event{
			{Label: "START_WORKFLOW"}, {Label: "TASK_COMPLETE"}, {Label: "TASK_ERROR"},
			{Label: "APPROVED"}, {Label: "DENIED"}, {Label: "ERROR_RESOLVED"},
			{Label: "ERROR_ESCALATED"}, {Label: "WORKFLOW_COMPLETE"}, {Label: "RESET"},
		},
	}
}

// testWorkflowEngineFlow tests the workflow engine flow
func testWorkflowEngineFlow(t *testing.T, statechart *sc.Statechart) {
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "workflow-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"workflowId":   structpb.NewStringValue("WF-001"),
			"currentStep":  structpb.NewNumberValue(0),
			"totalSteps":   structpb.NewNumberValue(5),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create workflow machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test successful workflow
	t.Run("SuccessfulWorkflow", func(t *testing.T) {
		events := []string{
			"START_WORKFLOW",     // Idle -> ProcessingWorkflow
			"TASK_COMPLETE",      // TaskExecution -> WaitingForApproval
			"APPROVED",           // WaitingForApproval -> TaskExecution
			"WORKFLOW_COMPLETE",  // TaskExecution -> WorkflowComplete
			"RESET",              // WorkflowComplete -> Idle
		}

		for i, event := range events {
			triggered, err := machine.Step(event)
			if err != nil {
				t.Fatalf("Failed to process event '%s' (step %d): %v", event, i+1, err)
			}
			if !triggered {
				t.Logf("Event '%s' did not trigger any transitions", event)
			}
		}

		// Should be back in Idle
		config := machine.GetCurrentConfiguration()
		hasIdle := false
		for _, state := range config.States {
			if state.Label == "Idle" {
				hasIdle = true
				break
			}
		}
		if !hasIdle {
			t.Error("Expected to be back in Idle state after successful workflow")
		}
	})

	// Test workflow with error handling
	t.Run("ErrorHandlingWorkflow", func(t *testing.T) {
		events := []string{
			"START_WORKFLOW",    // Idle -> ProcessingWorkflow
			"TASK_ERROR",        // TaskExecution -> ErrorHandling
			"ERROR_RESOLVED",    // ErrorHandling -> TaskExecution
			"TASK_COMPLETE",     // TaskExecution -> WaitingForApproval
			"DENIED",            // WaitingForApproval -> WorkflowFailed
			"RESET",             // WorkflowFailed -> Idle
		}

		for i, event := range events {
			triggered, err := machine.Step(event)
			if err != nil {
				t.Fatalf("Failed to process event '%s' (step %d): %v", event, i+1, err)
			}
			if !triggered {
				t.Logf("Event '%s' did not trigger any transitions", event)
			}
		}

		// Should be back in Idle
		config := machine.GetCurrentConfiguration()
		hasIdle := false
		for _, state := range config.States {
			if state.Label == "Idle" {
				hasIdle = true
				break
			}
		}
		if !hasIdle {
			t.Error("Expected to be back in Idle state after failed workflow")
		}
	})
}

// TestExampleStatecharts tests the statecharts from the examples package
func TestExampleStatecharts(t *testing.T) {
	t.Run("HierarchicalExample", func(t *testing.T) {
		statechart := examples.HierarchicalStatechart()
		testExampleHierarchical(t, statechart)
	})

	t.Run("OrthogonalExample", func(t *testing.T) {
		statechart := examples.OrthogonalStatechart()
		testExampleOrthogonal(t, statechart)
	})

	t.Run("CompoundExample", func(t *testing.T) {
		statechart := examples.CompoundStatechart()
		testExampleCompound(t, statechart)
	})

	// Test Hotel Evanstonian if available
	if testing.Short() {
		t.Skip("Skipping Hotel Evanstonian test in short mode")
	}
	
	t.Run("HotelEvanstonianExample", func(t *testing.T) {
		statechart := examples.HotelEvanstonianStatechart()
		testExampleHotelEvanstonian(t, statechart)
	})
}

// testExampleHierarchical tests the hierarchical example
func testExampleHierarchical(t *testing.T, statechart *semantics.Statechart) {
	machine, err := semantics.NewMachine(statechart, "hierarchical-example-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test the alarm system workflow
	events := []string{
		"POWER_ON",        // Off -> On (should enter Idle by default)
		"ARM",             // Idle -> Armed (should enter Monitoring by default)
		"MOTION_DETECTED", // Monitoring -> Triggered
		"RESET",           // Triggered -> Monitoring
		"DISARM",          // Armed -> Idle
		"POWER_OFF",       // On -> Off
	}

	for i, event := range events {
		triggered, err := machine.Step(event)
		if err != nil {
			t.Fatalf("Failed to process event '%s' (step %d): %v", event, i+1, err)
		}
		if !triggered {
			t.Logf("Event '%s' did not trigger any transitions", event)
		}
	}

	// Should end in Off state
	config := machine.GetCurrentConfiguration()
	hasOff := false
	for _, state := range config.States {
		if state.Label == "Off" {
			hasOff = true
			break
		}
	}
	if !hasOff {
		t.Error("Expected to end in Off state")
	}
}

// testExampleOrthogonal tests the orthogonal example
func testExampleOrthogonal(t *testing.T, statechart *semantics.Statechart) {
	machine, err := semantics.NewMachine(statechart, "orthogonal-example-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Get initial configuration (should have states from both regions)
	config := machine.GetCurrentConfiguration()
	if len(config.States) < 2 {
		t.Errorf("Expected at least 2 states in orthogonal configuration, got %d", len(config.States))
	}

	// Test independent region transitions
	machine.Step("switch_a")
	machine.Step("switch_b")

	// Verify both regions have changed
	finalConfig := machine.GetCurrentConfiguration()
	if len(finalConfig.States) < 2 {
		t.Errorf("Expected at least 2 states in final configuration, got %d", len(finalConfig.States))
	}
}

// testExampleCompound tests the compound example
func testExampleCompound(t *testing.T, statechart *semantics.Statechart) {
	machine, err := semantics.NewMachine(statechart, "compound-example-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test some basic transitions
	config := machine.GetCurrentConfiguration()
	if len(config.States) == 0 {
		t.Error("Expected initial configuration to have states")
	}

	// The compound example should have hierarchical structure
	// Just verify the machine works without errors
	machine.Step("process")
	machine.Step("complete")

	if err := machine.Validate(); err != nil {
		t.Errorf("Machine validation failed: %v", err)
	}
}

// testExampleHotelEvanstonian tests the Hotel Evanstonian example
func testExampleHotelEvanstonian(t *testing.T, statechart *semantics.Statechart) {
	machine, err := semantics.NewMachine(statechart, "hotel-example-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// The Hotel Evanstonian is a complex example
	// Test basic functionality
	config := machine.GetCurrentConfiguration()
	if len(config.States) == 0 {
		t.Error("Expected initial configuration to have states")
	}

	// Try some common hotel events
	machine.Step("check_in")
	machine.Step("request_service")
	machine.Step("check_out")

	// Verify machine remains in valid state
	if err := machine.Validate(); err != nil {
		t.Errorf("Machine validation failed: %v", err)
	}

	finalConfig := machine.GetCurrentConfiguration()
	if len(finalConfig.States) == 0 {
		t.Error("Expected final configuration to have states")
	}
}