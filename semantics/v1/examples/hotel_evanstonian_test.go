package examples

import (
	"testing"

	"github.com/tmc/sc"
)

// containsState checks if a configuration contains a state with the given label
func containsState(config *sc.Configuration, label string) bool {
	if config == nil || config.States == nil {
		return false
	}
	for _, state := range config.States {
		if state != nil && state.Label == label {
			return true
		}
	}
	return false
}

func TestHotelEvanstonianStatechart(t *testing.T) {
	chart := HotelEvanstonianStatechart()

	// Verify the statechart is valid according to semantic rules
	if err := chart.Validate(); err != nil {
		t.Errorf("Hotel Evanstonian statechart is invalid: %v", err)
	}

	// Test initial state - note that NewStatechart changes root label to __root__
	if state, err := chart.Default("__root__"); err != nil || state != "Idle" {
		t.Errorf("Expected Idle to be default state, got %s", state)
	}

	// Test orthogonal region defaults within PreparationPhase
	if state, err := chart.Default("KitchenPreparation"); err != nil || state != "MealPreparationIdle" {
		t.Errorf("Expected MealPreparationIdle to be default state within KitchenPreparation, got %s", state)
	}

	if state, err := chart.Default("WaiterPreparation"); err != nil || state != "CartPreparationIdle" {
		t.Errorf("Expected CartPreparationIdle to be default state within WaiterPreparation, got %s", state)
	}

	if state, err := chart.Default("SommelierTasks"); err != nil || state != "CheckingWineRequest" {
		t.Errorf("Expected CheckingWineRequest to be default state within SommelierTasks, got %s", state)
	}

	// Test that preparation phase regions are orthogonal
	orthogonal, err := chart.Orthogonal("KitchenPreparation", "WaiterPreparation")
	if err != nil || !orthogonal {
		t.Errorf("Expected KitchenPreparation and WaiterPreparation to be orthogonal")
	}

	orthogonal, err = chart.Orthogonal("KitchenPreparation", "SommelierTasks")
	if err != nil || !orthogonal {
		t.Errorf("Expected KitchenPreparation and SommelierTasks to be orthogonal")
	}

	orthogonal, err = chart.Orthogonal("WaiterPreparation", "SommelierTasks")
	if err != nil || !orthogonal {
		t.Errorf("Expected WaiterPreparation and SommelierTasks to be orthogonal")
	}

	// Test non-orthogonal states (different hierarchy levels)
	orthogonal, err = chart.Orthogonal("Idle", "OrderReceived")
	if err != nil || orthogonal {
		t.Errorf("Expected Idle and OrderReceived to NOT be orthogonal")
	}

	// Test ancestral relations
	related, err := chart.AncestrallyRelated("PreparationPhase", "MealPreparationIdle")
	if err != nil || !related {
		t.Errorf("Expected PreparationPhase and MealPreparationIdle to be ancestrally related")
	}

	related, err = chart.AncestrallyRelated("__root__", "CheckingWineRequest")
	if err != nil || !related {
		t.Errorf("Expected __root__ and CheckingWineRequest to be ancestrally related")
	}

	// Test that states in different orthogonal regions are not ancestrally related
	related, err = chart.AncestrallyRelated("MealPreparationIdle", "CartPreparationIdle")
	if err != nil || related {
		t.Errorf("Expected MealPreparationIdle and CartPreparationIdle to NOT be ancestrally related")
	}
}

func TestHotelEvanstonianInitialConfiguration(t *testing.T) {
	chart := HotelEvanstonianStatechart()

	// Test initial configuration
	config, err := chart.InitialConfiguration()
	if err != nil {
		t.Fatalf("Error getting initial configuration: %v", err)
	}


	// Verify initial state is Idle
	if !containsState(config, "Idle") {
		t.Errorf("Expected initial configuration to contain Idle state")
	}

	// Verify root state is included  
	if !containsState(config, "__root__") {
		t.Errorf("Expected initial configuration to contain __root__ root state")
	}
}

func TestHotelEvanstonianTransitions(t *testing.T) {
	chart := HotelEvanstonianStatechart()

	// Test that we have the expected number of transitions
	expectedTransitionCount := 20 // Based on the actual transitions defined in the statechart
	if len(chart.Transitions) != expectedTransitionCount {
		t.Errorf("Expected %d transitions, got %d", expectedTransitionCount, len(chart.Transitions))
	}

	// Test that we have the expected number of events
	expectedEventCount := 18 // Based on the events defined in the statechart
	if len(chart.Events) != expectedEventCount {
		t.Errorf("Expected %d events, got %d", expectedEventCount, len(chart.Events))
	}

	// Test key transitions exist
	transitionMap := make(map[string]*sc.Transition)
	for _, transition := range chart.Transitions {
		transitionMap[transition.Label] = transition
	}

	expectedTransitions := []string{
		"ReceiveOrder",
		"CreateOrderTicket", 
		"StartPreparation",
		"WineDesired",
		"WineNotDesired",
		"AllItemsPrepared",
		"DeliveryComplete",
		"NoMoreOrders",
		"MoreOrdersExist",
		"CancelOrder",
	}

	for _, transitionName := range expectedTransitions {
		if _, exists := transitionMap[transitionName]; !exists {
			t.Errorf("Expected transition '%s' to exist", transitionName)
		}
	}
}

func TestHotelEvanstonianBusinessLogic(t *testing.T) {
	chart := HotelEvanstonianStatechart()

	// Test the business logic transitions for order completion workflow
	t.Run("OrderCompletionLogic", func(t *testing.T) {
		// Find the transition for "more orders exist" which should loop back to Idle
		var moreOrdersTransition *sc.Transition
		for _, transition := range chart.Transitions {
			if transition.Label == "MoreOrdersExist" {
				moreOrdersTransition = transition
				break
			}
		}

		if moreOrdersTransition == nil {
			t.Fatal("Expected MoreOrdersExist transition to exist")
		}

		// Verify it transitions from CheckingForMoreOrders to Idle
		if len(moreOrdersTransition.From) != 1 || moreOrdersTransition.From[0] != "CheckingForMoreOrders" {
			t.Errorf("Expected MoreOrdersExist transition to be from CheckingForMoreOrders, got %v", moreOrdersTransition.From)
		}

		if len(moreOrdersTransition.To) != 1 || moreOrdersTransition.To[0] != "Idle" {
			t.Errorf("Expected MoreOrdersExist transition to go to Idle, got %v", moreOrdersTransition.To)
		}

		// Find the transition for "no more orders" which should go to debit
		var noMoreOrdersTransition *sc.Transition
		for _, transition := range chart.Transitions {
			if transition.Label == "NoMoreOrders" {
				noMoreOrdersTransition = transition
				break
			}
		}

		if noMoreOrdersTransition == nil {
			t.Fatal("Expected NoMoreOrders transition to exist")
		}

		// Verify it transitions from CheckingForMoreOrders to DebitingAccount
		if len(noMoreOrdersTransition.From) != 1 || noMoreOrdersTransition.From[0] != "CheckingForMoreOrders" {
			t.Errorf("Expected NoMoreOrders transition to be from CheckingForMoreOrders, got %v", noMoreOrdersTransition.From)
		}

		if len(noMoreOrdersTransition.To) != 1 || noMoreOrdersTransition.To[0] != "DebitingAccount" {
			t.Errorf("Expected NoMoreOrders transition to go to DebitingAccount, got %v", noMoreOrdersTransition.To)
		}
	})

	// Test sommelier workflow branches
	t.Run("SommelierWorkflowBranches", func(t *testing.T) {
		// Test wine requested path
		var wineRequestedTransition *sc.Transition
		for _, transition := range chart.Transitions {
			if transition.Label == "WineDesired" {
				wineRequestedTransition = transition
				break
			}
		}

		if wineRequestedTransition == nil {
			t.Fatal("Expected WineDesired transition to exist")
		}

		if wineRequestedTransition.Event != "WINE_REQUESTED" {
			t.Errorf("Expected WineDesired transition to have event WINE_REQUESTED, got %s", wineRequestedTransition.Event)
		}

		// Test wine not requested path
		var wineNotRequestedTransition *sc.Transition
		for _, transition := range chart.Transitions {
			if transition.Label == "WineNotDesired" {
				wineNotRequestedTransition = transition
				break
			}
		}

		if wineNotRequestedTransition == nil {
			t.Fatal("Expected WineNotDesired transition to exist")
		}

		if wineNotRequestedTransition.Event != "NO_WINE_REQUESTED" {
			t.Errorf("Expected WineNotDesired transition to have event NO_WINE_REQUESTED, got %s", wineNotRequestedTransition.Event)
		}
	})
}