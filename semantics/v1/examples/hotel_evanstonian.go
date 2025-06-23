// Package examples provides academic examples of statechart implementations.
// This file demonstrates a real-world business process modeling example.
package examples

import (
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// HotelEvanstonianStatechart creates a statechart that models the Hotel Evanstonian
// room service process. This demonstrates complex business process modeling with:
//   - Sequential workflow stages
//   - Parallel task execution (orthogonal regions)
//   - Conditional branching based on customer preferences
//   - Synchronization points for task consolidation
//
// The process flow:
//   1. Order received from guest
//   2. Room service manager creates order ticket
//   3. Parallel preparation tasks:
//      - Kitchen: Meal preparation
//      - Waiter: Cart and non-alcoholic beverage preparation
//      - Sommelier: Wine checking and preparation (conditional)
//      - Sommelier: Other alcoholic beverage preparation (conditional)
//   4. Consolidation of all prepared items
//   5. Delivery to guest
//   6. Check for additional orders
//   7. Account debit (if no more orders)
//   8. Order completion
//
// This example demonstrates:
// 1. Business process modeling with statecharts
// 2. Orthogonal regions for parallel task execution
// 3. Conditional transitions based on customer preferences
// 4. Synchronization and consolidation patterns
// 5. Multi-actor coordination (manager, kitchen, waiter, sommelier)
func HotelEvanstonianStatechart() *semantics.Statechart {
	return semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "RoomServiceProcess",
			Type:  sc.StateTypeNormal, // Root state must be Normal to have children
			Children: []*sc.State{
				{
					Label:     "Idle",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "OrderReceived",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "OrderCreated",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "PreparationPhase",
					Type:  sc.StateTypeOrthogonal, // Parallel preparation tasks
					Children: []*sc.State{
						{
							Label: "KitchenPreparation",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "MealPreparationIdle",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "PreparingMeals",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "MealsReady",
									Type:  sc.StateTypeBasic,
								},
							},
						},
						{
							Label: "WaiterPreparation",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "CartPreparationIdle",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "PreparingCart",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "CartReady",
									Type:  sc.StateTypeBasic,
								},
							},
						},
						{
							Label: "SommelierTasks",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{
									Label:     "CheckingWineRequest",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "FetchingWine",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "CheckingOtherAlcohol",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "PreparingAlcohol",
									Type:  sc.StateTypeBasic,
								},
								{
									Label: "BeveragePreparationComplete",
									Type:  sc.StateTypeBasic,
								},
							},
						},
					},
				},
				{
					Label: "ConsolidatingItems",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "DeliveringOrder",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "CheckingForMoreOrders",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "DebitingAccount",
					Type:  sc.StateTypeBasic,
				},
				{
					Label:   "OrderCompleted",
					Type:    sc.StateTypeBasic,
					IsFinal: true,
				},
			},
		},
		Transitions: []*sc.Transition{
			// Main workflow transitions
			{
				Label: "ReceiveOrder",
				From:  []string{"Idle"},
				To:    []string{"OrderReceived"},
				Event: "ORDER_RECEIVED",
			},
			{
				Label: "CreateOrderTicket",
				From:  []string{"OrderReceived"},
				To:    []string{"OrderCreated"},
				Event: "CREATE_ORDER_TICKET",
			},
			{
				Label: "StartPreparation",
				From:  []string{"OrderCreated"},
				To:    []string{"PreparationPhase"},
				Event: "BEGIN_PREPARATION",
			},

			// Kitchen preparation transitions
			{
				Label: "StartMealPreparation",
				From:  []string{"MealPreparationIdle"},
				To:    []string{"PreparingMeals"},
				Event: "BEGIN_PREPARATION",
			},
			{
				Label: "FinishMealPreparation",
				From:  []string{"PreparingMeals"},
				To:    []string{"MealsReady"},
				Event: "MEALS_PREPARED",
			},

			// Waiter preparation transitions
			{
				Label: "StartCartPreparation",
				From:  []string{"CartPreparationIdle"},
				To:    []string{"PreparingCart"},
				Event: "BEGIN_PREPARATION",
			},
			{
				Label: "FinishCartPreparation",
				From:  []string{"PreparingCart"},
				To:    []string{"CartReady"},
				Event: "CART_PREPARED",
			},

			// Sommelier wine workflow
			{
				Label: "WineNotDesired",
				From:  []string{"CheckingWineRequest"},
				To:    []string{"CheckingOtherAlcohol"},
				Event: "NO_WINE_REQUESTED",
			},
			{
				Label: "WineDesired",
				From:  []string{"CheckingWineRequest"},
				To:    []string{"FetchingWine"},
				Event: "WINE_REQUESTED",
			},
			{
				Label: "WineFetched",
				From:  []string{"FetchingWine"},
				To:    []string{"CheckingOtherAlcohol"},
				Event: "WINE_FETCHED",
			},

			// Sommelier other alcohol workflow
			{
				Label: "OtherAlcoholNotDesired",
				From:  []string{"CheckingOtherAlcohol"},
				To:    []string{"BeveragePreparationComplete"},
				Event: "NO_OTHER_ALCOHOL_REQUESTED",
			},
			{
				Label: "OtherAlcoholDesired",
				From:  []string{"CheckingOtherAlcohol"},
				To:    []string{"PreparingAlcohol"},
				Event: "OTHER_ALCOHOL_REQUESTED",
			},
			{
				Label: "AlcoholPrepared",
				From:  []string{"PreparingAlcohol"},
				To:    []string{"BeveragePreparationComplete"},
				Event: "ALCOHOL_PREPARED",
			},

			// Consolidation transition (all parallel tasks must be complete)
			{
				Label: "AllItemsPrepared",
				From:  []string{"PreparationPhase"},
				To:    []string{"ConsolidatingItems"},
				Event: "ALL_ITEMS_READY",
			},
			{
				Label: "ItemsConsolidated",
				From:  []string{"ConsolidatingItems"},
				To:    []string{"DeliveringOrder"},
				Event: "ITEMS_CONSOLIDATED",
			},

			// Delivery and completion workflow
			{
				Label: "DeliveryComplete",
				From:  []string{"DeliveringOrder"},
				To:    []string{"CheckingForMoreOrders"},
				Event: "ORDER_DELIVERED",
			},
			{
				Label: "NoMoreOrders",
				From:  []string{"CheckingForMoreOrders"},
				To:    []string{"DebitingAccount"},
				Event: "NO_MORE_ORDERS",
			},
			{
				Label: "MoreOrdersExist",
				From:  []string{"CheckingForMoreOrders"},
				To:    []string{"Idle"},
				Event: "MORE_ORDERS_PENDING",
			},
			{
				Label: "AccountDebited",
				From:  []string{"DebitingAccount"},
				To:    []string{"OrderCompleted"},
				Event: "ACCOUNT_DEBITED",
			},

			// Emergency/cancellation transitions
			{
				Label: "CancelOrder",
				From:  []string{"OrderReceived", "OrderCreated", "PreparationPhase", "ConsolidatingItems"},
				To:    []string{"OrderCompleted"},
				Event: "ORDER_CANCELLED",
			},
		},
		Events: []*sc.Event{
			{Label: "ORDER_RECEIVED"},
			{Label: "CREATE_ORDER_TICKET"},
			{Label: "BEGIN_PREPARATION"},
			{Label: "MEALS_PREPARED"},
			{Label: "CART_PREPARED"},
			{Label: "NO_WINE_REQUESTED"},
			{Label: "WINE_REQUESTED"},
			{Label: "WINE_FETCHED"},
			{Label: "NO_OTHER_ALCOHOL_REQUESTED"},
			{Label: "OTHER_ALCOHOL_REQUESTED"},
			{Label: "ALCOHOL_PREPARED"},
			{Label: "ALL_ITEMS_READY"},
			{Label: "ITEMS_CONSOLIDATED"},
			{Label: "ORDER_DELIVERED"},
			{Label: "NO_MORE_ORDERS"},
			{Label: "MORE_ORDERS_PENDING"},
			{Label: "ACCOUNT_DEBITED"},
			{Label: "ORDER_CANCELLED"},
		},
	})
}