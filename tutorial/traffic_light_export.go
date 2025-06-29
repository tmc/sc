package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// SmartTrafficLight creates a statechart for a smart traffic light system
func SmartTrafficLight() *semantics.Statechart {
	return semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "SmartTrafficLight",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Normal",
					Type:      sc.StateTypeNormal,
					IsInitial: true,
					Children: []*sc.State{
						{
							Label:     "Red",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "Green",
							Type:  sc.StateTypeBasic,
						},
						{
							Label: "Yellow",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Emergency",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "FlashingRed",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label: "FlashingOff",
							Type:  sc.StateTypeBasic,
						},
					},
				},
				{
					Label: "Maintenance",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "ToGreen",
				From:  []string{"Red"},
				To:    []string{"Green"},
				Event: "TIMER_EXPIRED",
			},
			{
				Label: "ToYellow",
				From:  []string{"Green"},
				To:    []string{"Yellow"},
				Event: "TIMER_EXPIRED",
			},
			{
				Label: "ToRed",
				From:  []string{"Yellow"},
				To:    []string{"Red"},
				Event: "TIMER_EXPIRED",
			},
			{
				Label: "EmergencyActivate",
				From:  []string{"Normal"},
				To:    []string{"Emergency"},
				Event: "EMERGENCY",
			},
			{
				Label: "EmergencyDeactivate",
				From:  []string{"Emergency"},
				To:    []string{"Normal"},
				Event: "EMERGENCY_CLEAR",
			},
			{
				Label: "FlashOn",
				From:  []string{"FlashingOff"},
				To:    []string{"FlashingRed"},
				Event: "FLASH_TIMER",
			},
			{
				Label: "FlashOff",
				From:  []string{"FlashingRed"},
				To:    []string{"FlashingOff"},
				Event: "FLASH_TIMER",
			},
			{
				Label: "EnterMaintenance",
				From:  []string{"Normal", "Emergency"},
				To:    []string{"Maintenance"},
				Event: "MAINTENANCE",
			},
			{
				Label: "ExitMaintenance",
				From:  []string{"Maintenance"},
				To:    []string{"Normal"},
				Event: "MAINTENANCE_COMPLETE",
			},
		},
		Events: []*sc.Event{
			{Label: "TIMER_EXPIRED"},
			{Label: "EMERGENCY"},
			{Label: "EMERGENCY_CLEAR"},
			{Label: "FLASH_TIMER"},
			{Label: "MAINTENANCE"},
			{Label: "MAINTENANCE_COMPLETE"},
		},
	})
}

// Export the traffic light statechart to JSON for the web visualizer
func main() {
	// Create the traffic light statechart
	chart := SmartTrafficLight()
	
	// Create a machine instance for export
	initialContext := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"timer":          structpb.NewNumberValue(0),
			"emergency_mode": structpb.NewBoolValue(false),
		},
	}
	
	machine, err := semantics.NewMachine(chart, "traffic-light-export", initialContext)
	if err != nil {
		log.Fatalf("Failed to create machine: %v", err)
	}
	
	// Export to JSON
	jsonData, err := json.MarshalIndent(machine, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal to JSON: %v", err)
	}
	
	// Write to file
	filename := "traffic_light_statechart.json"
	err = os.WriteFile(filename, jsonData, 0644)
	if err != nil {
		log.Fatalf("Failed to write file: %v", err)
	}
	
	fmt.Printf("✓ Exported traffic light statechart to %s\n", filename)
	fmt.Println("✓ Use this file with the web visualizer:")
	fmt.Println("  1. cd ../cmd/web-visualizer")
	fmt.Println("  2. go run main.go")
	fmt.Println("  3. Open http://localhost:8080")
	fmt.Println("  4. Import the JSON file")
}