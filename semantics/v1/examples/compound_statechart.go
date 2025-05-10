// Package examples provides academic examples of statechart implementations.
// This file demonstrates a compound statechart combining multiple statechart features.
package examples

import (
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// CompoundStatechart creates a complex statechart that combines multiple features:
// hierarchical composition, orthogonality, and transitions.
// It models a robotic control system with multiple subsystems:
//   - Robot
//   - Standby (initial)
//   - Operational
//   - MovementControl (orthogonal region)
//   - PositionControl
//   - Stationary (initial)
//   - Moving
//   - SpeedControl
//   - Slow (initial)
//   - Medium
//   - Fast
//   - SensorSystem (orthogonal region)
//   - Radar
//   - RadarIdle (initial)
//   - RadarActive
//   - Camera
//   - CameraOff (initial)
//   - CameraOn
//   - Error
//   - SoftError (initial)
//   - HardError
//
// The example demonstrates:
// 1. Hierarchical state composition (OR-states)
// 2. Orthogonal/parallel regions (AND-states)
// 3. Complex transitions between different hierarchy levels
// 4. Multi-level state nesting
//
// This combines features from Harel's statecharts paper and subsequent academic literature.
func CompoundStatechart() *semantics.Statechart {
	return semantics.NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "Robot",
			Children: []*sc.State{
				{
					Label:     "Standby",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label:     "Operational",
					Type:      sc.StateTypeNormal,
					IsInitial: false,
					Children: []*sc.State{
						{
							Label:     "MovementControl",
							Type:      sc.StateTypeOrthogonal, // Using orthogonal (AND) semantics
							IsInitial: true,
							Children: []*sc.State{
								{
									Label: "PositionControl",
									Type:  sc.StateTypeNormal,
									Children: []*sc.State{
										{
											Label:     "Stationary",
											Type:      sc.StateTypeBasic,
											IsInitial: true,
										},
										{
											Label: "Moving",
											Type:  sc.StateTypeBasic,
										},
									},
								},
								{
									Label: "SpeedControl",
									Type:  sc.StateTypeNormal,
									Children: []*sc.State{
										{
											Label:     "Slow",
											Type:      sc.StateTypeBasic,
											IsInitial: true,
										},
										{
											Label: "Medium",
											Type:  sc.StateTypeBasic,
										},
										{
											Label: "Fast",
											Type:  sc.StateTypeBasic,
										},
									},
								},
							},
						},
						{
							Label: "SensorSystem",
							Type:  sc.StateTypeOrthogonal, // Using orthogonal (AND) semantics
							Children: []*sc.State{
								{
									Label: "Radar",
									Type:  sc.StateTypeNormal,
									Children: []*sc.State{
										{
											Label:     "RadarIdle",
											Type:      sc.StateTypeBasic,
											IsInitial: true,
										},
										{
											Label: "RadarActive",
											Type:  sc.StateTypeBasic,
										},
									},
								},
								{
									Label: "Camera",
									Type:  sc.StateTypeNormal,
									Children: []*sc.State{
										{
											Label:     "CameraOff",
											Type:      sc.StateTypeBasic,
											IsInitial: true,
										},
										{
											Label: "CameraOn",
											Type:  sc.StateTypeBasic,
										},
									},
								},
							},
						},
					},
				},
				{
					Label: "Error",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label:     "SoftError",
							Type:      sc.StateTypeBasic,
							IsInitial: true,
						},
						{
							Label:   "HardError",
							Type:    sc.StateTypeBasic,
							IsFinal: true, // Terminal state
						},
					},
				},
			},
		},
		// Define a comprehensive set of transitions
		Transitions: []*sc.Transition{
			// High-level transitions
			{
				Label: "Activate",
				From:  []string{"Standby"},
				To:    []string{"Operational"},
				Event: "START",
			},
			{
				Label: "Deactivate",
				From:  []string{"Operational"},
				To:    []string{"Standby"},
				Event: "STOP",
			},
			{
				Label: "SystemFailure",
				From:  []string{"Operational"},
				To:    []string{"Error"},
				Event: "FAILURE",
			},
			{
				Label: "Recover",
				From:  []string{"SoftError"},
				To:    []string{"Standby"},
				Event: "RESET",
			},
			{
				Label: "EscalateError",
				From:  []string{"SoftError"},
				To:    []string{"HardError"},
				Event: "FAILURE",
			},

			// Movement control transitions
			{
				Label: "StartMoving",
				From:  []string{"Stationary"},
				To:    []string{"Moving"},
				Event: "MOVE",
			},
			{
				Label: "StopMoving",
				From:  []string{"Moving"},
				To:    []string{"Stationary"},
				Event: "HALT",
			},
			{
				Label: "IncreaseSpeed",
				From:  []string{"Slow"},
				To:    []string{"Medium"},
				Event: "FASTER",
			},
			{
				Label: "IncreaseToFast",
				From:  []string{"Medium"},
				To:    []string{"Fast"},
				Event: "FASTER",
			},
			{
				Label: "DecreaseSpeed",
				From:  []string{"Fast"},
				To:    []string{"Medium"},
				Event: "SLOWER",
			},
			{
				Label: "DecreaseToSlow",
				From:  []string{"Medium"},
				To:    []string{"Slow"},
				Event: "SLOWER",
			},

			// Sensor system transitions
			{
				Label: "ActivateRadar",
				From:  []string{"RadarIdle"},
				To:    []string{"RadarActive"},
				Event: "SCAN",
			},
			{
				Label: "DeactivateRadar",
				From:  []string{"RadarActive"},
				To:    []string{"RadarIdle"},
				Event: "SCAN_COMPLETE",
			},
			{
				Label: "TurnCameraOn",
				From:  []string{"CameraOff"},
				To:    []string{"CameraOn"},
				Event: "RECORD",
			},
			{
				Label: "TurnCameraOff",
				From:  []string{"CameraOn"},
				To:    []string{"CameraOff"},
				Event: "STOP_RECORDING",
			},

			// Cross-hierarchy transitions
			{
				Label: "EmergencyStop",
				From:  []string{"Moving"},
				To:    []string{"Standby"},
				Event: "EMERGENCY",
			},
			{
				Label: "SensorFailure",
				From:  []string{"RadarActive", "CameraOn"},
				To:    []string{"SoftError"},
				Event: "SENSOR_FAILURE",
			},
		},
		// Define the events in the statechart alphabet
		Events: []*sc.Event{
			{Label: "START"},
			{Label: "STOP"},
			{Label: "FAILURE"},
			{Label: "RESET"},
			{Label: "MOVE"},
			{Label: "HALT"},
			{Label: "FASTER"},
			{Label: "SLOWER"},
			{Label: "SCAN"},
			{Label: "SCAN_COMPLETE"},
			{Label: "RECORD"},
			{Label: "STOP_RECORDING"},
			{Label: "EMERGENCY"},
			{Label: "SENSOR_FAILURE"},
		},
	})
}
