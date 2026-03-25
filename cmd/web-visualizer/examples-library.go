package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	sc "github.com/tmc/sc/gen/statecharts/v1"
	"google.golang.org/protobuf/encoding/prototext"
)

// GetRealWorldExamples returns a comprehensive library of real-world statechart examples
func GetRealWorldExamples() map[string]*sc.Statechart {
	examples := make(map[string]*sc.Statechart)

	// Authentication Flow
	examples["authentication"] = createAuthenticationFlow()

	// E-commerce Checkout
	examples["ecommerce_checkout"] = createEcommerceCheckout()

	// IoT Device Lifecycle
	examples["iot_device"] = createIoTDeviceLifecycle()

	// Game State Machine
	examples["game_state"] = createGameStateMachine()

	// Financial Transaction
	examples["financial_transaction"] = createFinancialTransaction()

	// Traffic Light System
	examples["traffic_light"] = createTrafficLightSystem()

	// User Onboarding Flow
	examples["user_onboarding"] = createUserOnboardingFlow()

	// Video Player
	examples["video_player"] = createVideoPlayerStatechart()

	// Document Workflow
	examples["document_workflow"] = createDocumentWorkflow()

	// Chat Application
	examples["chat_application"] = createChatApplication()

	// Large-scale performance test examples
	examples["large_hierarchy"] = createLargeHierarchicalStatechart()
	examples["complex_workflow"] = createComplexWorkflowStatechart()

	// Load examples from testdata
	loadTestdataExamples(examples)

	return examples
}

func loadTestdataExamples(examples map[string]*sc.Statechart) {
	// Find the testdata directory relative to where we are running
	// We try a few common locations
	paths := []string{
		"testdata",
		"../testdata",
		"../../testdata",
		"semantics/v1/testdata",
	}

	var testdataDir string
	for _, p := range paths {
		if info, err := os.Stat(p); err == nil && info.IsDir() {
			testdataDir = p
			break
		}
	}

	if testdataDir == "" {
		log.Println("Could not find testdata directory for examples")
		return
	}

	log.Printf("Loading examples from: %s", testdataDir)

	err := filepath.Walk(testdataDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		ext := filepath.Ext(path)
		if ext != ".textproto" && ext != ".txt" && ext != ".json" {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			log.Printf("Failed to read example %s: %v", path, err)
			return nil
		}

		// Try to parse as JSON first if ext is .json
		if ext == ".json" {
			var scDef sc.Statechart
			if err := json.Unmarshal(content, &scDef); err == nil {
				name := strings.TrimSuffix(filepath.Base(path), ext)
				examples["file_"+name] = &scDef
				return nil
			}
		}

		// Parse .textproto and .txt using prototext
		if ext == ".textproto" || ext == ".txt" {
			var scDef sc.Statechart
			if err := prototext.Unmarshal(content, &scDef); err == nil {
				name := strings.TrimSuffix(filepath.Base(path), ext)
				examples["file_"+name] = &scDef
				return nil
			} else {
				log.Printf("Failed to unmarshal textproto %s: %v", path, err)
			}
		}

		return nil
	})

	if err != nil {
		log.Printf("Error walking testdata: %v", err)
	}
}

// Authentication Flow Statechart
func createAuthenticationFlow() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2, // compound
			Children: []*sc.State{
				{
					Label:     "Unauthenticated",
					Type:      2, // compound
					IsInitial: true,
					Children: []*sc.State{
						{
							Label:     "LoginForm",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "SignupForm",
							Type:  1, // atomic
						},
						{
							Label: "ForgotPassword",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "Authenticating",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "ValidatingCredentials",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "TwoFactorAuth",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "Authenticated",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "Dashboard",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "Profile",
							Type:  1, // atomic
						},
						{
							Label: "Settings",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "SessionExpired",
					Type:  1, // atomic
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "Login", From: []string{"LoginForm"}, To: []string{"ValidatingCredentials"}, Event: "SUBMIT_LOGIN"},
			{Label: "Signup", From: []string{"SignupForm"}, To: []string{"ValidatingCredentials"}, Event: "SUBMIT_SIGNUP"},
			{Label: "ForgotPwd", From: []string{"LoginForm"}, To: []string{"ForgotPassword"}, Event: "FORGOT_PASSWORD"},
			{Label: "BackToLogin", From: []string{"SignupForm", "ForgotPassword"}, To: []string{"LoginForm"}, Event: "BACK_TO_LOGIN"},
			{Label: "RequireTwoFactor", From: []string{"ValidatingCredentials"}, To: []string{"TwoFactorAuth"}, Event: "REQUIRE_2FA"},
			{Label: "LoginSuccess", From: []string{"ValidatingCredentials", "TwoFactorAuth"}, To: []string{"Dashboard"}, Event: "LOGIN_SUCCESS"},
			{Label: "LoginFailed", From: []string{"ValidatingCredentials", "TwoFactorAuth"}, To: []string{"LoginForm"}, Event: "LOGIN_FAILED"},
			{Label: "NavigateProfile", From: []string{"Dashboard"}, To: []string{"Profile"}, Event: "VIEW_PROFILE"},
			{Label: "NavigateSettings", From: []string{"Dashboard", "Profile"}, To: []string{"Settings"}, Event: "VIEW_SETTINGS"},
			{Label: "BackToDashboard", From: []string{"Profile", "Settings"}, To: []string{"Dashboard"}, Event: "BACK_TO_DASHBOARD"},
			{Label: "Logout", From: []string{"Dashboard", "Profile", "Settings"}, To: []string{"LoginForm"}, Event: "LOGOUT"},
			{Label: "SessionTimeout", From: []string{"Dashboard", "Profile", "Settings"}, To: []string{"SessionExpired"}, Event: "SESSION_TIMEOUT"},
			{Label: "ReloginFromExpired", From: []string{"SessionExpired"}, To: []string{"LoginForm"}, Event: "RELOGIN"},
		},
		Events: []*sc.Event{
			{Label: "SUBMIT_LOGIN"},
			{Label: "SUBMIT_SIGNUP"},
			{Label: "FORGOT_PASSWORD"},
			{Label: "BACK_TO_LOGIN"},
			{Label: "REQUIRE_2FA"},
			{Label: "LOGIN_SUCCESS"},
			{Label: "LOGIN_FAILED"},
			{Label: "VIEW_PROFILE"},
			{Label: "VIEW_SETTINGS"},
			{Label: "BACK_TO_DASHBOARD"},
			{Label: "LOGOUT"},
			{Label: "SESSION_TIMEOUT"},
			{Label: "RELOGIN"},
		},
	}
}

// E-commerce Checkout Statechart
func createEcommerceCheckout() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2, // compound
			Children: []*sc.State{
				{
					Label:     "Cart",
					Type:      1, // atomic
					IsInitial: true,
				},
				{
					Label: "Checkout",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "ShippingInfo",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "PaymentInfo",
							Type:  1, // atomic
						},
						{
							Label: "ReviewOrder",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "Processing",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "ValidatingPayment",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "ReservingInventory",
							Type:  1, // atomic
						},
						{
							Label: "CreatingOrder",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "Completed",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "OrderConfirmed",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "EmailSent",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "Failed",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "PaymentFailed",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "InventoryUnavailable",
							Type:  1, // atomic
						},
						{
							Label: "OrderError",
							Type:  1, // atomic
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartCheckout", From: []string{"Cart"}, To: []string{"ShippingInfo"}, Event: "PROCEED_TO_CHECKOUT"},
			{Label: "BackToCart", From: []string{"ShippingInfo"}, To: []string{"Cart"}, Event: "BACK_TO_CART"},
			{Label: "NextToPayment", From: []string{"ShippingInfo"}, To: []string{"PaymentInfo"}, Event: "CONTINUE_TO_PAYMENT"},
			{Label: "BackToShipping", From: []string{"PaymentInfo"}, To: []string{"ShippingInfo"}, Event: "BACK_TO_SHIPPING"},
			{Label: "NextToReview", From: []string{"PaymentInfo"}, To: []string{"ReviewOrder"}, Event: "CONTINUE_TO_REVIEW"},
			{Label: "BackToPayment", From: []string{"ReviewOrder"}, To: []string{"PaymentInfo"}, Event: "BACK_TO_PAYMENT"},
			{Label: "PlaceOrder", From: []string{"ReviewOrder"}, To: []string{"ValidatingPayment"}, Event: "PLACE_ORDER"},
			{Label: "PaymentValid", From: []string{"ValidatingPayment"}, To: []string{"ReservingInventory"}, Event: "PAYMENT_VALID"},
			{Label: "PaymentInvalid", From: []string{"ValidatingPayment"}, To: []string{"PaymentFailed"}, Event: "PAYMENT_INVALID"},
			{Label: "InventoryReserved", From: []string{"ReservingInventory"}, To: []string{"CreatingOrder"}, Event: "INVENTORY_RESERVED"},
			{Label: "InventoryFailed", From: []string{"ReservingInventory"}, To: []string{"InventoryUnavailable"}, Event: "INVENTORY_UNAVAILABLE"},
			{Label: "OrderCreated", From: []string{"CreatingOrder"}, To: []string{"OrderConfirmed"}, Event: "ORDER_CREATED"},
			{Label: "OrderFailed", From: []string{"CreatingOrder"}, To: []string{"OrderError"}, Event: "ORDER_CREATION_FAILED"},
			{Label: "EmailConfirmation", From: []string{"OrderConfirmed"}, To: []string{"EmailSent"}, Event: "SEND_CONFIRMATION"},
			{Label: "RetryPayment", From: []string{"PaymentFailed"}, To: []string{"PaymentInfo"}, Event: "RETRY_PAYMENT"},
			{Label: "RestartCheckout", From: []string{"InventoryUnavailable", "OrderError"}, To: []string{"Cart"}, Event: "RESTART_CHECKOUT"},
		},
		Events: []*sc.Event{
			{Label: "PROCEED_TO_CHECKOUT"},
			{Label: "BACK_TO_CART"},
			{Label: "CONTINUE_TO_PAYMENT"},
			{Label: "BACK_TO_SHIPPING"},
			{Label: "CONTINUE_TO_REVIEW"},
			{Label: "BACK_TO_PAYMENT"},
			{Label: "PLACE_ORDER"},
			{Label: "PAYMENT_VALID"},
			{Label: "PAYMENT_INVALID"},
			{Label: "INVENTORY_RESERVED"},
			{Label: "INVENTORY_UNAVAILABLE"},
			{Label: "ORDER_CREATED"},
			{Label: "ORDER_CREATION_FAILED"},
			{Label: "SEND_CONFIRMATION"},
			{Label: "RETRY_PAYMENT"},
			{Label: "RESTART_CHECKOUT"},
		},
	}
}

// IoT Device Lifecycle Statechart
func createIoTDeviceLifecycle() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2, // compound
			Children: []*sc.State{
				{
					Label:     "Inactive",
					Type:      1, // atomic
					IsInitial: true,
				},
				{
					Label: "Activating",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "ConnectingToWiFi",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "RegisteringDevice",
							Type:  1, // atomic
						},
						{
							Label: "DownloadingFirmware",
							Type:  1, // atomic
						},
					},
				},
				{
					Label: "Active",
					Type:  3, // parallel
					Children: []*sc.State{
						{
							Label: "OperationalState",
							Type:  2, // compound
							Children: []*sc.State{
								{
									Label:     "Monitoring",
									Type:      1, // atomic
									IsInitial: true,
								},
								{
									Label: "AlertMode",
									Type:  1, // atomic
								},
								{
									Label: "MaintenanceMode",
									Type:  1, // atomic
								},
							},
						},
						{
							Label: "ConnectivityState",
							Type:  2, // compound
							Children: []*sc.State{
								{
									Label:     "Connected",
									Type:      1, // atomic
									IsInitial: true,
								},
								{
									Label: "Reconnecting",
									Type:  1, // atomic
								},
								{
									Label: "Offline",
									Type:  1, // atomic
								},
							},
						},
						{
							Label: "PowerState",
							Type:  2, // compound
							Children: []*sc.State{
								{
									Label:     "Normal",
									Type:      1, // atomic
									IsInitial: true,
								},
								{
									Label: "LowBattery",
									Type:  1, // atomic
								},
								{
									Label: "Charging",
									Type:  1, // atomic
								},
							},
						},
					},
				},
				{
					Label: "Error",
					Type:  2, // compound
					Children: []*sc.State{
						{
							Label:     "SoftwareError",
							Type:      1, // atomic
							IsInitial: true,
						},
						{
							Label: "HardwareError",
							Type:  1, // atomic
						},
						{
							Label: "NetworkError",
							Type:  1, // atomic
						},
					},
				},
				{
					Label:   "Decommissioned",
					Type:    1, // atomic
					IsFinal: true,
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartActivation", From: []string{"Inactive"}, To: []string{"ConnectingToWiFi"}, Event: "POWER_ON"},
			{Label: "WiFiConnected", From: []string{"ConnectingToWiFi"}, To: []string{"RegisteringDevice"}, Event: "WIFI_CONNECTED"},
			{Label: "DeviceRegistered", From: []string{"RegisteringDevice"}, To: []string{"DownloadingFirmware"}, Event: "REGISTRATION_SUCCESS"},
			{Label: "FirmwareDownloaded", From: []string{"DownloadingFirmware"}, To: []string{"Monitoring"}, Event: "FIRMWARE_READY"},
			{Label: "TriggerAlert", From: []string{"Monitoring"}, To: []string{"AlertMode"}, Event: "THRESHOLD_EXCEEDED"},
			{Label: "ClearAlert", From: []string{"AlertMode"}, To: []string{"Monitoring"}, Event: "ALERT_CLEARED"},
			{Label: "EnterMaintenance", From: []string{"Monitoring", "AlertMode"}, To: []string{"MaintenanceMode"}, Event: "MAINTENANCE_REQUEST"},
			{Label: "ExitMaintenance", From: []string{"MaintenanceMode"}, To: []string{"Monitoring"}, Event: "MAINTENANCE_COMPLETE"},
			{Label: "ConnectionLost", From: []string{"Connected"}, To: []string{"Reconnecting"}, Event: "CONNECTION_LOST"},
			{Label: "ReconnectSuccess", From: []string{"Reconnecting"}, To: []string{"Connected"}, Event: "RECONNECT_SUCCESS"},
			{Label: "GoOffline", From: []string{"Reconnecting"}, To: []string{"Offline"}, Event: "RECONNECT_FAILED"},
			{Label: "BackOnline", From: []string{"Offline"}, To: []string{"Connected"}, Event: "NETWORK_RESTORED"},
			{Label: "BatteryLow", From: []string{"Normal"}, To: []string{"LowBattery"}, Event: "BATTERY_LOW"},
			{Label: "StartCharging", From: []string{"Normal", "LowBattery"}, To: []string{"Charging"}, Event: "CHARGER_CONNECTED"},
			{Label: "ChargingComplete", From: []string{"Charging"}, To: []string{"Normal"}, Event: "CHARGING_COMPLETE"},
			{Label: "SoftwareFailure", From: []string{"Monitoring", "AlertMode"}, To: []string{"SoftwareError"}, Event: "SOFTWARE_FAULT"},
			{Label: "HardwareFailure", From: []string{"Monitoring", "AlertMode"}, To: []string{"HardwareError"}, Event: "HARDWARE_FAULT"},
			{Label: "NetworkFailure", From: []string{"Connected", "Reconnecting"}, To: []string{"NetworkError"}, Event: "NETWORK_FAULT"},
			{Label: "RecoverFromSoftware", From: []string{"SoftwareError"}, To: []string{"Monitoring"}, Event: "SOFTWARE_RESET"},
			{Label: "RecoverFromNetwork", From: []string{"NetworkError"}, To: []string{"Connected"}, Event: "NETWORK_RESET"},
			{Label: "Decommission", From: []string{"HardwareError"}, To: []string{"Decommissioned"}, Event: "DECOMMISSION"},
			{Label: "PowerOff", From: []string{"Monitoring", "Error"}, To: []string{"Inactive"}, Event: "POWER_OFF"},
		},
		Events: []*sc.Event{
			{Label: "POWER_ON"},
			{Label: "WIFI_CONNECTED"},
			{Label: "REGISTRATION_SUCCESS"},
			{Label: "FIRMWARE_READY"},
			{Label: "THRESHOLD_EXCEEDED"},
			{Label: "ALERT_CLEARED"},
			{Label: "MAINTENANCE_REQUEST"},
			{Label: "MAINTENANCE_COMPLETE"},
			{Label: "CONNECTION_LOST"},
			{Label: "RECONNECT_SUCCESS"},
			{Label: "RECONNECT_FAILED"},
			{Label: "NETWORK_RESTORED"},
			{Label: "BATTERY_LOW"},
			{Label: "CHARGER_CONNECTED"},
			{Label: "CHARGING_COMPLETE"},
			{Label: "SOFTWARE_FAULT"},
			{Label: "HARDWARE_FAULT"},
			{Label: "NETWORK_FAULT"},
			{Label: "SOFTWARE_RESET"},
			{Label: "NETWORK_RESET"},
			{Label: "DECOMMISSION"},
			{Label: "POWER_OFF"},
		},
	}
}

// Additional example creation functions would follow the same pattern...
func createGameStateMachine() *sc.Statechart {
	// Game state machine implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2,
			Children: []*sc.State{
				{Label: "MainMenu", Type: 1, IsInitial: true},
				{Label: "InGame", Type: 2, Children: []*sc.State{
					{Label: "Playing", Type: 1, IsInitial: true},
					{Label: "Paused", Type: 1},
					{Label: "GameOver", Type: 1},
				}},
				{Label: "Settings", Type: 1},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartGame", From: []string{"MainMenu"}, To: []string{"Playing"}, Event: "START_GAME"},
			{Label: "PauseGame", From: []string{"Playing"}, To: []string{"Paused"}, Event: "PAUSE"},
			{Label: "ResumeGame", From: []string{"Paused"}, To: []string{"Playing"}, Event: "RESUME"},
			{Label: "EndGame", From: []string{"Playing"}, To: []string{"GameOver"}, Event: "GAME_OVER"},
			{Label: "BackToMenu", From: []string{"GameOver", "Settings"}, To: []string{"MainMenu"}, Event: "BACK_TO_MENU"},
			{Label: "OpenSettings", From: []string{"MainMenu"}, To: []string{"Settings"}, Event: "OPEN_SETTINGS"},
		},
		Events: []*sc.Event{
			{Label: "START_GAME"},
			{Label: "PAUSE"},
			{Label: "RESUME"},
			{Label: "GAME_OVER"},
			{Label: "BACK_TO_MENU"},
			{Label: "OPEN_SETTINGS"},
		},
	}
}

func createFinancialTransaction() *sc.Statechart {
	// Financial transaction processing implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2,
			Children: []*sc.State{
				{Label: "Initiated", Type: 1, IsInitial: true},
				{Label: "Validating", Type: 2, Children: []*sc.State{
					{Label: "CheckingBalance", Type: 1, IsInitial: true},
					{Label: "VerifyingIdentity", Type: 1},
					{Label: "FraudCheck", Type: 1},
				}},
				{Label: "Processing", Type: 1},
				{Label: "Completed", Type: 1, IsFinal: true},
				{Label: "Failed", Type: 2, Children: []*sc.State{
					{Label: "InsufficientFunds", Type: 1, IsInitial: true},
					{Label: "FraudDetected", Type: 1},
					{Label: "SystemError", Type: 1},
				}},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartValidation", From: []string{"Initiated"}, To: []string{"CheckingBalance"}, Event: "VALIDATE"},
			{Label: "BalanceOK", From: []string{"CheckingBalance"}, To: []string{"VerifyingIdentity"}, Event: "BALANCE_SUFFICIENT"},
			{Label: "BalanceFail", From: []string{"CheckingBalance"}, To: []string{"InsufficientFunds"}, Event: "BALANCE_INSUFFICIENT"},
			{Label: "IdentityOK", From: []string{"VerifyingIdentity"}, To: []string{"FraudCheck"}, Event: "IDENTITY_VERIFIED"},
			{Label: "FraudOK", From: []string{"FraudCheck"}, To: []string{"Processing"}, Event: "FRAUD_CHECK_PASSED"},
			{Label: "FraudFail", From: []string{"FraudCheck"}, To: []string{"FraudDetected"}, Event: "FRAUD_DETECTED"},
			{Label: "ProcessComplete", From: []string{"Processing"}, To: []string{"Completed"}, Event: "TRANSACTION_SUCCESS"},
			{Label: "ProcessFail", From: []string{"Processing"}, To: []string{"SystemError"}, Event: "SYSTEM_ERROR"},
		},
		Events: []*sc.Event{
			{Label: "VALIDATE"},
			{Label: "BALANCE_SUFFICIENT"},
			{Label: "BALANCE_INSUFFICIENT"},
			{Label: "IDENTITY_VERIFIED"},
			{Label: "FRAUD_CHECK_PASSED"},
			{Label: "FRAUD_DETECTED"},
			{Label: "TRANSACTION_SUCCESS"},
			{Label: "SYSTEM_ERROR"},
		},
	}
}

func createTrafficLightSystem() *sc.Statechart {
	// Traffic light system implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2,
			Children: []*sc.State{
				{Label: "Red", Type: 1, IsInitial: true},
				{Label: "Green", Type: 1},
				{Label: "Yellow", Type: 1},
				{Label: "Maintenance", Type: 1},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "ToGreen", From: []string{"Red"}, To: []string{"Green"}, Event: "TIMER_EXPIRED"},
			{Label: "ToYellow", From: []string{"Green"}, To: []string{"Yellow"}, Event: "TIMER_EXPIRED"},
			{Label: "ToRed", From: []string{"Yellow"}, To: []string{"Red"}, Event: "TIMER_EXPIRED"},
			{Label: "ToMaintenance", From: []string{"Red", "Green", "Yellow"}, To: []string{"Maintenance"}, Event: "MAINTENANCE_MODE"},
			{Label: "FromMaintenance", From: []string{"Maintenance"}, To: []string{"Red"}, Event: "MAINTENANCE_COMPLETE"},
		},
		Events: []*sc.Event{
			{Label: "TIMER_EXPIRED"},
			{Label: "MAINTENANCE_MODE"},
			{Label: "MAINTENANCE_COMPLETE"},
		},
	}
}

func createUserOnboardingFlow() *sc.Statechart {
	// User onboarding flow implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2,
			Children: []*sc.State{
				{Label: "Welcome", Type: 1, IsInitial: true},
				{Label: "CreateProfile", Type: 1},
				{Label: "EmailVerification", Type: 1},
				{Label: "Tutorial", Type: 2, Children: []*sc.State{
					{Label: "Step1", Type: 1, IsInitial: true},
					{Label: "Step2", Type: 1},
					{Label: "Step3", Type: 1},
				}},
				{Label: "Completed", Type: 1, IsFinal: true},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "StartProfile", From: []string{"Welcome"}, To: []string{"CreateProfile"}, Event: "GET_STARTED"},
			{Label: "ProfileCreated", From: []string{"CreateProfile"}, To: []string{"EmailVerification"}, Event: "PROFILE_COMPLETE"},
			{Label: "EmailVerified", From: []string{"EmailVerification"}, To: []string{"Step1"}, Event: "EMAIL_VERIFIED"},
			{Label: "NextStep", From: []string{"Step1"}, To: []string{"Step2"}, Event: "NEXT"},
			{Label: "NextStep2", From: []string{"Step2"}, To: []string{"Step3"}, Event: "NEXT"},
			{Label: "FinishTutorial", From: []string{"Step3"}, To: []string{"Completed"}, Event: "FINISH"},
			{Label: "SkipTutorial", From: []string{"Step1", "Step2", "Step3"}, To: []string{"Completed"}, Event: "SKIP"},
		},
		Events: []*sc.Event{
			{Label: "GET_STARTED"},
			{Label: "PROFILE_COMPLETE"},
			{Label: "EMAIL_VERIFIED"},
			{Label: "NEXT"},
			{Label: "FINISH"},
			{Label: "SKIP"},
		},
	}
}

func createVideoPlayerStatechart() *sc.Statechart {
	// Video player implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  3, // parallel
			Children: []*sc.State{
				{
					Label: "PlaybackState",
					Type:  2,
					Children: []*sc.State{
						{Label: "Stopped", Type: 1, IsInitial: true},
						{Label: "Playing", Type: 1},
						{Label: "Paused", Type: 1},
						{Label: "Buffering", Type: 1},
					},
				},
				{
					Label: "VolumeState",
					Type:  2,
					Children: []*sc.State{
						{Label: "Audible", Type: 1, IsInitial: true},
						{Label: "Muted", Type: 1},
					},
				},
				{
					Label: "QualityState",
					Type:  2,
					Children: []*sc.State{
						{Label: "Auto", Type: 1, IsInitial: true},
						{Label: "High", Type: 1},
						{Label: "Medium", Type: 1},
						{Label: "Low", Type: 1},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "Play", From: []string{"Stopped", "Paused"}, To: []string{"Playing"}, Event: "PLAY"},
			{Label: "Pause", From: []string{"Playing"}, To: []string{"Paused"}, Event: "PAUSE"},
			{Label: "Stop", From: []string{"Playing", "Paused"}, To: []string{"Stopped"}, Event: "STOP"},
			{Label: "Buffer", From: []string{"Playing"}, To: []string{"Buffering"}, Event: "BUFFER_START"},
			{Label: "BufferComplete", From: []string{"Buffering"}, To: []string{"Playing"}, Event: "BUFFER_COMPLETE"},
			{Label: "Mute", From: []string{"Audible"}, To: []string{"Muted"}, Event: "MUTE"},
			{Label: "Unmute", From: []string{"Muted"}, To: []string{"Audible"}, Event: "UNMUTE"},
			{Label: "SetHigh", From: []string{"Auto", "Medium", "Low"}, To: []string{"High"}, Event: "SET_HIGH_QUALITY"},
			{Label: "SetMedium", From: []string{"Auto", "High", "Low"}, To: []string{"Medium"}, Event: "SET_MEDIUM_QUALITY"},
			{Label: "SetLow", From: []string{"Auto", "High", "Medium"}, To: []string{"Low"}, Event: "SET_LOW_QUALITY"},
			{Label: "SetAuto", From: []string{"High", "Medium", "Low"}, To: []string{"Auto"}, Event: "SET_AUTO_QUALITY"},
		},
		Events: []*sc.Event{
			{Label: "PLAY"},
			{Label: "PAUSE"},
			{Label: "STOP"},
			{Label: "BUFFER_START"},
			{Label: "BUFFER_COMPLETE"},
			{Label: "MUTE"},
			{Label: "UNMUTE"},
			{Label: "SET_HIGH_QUALITY"},
			{Label: "SET_MEDIUM_QUALITY"},
			{Label: "SET_LOW_QUALITY"},
			{Label: "SET_AUTO_QUALITY"},
		},
	}
}

func createDocumentWorkflow() *sc.Statechart {
	// Document workflow implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  2,
			Children: []*sc.State{
				{Label: "Draft", Type: 1, IsInitial: true},
				{Label: "Review", Type: 2, Children: []*sc.State{
					{Label: "PendingReview", Type: 1, IsInitial: true},
					{Label: "InReview", Type: 1},
					{Label: "ChangesRequested", Type: 1},
				}},
				{Label: "Approved", Type: 1},
				{Label: "Published", Type: 1, IsFinal: true},
				{Label: "Archived", Type: 1, IsFinal: true},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "SubmitReview", From: []string{"Draft"}, To: []string{"PendingReview"}, Event: "SUBMIT_FOR_REVIEW"},
			{Label: "StartReview", From: []string{"PendingReview"}, To: []string{"InReview"}, Event: "START_REVIEW"},
			{Label: "RequestChanges", From: []string{"InReview"}, To: []string{"ChangesRequested"}, Event: "REQUEST_CHANGES"},
			{Label: "ApproveDocument", From: []string{"InReview"}, To: []string{"Approved"}, Event: "APPROVE"},
			{Label: "MakeChanges", From: []string{"ChangesRequested"}, To: []string{"Draft"}, Event: "MAKE_CHANGES"},
			{Label: "PublishDocument", From: []string{"Approved"}, To: []string{"Published"}, Event: "PUBLISH"},
			{Label: "ArchiveDocument", From: []string{"Published"}, To: []string{"Archived"}, Event: "ARCHIVE"},
			{Label: "WithdrawFromReview", From: []string{"PendingReview", "InReview"}, To: []string{"Draft"}, Event: "WITHDRAW"},
		},
		Events: []*sc.Event{
			{Label: "SUBMIT_FOR_REVIEW"},
			{Label: "START_REVIEW"},
			{Label: "REQUEST_CHANGES"},
			{Label: "APPROVE"},
			{Label: "MAKE_CHANGES"},
			{Label: "PUBLISH"},
			{Label: "ARCHIVE"},
			{Label: "WITHDRAW"},
		},
	}
}

func createChatApplication() *sc.Statechart {
	// Chat application implementation
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  3, // parallel
			Children: []*sc.State{
				{
					Label: "ConnectionState",
					Type:  2,
					Children: []*sc.State{
						{Label: "Disconnected", Type: 1, IsInitial: true},
						{Label: "Connecting", Type: 1},
						{Label: "Connected", Type: 1},
						{Label: "Reconnecting", Type: 1},
					},
				},
				{
					Label: "ChatState",
					Type:  2,
					Children: []*sc.State{
						{Label: "Idle", Type: 1, IsInitial: true},
						{Label: "Typing", Type: 1},
						{Label: "SendingMessage", Type: 1},
					},
				},
				{
					Label: "PresenceState",
					Type:  2,
					Children: []*sc.State{
						{Label: "Online", Type: 1, IsInitial: true},
						{Label: "Away", Type: 1},
						{Label: "Busy", Type: 1},
						{Label: "Invisible", Type: 1},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "Connect", From: []string{"Disconnected"}, To: []string{"Connecting"}, Event: "CONNECT"},
			{Label: "Connected", From: []string{"Connecting"}, To: []string{"Connected"}, Event: "CONNECTION_SUCCESS"},
			{Label: "ConnectionFailed", From: []string{"Connecting"}, To: []string{"Disconnected"}, Event: "CONNECTION_FAILED"},
			{Label: "Disconnect", From: []string{"Connected"}, To: []string{"Disconnected"}, Event: "DISCONNECT"},
			{Label: "ConnectionLost", From: []string{"Connected"}, To: []string{"Reconnecting"}, Event: "CONNECTION_LOST"},
			{Label: "Reconnected", From: []string{"Reconnecting"}, To: []string{"Connected"}, Event: "RECONNECTION_SUCCESS"},
			{Label: "StartTyping", From: []string{"Idle"}, To: []string{"Typing"}, Event: "START_TYPING"},
			{Label: "StopTyping", From: []string{"Typing"}, To: []string{"Idle"}, Event: "STOP_TYPING"},
			{Label: "SendMessage", From: []string{"Typing"}, To: []string{"SendingMessage"}, Event: "SEND_MESSAGE"},
			{Label: "MessageSent", From: []string{"SendingMessage"}, To: []string{"Idle"}, Event: "MESSAGE_SENT"},
			{Label: "GoAway", From: []string{"Online"}, To: []string{"Away"}, Event: "SET_AWAY"},
			{Label: "GoBusy", From: []string{"Online", "Away"}, To: []string{"Busy"}, Event: "SET_BUSY"},
			{Label: "GoInvisible", From: []string{"Online", "Away", "Busy"}, To: []string{"Invisible"}, Event: "SET_INVISIBLE"},
			{Label: "GoOnline", From: []string{"Away", "Busy", "Invisible"}, To: []string{"Online"}, Event: "SET_ONLINE"},
		},
		Events: []*sc.Event{
			{Label: "CONNECT"},
			{Label: "CONNECTION_SUCCESS"},
			{Label: "CONNECTION_FAILED"},
			{Label: "DISCONNECT"},
			{Label: "CONNECTION_LOST"},
			{Label: "RECONNECTION_SUCCESS"},
			{Label: "START_TYPING"},
			{Label: "STOP_TYPING"},
			{Label: "SEND_MESSAGE"},
			{Label: "MESSAGE_SENT"},
			{Label: "SET_AWAY"},
			{Label: "SET_BUSY"},
			{Label: "SET_INVISIBLE"},
			{Label: "SET_ONLINE"},
		},
	}
}

// Large Hierarchical Statechart for performance testing (150+ states)
func createLargeHierarchicalStatechart() *sc.Statechart {
	states := make([]*sc.State, 0)
	transitions := make([]*sc.Transition, 0)
	events := make([]*sc.Event, 0)

	// Create a deep hierarchical structure with many states
	for i := 0; i < 10; i++ {
		// Level 1 - Major subsystems
		subsystemStates := make([]*sc.State, 0)

		for j := 0; j < 15; j++ {
			// Level 2 - Components within subsystems
			componentStates := make([]*sc.State, 0)

			for k := 0; k < 5; k++ {
				// Level 3 - Individual states
				stateName := fmt.Sprintf("State_%d_%d_%d", i, j, k)
				state := &sc.State{
					Label:    stateName,
					Type:     1, // atomic
					Children: []*sc.State{},
				}

				if k == 0 {
					state.IsInitial = true
				}
				if k == 4 {
					state.IsFinal = true
				}

				componentStates = append(componentStates, state)

				// Add transitions between states in the same component
				if k > 0 {
					prevStateName := fmt.Sprintf("State_%d_%d_%d", i, j, k-1)
					eventName := fmt.Sprintf("NEXT_%d_%d_%d", i, j, k)

					transitions = append(transitions, &sc.Transition{
						From:  []string{prevStateName},
						To:    []string{stateName},
						Event: eventName,
					})

					events = append(events, &sc.Event{Label: eventName})
				}
			}

			componentName := fmt.Sprintf("Component_%d_%d", i, j)
			component := &sc.State{
				Label:    componentName,
				Type:     2, // compound
				Children: componentStates,
			}

			subsystemStates = append(subsystemStates, component)

			// Add transitions between components
			if j > 0 {
				prevComponentName := fmt.Sprintf("Component_%d_%d", i, j-1)
				eventName := fmt.Sprintf("COMPONENT_TRANSITION_%d_%d", i, j)

				transitions = append(transitions, &sc.Transition{
					From:  []string{prevComponentName},
					To:    []string{componentName},
					Event: eventName,
				})

				events = append(events, &sc.Event{Label: eventName})
			}
		}

		subsystemName := fmt.Sprintf("Subsystem_%d", i)
		subsystem := &sc.State{
			Label:    subsystemName,
			Type:     2, // compound
			Children: subsystemStates,
		}

		if i == 0 {
			subsystem.IsInitial = true
		}

		states = append(states, subsystem)

		// Add transitions between subsystems
		if i > 0 {
			prevSubsystemName := fmt.Sprintf("Subsystem_%d", i-1)
			eventName := fmt.Sprintf("SUBSYSTEM_TRANSITION_%d", i)

			transitions = append(transitions, &sc.Transition{
				From:  []string{prevSubsystemName},
				To:    []string{subsystemName},
				Event: eventName,
			})

			events = append(events, &sc.Event{Label: eventName})
		}
	}

	return &sc.Statechart{
		RootState: &sc.State{
			Label:    "__root__",
			Type:     2, // compound
			Children: states,
		},
		Transitions: transitions,
		Events:      events,
	}
}

// Complex Workflow Statechart for performance testing (200+ states, many transitions)
func createComplexWorkflowStatechart() *sc.Statechart {
	states := make([]*sc.State, 0)
	transitions := make([]*sc.Transition, 0)
	events := make([]*sc.Event, 0)

	// Create multiple parallel workflow tracks
	parallelTracks := make([]*sc.State, 0)

	for trackId := 0; trackId < 5; trackId++ {
		trackStates := make([]*sc.State, 0)

		// Each track has multiple stages
		for stageId := 0; stageId < 8; stageId++ {
			stageStates := make([]*sc.State, 0)

			// Each stage has multiple steps
			for stepId := 0; stepId < 10; stepId++ {
				stepName := fmt.Sprintf("Track%d_Stage%d_Step%d", trackId, stageId, stepId)
				step := &sc.State{
					Label:    stepName,
					Type:     1, // atomic
					Children: []*sc.State{},
				}

				if stepId == 0 && stageId == 0 {
					step.IsInitial = true
				}

				stageStates = append(stageStates, step)

				// Linear progression within stage
				if stepId > 0 {
					prevStepName := fmt.Sprintf("Track%d_Stage%d_Step%d", trackId, stageId, stepId-1)
					eventName := fmt.Sprintf("STEP_COMPLETE_%d_%d_%d", trackId, stageId, stepId)

					transitions = append(transitions, &sc.Transition{
						From:  []string{prevStepName},
						To:    []string{stepName},
						Event: eventName,
					})

					events = append(events, &sc.Event{Label: eventName})
				}

				// Add error transitions (every step can go to error state)
				errorStateName := fmt.Sprintf("Track%d_Error", trackId)
				errorEventName := fmt.Sprintf("ERROR_%d_%d_%d", trackId, stageId, stepId)

				transitions = append(transitions, &sc.Transition{
					From:  []string{stepName},
					To:    []string{errorStateName},
					Event: errorEventName,
				})

				events = append(events, &sc.Event{Label: errorEventName})
			}

			stageName := fmt.Sprintf("Track%d_Stage%d", trackId, stageId)
			stage := &sc.State{
				Label:    stageName,
				Type:     2, // compound
				Children: stageStates,
			}

			trackStates = append(trackStates, stage)

			// Stage to stage transitions
			if stageId > 0 {
				prevStageName := fmt.Sprintf("Track%d_Stage%d", trackId, stageId-1)
				eventName := fmt.Sprintf("STAGE_COMPLETE_%d_%d", trackId, stageId)

				transitions = append(transitions, &sc.Transition{
					From:  []string{prevStageName},
					To:    []string{stageName},
					Event: eventName,
				})

				events = append(events, &sc.Event{Label: eventName})
			}
		}

		// Add error and completed states for each track
		errorState := &sc.State{
			Label:    fmt.Sprintf("Track%d_Error", trackId),
			Type:     1, // atomic
			Children: []*sc.State{},
		}

		completedState := &sc.State{
			Label:    fmt.Sprintf("Track%d_Completed", trackId),
			Type:     1, // atomic
			IsFinal:  true,
			Children: []*sc.State{},
		}

		trackStates = append(trackStates, errorState, completedState)

		// Final stage to completed transition
		finalStageName := fmt.Sprintf("Track%d_Stage%d", trackId, 7)
		completedEventName := fmt.Sprintf("TRACK_COMPLETE_%d", trackId)

		transitions = append(transitions, &sc.Transition{
			From:  []string{finalStageName},
			To:    []string{fmt.Sprintf("Track%d_Completed", trackId)},
			Event: completedEventName,
		})

		events = append(events, &sc.Event{Label: completedEventName})

		// Recovery transitions from error state
		for stageId := 0; stageId < 8; stageId++ {
			stageName := fmt.Sprintf("Track%d_Stage%d", trackId, stageId)
			retryEventName := fmt.Sprintf("RETRY_%d_%d", trackId, stageId)

			transitions = append(transitions, &sc.Transition{
				From:  []string{fmt.Sprintf("Track%d_Error", trackId)},
				To:    []string{stageName},
				Event: retryEventName,
			})

			events = append(events, &sc.Event{Label: retryEventName})
		}

		trackName := fmt.Sprintf("Track_%d", trackId)
		track := &sc.State{
			Label:    trackName,
			Type:     2, // compound
			Children: trackStates,
		}

		parallelTracks = append(parallelTracks, track)
	}

	// Create the main parallel region
	mainRegion := &sc.State{
		Label:    "WorkflowRegion",
		Type:     3, // parallel
		Children: parallelTracks,
	}

	states = append(states, mainRegion)

	// Add coordination states
	coordinationStates := []*sc.State{
		{Label: "Initializing", Type: 1, IsInitial: true, Children: []*sc.State{}},
		{Label: "AllCompleted", Type: 1, IsFinal: true, Children: []*sc.State{}},
		{Label: "PartialFailure", Type: 1, Children: []*sc.State{}},
	}

	states = append(states, coordinationStates...)

	// Add coordination transitions
	transitions = append(transitions, &sc.Transition{
		From:  []string{"Initializing"},
		To:    []string{"WorkflowRegion"},
		Event: "START_WORKFLOW",
	})

	events = append(events, &sc.Event{Label: "START_WORKFLOW"})

	return &sc.Statechart{
		RootState: &sc.State{
			Label:    "__root__",
			Type:     2, // compound
			Children: states,
		},
		Transitions: transitions,
		Events:      events,
	}
}
