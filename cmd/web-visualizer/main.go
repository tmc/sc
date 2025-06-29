package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// Server represents the enhanced statechart web server
type Server struct {
	machines    map[string]*semantics.MachineWrapper
	upgrader    websocket.Upgrader
	mutex       sync.RWMutex
	clients     map[*websocket.Conn]*Client
	persistence *PersistenceManager
	metrics     *MetricsCollector
	startTime   time.Time
}

// Client represents a connected WebSocket client
type Client struct {
	Conn      *websocket.Conn
	ID        string
	MachineID string
	LastSeen  time.Time
}

// PersistenceManager handles data persistence
type PersistenceManager struct {
	dataDir string
	mutex   sync.RWMutex
}

// MetricsCollector collects server metrics
type MetricsCollector struct {
	requestCount    int64
	connectionCount int64
	errorCount      int64
	mutex           sync.RWMutex
}

// API Request/Response types
type CreateMachineRequest struct {
	ID         string                 `json:"id"`
	Statechart *sc.Statechart        `json:"statechart"`
	Context    map[string]interface{} `json:"context,omitempty"`
	Tags       []string               `json:"tags,omitempty"`
	Metadata   map[string]string      `json:"metadata,omitempty"`
}

type UpdateMachineRequest struct {
	Statechart *sc.Statechart        `json:"statechart,omitempty"`
	Context    map[string]interface{} `json:"context,omitempty"`
	Tags       []string               `json:"tags,omitempty"`
	Metadata   map[string]string      `json:"metadata,omitempty"`
}

type ProcessEventRequest struct {
	Event string                   `json:"event"`
	Data  map[string]interface{}   `json:"data,omitempty"`
}

type BatchEventRequest struct {
	Events []ProcessEventRequest `json:"events"`
}

type ValidationResponse struct {
	Valid    bool     `json:"valid"`
	Errors   []string `json:"errors,omitempty"`
	Warnings []string `json:"warnings,omitempty"`
}

type MachineSearchRequest struct {
	Query  string   `json:"query,omitempty"`
	Tags   []string `json:"tags,omitempty"`
	Limit  int      `json:"limit,omitempty"`
	Offset int      `json:"offset,omitempty"`
}

type WebSocketMessage struct {
	Type      string      `json:"type"`
	MachineID string      `json:"machine_id,omitempty"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	ClientID  string      `json:"client_id,omitempty"`
}

func NewServer() *Server {
	dataDir := getEnv("DATA_DIR", "./data")
	os.MkdirAll(dataDir, 0755)

	return &Server{
		machines:  make(map[string]*semantics.MachineWrapper),
		clients:   make(map[*websocket.Conn]*Client),
		startTime: time.Now(),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for development
			},
		},
		persistence: &PersistenceManager{
			dataDir: dataDir,
		},
		metrics: &MetricsCollector{},
	}
}

// Middleware for request logging and metrics
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		s.metrics.incrementRequestCount()

		next.ServeHTTP(w, r)

		log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

// CORS middleware
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *Server) GetExamples(w http.ResponseWriter, r *http.Request) {
	examples := GetRealWorldExamples() // Use the comprehensive real-world examples library

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(examples)
}

func (s *Server) CreateMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	var req CreateMachineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.ID == "" {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine ID is required", http.StatusBadRequest)
		return
	}

	if req.Statechart == nil {
		s.metrics.incrementErrorCount()
		http.Error(w, "Statechart is required", http.StatusBadRequest)
		return
	}

	// Check if machine already exists
	if _, exists := s.machines[req.ID]; exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine with this ID already exists", http.StatusConflict)
		return
	}

	// Create context from request
	var context *structpb.Struct
	if req.Context != nil {
		var err error
		context, err = structpb.NewStruct(req.Context)
		if err != nil {
			s.metrics.incrementErrorCount()
			http.Error(w, fmt.Sprintf("Invalid context: %v", err), http.StatusBadRequest)
			return
		}
	}

	// Create statechart wrapper
	statechart := semantics.NewStatechart(req.Statechart)

	// Validate statechart
	if err := statechart.ValidateAdvanced(); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid statechart: %v", err), http.StatusBadRequest)
		return
	}

	// Create machine
	machine, err := semantics.NewMachine(statechart, req.ID, context)
	if err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to create machine: %v", err), http.StatusBadRequest)
		return
	}

	// Start the machine
	if err := machine.Start(); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to start machine: %v", err), http.StatusInternalServerError)
		return
	}

	s.machines[req.ID] = machine

	// Persist machine
	if err := s.persistence.SaveMachine(req.ID, machine); err != nil {
		log.Printf("Failed to persist machine %s: %v", req.ID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(req.ID, "created", machine)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"tags":          req.Tags,
		"metadata":      req.Metadata,
		"created_at":    time.Now(),
	})
}

func (s *Server) GetMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"state":         machine.State.String(),
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"statechart":    machine.Statechart,
		"step_history":  machine.StepHistory,
		"errors":        machine.GetErrors(),
		"last_updated":  time.Now(),
	})
}

func (s *Server) UpdateMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	var req UpdateMachineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Update statechart if provided
	if req.Statechart != nil {
		statechart := semantics.NewStatechart(req.Statechart)
		if err := statechart.ValidateAdvanced(); err != nil {
			s.metrics.incrementErrorCount()
			http.Error(w, fmt.Sprintf("Invalid statechart: %v", err), http.StatusBadRequest)
			return
		}
		machine.Statechart = req.Statechart
	}

	// Update context if provided
	if req.Context != nil {
		context, err := structpb.NewStruct(req.Context)
		if err != nil {
			s.metrics.incrementErrorCount()
			http.Error(w, fmt.Sprintf("Invalid context: %v", err), http.StatusBadRequest)
			return
		}
		machine.Context = context
	}

	// Persist updated machine
	if err := s.persistence.SaveMachine(machineID, machine); err != nil {
		log.Printf("Failed to persist updated machine %s: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(machineID, "updated", machine)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"updated_at":    time.Now(),
	})
}

func (s *Server) DeleteMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	// Stop machine if running
	if machine.IsRunning() {
		machine.Stop()
	}

	delete(s.machines, machineID)

	// Remove from persistence
	if err := s.persistence.DeleteMachine(machineID); err != nil {
		log.Printf("Failed to delete persisted machine %s: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(machineID, "deleted", nil)

	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) ListMachines(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	machines := make([]map[string]interface{}, 0, len(s.machines))
	for id, machine := range s.machines {
		machines = append(machines, map[string]interface{}{
			"id":            id,
			"state":         machine.State.String(),
			"configuration": getStateLabels(machine.Configuration.States),
			"last_updated":  time.Now(),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"machines": machines,
		"total":    len(machines),
	})
}

func (s *Server) ProcessEvent(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	var req ProcessEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Process the event
	transitioned, err := machine.Step(req.Event)
	if err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to process event: %v", err), http.StatusInternalServerError)
		return
	}

	// Persist updated machine state
	if err := s.persistence.SaveMachine(machineID, machine); err != nil {
		log.Printf("Failed to persist machine %s after event: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastEventProcessed(machineID, req.Event, machine)

	response := map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"last_step":     getLastStep(machine.StepHistory),
		"transitioned":  transitioned,
		"timestamp":     time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) ProcessBatchEvents(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	var req BatchEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	results := make([]map[string]interface{}, 0, len(req.Events))

	for _, eventReq := range req.Events {
		transitioned, err := machine.Step(eventReq.Event)
		if err != nil {
			s.metrics.incrementErrorCount()
			results = append(results, map[string]interface{}{
				"event":   eventReq.Event,
				"error":   err.Error(),
				"success": false,
			})
			continue
		}

		results = append(results, map[string]interface{}{
			"event":        eventReq.Event,
			"transitioned": transitioned,
			"success":      true,
		})
	}

	// Persist updated machine state
	if err := s.persistence.SaveMachine(machineID, machine); err != nil {
		log.Printf("Failed to persist machine %s after batch events: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(machineID, "batch_processed", machine)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"results":       results,
		"timestamp":     time.Now(),
	})
}

func (s *Server) ResetMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	if err := machine.Reset(); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to reset machine: %v", err), http.StatusInternalServerError)
		return
	}

	// Persist reset machine state
	if err := s.persistence.SaveMachine(machineID, machine); err != nil {
		log.Printf("Failed to persist reset machine %s: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(machineID, "reset", machine)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"reset_at":      time.Now(),
	})
}

func (s *Server) ValidateMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	errors := []string{}
	warnings := []string{}

	// Validate machine
	if err := machine.Validate(); err != nil {
		errors = append(errors, err.Error())
	}

	// Get machine errors
	machineErrors := machine.GetErrors()
	for _, err := range machineErrors {
		errors = append(errors, err.Error())
	}

	// Additional validation checks
	if machine.IsStopped() {
		warnings = append(warnings, "Machine is currently stopped")
	}

	response := ValidationResponse{
		Valid:    len(errors) == 0,
		Errors:   errors,
		Warnings: warnings,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) SearchMachines(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	var req MachineSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Try query parameters if JSON decode fails
		req.Query = r.URL.Query().Get("query")
		req.Limit, _ = strconv.Atoi(r.URL.Query().Get("limit"))
		req.Offset, _ = strconv.Atoi(r.URL.Query().Get("offset"))
	}

	if req.Limit == 0 {
		req.Limit = 10
	}

	machines := make([]map[string]interface{}, 0)
	count := 0

	for id, machine := range s.machines {
		if count < req.Offset {
			count++
			continue
		}

		if len(machines) >= req.Limit {
			break
		}

		// Simple text search in machine ID
		if req.Query != "" && !strings.Contains(strings.ToLower(id), strings.ToLower(req.Query)) {
			continue
		}

		machines = append(machines, map[string]interface{}{
			"id":            id,
			"state":         machine.State.String(),
			"configuration": getStateLabels(machine.Configuration.States),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"machines": machines,
		"total":    len(s.machines),
		"offset":   req.Offset,
		"limit":    req.Limit,
	})
}

func (s *Server) GetMetrics(w http.ResponseWriter, r *http.Request) {
	s.metrics.mutex.RLock()
	defer s.metrics.mutex.RUnlock()

	s.mutex.RLock()
	machineCount := len(s.machines)
	clientCount := len(s.clients)
	s.mutex.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"request_count":    s.metrics.requestCount,
		"connection_count": s.metrics.connectionCount,
		"error_count":      s.metrics.errorCount,
		"active_machines":  machineCount,
		"active_clients":   clientCount,
		"uptime":           time.Since(s.startTime).String(),
		"timestamp":        time.Now(),
	})
}

func (s *Server) GetHealth(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	machineCount := len(s.machines)
	clientCount := len(s.clients)
	s.mutex.RUnlock()

	health := map[string]interface{}{
		"status":          "healthy",
		"timestamp":       time.Now(),
		"active_machines": machineCount,
		"active_clients":  clientCount,
		"uptime":          time.Since(s.startTime).String(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (s *Server) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		s.metrics.incrementErrorCount()
		return
	}
	defer conn.Close()

	// Create client
	clientID := generateClientID()
	client := &Client{
		Conn:     conn,
		ID:       clientID,
		LastSeen: time.Now(),
	}

	// Register client
	s.mutex.Lock()
	s.clients[conn] = client
	s.metrics.incrementConnectionCount()
	s.mutex.Unlock()

	defer func() {
		s.mutex.Lock()
		delete(s.clients, conn)
		s.mutex.Unlock()
	}()

	// Send welcome message
	welcomeMsg := WebSocketMessage{
		Type:      "welcome",
		ClientID:  clientID,
		Data:      map[string]interface{}{"connected_clients": len(s.clients)},
		Timestamp: time.Now(),
	}
	conn.WriteJSON(welcomeMsg)

	// Handle WebSocket messages
	for {
		var msg WebSocketMessage
		err := conn.ReadJSON(&msg)
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Update client last seen
		s.mutex.Lock()
		client.LastSeen = time.Now()
		s.mutex.Unlock()

		// Handle different message types
		switch msg.Type {
		case "subscribe":
			s.handleSubscribe(client, msg)
		case "unsubscribe":
			s.handleUnsubscribe(client, msg)
		case "ping":
			s.handlePing(client, msg)
		default:
			log.Printf("Unknown WebSocket message type: %s", msg.Type)
		}
	}
}

func (s *Server) ServeStaticFiles() http.Handler {
	return http.FileServer(http.Dir("./web/"))
}

// Utility functions

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func generateClientID() string {
	return fmt.Sprintf("client_%d", time.Now().UnixNano())
}

func getStateLabels(stateRefs []*sc.StateRef) []string {
	labels := make([]string, len(stateRefs))
	for i, ref := range stateRefs {
		labels[i] = ref.Label
	}
	return labels
}

func getLastStep(steps []*sc.Step) *sc.Step {
	if len(steps) == 0 {
		return nil
	}
	return steps[len(steps)-1]
}

// WebSocket message handlers

func (s *Server) handleSubscribe(client *Client, msg WebSocketMessage) {
	if machineID, ok := msg.Data.(string); ok {
		client.MachineID = machineID

		// Send current machine state
		s.mutex.RLock()
		if machine, exists := s.machines[machineID]; exists {
			response := WebSocketMessage{
				Type:      "machine_state",
				MachineID: machineID,
				Data: map[string]interface{}{
					"configuration": getStateLabels(machine.Configuration.States),
					"context":       machine.Context.AsMap(),
					"state":         machine.State.String(),
				},
				Timestamp: time.Now(),
			}
			client.Conn.WriteJSON(response)
		}
		s.mutex.RUnlock()
	}
}

func (s *Server) handleUnsubscribe(client *Client, msg WebSocketMessage) {
	client.MachineID = ""
}

func (s *Server) handlePing(client *Client, msg WebSocketMessage) {
	pong := WebSocketMessage{
		Type:      "pong",
		Timestamp: time.Now(),
	}
	client.Conn.WriteJSON(pong)
}

// Broadcast functions

func (s *Server) broadcastMachineUpdate(machineID, action string, machine *semantics.MachineWrapper) {
	msg := WebSocketMessage{
		Type:      "machine_updated",
		MachineID: machineID,
		Data: map[string]interface{}{
			"action": action,
		},
		Timestamp: time.Now(),
	}

	if machine != nil {
		msg.Data.(map[string]interface{})["configuration"] = getStateLabels(machine.Configuration.States)
		msg.Data.(map[string]interface{})["context"] = machine.Context.AsMap()
		msg.Data.(map[string]interface{})["state"] = machine.State.String()
	}

	for conn, client := range s.clients {
		if client.MachineID == machineID || client.MachineID == "" {
			if err := conn.WriteJSON(msg); err != nil {
				log.Printf("Failed to send WebSocket message to client %s: %v", client.ID, err)
			}
		}
	}
}

func (s *Server) broadcastEventProcessed(machineID, event string, machine *semantics.MachineWrapper) {
	msg := WebSocketMessage{
		Type:      "event_processed",
		MachineID: machineID,
		Data: map[string]interface{}{
			"event":         event,
			"configuration": getStateLabels(machine.Configuration.States),
			"context":       machine.Context.AsMap(),
			"last_step":     getLastStep(machine.StepHistory),
		},
		Timestamp: time.Now(),
	}

	for conn, client := range s.clients {
		if client.MachineID == machineID || client.MachineID == "" {
			if err := conn.WriteJSON(msg); err != nil {
				log.Printf("Failed to send WebSocket message to client %s: %v", client.ID, err)
			}
		}
	}
}

// Metrics methods

func (m *MetricsCollector) incrementRequestCount() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.requestCount++
}

func (m *MetricsCollector) incrementConnectionCount() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.connectionCount++
}

func (m *MetricsCollector) incrementErrorCount() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.errorCount++
}


func main() {
	server := NewServer()

	// Load persisted machines on startup
	if err := server.persistence.LoadMachines(server.machines); err != nil {
		log.Printf("Failed to load persisted machines: %v", err)
	}

	r := mux.NewRouter()

	// Apply middleware
	r.Use(server.corsMiddleware)
	r.Use(server.loggingMiddleware)

	// API routes
	api := r.PathPrefix("/api/v1").Subrouter()

	// Machine management
	api.HandleFunc("/machines", server.CreateMachine).Methods("POST")
	api.HandleFunc("/machines", server.ListMachines).Methods("GET")
	api.HandleFunc("/machines/search", server.SearchMachines).Methods("POST")
	api.HandleFunc("/machines/{id}", server.GetMachine).Methods("GET")
	api.HandleFunc("/machines/{id}", server.UpdateMachine).Methods("PUT")
	api.HandleFunc("/machines/{id}", server.DeleteMachine).Methods("DELETE")

	// Machine operations
	api.HandleFunc("/machines/{id}/events", server.ProcessEvent).Methods("POST")
	api.HandleFunc("/machines/{id}/events/batch", server.ProcessBatchEvents).Methods("POST")
	api.HandleFunc("/machines/{id}/reset", server.ResetMachine).Methods("POST")
	api.HandleFunc("/machines/{id}/validate", server.ValidateMachine).Methods("GET")

	// Export endpoints
	api.HandleFunc("/export/formats", server.GetSupportedFormats).Methods("GET")
	api.HandleFunc("/machines/{id}/export/{format}", server.ExportMachine).Methods("GET", "POST")
	api.HandleFunc("/machines/{id}/export/json", server.ExportMachineJSON).Methods("GET")
	api.HandleFunc("/machines/{id}/export/xstate", server.ExportMachineXState).Methods("GET")
	api.HandleFunc("/machines/{id}/export/svg", server.ExportMachineSVG).Methods("GET")

	// Examples and templates
	api.HandleFunc("/examples", server.GetExamples).Methods("GET")

	// Server metrics and health
	api.HandleFunc("/metrics", server.GetMetrics).Methods("GET")
	api.HandleFunc("/health", server.GetHealth).Methods("GET")

	// WebSocket for real-time updates
	r.HandleFunc("/ws", server.WebSocketHandler)

	// Static file serving
	r.PathPrefix("/").Handler(server.ServeStaticFiles())

	port := 8080
	if portStr := getEnv("PORT", ""); portStr != "" {
		if p, err := strconv.Atoi(portStr); err == nil {
			port = p
		}
	}

	log.Printf("Starting enhanced statechart server on port %d", port)
	log.Printf("Web interface: http://localhost:%d", port)
	log.Printf("API endpoints: http://localhost:%d/api/v1", port)
	log.Printf("WebSocket endpoint: ws://localhost:%d/ws", port)
	log.Printf("Data directory: %s", server.persistence.dataDir)

	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), r); err != nil {
		log.Fatal(err)
	}
}