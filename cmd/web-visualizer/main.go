package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/tmc/sc"
	"github.com/tmc/sc/internal/version"
	"github.com/tmc/sc/pkg/llm"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/structpb"
)

//go:embed web
var embeddedWeb embed.FS

// Server represents the enhanced statechart web server
type Server struct {
	machines    map[string]*semantics.MachineWrapper
	upgrader    websocket.Upgrader
	mutex       sync.RWMutex
	clients     map[*websocket.Conn]*Client
	persistence *PersistenceManager
	searchIndex *SearchIndex
	metrics     *MetricsCollector
	startTime   time.Time
	llmClient   llm.Client
	chartsDir   string
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
	Statechart json.RawMessage        `json:"statechart"`
	Context    map[string]interface{} `json:"context,omitempty"`
	Tags       []string               `json:"tags,omitempty"`
	Metadata   map[string]string      `json:"metadata,omitempty"`
}

type UpdateMachineRequest struct {
	Statechart *sc.Statechart         `json:"statechart,omitempty"`
	Context    map[string]interface{} `json:"context,omitempty"`
	Tags       []string               `json:"tags,omitempty"`
	Metadata   map[string]string      `json:"metadata,omitempty"`
}

type ProcessEventRequest struct {
	Event string                 `json:"event"`
	Data  map[string]interface{} `json:"data,omitempty"`
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

func NewServer(chartsDir string) *Server {
	dataDir := getEnv("DATA_DIR", "./data")
	os.MkdirAll(dataDir, 0755)

	// Ensure charts dir exists if specified
	if chartsDir != "" {
		if err := os.MkdirAll(chartsDir, 0755); err != nil {
			log.Printf("Warning: Failed to create charts directory: %v", err)
		}
	}

	// Initialize search index
	dbPath := filepath.Join(dataDir, "search.db")
	searchIndex, err := NewSearchIndex(dbPath)
	if err != nil {
		log.Printf("Failed to initialize search index: %v", err)
		searchIndex = nil
	}

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
		searchIndex: searchIndex,
		metrics:     &MetricsCollector{},
		chartsDir:   chartsDir,
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
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *Server) GetExamples(w http.ResponseWriter, r *http.Request) {
	examples := GetRealWorldExamples()

	// Marshal using protojson to ensure standard Protobuf JSON mapping (camelCase)
	resp := make(map[string]json.RawMessage)
	marshalOpts := protojson.MarshalOptions{
		EmitUnpopulated: false,
		UseProtoNames:   false, // Use camelCase (default)
	}

	for k, v := range examples {
		b, err := marshalOpts.Marshal(v)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to marshal example %s: %v", k, err), http.StatusInternalServerError)
			return
		}
		resp[k] = json.RawMessage(b)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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

	if len(req.Statechart) == 0 {
		s.metrics.incrementErrorCount()
		http.Error(w, "Statechart is required", http.StatusBadRequest)
		return
	}

	// Unmarshal statechart using protojson
	var scProto sc.Statechart
	if err := protojson.Unmarshal(req.Statechart, &scProto); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid statechart JSON: %v", err), http.StatusBadRequest)
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
	statechart := semantics.NewStatechart(&scProto)

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

	// Index machine for search
	if s.searchIndex != nil {
		if err := s.searchIndex.IndexMachine(req.ID, machine, req.Tags, req.Metadata); err != nil {
			log.Printf("Failed to index machine %s: %v", req.ID, err)
		}
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

func (s *Server) HandleSSESync(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Send initial ping
	fmt.Fprintf(w, "event: ping\ndata: %d\n\n", time.Now().UnixMilli())
	flusher.Flush()

	// Create watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Printf("Failed to create fsnotify watcher: %v", err)
		http.Error(w, "Watcher init failed", http.StatusInternalServerError)
		return
	}
	defer watcher.Close()

	// Paths to watch
	watchPaths := []string{}

	// Exact charts dir
	if s.chartsDir != "" {
		if info, err := os.Stat(s.chartsDir); err == nil && info.IsDir() {
			watchPaths = append(watchPaths, s.chartsDir)
			// Send initial chart list or ready signal?
			// For now just watch
		}
	}

	// Find testdata dir (legacy support)
	legacyPaths := []string{
		"testdata",
		"../testdata",
		"../../testdata",
		"semantics/v1/testdata",
	}
	for _, p := range legacyPaths {
		if info, err := os.Stat(p); err == nil && info.IsDir() {
			watchPaths = append(watchPaths, p)
			break
		}
	}

	if len(watchPaths) == 0 {
		fmt.Fprintf(w, "event: error\ndata: no directories found to watch\n\n")
		flusher.Flush()
		return
	}

	// Add paths to watcher
	for _, p := range watchPaths {
		if err := watcher.Add(p); err != nil {
			log.Printf("Failed to watch %s: %v", p, err)
		} else {
			log.Printf("SSE Warning: Watching %s", p)
		}
	}

	notify := r.Context().Done()

	// Event loop
	for {
		select {
		case <-notify:
			return
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			// Dedup/debounce could go here, but raw stream is fine for now
			if event.Op&fsnotify.Write == fsnotify.Write || event.Op&fsnotify.Create == fsnotify.Create {
				filename := filepath.Base(event.Name)
				// Determine type based on extension
				if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".sc.json") {
					// Is it a chart or a testdata file?
					// We just send "file_change" generic event or "chart_update"
					// Legacy visualizer expects "file_change" with specific format
					fmt.Fprintf(w, "event: file_change\ndata: {\"file\": \"%s\", \"action\": \"update\", \"path\": \"%s\"}\n\n", filename, event.Name)
					flusher.Flush()
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			log.Printf("fsnotify error: %v", err)
		case <-time.After(30 * time.Second):
			// Keep-alive ping
			fmt.Fprintf(w, "event: ping\ndata: %d\n\n", time.Now().UnixMilli())
			flusher.Flush()
		}
	}
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

	// Re-index machine for search
	if s.searchIndex != nil {
		if err := s.searchIndex.IndexMachine(machineID, machine, req.Tags, req.Metadata); err != nil {
			log.Printf("Failed to re-index machine %s: %v", machineID, err)
		}
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

	// Remove from search index
	if s.searchIndex != nil {
		if err := s.searchIndex.DeleteMachine(machineID); err != nil {
			log.Printf("Failed to remove machine %s from index: %v", machineID, err)
		}
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

// PatchStatechartRequest represents a partial update operation
type PatchStatechartRequest struct {
	Operation string                 `json:"operation"` // add_state, remove_state, update_state, add_transition, remove_transition, update_transition
	Payload   map[string]interface{} `json:"payload"`
}

// PatchStatechart applies partial updates to a machine's statechart
func (s *Server) PatchStatechart(w http.ResponseWriter, r *http.Request) {
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

	var req PatchStatechartRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Apply the operation
	var operationErr error
	switch req.Operation {
	case "add_state":
		operationErr = s.applyAddState(machine, req.Payload)
	case "remove_state":
		operationErr = s.applyRemoveState(machine, req.Payload)
	case "update_state":
		operationErr = s.applyUpdateState(machine, req.Payload)
	case "add_transition":
		operationErr = s.applyAddTransition(machine, req.Payload)
	case "remove_transition":
		operationErr = s.applyRemoveTransition(machine, req.Payload)
	case "update_transition":
		operationErr = s.applyUpdateTransition(machine, req.Payload)
	default:
		http.Error(w, fmt.Sprintf("Unknown operation: %s", req.Operation), http.StatusBadRequest)
		return
	}

	if operationErr != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Operation failed: %v", operationErr), http.StatusBadRequest)
		return
	}

	// Validate the updated statechart
	statechart := semantics.NewStatechart(machine.Statechart)
	if err := statechart.ValidateAdvanced(); err != nil {
		// Validation failed but we've already applied the change
		// Return success but include warnings
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":            machine.Id,
			"configuration": getStateLabels(machine.Configuration.States),
			"valid":         false,
			"warnings":      []string{err.Error()},
			"updated_at":    time.Now(),
		})
		return
	}

	// Persist updated machine
	if err := s.persistence.SaveMachine(machineID, machine); err != nil {
		log.Printf("Failed to persist patched machine %s: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(machineID, "patched", machine)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"valid":         true,
		"updated_at":    time.Now(),
	})
}

// Helper functions for patch operations
func (s *Server) applyAddState(machine *semantics.MachineWrapper, payload map[string]interface{}) error {
	label, _ := payload["label"].(string)
	parentLabel, _ := payload["parent"].(string)
	if label == "" {
		return fmt.Errorf("state label is required")
	}
	if parentLabel == "" {
		parentLabel = "__root__"
	}

	// Find parent state
	parent := findState(machine.Statechart.RootState, parentLabel)
	if parent == nil {
		return fmt.Errorf("parent state %s not found", parentLabel)
	}

	// Create new state
	stateType := 1 // Basic by default
	if t, ok := payload["type"].(float64); ok {
		stateType = int(t)
	}

	newState := &sc.State{
		Label:     label,
		Type:      sc.StateType(stateType),
		IsInitial: getBoolFromPayload(payload, "is_initial"),
		IsFinal:   getBoolFromPayload(payload, "is_final"),
	}

	parent.Children = append(parent.Children, newState)
	return nil
}

func (s *Server) applyRemoveState(machine *semantics.MachineWrapper, payload map[string]interface{}) error {
	label, _ := payload["label"].(string)
	if label == "" || label == "__root__" {
		return fmt.Errorf("invalid state label")
	}

	// Find and remove state from parent
	removed := removeState(machine.Statechart.RootState, label)
	if !removed {
		return fmt.Errorf("state %s not found", label)
	}

	// Also remove any transitions involving this state
	var filteredTransitions []*sc.Transition
	for _, t := range machine.Statechart.Transitions {
		involves := false
		for _, f := range t.From {
			if f == label {
				involves = true
				break
			}
		}
		for _, to := range t.To {
			if to == label {
				involves = true
				break
			}
		}
		if !involves {
			filteredTransitions = append(filteredTransitions, t)
		}
	}
	machine.Statechart.Transitions = filteredTransitions

	return nil
}

func (s *Server) applyUpdateState(machine *semantics.MachineWrapper, payload map[string]interface{}) error {
	label, _ := payload["label"].(string)
	if label == "" {
		return fmt.Errorf("state label is required")
	}

	state := findState(machine.Statechart.RootState, label)
	if state == nil {
		return fmt.Errorf("state %s not found", label)
	}

	// Apply updates
	if newLabel, ok := payload["new_label"].(string); ok && newLabel != "" {
		state.Label = newLabel
	}
	if t, ok := payload["type"].(float64); ok {
		state.Type = sc.StateType(int(t))
	}
	if isInitial, ok := payload["is_initial"].(bool); ok {
		state.IsInitial = isInitial
	}
	if isFinal, ok := payload["is_final"].(bool); ok {
		state.IsFinal = isFinal
	}

	return nil
}

func (s *Server) applyAddTransition(machine *semantics.MachineWrapper, payload map[string]interface{}) error {
	from, _ := payload["from"].([]interface{})
	to, _ := payload["to"].([]interface{})
	event, _ := payload["event"].(string)

	if len(from) == 0 || len(to) == 0 {
		return fmt.Errorf("from and to are required")
	}

	// Convert to string slices
	fromLabels := make([]string, len(from))
	for i, f := range from {
		fromLabels[i], _ = f.(string)
	}
	toLabels := make([]string, len(to))
	for i, t := range to {
		toLabels[i], _ = t.(string)
	}

	newTransition := &sc.Transition{
		Label: payload["label"].(string),
		From:  fromLabels,
		To:    toLabels,
		Event: event,
	}

	// Add guard if provided
	if guardExpr, ok := payload["guard"].(string); ok && guardExpr != "" {
		newTransition.Guard = &sc.Guard{Expression: guardExpr}
	}

	machine.Statechart.Transitions = append(machine.Statechart.Transitions, newTransition)
	return nil
}

func (s *Server) applyRemoveTransition(machine *semantics.MachineWrapper, payload map[string]interface{}) error {
	label, _ := payload["label"].(string)
	if label == "" {
		return fmt.Errorf("transition label is required")
	}

	var filteredTransitions []*sc.Transition
	found := false
	for _, t := range machine.Statechart.Transitions {
		if t.Label == label {
			found = true
			continue
		}
		filteredTransitions = append(filteredTransitions, t)
	}

	if !found {
		return fmt.Errorf("transition %s not found", label)
	}

	machine.Statechart.Transitions = filteredTransitions
	return nil
}

func (s *Server) applyUpdateTransition(machine *semantics.MachineWrapper, payload map[string]interface{}) error {
	label, _ := payload["label"].(string)
	if label == "" {
		return fmt.Errorf("transition label is required")
	}

	var transition *sc.Transition
	for _, t := range machine.Statechart.Transitions {
		if t.Label == label {
			transition = t
			break
		}
	}

	if transition == nil {
		return fmt.Errorf("transition %s not found", label)
	}

	// Apply updates
	if event, ok := payload["event"].(string); ok {
		transition.Event = event
	}
	if newLabel, ok := payload["new_label"].(string); ok && newLabel != "" {
		transition.Label = newLabel
	}
	if guardExpr, ok := payload["guard"].(string); ok {
		if guardExpr == "" {
			transition.Guard = nil
		} else {
			transition.Guard = &sc.Guard{Expression: guardExpr}
		}
	}

	return nil
}

// GetEnabledTransitions returns transitions that would fire for available events
func (s *Server) GetEnabledTransitions(w http.ResponseWriter, r *http.Request) {
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

	// Get all unique events from transitions
	eventSet := make(map[string]bool)
	for _, t := range machine.Statechart.Transitions {
		if t.Event != "" {
			eventSet[t.Event] = true
		}
	}

	// Check which transitions are enabled
	enabledTransitions := make([]map[string]interface{}, 0)
	configuration := getStateLabels(machine.Configuration.States)

	for _, t := range machine.Statechart.Transitions {
		// Check if any source state is in the current configuration
		enabled := false
		for _, from := range t.From {
			for _, active := range configuration {
				if from == active {
					enabled = true
					break
				}
			}
			if enabled {
				break
			}
		}

		if enabled {
			transitionInfo := map[string]interface{}{
				"label":   t.Label,
				"event":   t.Event,
				"from":    t.From,
				"to":      t.To,
				"enabled": true,
			}
			if t.Guard != nil {
				transitionInfo["guard"] = t.Guard.Expression
			}
			enabledTransitions = append(enabledTransitions, transitionInfo)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"transitions":   enabledTransitions,
		"configuration": configuration,
		"context":       machine.Context.AsMap(),
	})
}

// GetStepHistory returns the step at a specific index
func (s *Server) GetStepHistory(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	vars := mux.Vars(r)
	machineID := vars["id"]
	indexStr := vars["index"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	index, err := strconv.Atoi(indexStr)
	if err != nil {
		http.Error(w, "Invalid index", http.StatusBadRequest)
		return
	}

	if index < 0 || index >= len(machine.StepHistory) {
		http.Error(w, "Index out of range", http.StatusBadRequest)
		return
	}

	step := machine.StepHistory[index]

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"step":  step,
		"index": index,
		"total": len(machine.StepHistory),
	})
}

// RestoreToStep restores the machine to the state at a specific step index
func (s *Server) RestoreToStep(w http.ResponseWriter, r *http.Request) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	vars := mux.Vars(r)
	machineID := vars["id"]
	indexStr := vars["index"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	index, err := strconv.Atoi(indexStr)
	if err != nil {
		http.Error(w, "Invalid index", http.StatusBadRequest)
		return
	}

	if index < 0 || index >= len(machine.StepHistory) {
		http.Error(w, "Index out of range", http.StatusBadRequest)
		return
	}

	// Get the configuration from the step's resulting_configuration
	step := machine.StepHistory[index]
	if step.ResultingConfiguration != nil {
		machine.Configuration = step.ResultingConfiguration
	}

	// Truncate history to this point
	machine.StepHistory = machine.StepHistory[:index+1]

	// Persist updated machine
	if err := s.persistence.SaveMachine(machineID, machine); err != nil {
		log.Printf("Failed to persist restored machine %s: %v", machineID, err)
	}

	// Notify WebSocket clients
	s.broadcastMachineUpdate(machineID, "restored", machine)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"step_index":    index,
		"restored_at":   time.Now(),
	})
}

// Helper functions
func findState(state *sc.State, label string) *sc.State {
	if state == nil {
		return nil
	}
	if state.Label == label {
		return state
	}
	for _, child := range state.Children {
		if found := findState(child, label); found != nil {
			return found
		}
	}
	return nil
}

func removeState(parent *sc.State, label string) bool {
	if parent == nil {
		return false
	}
	for i, child := range parent.Children {
		if child.Label == label {
			parent.Children = append(parent.Children[:i], parent.Children[i+1:]...)
			return true
		}
		if removeState(child, label) {
			return true
		}
	}
	return false
}

func getBoolFromPayload(payload map[string]interface{}, key string) bool {
	if v, ok := payload[key].(bool); ok {
		return v
	}
	return false
}

func (s *Server) SearchMachines(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

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

	// Use FTS5 index if available and query is provided
	if s.searchIndex != nil && req.Query != "" {
		results, err := s.searchIndex.Search(req.Query, req.Limit, req.Offset)
		if err != nil {
			log.Printf("Search index error, falling back to linear scan: %v", err)
		} else {
			// Build response from search results
			machines := make([]map[string]interface{}, 0, len(results))
			for _, result := range results {
				machines = append(machines, map[string]interface{}{
					"id":            result.ID,
					"state":         result.State,
					"configuration": result.Configuration,
					"rank":          result.Rank,
					"snippet":       result.Snippet,
				})
			}

			total, _ := s.searchIndex.Count()

			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Search-Time-Ms", fmt.Sprintf("%.2f", float64(time.Since(start).Microseconds())/1000.0))
			json.NewEncoder(w).Encode(map[string]interface{}{
				"machines":   machines,
				"total":      total,
				"offset":     req.Offset,
				"limit":      req.Limit,
				"query_time": time.Since(start).String(),
			})
			return
		}
	}

	// Fallback to linear scan for backward compatibility
	s.mutex.RLock()
	defer s.mutex.RUnlock()

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
	w.Header().Set("X-Search-Time-Ms", fmt.Sprintf("%.2f", float64(time.Since(start).Microseconds())/1000.0))
	json.NewEncoder(w).Encode(map[string]interface{}{
		"machines":   machines,
		"total":      len(s.machines),
		"offset":     req.Offset,
		"limit":      req.Limit,
		"query_time": time.Since(start).String(),
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

// Chart API Handlers

func (s *Server) GetCharts(w http.ResponseWriter, r *http.Request) {
	if s.chartsDir == "" {
		http.Error(w, "Charts directory not configured", http.StatusNotImplemented)
		return
	}

	files, err := os.ReadDir(s.chartsDir)
	if err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to read charts directory: %v", err), http.StatusInternalServerError)
		return
	}

	var charts []map[string]interface{}
	for _, file := range files {
		if !file.IsDir() && (strings.HasSuffix(file.Name(), ".json") || strings.HasSuffix(file.Name(), ".sc.json")) {
			info, _ := file.Info()
			charts = append(charts, map[string]interface{}{
				"filename":      file.Name(),
				"last_modified": info.ModTime(),
				"size":          info.Size(),
			})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"charts": charts,
	})
}

func (s *Server) GetChartContent(w http.ResponseWriter, r *http.Request) {
	if s.chartsDir == "" {
		http.Error(w, "Charts directory not configured", http.StatusNotImplemented)
		return
	}

	vars := mux.Vars(r)
	filename := vars["filename"]

	// Basic security check to prevent directory traversal
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	path := filepath.Join(s.chartsDir, filename)
	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			http.Error(w, "Chart not found", http.StatusNotFound)
		} else {
			s.metrics.incrementErrorCount()
			http.Error(w, fmt.Sprintf("Failed to read chart: %v", err), http.StatusInternalServerError)
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(content)
}

func (s *Server) UpdateChartContent(w http.ResponseWriter, r *http.Request) {
	if s.chartsDir == "" {
		http.Error(w, "Charts directory not configured", http.StatusNotImplemented)
		return
	}

	vars := mux.Vars(r)
	filename := vars["filename"]

	// Basic security check
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	// Read body
	var content json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&content); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Create file
	path := filepath.Join(s.chartsDir, filename)

	// Create temp file for atomic write
	tmpFile, err := os.CreateTemp(s.chartsDir, "chart-*.tmp")
	if err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to create temp file: %v", err), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmpFile.Name()) // Cleanup on failure

	// Write formatted JSON
	encoder := json.NewEncoder(tmpFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(content); err != nil {
		tmpFile.Close()
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to write to file: %v", err), http.StatusInternalServerError)
		return
	}
	tmpFile.Close()

	// Rename temp file to target
	if err := os.Rename(tmpFile.Name(), path); err != nil {
		s.metrics.incrementErrorCount()
		http.Error(w, fmt.Sprintf("Failed to save chart: %v", err), http.StatusInternalServerError)
		return
	}

	// Notify WebSocket clients (optional, can also rely on file watcher)
	// s.broadcastChartUpdate(filename)

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "saved", "filename": filename})
}

func (s *Server) ServeStaticFiles() http.Handler {
	// Check if ./web/dist exists on disk (dev mode override)
	if _, err := os.Stat("./web/dist"); err == nil {
		log.Println("Serving static files from disk: ./web/dist")
		return http.FileServer(http.Dir("./web/dist"))
	}

	// Fallback to embedded filesystem
	log.Println("Serving static files from embedded FS")
	fsys, err := fs.Sub(embeddedWeb, "web")
	if err != nil {
		log.Fatalf("Failed to create sub filesystem: %v", err)
	}
	return http.FileServer(http.FS(fsys))
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
	versionFlag := flag.Bool("version", false, "Print version information")
	chartsDir := flag.String("charts-dir", "", "Directory containing chart files for bidi editing")
	flag.Parse()

	if *versionFlag {
		fmt.Println(version.Info())
		os.Exit(0)
	}

	server := NewServer(*chartsDir)

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
	api.HandleFunc("/sync", server.HandleSSESync).Methods("GET", "OPTIONS")

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

	// Editor operations
	api.HandleFunc("/machines/{id}/statechart", server.PatchStatechart).Methods("PATCH")

	// Simulation operations
	api.HandleFunc("/machines/{id}/enabled-transitions", server.GetEnabledTransitions).Methods("GET")
	api.HandleFunc("/machines/{id}/history/{index}", server.GetStepHistory).Methods("GET")
	api.HandleFunc("/machines/{id}/restore/{index}", server.RestoreToStep).Methods("POST")

	// Export endpoints
	api.HandleFunc("/export/formats", server.GetSupportedFormats).Methods("GET")
	api.HandleFunc("/machines/{id}/export/{format}", server.ExportMachine).Methods("GET", "POST")
	api.HandleFunc("/machines/{id}/export/json", server.ExportMachineJSON).Methods("GET")
	api.HandleFunc("/machines/{id}/export/xstate", server.ExportMachineXState).Methods("GET")
	api.HandleFunc("/machines/{id}/export/svg", server.ExportMachineSVG).Methods("GET")

	// Examples and templates
	api.HandleFunc("/examples", server.GetExamples).Methods("GET")

	// AI generation endpoints
	api.HandleFunc("/ai/status", server.GetAIStatus).Methods("GET")
	api.HandleFunc("/ai/generate", server.AIGenerate).Methods("POST")
	api.HandleFunc("/ai/improve", server.AIImprove).Methods("POST")
	api.HandleFunc("/ai/improve", server.AIImprove).Methods("POST")
	api.HandleFunc("/ai/explain", server.AIExplain).Methods("POST")

	// Chart file operations (Bidi Editing)
	api.HandleFunc("/charts", server.GetCharts).Methods("GET")
	api.HandleFunc("/charts/{filename}", server.GetChartContent).Methods("GET")
	api.HandleFunc("/charts/{filename}", server.UpdateChartContent).Methods("PUT")

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
