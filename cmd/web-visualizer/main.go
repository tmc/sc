package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/semantics/v1/examples"
	"google.golang.org/protobuf/types/known/structpb"
)

type Server struct {
	machines map[string]*semantics.MachineWrapper
	upgrader websocket.Upgrader
}

type CreateMachineRequest struct {
	ID         string                 `json:"id"`
	Statechart *sc.Statechart        `json:"statechart"`
	Context    map[string]interface{} `json:"context,omitempty"`
}

type ProcessEventRequest struct {
	Event string `json:"event"`
}

type SimulationStep struct {
	MachineID      string         `json:"machine_id"`
	Event          string         `json:"event"`
	Configuration  []string       `json:"configuration"`
	Transitions    []*sc.Transition `json:"transitions"`
	Context        map[string]interface{} `json:"context"`
	Timestamp      time.Time      `json:"timestamp"`
}

func NewServer() *Server {
	return &Server{
		machines: make(map[string]*semantics.MachineWrapper),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for development
			},
		},
	}
}

func (s *Server) GetExamples(w http.ResponseWriter, r *http.Request) {
	examples := map[string]*sc.Statechart{
		"hierarchical": examples.HierarchicalStatechart().Statechart,
		"orthogonal":   examples.OrthogonalStatechart().Statechart,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(examples)
}

func (s *Server) CreateMachine(w http.ResponseWriter, r *http.Request) {
	var req CreateMachineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.ID == "" {
		http.Error(w, "Machine ID is required", http.StatusBadRequest)
		return
	}

	if req.Statechart == nil {
		http.Error(w, "Statechart is required", http.StatusBadRequest)
		return
	}

	// Create context from request
	var context *structpb.Struct
	if req.Context != nil {
		var err error
		context, err = structpb.NewStruct(req.Context)
		if err != nil {
			http.Error(w, fmt.Sprintf("Invalid context: %v", err), http.StatusBadRequest)
			return
		}
	}

	// Create statechart wrapper
	statechart := semantics.NewStatechart(req.Statechart)
	
	// Create machine
	machine, err := semantics.NewMachine(statechart, req.ID, context)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create machine: %v", err), http.StatusBadRequest)
		return
	}

	// Start the machine
	if err := machine.Start(); err != nil {
		http.Error(w, fmt.Sprintf("Failed to start machine: %v", err), http.StatusInternalServerError)
		return
	}

	s.machines[req.ID] = machine

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
	})
}

func (s *Server) GetMachine(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
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
	})
}

func (s *Server) ProcessEvent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	var req ProcessEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Process the event
	transitioned, err := machine.Step(req.Event)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to process event: %v", err), http.StatusInternalServerError)
		return
	}
	
	_ = transitioned // We could use this to indicate if any transitions occurred

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"last_step":     getLastStep(machine.StepHistory),
	})
}

func (s *Server) ResetMachine(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	machineID := vars["id"]

	machine, exists := s.machines[machineID]
	if !exists {
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	if err := machine.Reset(); err != nil {
		http.Error(w, fmt.Sprintf("Failed to reset machine: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":            machine.Id,
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
	})
}

func (s *Server) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Handle WebSocket messages for real-time simulation updates
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Echo message back for now (can be extended for real-time simulation)
		if err := conn.WriteJSON(msg); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

func (s *Server) ServeStaticFiles() http.Handler {
	return http.FileServer(http.Dir("./web/"))
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

func main() {
	server := NewServer()

	r := mux.NewRouter()

	// API routes
	api := r.PathPrefix("/api").Subrouter()
	api.HandleFunc("/examples", server.GetExamples).Methods("GET")
	api.HandleFunc("/machines", server.CreateMachine).Methods("POST")
	api.HandleFunc("/machines/{id}", server.GetMachine).Methods("GET")
	api.HandleFunc("/machines/{id}/events", server.ProcessEvent).Methods("POST")
	api.HandleFunc("/machines/{id}/reset", server.ResetMachine).Methods("POST")
	
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

	log.Printf("Starting server on port %d", port)
	log.Printf("Web interface: http://localhost:%d", port)
	log.Printf("API endpoints: http://localhost:%d/api", port)
	
	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), r); err != nil {
		log.Fatal(err)
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}