package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/xstate"
	"github.com/tmc/sc/semantics/v1"
)

// ExportRequest represents export request parameters
type ExportRequest struct {
	Format  string            `json:"format"`
	Options map[string]interface{} `json:"options,omitempty"`
}

// ExportResponse represents export response
type ExportResponse struct {
	Success      bool                   `json:"success"`
	Format       string                 `json:"format"`
	Content      string                 `json:"content"`
	Filename     string                 `json:"filename"`
	MimeType     string                 `json:"mime_type"`
	Dependencies map[string]string      `json:"dependencies,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	ExportedAt   time.Time             `json:"exported_at"`
	Error        string                 `json:"error,omitempty"`
}

// GetSupportedFormats returns all supported export formats
func (s *Server) GetSupportedFormats(w http.ResponseWriter, r *http.Request) {
	formats := map[string]interface{}{
		"xstate": map[string]interface{}{
			"name":        "XState v5",
			"description": "Modern XState machine definition with TypeScript support",
			"extensions":  []string{"js", "ts"},
			"mime_types":  []string{"application/javascript", "application/typescript"},
			"features": []string{
				"Hierarchical states", "Parallel states", "Guards and actions",
				"TypeScript definitions", "Context management",
			},
		},
		"react": map[string]interface{}{
			"name":        "React Components",
			"description": "React components and hooks for statechart integration",
			"extensions":  []string{"jsx", "tsx"},
			"mime_types":  []string{"text/jsx", "text/tsx"},
			"features": []string{
				"React Hooks", "Context Provider", "TypeScript support", "State Components",
			},
		},
		"vue": map[string]interface{}{
			"name":        "Vue Components",
			"description": "Vue 3 components and composables for statechart integration",
			"extensions":  []string{"vue"},
			"mime_types":  []string{"text/x-vue"},
			"features": []string{
				"Composition API", "Composables", "TypeScript support", "Reactive State",
			},
		},
		"scxml": map[string]interface{}{
			"name":        "SCXML",
			"description": "W3C State Chart XML format for interoperability",
			"extensions":  []string{"scxml"},
			"mime_types":  []string{"application/xml"},
			"features": []string{
				"W3C Standard", "Tool Interoperability", "XML Format",
			},
		},
		"json": map[string]interface{}{
			"name":        "JSON",
			"description": "Clean JSON representation of statechart structure",
			"extensions":  []string{"json"},
			"mime_types":  []string{"application/json"},
			"features": []string{
				"Human Readable", "Language Agnostic", "Easy Parsing",
			},
		},
		"yaml": map[string]interface{}{
			"name":        "YAML",
			"description": "YAML representation of statechart structure",
			"extensions":  []string{"yaml", "yml"},
			"mime_types":  []string{"application/yaml"},
			"features": []string{
				"Human Readable", "Comments Support", "Configuration Format",
			},
		},
		"mermaid": map[string]interface{}{
			"name":        "Mermaid",
			"description": "Mermaid state diagram for documentation",
			"extensions":  []string{"mmd", "mermaid"},
			"mime_types":  []string{"text/plain"},
			"features": []string{
				"Diagram as Code", "GitHub Integration", "Documentation",
			},
		},
		"plantuml": map[string]interface{}{
			"name":        "PlantUML",
			"description": "PlantUML state diagram for documentation",
			"extensions":  []string{"puml", "plantuml"},
			"mime_types":  []string{"text/plain"},
			"features": []string{
				"Diagram Generation", "Documentation", "SVG/PNG Export",
			},
		},
		"graphviz": map[string]interface{}{
			"name":        "Graphviz DOT",
			"description": "Graphviz DOT format for high-quality diagrams",
			"extensions":  []string{"dot"},
			"mime_types":  []string{"text/vnd.graphviz"},
			"features": []string{
				"High Quality Diagrams", "Multiple Output Formats", "Layout Algorithms",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"formats": formats,
		"total":   len(formats),
	})
}

// ExportMachine exports a machine in the specified format
func (s *Server) ExportMachine(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	vars := mux.Vars(r)
	machineID := vars["id"]
	format := vars["format"]

	machine, exists := s.machines[machineID]
	if !exists {
		s.metrics.incrementErrorCount()
		http.Error(w, "Machine not found", http.StatusNotFound)
		return
	}

	// Parse options from query parameters or request body
	options := make(map[string]interface{})
	
	// Try to read options from request body
	if r.ContentLength > 0 {
		var exportReq ExportRequest
		if err := json.NewDecoder(r.Body).Decode(&exportReq); err == nil {
			if exportReq.Format != "" {
				format = exportReq.Format
			}
			if exportReq.Options != nil {
				options = exportReq.Options
			}
		}
	}

	// Add query parameters to options
	for key, values := range r.URL.Query() {
		if len(values) > 0 {
			if key == "pretty" || key == "typescript" || key == "comments" {
				options[key] = values[0] == "true"
			} else {
				options[key] = values[0]
			}
		}
	}

	// Generate export
	response := s.generateExport(machine, format, options)
	
	// Set appropriate headers based on format
	if response.Success {
		// Set content type
		w.Header().Set("Content-Type", response.MimeType)
		
		// Set filename for download
		if r.URL.Query().Get("download") == "true" {
			w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", response.Filename))
		}
		
		// Return content directly for download
		if r.URL.Query().Get("download") == "true" {
			w.Write([]byte(response.Content))
			return
		}
	}

	// Return JSON response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ExportMachineJSON exports a machine as JSON (legacy endpoint)
func (s *Server) ExportMachineJSON(w http.ResponseWriter, r *http.Request) {
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

	export := map[string]interface{}{
		"id":            machine.Id,
		"state":         machine.State.String(),
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"statechart":    machine.Statechart,
		"step_history":  machine.StepHistory,
		"exported_at":   time.Now().Format(time.RFC3339),
		"format":        "json",
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.json\"", machineID))
	json.NewEncoder(w).Encode(export)
}

// ExportMachineXState exports a machine in XState format (legacy endpoint)
func (s *Server) ExportMachineXState(w http.ResponseWriter, r *http.Request) {
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

	xstateConfig := convertToXState(machine.Statechart)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s-xstate.json\"", machineID))
	json.NewEncoder(w).Encode(xstateConfig)
}

// generateExport generates export content for a machine in the specified format
func (s *Server) generateExport(machine *semantics.MachineWrapper, format string, options map[string]interface{}) ExportResponse {
	response := ExportResponse{
		Format:     format,
		ExportedAt: time.Now(),
		Dependencies: make(map[string]string),
		Metadata:    make(map[string]interface{}),
	}

	// Set default options
	if options == nil {
		options = make(map[string]interface{})
	}

	// Add machine name to options if not present
	if _, exists := options["machineName"]; !exists {
		options["machineName"] = machine.Id
	}

	switch format {
	case "xstate", "xstate-js":
		content, err := s.generateXStateExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "application/javascript"
		response.Filename = fmt.Sprintf("%s.js", machine.Id)
		response.Dependencies["xstate"] = "^5.0.0"

	case "xstate-ts", "xstate-typescript":
		options["generateTypeDefinitions"] = true
		content, err := s.generateXStateExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "application/typescript"
		response.Filename = fmt.Sprintf("%s.ts", machine.Id)
		response.Dependencies["xstate"] = "^5.0.0"

	case "react", "react-jsx":
		content, err := s.generateReactExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "text/jsx"
		response.Filename = fmt.Sprintf("%s-components.jsx", machine.Id)
		response.Dependencies["react"] = "^18.0.0"
		response.Dependencies["@xstate/react"] = "^4.0.0"
		response.Dependencies["xstate"] = "^5.0.0"

	case "react-ts", "react-tsx":
		options["generateTypeDefinitions"] = true
		content, err := s.generateReactExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "text/tsx"
		response.Filename = fmt.Sprintf("%s-components.tsx", machine.Id)
		response.Dependencies["react"] = "^18.0.0"
		response.Dependencies["@types/react"] = "^18.0.0"
		response.Dependencies["@xstate/react"] = "^4.0.0"
		response.Dependencies["xstate"] = "^5.0.0"

	case "vue":
		content, err := s.generateVueExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "text/x-vue"
		response.Filename = fmt.Sprintf("%s-components.vue", machine.Id)
		response.Dependencies["vue"] = "^3.0.0"
		response.Dependencies["xstate"] = "^5.0.0"

	case "scxml":
		content, err := s.generateSCXMLExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "application/xml"
		response.Filename = fmt.Sprintf("%s.scxml", machine.Id)

	case "json":
		content, err := s.generateJSONExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "application/json"
		response.Filename = fmt.Sprintf("%s.json", machine.Id)

	case "yaml", "yml":
		content, err := s.generateYAMLExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "application/yaml"
		response.Filename = fmt.Sprintf("%s.yaml", machine.Id)

	case "mermaid":
		content, err := s.generateMermaidExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "text/plain"
		response.Filename = fmt.Sprintf("%s.mmd", machine.Id)

	case "plantuml":
		content, err := s.generatePlantUMLExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "text/plain"
		response.Filename = fmt.Sprintf("%s.puml", machine.Id)

	case "graphviz", "dot":
		content, err := s.generateGraphvizExport(machine, options)
		if err != nil {
			response.Error = err.Error()
			return response
		}
		response.Success = true
		response.Content = content
		response.MimeType = "text/vnd.graphviz"
		response.Filename = fmt.Sprintf("%s.dot", machine.Id)

	default:
		response.Error = fmt.Sprintf("Unsupported format: %s", format)
		return response
	}

	// Add metadata
	response.Metadata["stateCount"] = s.countStates(machine.Statechart.RootState)
	response.Metadata["transitionCount"] = len(machine.Statechart.Transitions)
	response.Metadata["eventCount"] = len(machine.Statechart.Events)
	response.Metadata["machineId"] = machine.Id

	return response
}

// generateXStateExport generates XState machine code
func (s *Server) generateXStateExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	converter := xstate.NewConverter()
	xstateMachine, err := converter.Export(machine.Statechart)
	if err != nil {
		return "", fmt.Errorf("failed to convert to XState: %v", err)
	}

	// Generate JavaScript/TypeScript code
	var code strings.Builder
	
	// Add header
	code.WriteString("/**\n")
	code.WriteString(fmt.Sprintf(" * Generated XState Machine: %s\n", machine.Id))
	code.WriteString(fmt.Sprintf(" * Generated at: %s\n", time.Now().Format(time.RFC3339)))
	code.WriteString(" * Generator: Statechart Web Visualizer\n")
	code.WriteString(" */\n\n")

	// Add imports
	if options["generateTypeDefinitions"] == true {
		code.WriteString("import { createMachine, interpret, assign } from 'xstate';\n")
		code.WriteString("import type { StateMachine, Interpreter } from 'xstate';\n\n")
	} else {
		code.WriteString("import { createMachine, interpret, assign } from 'xstate';\n\n")
	}

	// Add machine definition
	machineJSON, err := json.MarshalIndent(xstateMachine, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to serialize XState machine: %v", err)
	}

	code.WriteString(fmt.Sprintf("const %s = createMachine(%s);\n\n", machine.Id, string(machineJSON)))

	// Add service creation
	code.WriteString(fmt.Sprintf("// Create and export service\n"))
	code.WriteString(fmt.Sprintf("export const %sService = interpret(%s);\n\n", machine.Id, machine.Id))

	// Add exports
	code.WriteString(fmt.Sprintf("export { %s };\n", machine.Id))
	code.WriteString(fmt.Sprintf("export default %s;\n", machine.Id))

	return code.String(), nil
}

// generateReactExport generates React components
func (s *Server) generateReactExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	// This is a simplified implementation
	// In a full implementation, this would use the React generator
	var code strings.Builder
	
	code.WriteString("/**\n")
	code.WriteString(fmt.Sprintf(" * React Components for %s\n", machine.Id))
	code.WriteString(fmt.Sprintf(" * Generated at: %s\n", time.Now().Format(time.RFC3339)))
	code.WriteString(" */\n\n")

	code.WriteString("import React from 'react';\n")
	code.WriteString("import { useMachine } from '@xstate/react';\n")
	code.WriteString(fmt.Sprintf("import { %s } from './%s';\n\n", machine.Id, machine.Id))

	// Generate machine hook
	machineHook := strings.Title(machine.Id)
	code.WriteString(fmt.Sprintf("export function use%s(initialContext = {}) {\n", machineHook))
	code.WriteString(fmt.Sprintf("  const [state, send] = useMachine(%s, {\n", machine.Id))
	code.WriteString("    context: {\n")
	code.WriteString(fmt.Sprintf("      ...%s.context,\n", machine.Id))
	code.WriteString("      ...initialContext\n")
	code.WriteString("    }\n")
	code.WriteString("  });\n\n")
	code.WriteString("  return { state, send };\n")
	code.WriteString("}\n\n")

	// Generate component
	code.WriteString(fmt.Sprintf("export const %sComponent = ({ children, ...props }) => {\n", machineHook))
	code.WriteString(fmt.Sprintf("  const machine = use%s();\n", machineHook))
	code.WriteString("  \n")
	code.WriteString("  return (\n")
	code.WriteString(fmt.Sprintf("    <div className=\"%s-component\" {...props}>\n", strings.ToLower(machine.Id)))
	code.WriteString("      {typeof children === 'function' ? children(machine) : children}\n")
	code.WriteString("    </div>\n")
	code.WriteString("  );\n")
	code.WriteString("};\n")

	return code.String(), nil
}

// generateVueExport generates Vue components
func (s *Server) generateVueExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	// Simplified Vue component generation
	var code strings.Builder
	
	code.WriteString("<template>\n")
	code.WriteString(fmt.Sprintf("  <div class=\"%s-component\">\n", strings.ToLower(machine.Id)))
	code.WriteString("    <slot :machine=\"machine\" :state=\"machine.state\" :send=\"machine.send\" />\n")
	code.WriteString("  </div>\n")
	code.WriteString("</template>\n\n")

	code.WriteString("<script>\n")
	code.WriteString("import { ref, computed, onMounted, onUnmounted } from 'vue';\n")
	code.WriteString("import { interpret } from 'xstate';\n")
	code.WriteString(fmt.Sprintf("import { %s } from './%s';\n\n", machine.Id, machine.Id))

	code.WriteString("export default {\n")
	code.WriteString(fmt.Sprintf("  name: '%sComponent',\n", strings.Title(machine.Id)))
	code.WriteString("  setup(props) {\n")
	code.WriteString("    const service = ref(null);\n")
	code.WriteString("    const state = ref(null);\n\n")
	
	code.WriteString("    onMounted(() => {\n")
	code.WriteString(fmt.Sprintf("      service.value = interpret(%s);\n", machine.Id))
	code.WriteString("      service.value.onTransition((newState) => {\n")
	code.WriteString("        state.value = newState;\n")
	code.WriteString("      });\n")
	code.WriteString("      service.value.start();\n")
	code.WriteString("    });\n\n")
	
	code.WriteString("    onUnmounted(() => {\n")
	code.WriteString("      if (service.value) {\n")
	code.WriteString("        service.value.stop();\n")
	code.WriteString("      }\n")
	code.WriteString("    });\n\n")
	
	code.WriteString("    const send = (event) => {\n")
	code.WriteString("      if (service.value) {\n")
	code.WriteString("        service.value.send(event);\n")
	code.WriteString("      }\n")
	code.WriteString("    };\n\n")
	
	code.WriteString("    return {\n")
	code.WriteString("      machine: {\n")
	code.WriteString("        state: computed(() => state.value),\n")
	code.WriteString("        send\n")
	code.WriteString("      }\n")
	code.WriteString("    };\n")
	code.WriteString("  }\n")
	code.WriteString("};\n")
	code.WriteString("</script>\n")

	return code.String(), nil
}

// Additional export format generators

func (s *Server) generateSCXMLExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	var code strings.Builder
	
	code.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
	code.WriteString("<scxml xmlns=\"http://www.w3.org/2005/07/scxml\" version=\"1.0\" datamodel=\"ecmascript\"")
	code.WriteString(fmt.Sprintf(" name=\"%s\"", machine.Id))
	
	// Find initial state
	initialState := s.findInitialState(machine.Statechart.RootState)
	if initialState != "" {
		code.WriteString(fmt.Sprintf(" initial=\"%s\"", initialState))
	}
	
	code.WriteString(">\n")
	
	// Add states
	code.WriteString(s.generateSCXMLStates(machine.Statechart.RootState, 1))
	
	code.WriteString("</scxml>\n")
	return code.String(), nil
}

func (s *Server) generateJSONExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	export := map[string]interface{}{
		"id":            machine.Id,
		"statechart":    machine.Statechart,
		"state":         machine.State.String(),
		"configuration": getStateLabels(machine.Configuration.States),
		"context":       machine.Context.AsMap(),
		"exported_at":   time.Now().Format(time.RFC3339),
		"format":        "json",
		"version":       "1.0.0",
	}
	
	if options["pretty"] == true {
		jsonBytes, err := json.MarshalIndent(export, "", "  ")
		if err != nil {
			return "", err
		}
		return string(jsonBytes), nil
	}
	
	jsonBytes, err := json.Marshal(export)
	if err != nil {
		return "", err
	}
	return string(jsonBytes), nil
}

func (s *Server) generateYAMLExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	// Simple YAML generation - in a full implementation, use a YAML library
	var code strings.Builder
	
	code.WriteString(fmt.Sprintf("id: %s\n", machine.Id))
	code.WriteString(fmt.Sprintf("exported_at: %s\n", time.Now().Format(time.RFC3339)))
	code.WriteString("statechart:\n")
	code.WriteString(fmt.Sprintf("  name: %s\n", machine.Statechart.Name))
	code.WriteString("  root_state:\n")
	code.WriteString(fmt.Sprintf("    label: %s\n", machine.Statechart.RootState.Label))
	
	return code.String(), nil
}

func (s *Server) generateMermaidExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	var code strings.Builder
	
	code.WriteString("stateDiagram-v2\n")
	code.WriteString(s.generateMermaidStates(machine.Statechart.RootState))
	
	// Add transitions
	for _, transition := range machine.Statechart.Transitions {
		if len(transition.From) > 0 && len(transition.To) > 0 {
			for _, from := range transition.From {
				for _, to := range transition.To {
					code.WriteString(fmt.Sprintf("  %s --> %s", s.sanitizeID(from), s.sanitizeID(to)))
					if transition.Event != "" {
						code.WriteString(fmt.Sprintf(" : %s", transition.Event))
					}
					code.WriteString("\n")
				}
			}
		}
	}
	
	return code.String(), nil
}

func (s *Server) generatePlantUMLExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	var code strings.Builder
	
	code.WriteString("@startuml\n")
	if machine.Id != "" {
		code.WriteString(fmt.Sprintf("title %s\n", machine.Id))
	}
	code.WriteString("\n")
	
	code.WriteString(s.generatePlantUMLStates(machine.Statechart.RootState))
	
	// Add transitions
	for _, transition := range machine.Statechart.Transitions {
		if len(transition.From) > 0 && len(transition.To) > 0 {
			for _, from := range transition.From {
				for _, to := range transition.To {
					code.WriteString(fmt.Sprintf("%s --> %s", s.sanitizeID(from), s.sanitizeID(to)))
					if transition.Event != "" {
						code.WriteString(fmt.Sprintf(" : %s", transition.Event))
					}
					code.WriteString("\n")
				}
			}
		}
	}
	
	code.WriteString("\n@enduml\n")
	return code.String(), nil
}

func (s *Server) generateGraphvizExport(machine *semantics.MachineWrapper, options map[string]interface{}) (string, error) {
	var code strings.Builder
	
	code.WriteString(fmt.Sprintf("digraph %s {\n", s.sanitizeID(machine.Id)))
	code.WriteString("  rankdir=TB;\n")
	code.WriteString("  node [shape=box, style=rounded];\n")
	code.WriteString("  edge [fontsize=10];\n\n")
	
	// Add states
	code.WriteString("  // States\n")
	code.WriteString(s.generateGraphvizStates(machine.Statechart.RootState))
	
	// Add transitions
	code.WriteString("\n  // Transitions\n")
	for _, transition := range machine.Statechart.Transitions {
		if len(transition.From) > 0 && len(transition.To) > 0 {
			for _, from := range transition.From {
				for _, to := range transition.To {
					code.WriteString(fmt.Sprintf("  %s -> %s", s.sanitizeID(from), s.sanitizeID(to)))
					if transition.Event != "" {
						code.WriteString(fmt.Sprintf(" [label=\"%s\"]", transition.Event))
					}
					code.WriteString(";\n")
				}
			}
		}
	}
	
	code.WriteString("}\n")
	return code.String(), nil
}

// Helper functions

func (s *Server) findInitialState(state *sc.State) string {
	if state == nil || state.Children == nil {
		return ""
	}
	
	for _, child := range state.Children {
		if child.IsInitial {
			return child.Label
		}
	}
	
	// If no initial state marked, return first child
	if len(state.Children) > 0 {
		return state.Children[0].Label
	}
	
	return ""
}

func (s *Server) generateSCXMLStates(state *sc.State, level int) string {
	if state == nil || state.Children == nil {
		return ""
	}
	
	var code strings.Builder
	indent := strings.Repeat("  ", level)
	
	for _, child := range state.Children {
		code.WriteString(fmt.Sprintf("%s<state id=\"%s\"", indent, child.Label))
		
		if child.Children != nil && len(child.Children) > 0 {
			initialChild := s.findInitialState(child)
			if initialChild != "" {
				code.WriteString(fmt.Sprintf(" initial=\"%s\"", initialChild))
			}
		}
		
		code.WriteString(">\n")
		
		// Add nested states
		if child.Children != nil && len(child.Children) > 0 {
			code.WriteString(s.generateSCXMLStates(child, level+1))
		}
		
		code.WriteString(fmt.Sprintf("%s</state>\n", indent))
	}
	
	return code.String()
}

func (s *Server) generateMermaidStates(state *sc.State) string {
	if state == nil || state.Children == nil {
		return ""
	}
	
	var code strings.Builder
	
	for _, child := range state.Children {
		code.WriteString(fmt.Sprintf("  %s : %s\n", s.sanitizeID(child.Label), child.Label))
		
		if child.IsInitial {
			code.WriteString(fmt.Sprintf("  [*] --> %s\n", s.sanitizeID(child.Label)))
		}
		
		if child.IsFinal {
			code.WriteString(fmt.Sprintf("  %s --> [*]\n", s.sanitizeID(child.Label)))
		}
	}
	
	return code.String()
}

func (s *Server) generatePlantUMLStates(state *sc.State) string {
	if state == nil || state.Children == nil {
		return ""
	}
	
	var code strings.Builder
	
	for _, child := range state.Children {
		code.WriteString(fmt.Sprintf("state \"%s\" as %s\n", child.Label, s.sanitizeID(child.Label)))
		
		if child.IsInitial {
			code.WriteString(fmt.Sprintf("[*] --> %s\n", s.sanitizeID(child.Label)))
		}
		
		if child.IsFinal {
			code.WriteString(fmt.Sprintf("%s --> [*]\n", s.sanitizeID(child.Label)))
		}
	}
	
	return code.String()
}

func (s *Server) generateGraphvizStates(state *sc.State) string {
	if state == nil || state.Children == nil {
		return ""
	}
	
	var code strings.Builder
	
	for _, child := range state.Children {
		code.WriteString(fmt.Sprintf("  %s [label=\"%s\"", s.sanitizeID(child.Label), child.Label))
		
		if child.IsInitial {
			code.WriteString(", color=green")
		}
		
		if child.IsFinal {
			code.WriteString(", shape=doublecircle")
		}
		
		code.WriteString("];\n")
	}
	
	return code.String()
}

func (s *Server) sanitizeID(id string) string {
	// Replace non-alphanumeric characters with underscores
	result := ""
	for _, char := range id {
		if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
			result += string(char)
		} else {
			result += "_"
		}
	}
	return result
}

func (s *Server) countStates(state *sc.State) int {
	if state == nil {
		return 0
	}
	
	count := 1
	if state.Children != nil {
		for _, child := range state.Children {
			count += s.countStates(child)
		}
	}
	return count
}

// convertToXState converts a statechart to XState format
func convertToXState(statechart *sc.Statechart) map[string]interface{} {
	if statechart == nil || statechart.RootState == nil {
		return map[string]interface{}{}
	}

	config := map[string]interface{}{
		"id":      "converted_machine",
		"initial": findInitialState(statechart.RootState),
		"states":  convertStates(statechart.RootState),
	}

	// Add transitions if any
	if len(statechart.Transitions) > 0 {
		// Group transitions by source state
		transitionsByState := make(map[string][]map[string]interface{})
		for _, transition := range statechart.Transitions {
			for _, from := range transition.From {
				if transition.Event != "" {
					// Use event as the key in XState format
					if _, exists := transitionsByState[from]; !exists {
						transitionsByState[from] = make([]map[string]interface{}, 0)
					}
					
					eventTransition := map[string]interface{}{
						"target": transition.To,
					}
					if transition.Guard != nil && transition.Guard.Expression != "" {
						eventTransition["cond"] = transition.Guard.Expression
					}
					if len(transition.Actions) > 0 {
						actions := make([]string, len(transition.Actions))
						for i, action := range transition.Actions {
							actions[i] = action.Label
						}
						eventTransition["actions"] = actions
					}
					
					transitionsByState[from] = append(transitionsByState[from], map[string]interface{}{
						transition.Event: eventTransition,
					})
				}
			}
		}
	}

	return config
}

// convertStates recursively converts states to XState format
func convertStates(state *sc.State) map[string]interface{} {
	if state == nil {
		return map[string]interface{}{}
	}

	states := make(map[string]interface{})
	
	for _, child := range state.Children {
		childConfig := map[string]interface{}{
			"type": getXStateType(child.Type),
		}
		
		if len(child.Children) > 0 {
			childConfig["states"] = convertStates(child)
			childConfig["initial"] = findInitialState(child)
		}
		
		states[child.Label] = childConfig
	}
	
	return states
}

// getXStateType converts sc.StateType to XState type
func getXStateType(stateType sc.StateType) string {
	switch stateType {
	case sc.StateTypeBasic:
		return "atomic"
	case sc.StateTypeNormal:
		return "compound"
	case sc.StateTypeParallel:
		return "parallel"
	default:
		return "atomic"
	}
}

// findInitialState finds the initial state in a compound state
func findInitialState(state *sc.State) string {
	if state == nil || len(state.Children) == 0 {
		return ""
	}
	
	for _, child := range state.Children {
		if child.IsInitial {
			return child.Label
		}
	}
	
	// If no initial state is marked, use the first child
	return state.Children[0].Label
}

// ExportMachineSVG exports a machine as SVG (placeholder implementation)
func (s *Server) ExportMachineSVG(w http.ResponseWriter, r *http.Request) {
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

	// Generate basic SVG representation
	svg := generateBasicSVG(machine.Statechart)

	w.Header().Set("Content-Type", "image/svg+xml")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.svg\"", machineID))
	w.Write([]byte(svg))
}

// generateBasicSVG creates a basic SVG representation of the statechart
func generateBasicSVG(statechart *sc.Statechart) string {
	if statechart == nil || statechart.RootState == nil {
		return `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
			<text x="10" y="50">Empty Statechart</text>
		</svg>`
	}

	var svg strings.Builder
	svg.WriteString(`<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">`)
	svg.WriteString(`<style>
		.state { fill: #e1f5fe; stroke: #0277bd; stroke-width: 2; }
		.state-text { font-family: Arial; font-size: 12px; text-anchor: middle; }
		.transition { stroke: #666; stroke-width: 1.5; marker-end: url(#arrowhead); }
		.transition-text { font-family: Arial; font-size: 10px; text-anchor: middle; }
	</style>`)
	svg.WriteString(`<defs>
		<marker id="arrowhead" markerWidth="10" markerHeight="7" 
			refX="9" refY="3.5" orient="auto">
			<polygon points="0 0, 10 3.5, 0 7" fill="#666" />
		</marker>
	</defs>`)

	// Draw states
	y := 50
	for _, child := range statechart.RootState.Children {
		svg.WriteString(fmt.Sprintf(`<rect class="state" x="50" y="%d" width="120" height="40" rx="5"/>`, y))
		svg.WriteString(fmt.Sprintf(`<text class="state-text" x="110" y="%d">%s</text>`, y+25, child.Label))
		y += 80
	}

	// Draw transitions (simplified)
	for i, transition := range statechart.Transitions {
		if len(transition.From) > 0 && len(transition.To) > 0 {
			startY := 70 + (i%3)*80
			endY := 70 + ((i+1)%3)*80
			svg.WriteString(fmt.Sprintf(`<line class="transition" x1="170" y1="%d" x2="220" y2="%d"/>`, startY, endY))
			if transition.Event != "" {
				svg.WriteString(fmt.Sprintf(`<text class="transition-text" x="195" y="%d">%s</text>`, (startY+endY)/2-5, transition.Event))
			}
		}
	}

	svg.WriteString(`</svg>`)
	return svg.String()
}