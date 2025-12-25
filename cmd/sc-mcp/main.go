// Command sc-mcp provides an MCP (Model Context Protocol) server for statechart operations.
//
// It exposes tools, resources, and prompts for working with statecharts,
// allowing LLMs to validate, query, visualize, and simulate statechart definitions.
//
// Usage:
//
//	sc-mcp [flags]
//
// Flags:
//
//	-dir string    Directory to serve statechart files from (enables resources)
//
// The server communicates via stdin/stdout using the MCP protocol.
//
// Tools:
//   - sc_validate: Validate a statechart definition
//   - sc_info: Get summary information about a statechart
//   - sc_states: List all states in a statechart
//   - sc_events: List all events in a statechart
//   - sc_mermaid: Generate a Mermaid diagram
//   - sc_step: Send events and get resulting configuration
//   - sc_transitions: List all transitions
//   - sc_dot: Generate a Graphviz DOT diagram
//
// Resources:
//   - statechart://<path>: Access statechart files from the configured directory
//
// Prompts:
//   - analyze_statechart: Comprehensive analysis of a statechart
//   - design_statechart: Help design a new statechart
//   - debug_transition: Debug why a transition isn't firing
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/encoding/protojson"
)

var (
	chartDir = flag.String("dir", "", "Directory to serve statechart files from")
)

func main() {
	flag.Parse()

	server := mcp.NewServer(&mcp.Implementation{
		Name:    "sc-mcp",
		Version: "1.0.0",
	}, &mcp.ServerOptions{
		Instructions: `Statechart MCP Server - Tools for working with Harel statecharts.

Use the tools to validate, analyze, and visualize statechart definitions.
Statecharts should be provided as JSON following the sc proto format.

Example statechart:
{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "Off", "type": 1, "is_initial": true},
      {"label": "On", "type": 1}
    ]
  },
  "transitions": [
    {"from": ["Off"], "to": ["On"], "event": "TURN_ON"}
  ]
}`,
	})

	// Register tools
	registerTools(server)

	// Register resources if directory specified
	if *chartDir != "" {
		registerResources(server, *chartDir)
	}

	// Register prompts
	registerPrompts(server)

	// Run the server on stdio
	if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}

// Helper to create text result
func textResult(text string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: text},
		},
	}
}

// Helper to create error result
func errorResult(text string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: text},
		},
		IsError: true,
	}
}

// ============================================================================
// Tools
// ============================================================================

type validateInput struct {
	Chart string `json:"chart" jsonschema:"The statechart definition as JSON string"`
}

type infoInput struct {
	Chart string `json:"chart" jsonschema:"The statechart definition as JSON string"`
}

type statesInput struct {
	Chart    string `json:"chart" jsonschema:"The statechart definition as JSON string"`
	LeafOnly bool   `json:"leaf_only,omitempty" jsonschema:"If true only return leaf states"`
	ShowType bool   `json:"show_type,omitempty" jsonschema:"If true include state type"`
}

type eventsInput struct {
	Chart string `json:"chart" jsonschema:"The statechart definition as JSON string"`
}

type mermaidInput struct {
	Chart     string `json:"chart" jsonschema:"The statechart definition as JSON string"`
	Direction string `json:"direction,omitempty" jsonschema:"Diagram direction: TB LR BT RL"`
}

type stepInput struct {
	Chart  string   `json:"chart" jsonschema:"The statechart definition as JSON string"`
	Events []string `json:"events" jsonschema:"Array of event names to send in sequence"`
}

type transitionsInput struct {
	Chart string `json:"chart" jsonschema:"The statechart definition as JSON string"`
}

type dotInput struct {
	Chart   string `json:"chart" jsonschema:"The statechart definition as JSON string"`
	Rankdir string `json:"rankdir,omitempty" jsonschema:"Graph direction: TB LR BT RL"`
}

func registerTools(server *mcp.Server) {
	// sc_validate
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_validate",
		Description: "Validate a statechart definition. Returns VALID or describes validation errors.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input validateInput) (*mcp.CallToolResult, any, error) {
		_, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		wrapper := semantics.NewStatechart(chart)
		if err := wrapper.Validate(); err != nil {
			return textResult(fmt.Sprintf("INVALID: %v", err)), nil, nil
		}

		return textResult("VALID"), nil, nil
	})

	// sc_info
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_info",
		Description: "Get summary information about a statechart including state counts, transitions, and events.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input infoInput) (*mcp.CallToolResult, any, error) {
		machine, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		wrapper := semantics.NewStatechart(chart)

		stateCount, leafCount, parallelCount := countStates(chart.RootState)

		var sb strings.Builder
		fmt.Fprintf(&sb, "Name: %s\n", chart.Name)
		fmt.Fprintf(&sb, "States: %d total, %d leaf, %d parallel\n", stateCount, leafCount, parallelCount)
		fmt.Fprintf(&sb, "Transitions: %d\n", len(chart.Transitions))
		fmt.Fprintf(&sb, "Events: %d\n", len(chart.Events))

		if err := wrapper.Validate(); err != nil {
			fmt.Fprintf(&sb, "Valid: no (%v)\n", err)
		} else {
			fmt.Fprintf(&sb, "Valid: yes\n")
		}

		if machine != nil {
			fmt.Fprintf(&sb, "Machine ID: %s\n", machine.Id)
			fmt.Fprintf(&sb, "State: %s\n", machine.State.String())
			if machine.Configuration != nil {
				var labels []string
				for _, s := range machine.Configuration.States {
					labels = append(labels, s.Label)
				}
				fmt.Fprintf(&sb, "Configuration: %s\n", strings.Join(labels, ", "))
			}
		}

		return textResult(sb.String()), nil, nil
	})

	// sc_states
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_states",
		Description: "List all states in a statechart, one per line.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input statesInput) (*mcp.CallToolResult, any, error) {
		_, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		states := collectStates(chart.RootState, input.LeafOnly, input.ShowType)
		return textResult(strings.Join(states, "\n")), nil, nil
	})

	// sc_events
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_events",
		Description: "List all events defined in a statechart.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input eventsInput) (*mcp.CallToolResult, any, error) {
		_, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		events := collectEvents(chart)
		return textResult(strings.Join(events, "\n")), nil, nil
	})

	// sc_mermaid
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_mermaid",
		Description: "Generate a Mermaid stateDiagram-v2 visualization of the statechart.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input mermaidInput) (*mcp.CallToolResult, any, error) {
		_, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		direction := input.Direction
		if direction == "" {
			direction = "TB"
		}

		mermaid := generateMermaid(chart, direction)
		return textResult(mermaid), nil, nil
	})

	// sc_step
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_step",
		Description: "Send events to a statechart machine and return the resulting configuration.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input stepInput) (*mcp.CallToolResult, any, error) {
		machine, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		result, err := runSteps(machine, chart, input.Events)
		if err != nil {
			return errorResult(err.Error()), nil, nil
		}

		return textResult(result), nil, nil
	})

	// sc_transitions
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_transitions",
		Description: "List all transitions in a statechart with source, target, event, and guard.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input transitionsInput) (*mcp.CallToolResult, any, error) {
		_, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		result := formatTransitions(chart)
		return textResult(result), nil, nil
	})

	// sc_dot
	mcp.AddTool(server, &mcp.Tool{
		Name:        "sc_dot",
		Description: "Generate a Graphviz DOT format visualization of the statechart.",
	}, func(ctx context.Context, req *mcp.CallToolRequest, input dotInput) (*mcp.CallToolResult, any, error) {
		_, chart, err := parseChart(input.Chart)
		if err != nil {
			return errorResult(fmt.Sprintf("Parse error: %v", err)), nil, nil
		}

		rankdir := input.Rankdir
		if rankdir == "" {
			rankdir = "TB"
		}

		dot := generateDot(chart, rankdir)
		return textResult(dot), nil, nil
	})
}

// ============================================================================
// Resources
// ============================================================================

func registerResources(server *mcp.Server, dir string) {
	// Find all JSON files in the directory
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		log.Printf("Warning: could not scan directory %s: %v", dir, err)
		return
	}

	// Resource handler for reading files
	handler := func(ctx context.Context, req *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
		uri := req.Params.URI

		// Parse the URI to get the file path
		var filePath string
		if strings.HasPrefix(uri, "statechart://") {
			filePath = filepath.Join(dir, strings.TrimPrefix(uri, "statechart://"))
		} else if strings.HasPrefix(uri, "file://") {
			filePath = strings.TrimPrefix(uri, "file://")
		} else {
			return nil, fmt.Errorf("unsupported URI scheme: %s", uri)
		}

		// Security: ensure the path is within the allowed directory
		absPath, err := filepath.Abs(filePath)
		if err != nil {
			return nil, fmt.Errorf("invalid path: %v", err)
		}
		absDir, err := filepath.Abs(dir)
		if err != nil {
			return nil, fmt.Errorf("invalid directory: %v", err)
		}
		if !strings.HasPrefix(absPath, absDir) {
			return nil, fmt.Errorf("access denied: path outside allowed directory")
		}

		content, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("read file: %v", err)
		}

		return &mcp.ReadResourceResult{
			Contents: []*mcp.ResourceContents{
				{
					URI:      uri,
					MIMEType: "application/json",
					Text:     string(content),
				},
			},
		}, nil
	}

	// Register each file as a resource
	for _, file := range files {
		name := filepath.Base(file)
		uri := "statechart://" + name

		server.AddResource(&mcp.Resource{
			URI:         uri,
			Name:        name,
			Description: fmt.Sprintf("Statechart definition: %s", name),
			MIMEType:    "application/json",
		}, handler)
	}

	// Register a resource template for dynamic access
	server.AddResourceTemplate(&mcp.ResourceTemplate{
		URITemplate: "statechart://{filename}",
		Name:        "Statechart files",
		Description: "Access statechart JSON files by name",
		MIMEType:    "application/json",
	}, handler)
}

// ============================================================================
// Prompts
// ============================================================================

func registerPrompts(server *mcp.Server) {
	// analyze_statechart - Comprehensive analysis
	server.AddPrompt(&mcp.Prompt{
		Name:        "analyze_statechart",
		Description: "Perform a comprehensive analysis of a statechart including structure, reachability, and potential issues.",
		Arguments: []*mcp.PromptArgument{
			{
				Name:        "chart",
				Description: "The statechart definition as JSON",
				Required:    true,
			},
		},
	}, func(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
		chart := req.Params.Arguments["chart"]

		return &mcp.GetPromptResult{
			Description: "Comprehensive statechart analysis",
			Messages: []*mcp.PromptMessage{
				{
					Role: "user",
					Content: &mcp.TextContent{
						Text: fmt.Sprintf(`Please analyze this statechart comprehensively:

%s

Perform the following analysis:
1. **Validation**: Is the statechart well-formed? Check for:
   - Proper hierarchy (tree structure)
   - Default states in composite states
   - Valid transition targets

2. **Structure Analysis**:
   - Total states, leaf states, composite states, parallel states
   - Hierarchy depth
   - State type distribution

3. **Transition Analysis**:
   - All events used
   - Guard conditions
   - Potential unreachable states
   - Missing transitions (dead ends)

4. **Semantic Issues**:
   - Non-deterministic transitions (same event, no guards)
   - Conflicting guards
   - States with no outgoing transitions (terminal states)

5. **Recommendations**:
   - Potential improvements
   - Missing error handling states
   - Suggestions for better organization

Use the sc_validate, sc_info, sc_states, sc_events, and sc_transitions tools to gather information.`, chart),
					},
				},
			},
		}, nil
	})

	// design_statechart - Help design a new statechart
	server.AddPrompt(&mcp.Prompt{
		Name:        "design_statechart",
		Description: "Interactive help for designing a new statechart from requirements.",
		Arguments: []*mcp.PromptArgument{
			{
				Name:        "requirements",
				Description: "Description of the system or behavior to model",
				Required:    true,
			},
			{
				Name:        "domain",
				Description: "Domain context (e.g., UI, game, workflow, protocol)",
				Required:    false,
			},
		},
	}, func(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
		requirements := req.Params.Arguments["requirements"]
		domain := req.Params.Arguments["domain"]
		if domain == "" {
			domain = "general"
		}

		return &mcp.GetPromptResult{
			Description: "Statechart design assistant",
			Messages: []*mcp.PromptMessage{
				{
					Role: "user",
					Content: &mcp.TextContent{
						Text: fmt.Sprintf(`Help me design a statechart for the following requirements:

**Domain**: %s

**Requirements**:
%s

Please:
1. Identify the main states and their hierarchy
2. Determine if parallel (orthogonal) regions are needed
3. List the events that trigger transitions
4. Identify guard conditions
5. Consider error states and recovery
6. Generate the statechart JSON in the sc format

After generating, use sc_validate to verify and sc_mermaid to visualize.

The output format should be:
{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [...]
  },
  "transitions": [...],
  "events": [...]
}`, domain, requirements),
					},
				},
			},
		}, nil
	})

	// debug_transition - Debug why a transition isn't firing
	server.AddPrompt(&mcp.Prompt{
		Name:        "debug_transition",
		Description: "Debug why a specific transition is not firing as expected.",
		Arguments: []*mcp.PromptArgument{
			{
				Name:        "chart",
				Description: "The statechart definition as JSON",
				Required:    true,
			},
			{
				Name:        "current_state",
				Description: "The current state(s) of the machine",
				Required:    true,
			},
			{
				Name:        "event",
				Description: "The event being sent",
				Required:    true,
			},
			{
				Name:        "expected_state",
				Description: "The expected target state",
				Required:    false,
			},
		},
	}, func(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
		chart := req.Params.Arguments["chart"]
		currentState := req.Params.Arguments["current_state"]
		event := req.Params.Arguments["event"]
		expectedState := req.Params.Arguments["expected_state"]

		expectedPart := ""
		if expectedState != "" {
			expectedPart = fmt.Sprintf("\n**Expected target state**: %s", expectedState)
		}

		return &mcp.GetPromptResult{
			Description: "Transition debugging",
			Messages: []*mcp.PromptMessage{
				{
					Role: "user",
					Content: &mcp.TextContent{
						Text: fmt.Sprintf(`Debug why a transition is not firing:

**Statechart**:
%s

**Current state(s)**: %s
**Event sent**: %s%s

Please investigate:
1. Use sc_transitions to list all transitions
2. Check if there's a transition from the current state with this event
3. If there's a guard, check if it would evaluate to true
4. Check if the current state is actually active (consider hierarchy)
5. Look for conflicting transitions with higher priority
6. Verify the event name is correct (case-sensitive)

Provide a diagnosis and suggested fix.`, chart, currentState, event, expectedPart),
					},
				},
			},
		}, nil
	})

	// explain_statechart - Explain a statechart in plain language
	server.AddPrompt(&mcp.Prompt{
		Name:        "explain_statechart",
		Description: "Generate a plain-language explanation of what a statechart does.",
		Arguments: []*mcp.PromptArgument{
			{
				Name:        "chart",
				Description: "The statechart definition as JSON",
				Required:    true,
			},
			{
				Name:        "audience",
				Description: "Target audience: technical, non-technical, or documentation",
				Required:    false,
			},
		},
	}, func(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
		chart := req.Params.Arguments["chart"]
		audience := req.Params.Arguments["audience"]
		if audience == "" {
			audience = "technical"
		}

		return &mcp.GetPromptResult{
			Description: "Statechart explanation",
			Messages: []*mcp.PromptMessage{
				{
					Role: "user",
					Content: &mcp.TextContent{
						Text: fmt.Sprintf(`Explain this statechart in plain language for a %s audience:

%s

Use sc_info, sc_states, sc_events, and sc_mermaid to understand the structure.

Provide:
1. A high-level summary of what this statechart models
2. Description of each major state and what it represents
3. Explanation of the key transitions and when they occur
4. Any notable patterns (hierarchical states, parallel regions, guards)
5. A Mermaid diagram for visualization`, audience, chart),
					},
				},
			},
		}, nil
	})
}

// ============================================================================
// Helper functions
// ============================================================================

func parseChart(chartJSON string) (*sc.Machine, *sc.Statechart, error) {
	data := []byte(chartJSON)
	opts := protojson.UnmarshalOptions{DiscardUnknown: true}

	// Try Machine format first
	machine := &sc.Machine{}
	if err := opts.Unmarshal(data, machine); err == nil && machine.Statechart != nil {
		return machine, machine.Statechart, nil
	}

	// Try raw Statechart format
	chart := &sc.Statechart{}
	if err := opts.Unmarshal(data, chart); err == nil && chart.RootState != nil {
		return nil, chart, nil
	}

	return nil, nil, fmt.Errorf("invalid JSON: not a Machine or Statechart")
}

func countStates(s *sc.State) (total, leaf, parallel int) {
	if s == nil {
		return 0, 0, 0
	}
	total = 1
	if len(s.Children) == 0 {
		leaf = 1
	}
	if s.Type == sc.StateTypeParallel {
		parallel = 1
	}
	for _, child := range s.Children {
		t, l, p := countStates(child)
		total += t
		leaf += l
		parallel += p
	}
	return
}

func collectStates(s *sc.State, leafOnly, showType bool) []string {
	var states []string
	var collect func(*sc.State)
	collect = func(s *sc.State) {
		if s == nil {
			return
		}
		isLeaf := len(s.Children) == 0
		if !leafOnly || isLeaf {
			if showType {
				states = append(states, fmt.Sprintf("%s\t%s", s.Label, stateTypeString(s.Type)))
			} else {
				states = append(states, s.Label)
			}
		}
		for _, child := range s.Children {
			collect(child)
		}
	}
	collect(s)
	return states
}

func stateTypeString(t sc.StateType) string {
	switch t {
	case sc.StateTypeBasic:
		return "basic"
	case sc.StateTypeOR:
		return "or"
	case sc.StateTypeAND:
		return "parallel"
	default:
		return "unknown"
	}
}

func collectEvents(chart *sc.Statechart) []string {
	seen := make(map[string]bool)
	var events []string

	for _, e := range chart.Events {
		if !seen[e.Label] {
			events = append(events, e.Label)
			seen[e.Label] = true
		}
	}

	for _, t := range chart.Transitions {
		if t.Event != "" && !seen[t.Event] {
			events = append(events, t.Event)
			seen[t.Event] = true
		}
	}

	return events
}

func generateMermaid(chart *sc.Statechart, direction string) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "stateDiagram-v2\n")
	fmt.Fprintf(&sb, "  direction %s\n\n", direction)

	generateMermaidStates(&sb, chart.RootState, "  ")

	for _, t := range chart.Transitions {
		for _, from := range t.From {
			for _, to := range t.To {
				label := t.Event
				if t.Guard != nil && t.Guard.Expression != "" {
					label = fmt.Sprintf("%s [%s]", label, sanitizeLabel(t.Guard.Expression))
				}
				if label != "" {
					fmt.Fprintf(&sb, "  %s --> %s : %s\n", sanitizeID(from), sanitizeID(to), sanitizeLabel(label))
				} else {
					fmt.Fprintf(&sb, "  %s --> %s\n", sanitizeID(from), sanitizeID(to))
				}
			}
		}
	}

	return sb.String()
}

func generateMermaidStates(sb *strings.Builder, s *sc.State, indent string) {
	if s == nil || s.Label == "__root__" {
		for _, child := range s.Children {
			generateMermaidStates(sb, child, indent)
		}
		return
	}

	id := sanitizeID(s.Label)

	if len(s.Children) > 0 {
		fmt.Fprintf(sb, "%sstate %s {\n", indent, id)
		for _, child := range s.Children {
			generateMermaidStates(sb, child, indent+"  ")
		}
		for _, child := range s.Children {
			if child.IsInitial {
				fmt.Fprintf(sb, "%s  [*] --> %s\n", indent, sanitizeID(child.Label))
				break
			}
		}
		fmt.Fprintf(sb, "%s}\n", indent)
	} else if s.IsFinal {
		fmt.Fprintf(sb, "%s%s --> [*]\n", indent, id)
	}
}

func sanitizeID(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, " ", "_"), "-", "_")
}

func sanitizeLabel(s string) string {
	// Mermaid/DOT labels can't contain certain characters
	s = strings.ReplaceAll(s, "&&", " AND ")
	s = strings.ReplaceAll(s, "||", " OR ")
	s = strings.ReplaceAll(s, "&", " and ")
	s = strings.ReplaceAll(s, "|", " or ")
	s = strings.ReplaceAll(s, "<", " lt ")
	s = strings.ReplaceAll(s, ">", " gt ")
	s = strings.ReplaceAll(s, "\"", "'")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")
	s = strings.ReplaceAll(s, "#", "")
	s = strings.ReplaceAll(s, ";", "")
	if len(s) > 80 {
		s = s[:77] + "..."
	}
	return s
}

func runSteps(machine *sc.Machine, chart *sc.Statechart, events []string) (string, error) {
	wrapper := semantics.NewStatechart(chart)

	var mw *semantics.MachineWrapper
	var err error
	if machine != nil && machine.Id != "" {
		mw, err = semantics.NewMachine(wrapper, machine.Id, machine.Context)
	} else {
		mw, err = semantics.NewMachine(wrapper, "mcp-machine", nil)
	}
	if err != nil {
		return "", fmt.Errorf("create machine: %w", err)
	}

	if err := mw.Start(); err != nil {
		return "", fmt.Errorf("start machine: %w", err)
	}

	var sb strings.Builder
	for _, event := range events {
		transitioned, err := mw.Step(event)
		if err != nil {
			fmt.Fprintf(&sb, "Event %s: error - %v\n", event, err)
			continue
		}

		config := mw.GetCurrentConfiguration()
		var labels []string
		for _, s := range config.States {
			labels = append(labels, s.Label)
		}

		if transitioned {
			fmt.Fprintf(&sb, "Event %s: -> %s\n", event, strings.Join(labels, ", "))
		} else {
			fmt.Fprintf(&sb, "Event %s: (no transition)\n", event)
		}
	}

	config := mw.GetCurrentConfiguration()
	var labels []string
	for _, s := range config.States {
		labels = append(labels, s.Label)
	}
	fmt.Fprintf(&sb, "\nFinal configuration: %s", strings.Join(labels, ", "))

	return sb.String(), nil
}

func formatTransitions(chart *sc.Statechart) string {
	type transitionInfo struct {
		Label string   `json:"label,omitempty"`
		From  []string `json:"from"`
		To    []string `json:"to"`
		Event string   `json:"event,omitempty"`
		Guard string   `json:"guard,omitempty"`
	}

	var transitions []transitionInfo
	for _, t := range chart.Transitions {
		ti := transitionInfo{
			Label: t.Label,
			From:  t.From,
			To:    t.To,
			Event: t.Event,
		}
		if t.Guard != nil {
			ti.Guard = t.Guard.Expression
		}
		transitions = append(transitions, ti)
	}

	data, _ := json.MarshalIndent(transitions, "", "  ")
	return string(data)
}

func generateDot(chart *sc.Statechart, rankdir string) string {
	var sb strings.Builder
	fmt.Fprintln(&sb, "digraph statechart {")
	fmt.Fprintf(&sb, "  rankdir=%s;\n", rankdir)
	fmt.Fprintln(&sb, "  node [shape=box, style=rounded];")
	fmt.Fprintln(&sb)

	// Generate nodes
	var genNodes func(*sc.State)
	genNodes = func(s *sc.State) {
		if s == nil {
			return
		}
		id := sanitizeID(s.Label)
		if s.Label != "__root__" {
			attrs := ""
			if s.IsInitial {
				attrs = ", style=\"rounded,bold\""
			}
			if s.Type == sc.StateTypeAND {
				attrs = ", shape=box, style=\"rounded,dashed\""
			}
			fmt.Fprintf(&sb, "  %s [label=\"%s\"%s];\n", id, s.Label, attrs)
		}
		for _, child := range s.Children {
			genNodes(child)
		}
	}
	genNodes(chart.RootState)

	fmt.Fprintln(&sb)

	// Generate edges
	for _, t := range chart.Transitions {
		for _, from := range t.From {
			for _, to := range t.To {
				label := t.Event
				if t.Guard != nil && t.Guard.Expression != "" {
					label = fmt.Sprintf("%s\\n[%s]", label, sanitizeLabel(t.Guard.Expression))
				}
				fmt.Fprintf(&sb, "  %s -> %s [label=\"%s\"];\n", sanitizeID(from), sanitizeID(to), sanitizeLabel(label))
			}
		}
	}

	fmt.Fprintln(&sb, "}")
	return sb.String()
}
