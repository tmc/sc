package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/tmc/sc/pkg/llm"
)

// AIGenerateRequest represents a request to generate a statechart
type AIGenerateRequest struct {
	Description string   `json:"description"`
	Domain      string   `json:"domain,omitempty"`
	Scenarios   []string `json:"scenarios,omitempty"`
}

// AIImproveRequest represents a request to improve an existing statechart
type AIImproveRequest struct {
	Statechart json.RawMessage `json:"statechart"`
}

// AIExplainRequest represents a request to explain a statechart
type AIExplainRequest struct {
	Statechart json.RawMessage `json:"statechart"`
}

// AIStatusResponse represents the AI availability status
type AIStatusResponse struct {
	Available bool   `json:"available"`
	Provider  string `json:"provider,omitempty"`
	Model     string `json:"model,omitempty"`
	Error     string `json:"error,omitempty"`
}

// getLLMClient returns the configured LLM client
func (s *Server) getLLMClient() (llm.Client, error) {
	if s.llmClient != nil {
		return s.llmClient, nil
	}

	client, err := llm.NewClientFromEnv()
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM client: %w", err)
	}

	s.llmClient = client
	return client, nil
}

// GetAIStatus returns the AI service availability status
func (s *Server) GetAIStatus(w http.ResponseWriter, r *http.Request) {
	client, err := s.getLLMClient()
	if err != nil {
		json.NewEncoder(w).Encode(AIStatusResponse{
			Available: false,
			Error:     err.Error(),
		})
		return
	}

	json.NewEncoder(w).Encode(AIStatusResponse{
		Available: true,
		Provider:  client.Name(),
	})
}

// AIGenerate generates a statechart from a natural language description
func (s *Server) AIGenerate(w http.ResponseWriter, r *http.Request) {
	var req AIGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Description == "" {
		http.Error(w, "description is required", http.StatusBadRequest)
		return
	}

	client, err := s.getLLMClient()
	if err != nil {
		http.Error(w, fmt.Sprintf("AI not available: %v", err), http.StatusServiceUnavailable)
		return
	}

	domain := req.Domain
	if domain == "" {
		domain = "general"
	}

	domainHint := DomainHints[domain]
	if domainHint == "" {
		domainHint = DomainHints["general"]
	}

	scenarios := "None specified"
	if len(req.Scenarios) > 0 {
		scenarios = strings.Join(req.Scenarios, "\n- ")
		scenarios = "- " + scenarios
	}

	prompt := fmt.Sprintf(GeneratePromptTemplate, req.Description, domainHint, scenarios)

	// Check if streaming is requested
	if r.Header.Get("Accept") == "text/event-stream" {
		s.streamAIResponse(w, r, client, prompt)
		return
	}

	// Non-streaming response
	ctx := r.Context()
	resp, err := client.Generate(ctx, llm.GenerateRequest{
		Prompt:       prompt,
		SystemPrompt: SystemPrompt,
		Temperature:  0.7,
		MaxTokens:    4096,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Validate the response is valid JSON
	var statechart json.RawMessage
	if err := json.Unmarshal([]byte(resp.Content), &statechart); err != nil {
		http.Error(w, fmt.Sprintf("invalid statechart generated: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(statechart)
}

// AIImprove suggests improvements to an existing statechart
func (s *Server) AIImprove(w http.ResponseWriter, r *http.Request) {
	var req AIImproveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	client, err := s.getLLMClient()
	if err != nil {
		http.Error(w, fmt.Sprintf("AI not available: %v", err), http.StatusServiceUnavailable)
		return
	}

	prompt := fmt.Sprintf(ImprovePromptTemplate, string(req.Statechart))

	if r.Header.Get("Accept") == "text/event-stream" {
		s.streamAIResponse(w, r, client, prompt)
		return
	}

	ctx := r.Context()
	resp, err := client.Generate(ctx, llm.GenerateRequest{
		Prompt:       prompt,
		SystemPrompt: SystemPrompt,
		Temperature:  0.7,
		MaxTokens:    4096,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("improvement failed: %v", err), http.StatusInternalServerError)
		return
	}

	var statechart json.RawMessage
	if err := json.Unmarshal([]byte(resp.Content), &statechart); err != nil {
		http.Error(w, fmt.Sprintf("invalid statechart generated: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(statechart)
}

// AIExplain provides an explanation of a statechart's behavior
func (s *Server) AIExplain(w http.ResponseWriter, r *http.Request) {
	var req AIExplainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	client, err := s.getLLMClient()
	if err != nil {
		http.Error(w, fmt.Sprintf("AI not available: %v", err), http.StatusServiceUnavailable)
		return
	}

	prompt := fmt.Sprintf(ExplainPromptTemplate, string(req.Statechart))

	if r.Header.Get("Accept") == "text/event-stream" {
		s.streamAIResponse(w, r, client, prompt)
		return
	}

	ctx := r.Context()
	resp, err := client.Generate(ctx, llm.GenerateRequest{
		Prompt:       prompt,
		SystemPrompt: "You are an expert at explaining statechart behavior in clear, concise terms.",
		Temperature:  0.7,
		MaxTokens:    2048,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("explanation failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"explanation": resp.Content,
	})
}

// streamAIResponse handles SSE streaming for AI responses
func (s *Server) streamAIResponse(w http.ResponseWriter, r *http.Request, client llm.Client, prompt string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ctx := r.Context()
	chunks, err := client.StreamGenerate(ctx, llm.GenerateRequest{
		Prompt:       prompt,
		SystemPrompt: SystemPrompt,
		Temperature:  0.7,
		MaxTokens:    4096,
	})
	if err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		flusher.Flush()
		return
	}

	for chunk := range chunks {
		if chunk.Error != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", chunk.Error.Error())
			flusher.Flush()
			return
		}

		if chunk.Done {
			fmt.Fprintf(w, "event: done\ndata: \n\n")
			flusher.Flush()
			return
		}

		if chunk.Content != "" {
			// Escape newlines for SSE
			escaped := strings.ReplaceAll(chunk.Content, "\n", "\\n")
			fmt.Fprintf(w, "data: %s\n\n", escaped)
			flusher.Flush()
		}
	}
}

// validateStatechartJSON validates that a string contains valid statechart JSON
func validateStatechartJSON(content string) error {
	var sc struct {
		RootState struct {
			Label    string `json:"label"`
			Type     int    `json:"type"`
			Children []struct {
				Label string `json:"label"`
			} `json:"children"`
		} `json:"root_state"`
	}

	if err := json.Unmarshal([]byte(content), &sc); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if sc.RootState.Label == "" {
		return fmt.Errorf("missing root_state.label")
	}

	return nil
}

// extractJSONFromResponse extracts JSON from a potentially markdown-wrapped response
func extractJSONFromResponse(content string) string {
	// Try to find JSON in code blocks
	if idx := strings.Index(content, "```json"); idx != -1 {
		start := idx + 7
		if end := strings.Index(content[start:], "```"); end != -1 {
			return strings.TrimSpace(content[start : start+end])
		}
	}

	if idx := strings.Index(content, "```"); idx != -1 {
		start := idx + 3
		// Skip optional language identifier
		if nlIdx := strings.Index(content[start:], "\n"); nlIdx != -1 {
			start += nlIdx + 1
		}
		if end := strings.Index(content[start:], "```"); end != -1 {
			return strings.TrimSpace(content[start : start+end])
		}
	}

	// Try to find raw JSON
	if idx := strings.Index(content, "{"); idx != -1 {
		// Find matching closing brace
		depth := 0
		for i := idx; i < len(content); i++ {
			switch content[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return content[idx : i+1]
				}
			}
		}
	}

	return content
}

// streamWithJSONExtraction streams and accumulates content, extracting JSON at the end
func (s *Server) streamWithJSONExtraction(w http.ResponseWriter, r *http.Request, client llm.Client, prompt string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	chunks, err := client.StreamGenerate(ctx, llm.GenerateRequest{
		Prompt:       prompt,
		SystemPrompt: SystemPrompt,
		Temperature:  0.7,
		MaxTokens:    4096,
	})
	if err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		flusher.Flush()
		return
	}

	var accumulated strings.Builder

	for chunk := range chunks {
		if chunk.Error != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", chunk.Error.Error())
			flusher.Flush()
			return
		}

		if chunk.Done {
			// Extract and validate JSON
			content := accumulated.String()
			jsonContent := extractJSONFromResponse(content)

			if err := validateStatechartJSON(jsonContent); err != nil {
				fmt.Fprintf(w, "event: error\ndata: invalid statechart: %s\n\n", err.Error())
			} else {
				fmt.Fprintf(w, "event: complete\ndata: %s\n\n", strings.ReplaceAll(jsonContent, "\n", "\\n"))
			}
			flusher.Flush()
			return
		}

		if chunk.Content != "" {
			accumulated.WriteString(chunk.Content)
			escaped := strings.ReplaceAll(chunk.Content, "\n", "\\n")
			fmt.Fprintf(w, "data: %s\n\n", escaped)
			flusher.Flush()
		}
	}
}
