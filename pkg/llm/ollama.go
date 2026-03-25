package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OllamaClient implements the Client interface for local Ollama models
type OllamaClient struct {
	model   string
	baseURL string
	client  *http.Client
}

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(cfg Config) (*OllamaClient, error) {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	model := cfg.Model
	if model == "" {
		model = "llama3.2"
	}

	return &OllamaClient{
		model:   model,
		baseURL: baseURL,
		client:  &http.Client{},
	}, nil
}

func (c *OllamaClient) Name() string {
	return "ollama"
}

func (c *OllamaClient) Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}

	// Build prompt with system prompt if provided
	prompt := req.Prompt
	if req.SystemPrompt != "" {
		prompt = req.SystemPrompt + "\n\n" + req.Prompt
	}

	payload := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": false,
	}

	if req.Temperature > 0 {
		payload["options"] = map[string]interface{}{
			"temperature": req.Temperature,
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Response string `json:"response"`
		Done     bool   `json:"done"`
		Context  []int  `json:"context"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &GenerateResponse{
		Content:      result.Response,
		FinishReason: "stop",
		// Ollama doesn't provide token counts in non-streaming mode
	}, nil
}

func (c *OllamaClient) StreamGenerate(ctx context.Context, req GenerateRequest) (<-chan StreamChunk, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}

	prompt := req.Prompt
	if req.SystemPrompt != "" {
		prompt = req.SystemPrompt + "\n\n" + req.Prompt
	}

	payload := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": true,
	}

	if req.Temperature > 0 {
		payload["options"] = map[string]interface{}{
			"temperature": req.Temperature,
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	chunks := make(chan StreamChunk, 100)

	go func() {
		defer close(chunks)
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			var event struct {
				Response string `json:"response"`
				Done     bool   `json:"done"`
			}

			if err := json.Unmarshal([]byte(line), &event); err != nil {
				continue
			}

			if event.Response != "" {
				chunks <- StreamChunk{Content: event.Response}
			}

			if event.Done {
				chunks <- StreamChunk{Done: true}
				return
			}
		}

		if err := scanner.Err(); err != nil {
			chunks <- StreamChunk{Error: err}
		}
	}()

	return chunks, nil
}
