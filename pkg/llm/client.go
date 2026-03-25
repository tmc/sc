// Package llm provides a multi-provider LLM client abstraction for AI-assisted
// statechart generation.
package llm

import (
	"context"
	"fmt"
	"os"
)

// Client is the interface for LLM providers
type Client interface {
	// Generate sends a prompt to the LLM and returns the response
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)

	// StreamGenerate sends a prompt and streams the response
	StreamGenerate(ctx context.Context, req GenerateRequest) (<-chan StreamChunk, error)

	// Name returns the provider name
	Name() string
}

// GenerateRequest represents a request to the LLM
type GenerateRequest struct {
	Prompt       string  `json:"prompt"`
	SystemPrompt string  `json:"system_prompt,omitempty"`
	Temperature  float64 `json:"temperature,omitempty"`
	MaxTokens    int     `json:"max_tokens,omitempty"`
	Model        string  `json:"model,omitempty"`
}

// GenerateResponse represents the LLM response
type GenerateResponse struct {
	Content      string `json:"content"`
	FinishReason string `json:"finish_reason,omitempty"`
	Usage        *Usage `json:"usage,omitempty"`
}

// Usage tracks token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunk represents a chunk of streaming response
type StreamChunk struct {
	Content string `json:"content"`
	Done    bool   `json:"done"`
	Error   error  `json:"error,omitempty"`
}

// Config holds LLM client configuration
type Config struct {
	Provider string // "openai", "anthropic", "ollama"
	APIKey   string
	Model    string
	BaseURL  string // Optional override for API endpoint
}

// NewClient creates a new LLM client based on configuration
func NewClient(cfg Config) (Client, error) {
	switch cfg.Provider {
	case "openai":
		return NewOpenAIClient(cfg)
	case "anthropic":
		return NewAnthropicClient(cfg)
	case "ollama":
		return NewOllamaClient(cfg)
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}
}

// NewClientFromEnv creates a client from environment variables
func NewClientFromEnv() (Client, error) {
	cfg := Config{
		Provider: getEnv("SC_LLM_PROVIDER", "openai"),
		APIKey:   getEnv("SC_LLM_API_KEY", ""),
		Model:    getEnv("SC_LLM_MODEL", ""),
		BaseURL:  getEnv("SC_LLM_BASE_URL", ""),
	}

	return NewClient(cfg)
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
