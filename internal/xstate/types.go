// Package xstate provides import/export functionality for XState/Stately machine JSON format.
//
// This package enables round-trip conversion between XState JSON and the sc proto format,
// preserving all XState-specific metadata (visual layout, invoke, assets) in typed extensions.
package xstate

import (
	"encoding/json"
)

// Machine represents a complete XState/Stately machine from the editor.
type Machine struct {
	ID               string     `json:"id"`
	Name             string     `json:"name"`
	Definition       Definition `json:"definition"`
	ProjectVersionID string     `json:"projectVersionId,omitempty"`
	ForkParentID     string     `json:"forkParentId,omitempty"`
	LastEditedByID   string     `json:"lastEditedById,omitempty"`
	OriginalCode     string     `json:"originalCode,omitempty"`
	CreatedAt        string     `json:"createdAt,omitempty"`
	UpdatedAt        string     `json:"updatedAt,omitempty"`
	FileID           string     `json:"fileId,omitempty"`
	FileIndex        *int       `json:"fileIndex,omitempty"`
}

// Definition contains the machine structure.
type Definition struct {
	ID              string                 `json:"id"`
	Edges           []Edge                 `json:"edges"`
	Context         map[string]interface{} `json:"context,omitempty"`
	Schemas         Schemas                `json:"schemas,omitempty"`
	RootNode        Node                   `json:"rootNode"`
	Implementations Implementations        `json:"implementations,omitempty"`
}

// Node represents a state node in the hierarchy.
type Node struct {
	ID       string   `json:"id"`
	Data     NodeData `json:"data"`
	Size     Size     `json:"size"`
	Nodes    []Node   `json:"nodes,omitempty"`
	Position Position `json:"position"`
	UniqueID string   `json:"uniqueId"`
}

// NodeData contains state-specific properties.
type NodeData struct {
	Key         string          `json:"key"`
	Exit        []ActionRef     `json:"exit,omitempty"`
	Tags        []string        `json:"tags,omitempty"`
	Color       string          `json:"color,omitempty"`
	Entry       []ActionRef     `json:"entry,omitempty"`
	Assets      []Asset         `json:"assets,omitempty"`
	Invoke      []InvokeConfig  `json:"invoke,omitempty"`
	Initial     string          `json:"initial,omitempty"`
	Type        string          `json:"type,omitempty"` // "parallel" for AND states
	Description string          `json:"description,omitempty"`
	MetaEntries json.RawMessage `json:"metaEntries,omitempty"` // [[key, value], ...]
}

// Edge represents a transition between states.
type Edge struct {
	ID       string   `json:"id"`
	Data     EdgeData `json:"data"`
	Size     Size     `json:"size"`
	Source   string   `json:"source"`
	Target   string   `json:"target,omitempty"` // Empty for self-transitions at root level
	Position Position `json:"position"`
	UniqueID string   `json:"uniqueId"`
}

// EdgeData contains transition-specific properties.
type EdgeData struct {
	Color         string          `json:"color,omitempty"`
	Actions       []ActionRef     `json:"actions,omitempty"`
	Guard         *GuardRef       `json:"guard,omitempty"`
	EventTypeData EventTypeData   `json:"eventTypeData"`
	Description   string          `json:"description,omitempty"`
	Internal      bool            `json:"internal,omitempty"`
	MetaEntries   json.RawMessage `json:"metaEntries,omitempty"`
}

// EventTypeData describes the trigger type for a transition.
type EventTypeData struct {
	Type      string `json:"type"`                // "named", "always", "after", "invocation.done", "invocation.error", "state.done"
	EventType string `json:"eventType,omitempty"` // Event name for "named" type
}

// ActionRef references an action by type and parameters.
type ActionRef struct {
	Kind   string       `json:"kind,omitempty"` // "named", etc.
	Action ActionConfig `json:"action,omitempty"`
}

// ActionConfig contains action type and parameters.
type ActionConfig struct {
	Type   string                 `json:"type"`
	Params map[string]interface{} `json:"params,omitempty"`
}

// GuardRef references a guard condition.
type GuardRef struct {
	Kind   string                 `json:"kind,omitempty"` // "named"
	Name   string                 `json:"name,omitempty"`
	Type   string                 `json:"type,omitempty"`
	Params map[string]interface{} `json:"params,omitempty"`
}

// InvokeConfig describes an actor invocation.
type InvokeConfig struct {
	ID       string                 `json:"id"`
	Src      string                 `json:"src"`
	Kind     string                 `json:"kind,omitempty"`
	Input    map[string]interface{} `json:"input,omitempty"`
	Settings map[string]interface{} `json:"settings,omitempty"`
}

// Asset represents a visual asset attached to a state.
type Asset struct {
	Name       string                 `json:"name"`
	Template   string                 `json:"template"` // "@figma.link", "@stately.image"
	Properties map[string]interface{} `json:"properties"`
}

// Position represents x,y coordinates.
type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Size represents dimensions. Can have different formats in XState JSON.
type Size struct {
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
	// Some Size objects also have x,y,top,left,right,bottom - we ignore those
}

// Schemas contains type schemas for machine elements.
type Schemas struct {
	Tags    map[string]interface{} `json:"tags,omitempty"`
	Input   interface{}            `json:"input,omitempty"`
	Actors  map[string]interface{} `json:"actors,omitempty"`
	Delays  map[string]interface{} `json:"delays,omitempty"`
	Events  map[string]interface{} `json:"events,omitempty"`
	Guards  map[string]interface{} `json:"guards,omitempty"`
	Output  interface{}            `json:"output,omitempty"`
	Actions map[string]interface{} `json:"actions,omitempty"`
	Context map[string]interface{} `json:"context,omitempty"`
}

// Implementations contains action, guard, and actor implementations.
type Implementations struct {
	Actions map[string]ActionImpl `json:"actions,omitempty"`
	Guards  map[string]GuardImpl  `json:"guards,omitempty"`
	Actors  map[string]ActorImpl  `json:"actors,omitempty"`
}

// ActionImpl describes an action implementation.
type ActionImpl struct {
	ID      string                 `json:"id"`
	Name    string                 `json:"name,omitempty"`
	Type    string                 `json:"type"` // "action"
	Code    string                 `json:"code,omitempty"`
	Schema  map[string]interface{} `json:"schema,omitempty"`
	Imports []string               `json:"imports,omitempty"`
}

// GuardImpl describes a guard implementation.
type GuardImpl struct {
	ID      string                 `json:"id"`
	Name    string                 `json:"name,omitempty"`
	Type    string                 `json:"type"` // "guard"
	Params  map[string]interface{} `json:"params,omitempty"`
	Imports []string               `json:"imports,omitempty"`
}

// ActorImpl describes an actor implementation.
type ActorImpl struct {
	ID      string                 `json:"id"`
	Kind    string                 `json:"kind,omitempty"`
	Name    string                 `json:"name,omitempty"`
	Type    string                 `json:"type"` // "actor"
	Input   map[string]interface{} `json:"input,omitempty"`
	Output  map[string]interface{} `json:"output,omitempty"`
	Imports []string               `json:"imports,omitempty"`
}
