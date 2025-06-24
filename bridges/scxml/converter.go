package scxml

import (
	"encoding/xml"
	"fmt"
	"strings"

	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/common"
	"google.golang.org/protobuf/types/known/structpb"
)

// Document represents an SCXML document structure.
type Document struct {
	XMLName    xml.Name   `xml:"scxml"`
	Namespace  string     `xml:"xmlns,attr"`
	Version    string     `xml:"version,attr"`
	Initial    string     `xml:"initial,attr,omitempty"`
	Name       string     `xml:"name,attr,omitempty"`
	Datamodel  *Datamodel `xml:"datamodel,omitempty"`
	States     []*State   `xml:"state"`
	Parallels  []*State   `xml:"parallel"`
	Finals     []*State   `xml:"final"`
}

// State represents an SCXML state element.
type State struct {
	XMLName     xml.Name      `xml:"state"`
	ID          string        `xml:"id,attr"`
	Initial     string        `xml:"initial,attr,omitempty"`
	States      []*State      `xml:"state"`
	Parallels   []*State      `xml:"parallel"`
	Finals      []*State      `xml:"final"`
	Transitions []*Transition `xml:"transition"`
	OnEntry     []*OnEntry    `xml:"onentry"`
	OnExit      []*OnExit     `xml:"onexit"`
}

// Transition represents an SCXML transition element.
type Transition struct {
	XMLName xml.Name `xml:"transition"`
	Event   string   `xml:"event,attr,omitempty"`
	Cond    string   `xml:"cond,attr,omitempty"`
	Target  string   `xml:"target,attr,omitempty"`
	Type    string   `xml:"type,attr,omitempty"`
	Actions []Action `xml:",any"`
}

// OnEntry represents SCXML onentry executable content.
type OnEntry struct {
	XMLName xml.Name `xml:"onentry"`
	Actions []Action `xml:",any"`
}

// OnExit represents SCXML onexit executable content.
type OnExit struct {
	XMLName xml.Name `xml:"onexit"`
	Actions []Action `xml:",any"`
}

// Action represents SCXML executable content.
type Action struct {
	XMLName xml.Name `xml:",any"`
	Content string   `xml:",chardata"`
	Attrs   []xml.Attr `xml:",any,attr"`
}

// Datamodel represents SCXML datamodel.
type Datamodel struct {
	XMLName xml.Name `xml:"datamodel"`
	Data    []*Data  `xml:"data"`
}

// Data represents SCXML data element.
type Data struct {
	XMLName xml.Name `xml:"data"`
	ID      string   `xml:"id,attr"`
	Expr    string   `xml:"expr,attr,omitempty"`
	Src     string   `xml:"src,attr,omitempty"`
	Content string   `xml:",chardata"`
}

// Converter implements bidirectional conversion between SCXML and Harel formats.
type Converter struct {
	mapping *common.SemanticMapping
}

// NewConverter creates a new SCXML converter with default semantic mappings.
func NewConverter() *Converter {
	return &Converter{
		mapping: &common.SemanticMapping{
			StateTypeMapping: map[string]sc.StateType{
				"state":    sc.StateTypeNormal,
				"parallel": sc.StateTypeParallel,
				"final":    sc.StateTypeBasic,
			},
			EventMapping:  make(map[string]string),
			ActionMapping: make(map[string]string),
			UnsupportedFeatures: []string{
				"invoke", "send", "script", "foreach", "if", "elseif", "else",
			},
		},
	}
}

// Import converts an SCXML document to Harel core semantics.
func (c *Converter) Import(doc Document) (*sc.Statechart, error) {
	// Create root state
	rootState := &sc.State{
		Label: "__root__",
		Type:  sc.StateTypeNormal,
	}

	var allTransitions []*sc.Transition
	var allEvents []*sc.Event
	eventSet := make(map[string]bool)

	// Convert all states (regular, parallel, final)
	allStates := append(doc.States, doc.Parallels...)
	allStates = append(allStates, doc.Finals...)

	for _, scxmlState := range allStates {
		state, transitions, events, err := c.convertState(scxmlState, doc.Initial)
		if err != nil {
			return nil, err
		}

		rootState.Children = append(rootState.Children, state)
		allTransitions = append(allTransitions, transitions...)

		// Deduplicate events
		for _, event := range events {
			if !eventSet[event.Label] {
				allEvents = append(allEvents, event)
				eventSet[event.Label] = true
			}
		}
	}

	// Create statechart
	statechart := &sc.Statechart{
		RootState:   rootState,
		Transitions: allTransitions,
		Events:      allEvents,
	}

	return statechart, nil
}

// Export converts Harel core semantics to SCXML document format.
func (c *Converter) Export(statechart *sc.Statechart) (Document, error) {
	if statechart.RootState == nil {
		return Document{}, common.NewConversionError("scxml", "export", "rootState",
			fmt.Errorf("root state is required"))
	}

	doc := Document{
		Namespace: "http://www.w3.org/2005/07/scxml",
		Version:   "1.0",
		Name:      "exported_statechart",
	}

	// Find initial state
	if len(statechart.RootState.Children) > 0 {
		for _, child := range statechart.RootState.Children {
			if child.IsInitial {
				doc.Initial = child.ID
				break
			}
		}
	}

	// Convert states
	for _, state := range statechart.RootState.Children {
		scxmlState, err := c.exportState(state, statechart.Transitions)
		if err != nil {
			return Document{}, err
		}

		switch state.Type {
		case sc.StateTypeParallel:
			doc.Parallels = append(doc.Parallels, scxmlState)
		case sc.StateTypeBasic:
			if state.IsFinal {
				doc.Finals = append(doc.Finals, scxmlState)
			} else {
				doc.States = append(doc.States, scxmlState)
			}
		default:
			doc.States = append(doc.States, scxmlState)
		}
	}

	return doc, nil
}

// Validate checks semantic consistency of SCXML document.
func (c *Converter) Validate(doc Document) error {
	result := &common.ValidationResult{Valid: true}

	// Check required namespace
	if doc.Namespace != "http://www.w3.org/2005/07/scxml" {
		result.AddWarning("non-standard SCXML namespace: %s", doc.Namespace)
	}

	// Check version
	if doc.Version != "1.0" {
		result.AddWarning("non-standard SCXML version: %s", doc.Version)
	}

	// Validate states
	allStates := append(doc.States, doc.Parallels...)
	allStates = append(allStates, doc.Finals...)

	if len(allStates) == 0 {
		result.AddError("document must have at least one state")
	}

	for _, state := range allStates {
		c.validateState(state, result)
	}

	if !result.Valid {
		return fmt.Errorf("validation failed: %v", result.Errors)
	}

	return nil
}

// FormatName returns the name of the SCXML format.
func (c *Converter) FormatName() string {
	return "SCXML"
}

// convertState converts an SCXML state to Harel state.
func (c *Converter) convertState(scxmlState *State, initial string) (*sc.State, []*sc.Transition, []*sc.Event, error) {
	state := &sc.State{
		Label:     scxmlState.ID,
		IsInitial: scxmlState.ID == initial,
	}

	// Determine state type
	switch scxmlState.XMLName.Local {
	case "parallel":
		state.Type = sc.StateTypeParallel
	case "final":
		state.Type = sc.StateTypeBasic
		state.IsFinal = true
	default:
		if len(scxmlState.States)+len(scxmlState.Parallels)+len(scxmlState.Finals) > 0 {
			state.Type = sc.StateTypeNormal
		} else {
			state.Type = sc.StateTypeBasic
		}
	}

	var allTransitions []*sc.Transition
	var allEvents []*sc.Event
	eventSet := make(map[string]bool)

	// Convert child states
	childStates := append(scxmlState.States, scxmlState.Parallels...)
	childStates = append(childStates, scxmlState.Finals...)

	for _, childState := range childStates {
		child, transitions, events, err := c.convertState(childState, scxmlState.Initial)
		if err != nil {
			return nil, nil, nil, err
		}

		state.Children = append(state.Children, child)
		allTransitions = append(allTransitions, transitions...)

		for _, event := range events {
			if !eventSet[event.Label] {
				allEvents = append(allEvents, event)
				eventSet[event.Label] = true
			}
		}
	}

	// Convert transitions
	for _, scxmlTransition := range scxmlState.Transitions {
		transition := &sc.Transition{
			Label: fmt.Sprintf("%s_transition", scxmlState.ID),
			From:  []string{scxmlState.ID},
			Event: scxmlTransition.Event,
		}

		// Handle target
		if scxmlTransition.Target != "" {
			transition.To = strings.Fields(scxmlTransition.Target)
		} else {
			// Internal transition
			transition.To = []string{scxmlState.ID}
		}

		// Handle condition
		if scxmlTransition.Cond != "" {
			transition.Guard = &sc.Guard{Expression: scxmlTransition.Cond}
		}

		// Convert actions
		for _, action := range scxmlTransition.Actions {
			transition.Actions = append(transition.Actions, &sc.Action{
				Label: action.XMLName.Local,
			})
		}

		allTransitions = append(allTransitions, transition)

		// Create event if not empty
		if scxmlTransition.Event != "" && !eventSet[scxmlTransition.Event] {
			allEvents = append(allEvents, &sc.Event{Label: scxmlTransition.Event})
			eventSet[scxmlTransition.Event] = true
		}
	}

	return state, allTransitions, allEvents, nil
}

// exportState converts Harel state to SCXML state.
func (c *Converter) exportState(state *sc.State, transitions []*sc.Transition) (*State, error) {
	scxmlState := &State{
		ID: state.Label,
	}

	// Set XML name based on state type
	switch state.Type {
	case sc.StateTypeParallel:
		scxmlState.XMLName = xml.Name{Local: "parallel"}
	case sc.StateTypeBasic:
		if state.IsFinal {
			scxmlState.XMLName = xml.Name{Local: "final"}
		} else {
			scxmlState.XMLName = xml.Name{Local: "state"}
		}
	default:
		scxmlState.XMLName = xml.Name{Local: "state"}
	}

	// Find initial child state
	for _, child := range state.Children {
		if child.IsInitial {
			scxmlState.Initial = child.Label
			break
		}
	}

	// Convert child states
	for _, child := range state.Children {
		childState, err := c.exportState(child, transitions)
		if err != nil {
			return nil, err
		}

		switch child.Type {
		case sc.StateTypeParallel:
			scxmlState.Parallels = append(scxmlState.Parallels, childState)
		case sc.StateTypeBasic:
			if child.IsFinal {
				scxmlState.Finals = append(scxmlState.Finals, childState)
			} else {
				scxmlState.States = append(scxmlState.States, childState)
			}
		default:
			scxmlState.States = append(scxmlState.States, childState)
		}
	}

	// Convert transitions
	for _, transition := range transitions {
		for _, from := range transition.From {
			if from == state.Label {
				scxmlTransition := &Transition{
					Event:  transition.Event,
					Target: strings.Join(transition.To, " "),
				}

				if transition.Guard != nil {
					scxmlTransition.Cond = transition.Guard.Expression
				}

				// Convert actions
				for _, action := range transition.Actions {
					scxmlTransition.Actions = append(scxmlTransition.Actions, Action{
						XMLName: xml.Name{Local: action.Label},
					})
				}

				scxmlState.Transitions = append(scxmlState.Transitions, scxmlTransition)
			}
		}
	}

	return scxmlState, nil
}

// validateState recursively validates SCXML state structure.
func (c *Converter) validateState(state *State, result *common.ValidationResult) {
	if state.ID == "" {
		result.AddError("state must have an ID")
	}

	// Check compound states have initial
	if len(state.States)+len(state.Parallels) > 0 && state.Initial == "" && state.XMLName.Local != "parallel" {
		result.AddWarning("compound state '%s' should specify initial state", state.ID)
	}

	// Validate child states
	for _, child := range state.States {
		c.validateState(child, result)
	}
	for _, child := range state.Parallels {
		c.validateState(child, result)
	}
	for _, child := range state.Finals {
		c.validateState(child, result)
	}
}

// ParseXML parses SCXML document from XML string.
func ParseXML(xmlStr string) (Document, error) {
	var doc Document
	err := xml.Unmarshal([]byte(xmlStr), &doc)
	if err != nil {
		return Document{}, common.NewConversionError("scxml", "parse", "xml", err)
	}
	return doc, nil
}

// ToXML converts SCXML document to XML string.
func ToXML(doc Document) (string, error) {
	xmlBytes, err := xml.MarshalIndent(doc, "", "  ")
	if err != nil {
		return "", common.NewConversionError("scxml", "serialize", "xml", err)
	}
	return xml.Header + string(xmlBytes), nil
}