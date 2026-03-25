package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/sc"
	statechartspb "github.com/tmc/sc/gen/statecharts/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/structpb"
)

var textTransitionRE = regexp.MustCompile(`^\s*(\S+)\s*->\s*(\S+)\s*\[(.+)\]\s*$`)

type buildOptions struct {
	MinConfidence  float64
	InferHierarchy bool
	InferParallel  bool
}

type inferer struct {
	rootLabel string

	observations int
	configs      int

	states      map[string]*stateStats
	transitions map[string]*transitionStats
	events      map[string]bool

	outgoingTotals map[string]int
	endpoints      map[string]bool

	cooccurrence map[string]map[string]int
	beforeCount  map[string]map[string]int
	distanceSum  map[string]map[string]int

	initialConfigs  [][]string
	initialFallback []string
}

type stateStats struct {
	Label        string
	Observations int
	Configs      int
	FirstSeen    int
	LastSeen     int
}

type transitionStats struct {
	From         []string
	To           []string
	Event        string
	Label        string
	Observations int
	FirstSeen    int
	LastSeen     int
}

type transitionObservation struct {
	From  []string
	To    []string
	Event string
	Label string
}

type traceDataset struct {
	Traces []json.RawMessage `json:"traces"`
}

type jsonLineTransition struct {
	From  json.RawMessage `json:"from"`
	To    json.RawMessage `json:"to"`
	Event string          `json:"event"`
}

type inferredModel struct {
	rootLabel       string
	children        map[string][]string
	parent          map[string]string
	leafs           map[string]bool
	subtreeLeaves   map[string]map[string]bool
	initialLeafHits map[string]int
	inferParallel   bool
}

func newInferer() *inferer {
	return &inferer{
		states:         make(map[string]*stateStats),
		transitions:    make(map[string]*transitionStats),
		events:         make(map[string]bool),
		outgoingTotals: make(map[string]int),
		endpoints:      make(map[string]bool),
		cooccurrence:   make(map[string]map[string]int),
		beforeCount:    make(map[string]map[string]int),
		distanceSum:    make(map[string]map[string]int),
	}
}

func (i *inferer) ObservationCount() int {
	return i.observations
}

func (i *inferer) ObserveInitialConfig(labels []string) bool {
	labels = cleanLabels(labels)
	if len(labels) == 0 {
		return false
	}
	i.initialConfigs = append(i.initialConfigs, append([]string(nil), labels...))
	return i.observeConfig(labels, i.observations)
}

func (i *inferer) RememberInitialFallback(labels []string) {
	if len(i.initialFallback) != 0 {
		return
	}
	i.initialFallback = append([]string(nil), cleanLabels(labels)...)
}

func (i *inferer) ProcessObservation(targetConfig []string, transitions []transitionObservation) bool {
	step := i.observations
	structuralChange := false

	targetConfig = cleanLabels(targetConfig)
	if len(targetConfig) > 0 {
		structuralChange = i.observeConfig(targetConfig, step) || structuralChange
	}
	for _, observation := range transitions {
		structuralChange = i.observeTransition(observation, step) || structuralChange
	}

	i.observations++
	return structuralChange
}

func (i *inferer) observeConfig(labels []string, step int) bool {
	if len(labels) == 0 {
		return false
	}
	if i.rootLabel == "" {
		i.rootLabel = labels[0]
	}

	i.configs++
	structuralChange := false
	for _, label := range labels {
		stats, created := i.ensureState(label, step)
		if created {
			structuralChange = true
		}
		stats.Observations++
		stats.Configs++
		stats.LastSeen = step
	}

	for idx, label := range labels {
		for next := idx + 1; next < len(labels); next++ {
			other := labels[next]
			i.incrementPair(i.cooccurrence, label, other, 1)
			i.incrementPair(i.cooccurrence, other, label, 1)
			i.incrementPair(i.beforeCount, label, other, 1)
			i.incrementPair(i.distanceSum, label, other, next-idx)
		}
	}
	return structuralChange
}

func (i *inferer) observeTransition(observation transitionObservation, step int) bool {
	from := cleanLabels(observation.From)
	to := cleanLabels(observation.To)
	if len(from) == 0 && len(to) == 0 {
		return false
	}

	structuralChange := false
	for _, label := range append(append([]string(nil), from...), to...) {
		stats, created := i.ensureState(label, step)
		if created {
			structuralChange = true
		}
		stats.Observations++
		stats.LastSeen = step
		i.endpoints[label] = true
	}

	if observation.Event != "" {
		i.events[observation.Event] = true
	}

	key := transitionKey(from, to, observation.Event)
	stats, ok := i.transitions[key]
	if !ok {
		stats = &transitionStats{
			From:         append([]string(nil), from...),
			To:           append([]string(nil), to...),
			Event:        observation.Event,
			Label:        observation.Label,
			FirstSeen:    step,
			LastSeen:     step,
			Observations: 0,
		}
		i.transitions[key] = stats
		structuralChange = true
	}
	stats.Observations++
	stats.LastSeen = step
	if stats.Label == "" {
		stats.Label = observation.Label
	}
	i.outgoingTotals[sourceKey(from)]++
	return structuralChange
}

func (i *inferer) ensureState(label string, step int) (*stateStats, bool) {
	stats, ok := i.states[label]
	if ok {
		return stats, false
	}
	stats = &stateStats{
		Label:     label,
		FirstSeen: step,
		LastSeen:  step,
	}
	i.states[label] = stats
	return stats, true
}

func (i *inferer) incrementPair(dst map[string]map[string]int, a, b string, delta int) {
	row := dst[a]
	if row == nil {
		row = make(map[string]int)
		dst[a] = row
	}
	row[b] += delta
}

func (i *inferer) ProgressLine() string {
	return fmt.Sprintf(
		"sc-infer: %d observations | %d states | %d transitions | confidence: %.2f avg",
		i.observations,
		len(i.states),
		len(i.transitions),
		i.averageTransitionConfidence(),
	)
}

func (i *inferer) averageTransitionConfidence() float64 {
	if len(i.transitions) == 0 {
		return 0
	}
	total := 0.0
	count := 0
	for _, transition := range i.transitions {
		denom := i.outgoingTotals[sourceKey(transition.From)]
		if denom == 0 {
			continue
		}
		total += float64(transition.Observations) / float64(denom)
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

func (i *inferer) BuildChart(opts buildOptions) (*sc.Statechart, error) {
	rootLabel := i.rootLabel
	if rootLabel == "" {
		rootLabel = "__root__"
	}

	leafs := i.inferLeaves(rootLabel)
	if len(leafs) == 0 {
		return nil, fmt.Errorf("no states observed")
	}

	model := i.buildModel(rootLabel, leafs, opts)
	root := i.buildState(rootLabel, model)
	transitions, usedEvents := i.buildTransitions(opts.MinConfidence)

	events := make([]*sc.Event, 0, len(usedEvents))
	if len(usedEvents) == 0 {
		for event := range i.events {
			usedEvents[event] = true
		}
	}
	eventLabels := make([]string, 0, len(usedEvents))
	for event := range usedEvents {
		eventLabels = append(eventLabels, event)
	}
	slices.Sort(eventLabels)
	for _, event := range eventLabels {
		if event == "" {
			continue
		}
		events = append(events, &sc.Event{Label: event})
	}

	return &sc.Statechart{
		Name:        "inferred",
		RootState:   root,
		Transitions: transitions,
		Events:      events,
	}, nil
}

func (i *inferer) buildModel(rootLabel string, leafLabels []string, opts buildOptions) *inferredModel {
	leafs := make(map[string]bool, len(leafLabels))
	for _, label := range leafLabels {
		leafs[label] = true
	}

	model := &inferredModel{
		rootLabel:       rootLabel,
		children:        make(map[string][]string),
		parent:          make(map[string]string),
		leafs:           leafs,
		subtreeLeaves:   make(map[string]map[string]bool),
		initialLeafHits: i.initialLeafHits(leafs),
		inferParallel:   opts.InferParallel,
	}

	if !opts.InferHierarchy || i.configs == 0 {
		for _, label := range leafLabels {
			model.parent[label] = rootLabel
			model.children[rootLabel] = append(model.children[rootLabel], label)
		}
		i.sortChildren(model.children)
		i.fillSubtreeLeaves(model, rootLabel)
		return model
	}

	composites := make(map[string]bool)
	for label := range i.states {
		if label == rootLabel || !leafs[label] {
			composites[label] = true
		}
	}
	composites[rootLabel] = true

	descendants := make(map[string]map[string]bool)
	for composite := range composites {
		desc := make(map[string]bool)
		for leaf := range leafs {
			if composite == rootLabel {
				desc[leaf] = true
				continue
			}
			if i.isOrderedAncestor(composite, leaf) {
				desc[leaf] = true
			}
		}
		if composite == rootLabel || len(desc) > 0 {
			descendants[composite] = desc
		}
	}

	directLeafs := make(map[string]map[string]bool)
	for composite := range descendants {
		directLeafs[composite] = make(map[string]bool)
	}

	for _, leaf := range leafLabels {
		parent := rootLabel
		for composite, desc := range descendants {
			if !desc[leaf] {
				continue
			}
			if parent == rootLabel || betterLeafParent(i, descendants, composite, parent, leaf, rootLabel) {
				parent = composite
			}
		}
		model.parent[leaf] = parent
		model.children[parent] = append(model.children[parent], leaf)
		directLeafs[parent][leaf] = true
	}

	for composite := range descendants {
		if composite == rootLabel {
			continue
		}
		cluster := directLeafs[composite]
		if len(cluster) == 0 {
			continue
		}

		var preferred []string
		var fallback []string
		for candidate := range descendants {
			if candidate == composite {
				continue
			}
			if !i.isOrderedAncestor(candidate, composite) {
				continue
			}
			if !containsAll(descendants[candidate], cluster) {
				continue
			}
			if candidate == rootLabel || len(directLeafs[candidate]) == 0 || intersects(directLeafs[candidate], cluster) {
				preferred = append(preferred, candidate)
			} else {
				fallback = append(fallback, candidate)
			}
		}

		candidates := preferred
		if len(candidates) == 0 {
			candidates = fallback
		}
		if len(candidates) == 0 {
			model.parent[composite] = rootLabel
			model.children[rootLabel] = append(model.children[rootLabel], composite)
			continue
		}

		best := candidates[0]
		for _, candidate := range candidates[1:] {
			if betterCompositeParent(i, descendants, candidate, best, composite) {
				best = candidate
			}
		}
		model.parent[composite] = best
		model.children[best] = append(model.children[best], composite)
	}

	for child, parent := range model.parent {
		if child == rootLabel || parent == "" {
			continue
		}
		if !containsString(model.children[parent], child) {
			model.children[parent] = append(model.children[parent], child)
		}
	}

	i.sortChildren(model.children)
	i.fillSubtreeLeaves(model, rootLabel)
	return model
}

func (i *inferer) buildState(label string, model *inferredModel) *sc.State {
	children := model.children[label]
	state := &sc.State{
		Label:    label,
		Metadata: i.stateMetadata(label),
	}

	if len(children) == 0 || model.leafs[label] {
		state.Type = sc.StateTypeBasic
		return state
	}

	childStates := make([]*sc.State, 0, len(children))
	for _, child := range children {
		childStates = append(childStates, i.buildState(child, model))
	}

	state.Children = childStates
	if model.inferParallel && i.hasParallelChildren(children, model) {
		state.Type = sc.StateTypeAND
	} else {
		state.Type = sc.StateTypeOR
		best := i.initialChild(children, model)
		for _, child := range state.Children {
			if child.Label == best {
				child.IsInitial = true
				break
			}
		}
	}
	return state
}

func (i *inferer) hasParallelChildren(children []string, model *inferredModel) bool {
	if len(children) < 2 {
		return false
	}
	for idx := 0; idx < len(children); idx++ {
		leftLeaves := model.subtreeLeaves[children[idx]]
		for next := idx + 1; next < len(children); next++ {
			rightLeaves := model.subtreeLeaves[children[next]]
			for left := range leftLeaves {
				for right := range rightLeaves {
					if left == right {
						continue
					}
					if i.pairCount(i.cooccurrence, left, right) > 0 {
						return true
					}
				}
			}
		}
	}
	return false
}

func (i *inferer) initialChild(children []string, model *inferredModel) string {
	best := children[0]
	bestHits := -1
	for _, child := range children {
		hits := 0
		for leaf := range model.subtreeLeaves[child] {
			hits += model.initialLeafHits[leaf]
		}
		if hits > bestHits {
			best = child
			bestHits = hits
			continue
		}
		if hits == bestHits && i.compareLabels(child, best) < 0 {
			best = child
		}
	}
	return best
}

func (i *inferer) fillSubtreeLeaves(model *inferredModel, label string) map[string]bool {
	if existing := model.subtreeLeaves[label]; existing != nil {
		return existing
	}
	out := make(map[string]bool)
	if model.leafs[label] || len(model.children[label]) == 0 {
		if model.leafs[label] {
			out[label] = true
		}
		model.subtreeLeaves[label] = out
		return out
	}
	for _, child := range model.children[label] {
		for leaf := range i.fillSubtreeLeaves(model, child) {
			out[leaf] = true
		}
	}
	model.subtreeLeaves[label] = out
	return out
}

func (i *inferer) initialLeafHits(leafs map[string]bool) map[string]int {
	hits := make(map[string]int)
	for _, labels := range i.initialConfigs {
		for _, label := range labels {
			if leafs[label] {
				hits[label]++
			}
		}
	}
	if len(hits) != 0 {
		return hits
	}
	for _, label := range i.initialFallback {
		if leafs[label] {
			hits[label]++
		}
	}
	return hits
}

func (i *inferer) inferLeaves(rootLabel string) []string {
	labels := make([]string, 0, len(i.endpoints))
	for label := range i.endpoints {
		if label == rootLabel {
			continue
		}
		labels = append(labels, label)
	}
	if len(labels) == 0 {
		for label := range i.states {
			if label == rootLabel {
				continue
			}
			labels = append(labels, label)
		}
	}
	slices.SortFunc(labels, func(a, b string) int {
		return i.compareLabels(a, b)
	})
	return labels
}

func (i *inferer) compareLabels(a, b string) int {
	aStats := i.states[a]
	bStats := i.states[b]
	aSeen := math.MaxInt
	bSeen := math.MaxInt
	if aStats != nil {
		aSeen = aStats.FirstSeen
	}
	if bStats != nil {
		bSeen = bStats.FirstSeen
	}
	if aSeen != bSeen {
		if aSeen < bSeen {
			return -1
		}
		return 1
	}
	return strings.Compare(a, b)
}

func (i *inferer) isOrderedAncestor(ancestor, descendant string) bool {
	if ancestor == descendant {
		return false
	}
	required := i.stateConfigs(descendant)
	if required == 0 {
		return false
	}
	return i.pairCount(i.cooccurrence, ancestor, descendant) == required &&
		i.pairCount(i.beforeCount, ancestor, descendant) == required
}

func (i *inferer) stateConfigs(label string) int {
	stats := i.states[label]
	if stats == nil {
		return 0
	}
	return stats.Configs
}

func (i *inferer) averageDistance(ancestor, descendant string) float64 {
	count := i.pairCount(i.beforeCount, ancestor, descendant)
	if count == 0 {
		return math.MaxFloat64
	}
	return float64(i.pairCount(i.distanceSum, ancestor, descendant)) / float64(count)
}

func (i *inferer) pairCount(src map[string]map[string]int, a, b string) int {
	row := src[a]
	if row == nil {
		return 0
	}
	return row[b]
}

func (i *inferer) sortChildren(children map[string][]string) {
	for label := range children {
		slices.SortFunc(children[label], func(a, b string) int {
			return i.compareLabels(a, b)
		})
	}
}

func (i *inferer) buildTransitions(minConfidence float64) ([]*sc.Transition, map[string]bool) {
	keys := make([]string, 0, len(i.transitions))
	for key := range i.transitions {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b string) int {
		left := i.transitions[a]
		right := i.transitions[b]
		if left.FirstSeen != right.FirstSeen {
			if left.FirstSeen < right.FirstSeen {
				return -1
			}
			return 1
		}
		if left.Event != right.Event {
			return strings.Compare(left.Event, right.Event)
		}
		if cmp := strings.Compare(sourceKey(left.From), sourceKey(right.From)); cmp != 0 {
			return cmp
		}
		return strings.Compare(sourceKey(left.To), sourceKey(right.To))
	})

	transitions := make([]*sc.Transition, 0, len(keys))
	usedEvents := make(map[string]bool)
	for _, key := range keys {
		stats := i.transitions[key]
		confidence := i.transitionConfidence(stats)
		if confidence < minConfidence {
			continue
		}
		metadata := metadataStruct(confidence, stats.Observations, stats.FirstSeen, stats.LastSeen)
		transitions = append(transitions, &sc.Transition{
			Label:    stats.Label,
			From:     append([]string(nil), stats.From...),
			To:       append([]string(nil), stats.To...),
			Event:    stats.Event,
			Metadata: metadata,
		})
		if stats.Event != "" {
			usedEvents[stats.Event] = true
		}
	}
	return transitions, usedEvents
}

func (i *inferer) transitionConfidence(stats *transitionStats) float64 {
	if stats == nil {
		return 0
	}
	denom := i.outgoingTotals[sourceKey(stats.From)]
	if denom == 0 {
		return 0
	}
	return float64(stats.Observations) / float64(denom)
}

func (i *inferer) stateMetadata(label string) *structpb.Struct {
	stats := i.states[label]
	if stats == nil {
		return nil
	}
	maxObservations := 0
	for _, state := range i.states {
		if state.Observations > maxObservations {
			maxObservations = state.Observations
		}
	}
	if maxObservations == 0 {
		maxObservations = 1
	}
	confidence := float64(stats.Observations) / float64(maxObservations)
	return metadataStruct(confidence, stats.Observations, stats.FirstSeen, stats.LastSeen)
}

func metadataStruct(confidence float64, observations, firstSeen, lastSeen int) *structpb.Struct {
	fields, err := structpb.NewStruct(map[string]any{
		"confidence":      formatConfidence(confidence),
		"observations":    strconv.Itoa(observations),
		"first_seen_step": strconv.Itoa(firstSeen),
		"last_seen_step":  strconv.Itoa(lastSeen),
	})
	if err != nil {
		return nil
	}
	return fields
}

func formatConfidence(v float64) string {
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return strconv.FormatFloat(v, 'f', 6, 64)
}

func betterLeafParent(i *inferer, descendants map[string]map[string]bool, candidate, current, leaf, rootLabel string) bool {
	if current == "" {
		return true
	}
	candidateSize := len(descendants[candidate])
	currentSize := len(descendants[current])
	if candidate == rootLabel {
		candidateSize = len(descendants[rootLabel])
	}
	if current == rootLabel {
		currentSize = len(descendants[rootLabel])
	}
	if candidateSize != currentSize {
		return candidateSize < currentSize
	}
	candidateDistance := i.averageDistance(candidate, leaf)
	currentDistance := i.averageDistance(current, leaf)
	if candidateDistance != currentDistance {
		return candidateDistance < currentDistance
	}
	return i.compareLabels(candidate, current) < 0
}

func betterCompositeParent(i *inferer, descendants map[string]map[string]bool, candidate, current, child string) bool {
	candidateSize := len(descendants[candidate])
	currentSize := len(descendants[current])
	if candidateSize != currentSize {
		return candidateSize < currentSize
	}
	candidateDistance := i.averageDistance(candidate, child)
	currentDistance := i.averageDistance(current, child)
	if candidateDistance != currentDistance {
		return candidateDistance < currentDistance
	}
	return i.compareLabels(candidate, current) < 0
}

func containsAll(haystack, needles map[string]bool) bool {
	for needle := range needles {
		if !haystack[needle] {
			return false
		}
	}
	return true
}

func intersects(left, right map[string]bool) bool {
	for label := range left {
		if right[label] {
			return true
		}
	}
	return false
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func detectFormat(data []byte) (string, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return "", fmt.Errorf("input is empty")
	}

	if trimmed[0] == '{' {
		var dataset traceDataset
		if err := json.Unmarshal(trimmed, &dataset); err == nil && len(dataset.Traces) > 0 {
			return "trace", nil
		}
	}

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if textTransitionRE.MatchString(line) {
			return "text", nil
		}
		var row jsonLineTransition
		if err := json.Unmarshal([]byte(line), &row); err == nil {
			return "jsonl", nil
		}
		break
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scan input: %w", err)
	}
	return "", fmt.Errorf("could not detect input format")
}

func parseTextInput(data []byte, inferer *inferer, onObservation func(bool) error) error {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		match := textTransitionRE.FindStringSubmatch(line)
		if match == nil {
			return fmt.Errorf("parse text line %d: invalid transition %q", lineNo, line)
		}
		observation := transitionObservation{
			From:  []string{match[1]},
			To:    []string{match[2]},
			Event: strings.TrimSpace(match[3]),
		}
		inferer.RememberInitialFallback(observation.From)
		if err := onObservation(inferer.ProcessObservation(nil, []transitionObservation{observation})); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan text input: %w", err)
	}
	return nil
}

func parseJSONLInput(data []byte, inferer *inferer, onObservation func(bool) error) error {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row jsonLineTransition
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return fmt.Errorf("parse jsonl line %d: %w", lineNo, err)
		}
		from, err := parseEndpoint(row.From)
		if err != nil {
			return fmt.Errorf("parse jsonl line %d from: %w", lineNo, err)
		}
		to, err := parseEndpoint(row.To)
		if err != nil {
			return fmt.Errorf("parse jsonl line %d to: %w", lineNo, err)
		}
		observation := transitionObservation{
			From:  from,
			To:    to,
			Event: row.Event,
		}
		inferer.RememberInitialFallback(observation.From)
		if err := onObservation(inferer.ProcessObservation(nil, []transitionObservation{observation})); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan jsonl input: %w", err)
	}
	return nil
}

func parseTraceInput(data []byte, inferer *inferer, onObservation func(bool) error) error {
	var dataset traceDataset
	if err := json.Unmarshal(data, &dataset); err != nil {
		return fmt.Errorf("parse trace dataset: %w", err)
	}
	if len(dataset.Traces) == 0 {
		return fmt.Errorf("trace dataset contains no traces")
	}

	opts := protojson.UnmarshalOptions{DiscardUnknown: true}
	for index, rawTrace := range dataset.Traces {
		var trace statechartspb.ExecutionTrace
		if err := opts.Unmarshal(rawTrace, &trace); err != nil {
			return fmt.Errorf("parse trace %d: %w", index, err)
		}

		initial := configLabels(trace.InitialConfig)
		if len(initial) == 0 && len(trace.Entries) > 0 {
			initial = configLabels(trace.Entries[0].SourceConfig)
		}
		if len(initial) > 0 {
			inferer.ObserveInitialConfig(initial)
		}

		for _, entry := range trace.Entries {
			target := configLabels(entry.TargetConfig)
			if len(target) == 0 {
				target = configLabels(entry.SourceConfig)
			}
			if err := onObservation(inferer.ProcessObservation(target, transitionsFromEntry(entry))); err != nil {
				return err
			}
		}
	}
	return nil
}

func transitionsFromEntry(entry *statechartspb.TransitionLogEntry) []transitionObservation {
	transitions := make([]transitionObservation, 0, len(entry.GetTransitionsFired()))
	for _, fired := range entry.GetTransitionsFired() {
		if fired == nil {
			continue
		}
		transitions = append(transitions, transitionObservation{
			From:  append([]string(nil), fired.GetFrom()...),
			To:    append([]string(nil), fired.GetTo()...),
			Event: fired.GetEvent(),
			Label: fired.GetLabel(),
		})
	}
	if len(transitions) != 0 {
		return transitions
	}

	source := configLabels(entry.GetSourceConfig())
	target := configLabels(entry.GetTargetConfig())
	if slices.Equal(source, target) {
		return nil
	}
	from := diffLabels(source, target)
	to := diffLabels(target, source)
	if len(from) == 0 && len(to) == 0 {
		return nil
	}
	return []transitionObservation{{
		From:  from,
		To:    to,
		Event: entry.GetTriggerEvent().GetLabel(),
	}}
}

func parseEndpoint(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		return cleanLabels(many), nil
	}
	return nil, fmt.Errorf("expected string or []string")
}

func configLabels(config *sc.Configuration) []string {
	if config == nil {
		return nil
	}
	labels := make([]string, 0, len(config.GetStates()))
	for _, state := range config.GetStates() {
		if state == nil || state.GetLabel() == "" {
			continue
		}
		labels = append(labels, state.GetLabel())
	}
	return cleanLabels(labels)
}

func diffLabels(left, right []string) []string {
	rightSet := make(map[string]bool, len(right))
	for _, label := range right {
		rightSet[label] = true
	}
	out := make([]string, 0)
	for _, label := range left {
		if !rightSet[label] {
			out = append(out, label)
		}
	}
	return cleanLabels(out)
}

func cleanLabels(labels []string) []string {
	if len(labels) == 0 {
		return nil
	}
	out := make([]string, 0, len(labels))
	seen := make(map[string]bool, len(labels))
	for _, label := range labels {
		label = strings.TrimSpace(label)
		if label == "" || seen[label] {
			continue
		}
		seen[label] = true
		out = append(out, label)
	}
	return out
}

func transitionKey(from, to []string, event string) string {
	return sourceKey(from) + "\x1e" + sourceKey(to) + "\x1e" + event
}

func sourceKey(labels []string) string {
	return strings.Join(cleanLabels(labels), "\x1f")
}
