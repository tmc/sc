package profiling

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/testing"
)

// LargeStatechartMemoryOptimizer provides specialized memory optimizations for very large statecharts.
type LargeStatechartMemoryOptimizer struct {
	// String interning for labels
	stringIntern     map[string]string
	stringInternLock sync.RWMutex

	// Shared empty slices to reduce allocations
	emptyStrings     []string
	emptyStates      []*sc.State
	emptyTransitions []*sc.Transition
	emptyActions     []*sc.Action

	// Statistics
	stats OptimizationStats
}

// OptimizationStats tracks optimization effectiveness.
type OptimizationStats struct {
	StringsInterned      int64
	BytesSavedStrings    int64
	SlicesOptimized      int64
	BytesSavedSlices     int64
	StatesOptimized      int64
	TransitionsOptimized int64
	TotalBytesSaved      int64
}

// NewLargeStatechartMemoryOptimizer creates a new optimizer for large statecharts.
func NewLargeStatechartMemoryOptimizer() *LargeStatechartMemoryOptimizer {
	return &LargeStatechartMemoryOptimizer{
		stringIntern:     make(map[string]string, 1000),
		emptyStrings:     make([]string, 0),
		emptyStates:      make([]*sc.State, 0),
		emptyTransitions: make([]*sc.Transition, 0),
		emptyActions:     make([]*sc.Action, 0),
	}
}

// OptimizeLargeStatechart applies memory optimizations specifically for large statecharts.
func (lsmo *LargeStatechartMemoryOptimizer) OptimizeLargeStatechart(statechart *sc.Statechart) (*sc.Statechart, OptimizationReport) {
	start := time.Now()
	beforeSize := lsmo.estimateStatechartSize(statechart)

	// Create optimized copy
	optimized := &sc.Statechart{
		RootState:   lsmo.optimizeStateDeep(statechart.RootState),
		Transitions: lsmo.optimizeTransitionsDeep(statechart.Transitions),
		Events:      lsmo.optimizeEventsDeep(statechart.Events),
	}

	afterSize := lsmo.estimateStatechartSize(optimized)

	report := OptimizationReport{
		Duration:         time.Since(start),
		OriginalSize:     beforeSize,
		OptimizedSize:    afterSize,
		SizeReduction:    beforeSize - afterSize,
		ReductionPercent: float64(beforeSize-afterSize) / float64(beforeSize) * 100,
		Stats:            lsmo.stats,
	}

	return optimized, report
}

// optimizeStateDeep performs deep optimization of state structures.
func (lsmo *LargeStatechartMemoryOptimizer) optimizeStateDeep(state *sc.State) *sc.State {
	if state == nil {
		return nil
	}

	atomic.AddInt64(&lsmo.stats.StatesOptimized, 1)

	optimized := &sc.State{
		Label:     lsmo.internString(state.Label),
		Type:      state.Type,
		IsInitial: state.IsInitial,
		IsFinal:   state.IsFinal,
	}

	// Optimize children
	if len(state.Children) > 0 {
		// Pre-allocate exact size
		optimized.Children = make([]*sc.State, 0, len(state.Children))
		for _, child := range state.Children {
			optimized.Children = append(optimized.Children, lsmo.optimizeStateDeep(child))
		}

		// Trim excess capacity
		if cap(optimized.Children) > len(optimized.Children) {
			newChildren := make([]*sc.State, len(optimized.Children))
			copy(newChildren, optimized.Children)
			optimized.Children = newChildren
			atomic.AddInt64(&lsmo.stats.SlicesOptimized, 1)
		}
	} else {
		// Use shared empty slice
		optimized.Children = lsmo.emptyStates
	}

	return optimized
}

// optimizeTransitionsDeep performs deep optimization of transitions.
func (lsmo *LargeStatechartMemoryOptimizer) optimizeTransitionsDeep(transitions []*sc.Transition) []*sc.Transition {
	if len(transitions) == 0 {
		return lsmo.emptyTransitions
	}

	// Pre-allocate exact size
	optimized := make([]*sc.Transition, 0, len(transitions))

	for _, t := range transitions {
		atomic.AddInt64(&lsmo.stats.TransitionsOptimized, 1)

		opt := &sc.Transition{
			Label: lsmo.internString(t.Label),
			Event: lsmo.internString(t.Event),
			Guard: t.Guard, // Guards are typically unique, so we don't intern
		}

		// Optimize From slice
		if len(t.From) > 0 {
			opt.From = make([]string, 0, len(t.From))
			for _, from := range t.From {
				opt.From = append(opt.From, lsmo.internString(from))
			}
			// Trim excess capacity
			if cap(opt.From) > len(opt.From) {
				newFrom := make([]string, len(opt.From))
				copy(newFrom, opt.From)
				opt.From = newFrom
			}
		} else {
			opt.From = lsmo.emptyStrings
		}

		// Optimize To slice
		if len(t.To) > 0 {
			opt.To = make([]string, 0, len(t.To))
			for _, to := range t.To {
				opt.To = append(opt.To, lsmo.internString(to))
			}
			// Trim excess capacity
			if cap(opt.To) > len(opt.To) {
				newTo := make([]string, len(opt.To))
				copy(newTo, opt.To)
				opt.To = newTo
			}
		} else {
			opt.To = lsmo.emptyStrings
		}

		// Optimize Actions
		if len(t.Actions) > 0 {
			opt.Actions = make([]*sc.Action, 0, len(t.Actions))
			for _, action := range t.Actions {
				opt.Actions = append(opt.Actions, &sc.Action{
					Label: lsmo.internString(action.Label),
				})
			}
		} else {
			opt.Actions = lsmo.emptyActions
		}

		optimized = append(optimized, opt)
	}

	return optimized
}

// optimizeEventsDeep performs deep optimization of events.
func (lsmo *LargeStatechartMemoryOptimizer) optimizeEventsDeep(events []*sc.Event) []*sc.Event {
	if len(events) == 0 {
		return nil
	}

	optimized := make([]*sc.Event, 0, len(events))
	for _, event := range events {
		opt := &sc.Event{
			Label:      lsmo.internString(event.Label),
			Parameters: event.Parameters, // Parameters are typically unique
		}
		optimized = append(optimized, opt)
	}

	return optimized
}

// internString performs string interning to reduce memory usage.
func (lsmo *LargeStatechartMemoryOptimizer) internString(s string) string {
	if s == "" {
		return ""
	}

	lsmo.stringInternLock.RLock()
	if interned, exists := lsmo.stringIntern[s]; exists {
		lsmo.stringInternLock.RUnlock()
		return interned
	}
	lsmo.stringInternLock.RUnlock()

	lsmo.stringInternLock.Lock()
	defer lsmo.stringInternLock.Unlock()

	// Double-check after acquiring write lock
	if interned, exists := lsmo.stringIntern[s]; exists {
		return interned
	}

	// Intern the string
	lsmo.stringIntern[s] = s
	atomic.AddInt64(&lsmo.stats.StringsInterned, 1)

	// Estimate bytes saved (assuming average duplication factor of 3)
	atomic.AddInt64(&lsmo.stats.BytesSavedStrings, int64(len(s)*2))

	return s
}

// estimateStatechartSize estimates the memory size of a statechart.
func (lsmo *LargeStatechartMemoryOptimizer) estimateStatechartSize(statechart *sc.Statechart) int64 {
	if statechart == nil {
		return 0
	}

	size := int64(unsafe.Sizeof(*statechart))
	size += lsmo.estimateStateSize(statechart.RootState)

	for _, t := range statechart.Transitions {
		size += lsmo.estimateTransitionSize(t)
	}

	for _, e := range statechart.Events {
		size += lsmo.estimateEventSize(e)
	}

	return size
}

// estimateStateSize estimates the memory size of a state.
func (lsmo *LargeStatechartMemoryOptimizer) estimateStateSize(state *sc.State) int64 {
	if state == nil {
		return 0
	}

	size := int64(unsafe.Sizeof(*state))
	size += int64(len(state.Label))

	// Slice overhead
	size += int64(cap(state.Children)) * int64(unsafe.Sizeof(&sc.State{}))

	// Recursively count children
	for _, child := range state.Children {
		size += lsmo.estimateStateSize(child)
	}

	return size
}

// estimateTransitionSize estimates the memory size of a transition.
func (lsmo *LargeStatechartMemoryOptimizer) estimateTransitionSize(t *sc.Transition) int64 {
	if t == nil {
		return 0
	}

	size := int64(unsafe.Sizeof(*t))
	size += int64(len(t.Label))
	size += int64(len(t.Event))

	// From/To slices
	size += int64(cap(t.From)) * int64(unsafe.Sizeof(""))
	for _, s := range t.From {
		size += int64(len(s))
	}

	size += int64(cap(t.To)) * int64(unsafe.Sizeof(""))
	for _, s := range t.To {
		size += int64(len(s))
	}

	// Actions
	size += int64(cap(t.Actions)) * int64(unsafe.Sizeof(&sc.Action{}))
	for _, a := range t.Actions {
		if a != nil {
			size += int64(unsafe.Sizeof(*a)) + int64(len(a.Label))
		}
	}

	return size
}

// estimateEventSize estimates the memory size of an event.
func (lsmo *LargeStatechartMemoryOptimizer) estimateEventSize(e *sc.Event) int64 {
	if e == nil {
		return 0
	}

	size := int64(unsafe.Sizeof(*e))
	size += int64(len(e.Label))

	// Parameters size (simplified estimation)
	if e.Parameters != nil {
		size += 100 // Rough estimate for structpb overhead
	}

	return size
}

// OptimizationReport contains the results of an optimization operation.
type OptimizationReport struct {
	Duration         time.Duration
	OriginalSize     int64
	OptimizedSize    int64
	SizeReduction    int64
	ReductionPercent float64
	Stats            OptimizationStats
}

// String returns a human-readable representation of the optimization report.
func (or OptimizationReport) String() string {
	return fmt.Sprintf(
		"Optimization Report:\n"+
			"  Duration: %v\n"+
			"  Original Size: %d bytes\n"+
			"  Optimized Size: %d bytes\n"+
			"  Size Reduction: %d bytes (%.2f%%)\n"+
			"  Strings Interned: %d (saved %d bytes)\n"+
			"  Slices Optimized: %d\n"+
			"  States Optimized: %d\n"+
			"  Transitions Optimized: %d",
		or.Duration,
		or.OriginalSize,
		or.OptimizedSize,
		or.SizeReduction,
		or.ReductionPercent,
		or.Stats.StringsInterned,
		or.Stats.BytesSavedStrings,
		or.Stats.SlicesOptimized,
		or.Stats.StatesOptimized,
		or.Stats.TransitionsOptimized,
	)
}

// StateChartMemoryIndex provides memory-efficient indexing for large statecharts.
type StateChartMemoryIndex struct {
	// Compact representation using arrays instead of maps for better memory efficiency
	stateLabels      []string
	stateIndices     map[string]int32 // Use int32 to save memory
	stateParents     []int32
	stateDepths      []uint8 // Most statecharts won't exceed 255 levels
	stateChildCounts []uint16
	stateFirstChild  []int32

	// Transition index
	transitionFromIndices []int32
	transitionToIndices   []int32
	transitionEvents      []string
	transitionCount       int32

	// Memory stats
	totalMemory int64
}

// BuildMemoryEfficientIndex creates a memory-efficient index for large statecharts.
func BuildMemoryEfficientIndex(statechart *sc.Statechart) *StateChartMemoryIndex {
	index := &StateChartMemoryIndex{
		stateIndices: make(map[string]int32),
	}

	// First pass: count states
	stateCount := countStates(statechart.RootState)

	// Pre-allocate arrays
	index.stateLabels = make([]string, 0, stateCount)
	index.stateParents = make([]int32, 0, stateCount)
	index.stateDepths = make([]uint8, 0, stateCount)
	index.stateChildCounts = make([]uint16, 0, stateCount)
	index.stateFirstChild = make([]int32, 0, stateCount)

	// Build state index
	index.indexStateCompact(statechart.RootState, -1, 0)

	// Build transition index
	index.indexTransitionsCompact(statechart.Transitions)

	// Calculate memory usage
	index.calculateMemoryUsage()

	return index
}

// indexStateCompact builds a compact state index.
func (index *StateChartMemoryIndex) indexStateCompact(state *sc.State, parentIdx int32, depth uint8) int32 {
	if state == nil {
		return -1
	}

	// Add state to arrays
	stateIdx := int32(len(index.stateLabels))
	index.stateLabels = append(index.stateLabels, state.Label)
	index.stateParents = append(index.stateParents, parentIdx)
	index.stateDepths = append(index.stateDepths, depth)
	index.stateChildCounts = append(index.stateChildCounts, uint16(len(state.Children)))

	// Map label to index
	index.stateIndices[state.Label] = stateIdx

	// Record first child index
	if len(state.Children) > 0 {
		index.stateFirstChild = append(index.stateFirstChild, int32(len(index.stateLabels)))
	} else {
		index.stateFirstChild = append(index.stateFirstChild, -1)
	}

	// Index children
	for _, child := range state.Children {
		index.indexStateCompact(child, stateIdx, depth+1)
	}

	return stateIdx
}

// indexTransitionsCompact builds a compact transition index.
func (index *StateChartMemoryIndex) indexTransitionsCompact(transitions []*sc.Transition) {
	// Pre-allocate
	totalTransitions := 0
	for _, t := range transitions {
		totalTransitions += len(t.From) * len(t.To)
	}

	index.transitionFromIndices = make([]int32, 0, totalTransitions)
	index.transitionToIndices = make([]int32, 0, totalTransitions)
	index.transitionEvents = make([]string, 0, totalTransitions)

	// Build index
	for _, t := range transitions {
		for _, from := range t.From {
			fromIdx, exists := index.stateIndices[from]
			if !exists {
				continue
			}

			for _, to := range t.To {
				toIdx, exists := index.stateIndices[to]
				if !exists {
					continue
				}

				index.transitionFromIndices = append(index.transitionFromIndices, fromIdx)
				index.transitionToIndices = append(index.transitionToIndices, toIdx)
				index.transitionEvents = append(index.transitionEvents, t.Event)
				index.transitionCount++
			}
		}
	}
}

// GetStateInfo returns information about a state using the compact index.
func (index *StateChartMemoryIndex) GetStateInfo(label string) (parentLabel string, depth uint8, childCount uint16, found bool) {
	idx, exists := index.stateIndices[label]
	if !exists {
		return "", 0, 0, false
	}

	parentIdx := index.stateParents[idx]
	if parentIdx >= 0 {
		parentLabel = index.stateLabels[parentIdx]
	}

	return parentLabel, index.stateDepths[idx], index.stateChildCounts[idx], true
}

// GetTransitionsFrom returns transitions from a specific state.
func (index *StateChartMemoryIndex) GetTransitionsFrom(label string) []TransitionInfo {
	fromIdx, exists := index.stateIndices[label]
	if !exists {
		return nil
	}

	var transitions []TransitionInfo
	for i := int32(0); i < index.transitionCount; i++ {
		if index.transitionFromIndices[i] == fromIdx {
			toLabel := index.stateLabels[index.transitionToIndices[i]]
			transitions = append(transitions, TransitionInfo{
				From:  label,
				To:    toLabel,
				Event: index.transitionEvents[i],
			})
		}
	}

	return transitions
}

// TransitionInfo represents compact transition information.
type TransitionInfo struct {
	From  string
	To    string
	Event string
}

// calculateMemoryUsage calculates the total memory used by the index.
func (index *StateChartMemoryIndex) calculateMemoryUsage() {
	index.totalMemory = 0

	// State arrays
	index.totalMemory += int64(len(index.stateLabels)) * int64(unsafe.Sizeof(""))
	for _, label := range index.stateLabels {
		index.totalMemory += int64(len(label))
	}

	index.totalMemory += int64(len(index.stateParents)) * 4     // int32
	index.totalMemory += int64(len(index.stateDepths)) * 1      // uint8
	index.totalMemory += int64(len(index.stateChildCounts)) * 2 // uint16
	index.totalMemory += int64(len(index.stateFirstChild)) * 4  // int32

	// State index map
	index.totalMemory += int64(len(index.stateIndices)) * (int64(unsafe.Sizeof("")) + 4)

	// Transition arrays
	index.totalMemory += int64(len(index.transitionFromIndices)) * 4
	index.totalMemory += int64(len(index.transitionToIndices)) * 4
	index.totalMemory += int64(len(index.transitionEvents)) * int64(unsafe.Sizeof(""))
	for _, event := range index.transitionEvents {
		index.totalMemory += int64(len(event))
	}
}

// GetMemoryUsage returns the total memory used by the index.
func (index *StateChartMemoryIndex) GetMemoryUsage() int64 {
	return index.totalMemory
}

// countStates recursively counts the number of states.
func countStates(state *sc.State) int {
	if state == nil {
		return 0
	}

	count := 1
	for _, child := range state.Children {
		count += countStates(child)
	}

	return count
}

// LargeStatechartValidator provides memory-efficient validation for large statecharts.
type LargeStatechartValidator struct {
	index     *StateChartMemoryIndex
	optimizer *LargeStatechartMemoryOptimizer
	pool      *ObjectPool
}

// NewLargeStatechartValidator creates a validator optimized for large statecharts.
func NewLargeStatechartValidator() *LargeStatechartValidator {
	return &LargeStatechartValidator{
		optimizer: NewLargeStatechartMemoryOptimizer(),
		pool:      NewObjectPool(),
	}
}

// ValidateEfficiently performs memory-efficient validation of large statecharts.
func (lsv *LargeStatechartValidator) ValidateEfficiently(statechart *sc.Statechart) error {
	// Build compact index
	lsv.index = BuildMemoryEfficientIndex(statechart)

	// Perform validation using the index
	// This avoids traversing the full statechart multiple times

	// Check for duplicate state labels efficiently
	if err := lsv.checkDuplicateStates(); err != nil {
		return err
	}

	// Validate transitions efficiently
	if err := lsv.validateTransitionsEfficiently(statechart.Transitions); err != nil {
		return err
	}

	// Additional validations...

	return nil
}

// checkDuplicateStates checks for duplicate state labels using the index.
func (lsv *LargeStatechartValidator) checkDuplicateStates() error {
	seen := make(map[string]bool, len(lsv.index.stateLabels))

	for _, label := range lsv.index.stateLabels {
		if seen[label] {
			return fmt.Errorf("duplicate state label: %s", label)
		}
		seen[label] = true
	}

	return nil
}

// validateTransitionsEfficiently validates transitions using the index.
func (lsv *LargeStatechartValidator) validateTransitionsEfficiently(transitions []*sc.Transition) error {
	for _, t := range transitions {
		// Check that all source states exist
		for _, from := range t.From {
			if _, exists := lsv.index.stateIndices[from]; !exists {
				return fmt.Errorf("transition source state not found: %s", from)
			}
		}

		// Check that all target states exist
		for _, to := range t.To {
			if _, exists := lsv.index.stateIndices[to]; !exists {
				return fmt.Errorf("transition target state not found: %s", to)
			}
		}
	}

	return nil
}

// ConcurrentStatechartOptimizer provides concurrent optimization for very large statecharts.
type ConcurrentStatechartOptimizer struct {
	workers   int
	optimizer *LargeStatechartMemoryOptimizer
	pool      *ObjectPool
}

// NewConcurrentStatechartOptimizer creates a concurrent optimizer.
func NewConcurrentStatechartOptimizer(workers int) *ConcurrentStatechartOptimizer {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	return &ConcurrentStatechartOptimizer{
		workers:   workers,
		optimizer: NewLargeStatechartMemoryOptimizer(),
		pool:      NewObjectPool(),
	}
}

// OptimizeConcurrently performs concurrent optimization of large statecharts.
func (cso *ConcurrentStatechartOptimizer) OptimizeConcurrently(statechart *sc.Statechart) (*sc.Statechart, OptimizationReport) {
	start := time.Now()

	// For very large statecharts, parallelize the optimization
	optimized := &sc.Statechart{}

	var wg sync.WaitGroup

	// Optimize root state in main goroutine
	optimized.RootState = cso.optimizer.optimizeStateDeep(statechart.RootState)

	// Optimize transitions concurrently
	if len(statechart.Transitions) > 100 {
		// Split transitions into chunks
		chunkSize := len(statechart.Transitions) / cso.workers
		if chunkSize < 10 {
			chunkSize = 10
		}

		transitionChunks := make([][]*sc.Transition, 0)
		for i := 0; i < len(statechart.Transitions); i += chunkSize {
			end := i + chunkSize
			if end > len(statechart.Transitions) {
				end = len(statechart.Transitions)
			}
			transitionChunks = append(transitionChunks, statechart.Transitions[i:end])
		}

		// Process chunks concurrently
		results := make([][]*sc.Transition, len(transitionChunks))
		for i, chunk := range transitionChunks {
			wg.Add(1)
			go func(idx int, transitions []*sc.Transition) {
				defer wg.Done()
				results[idx] = cso.optimizer.optimizeTransitionsDeep(transitions)
			}(i, chunk)
		}

		wg.Wait()

		// Merge results
		optimized.Transitions = make([]*sc.Transition, 0, len(statechart.Transitions))
		for _, result := range results {
			optimized.Transitions = append(optimized.Transitions, result...)
		}
	} else {
		optimized.Transitions = cso.optimizer.optimizeTransitionsDeep(statechart.Transitions)
	}

	// Optimize events
	optimized.Events = cso.optimizer.optimizeEventsDeep(statechart.Events)

	report := OptimizationReport{
		Duration: time.Since(start),
		Stats:    cso.optimizer.stats,
	}

	return optimized, report
}

// MemoryBudgetManager manages memory usage within specified budgets.
type MemoryBudgetManager struct {
	maxMemory        uint64
	warningThreshold uint64
	currentUsage     uint64
	mu               sync.RWMutex
	callbacks        []MemoryBudgetCallback
}

// MemoryBudgetCallback is called when memory usage changes.
type MemoryBudgetCallback func(current, max uint64, exceeded bool)

// NewMemoryBudgetManager creates a new memory budget manager.
func NewMemoryBudgetManager(maxMemory uint64) *MemoryBudgetManager {
	return &MemoryBudgetManager{
		maxMemory:        maxMemory,
		warningThreshold: uint64(float64(maxMemory) * 0.8),
		callbacks:        make([]MemoryBudgetCallback, 0),
	}
}

// CheckBudget checks if an operation would exceed the memory budget.
func (mbm *MemoryBudgetManager) CheckBudget(estimatedSize uint64) (bool, uint64) {
	mbm.mu.RLock()
	defer mbm.mu.RUnlock()

	if mbm.currentUsage+estimatedSize > mbm.maxMemory {
		return false, mbm.maxMemory - mbm.currentUsage
	}

	return true, 0
}

// AllocateMemory records memory allocation.
func (mbm *MemoryBudgetManager) AllocateMemory(size uint64) error {
	mbm.mu.Lock()

	if mbm.currentUsage+size > mbm.maxMemory {
		mbm.mu.Unlock()
		return fmt.Errorf("memory budget exceeded: requested %d, available %d", size, mbm.maxMemory-mbm.currentUsage)
	}

	mbm.currentUsage += size
	currentUsage := mbm.currentUsage
	maxMemory := mbm.maxMemory
	exceeded := mbm.currentUsage >= mbm.warningThreshold
	callbacks := append([]MemoryBudgetCallback(nil), mbm.callbacks...)
	mbm.mu.Unlock()

	// Notify callbacks synchronously to provide deterministic behavior.
	for _, cb := range callbacks {
		cb(currentUsage, maxMemory, exceeded)
	}

	return nil
}

// ReleaseMemory records memory release.
func (mbm *MemoryBudgetManager) ReleaseMemory(size uint64) {
	mbm.mu.Lock()
	defer mbm.mu.Unlock()

	if size > mbm.currentUsage {
		mbm.currentUsage = 0
	} else {
		mbm.currentUsage -= size
	}
}

// AddCallback adds a memory budget callback.
func (mbm *MemoryBudgetManager) AddCallback(cb MemoryBudgetCallback) {
	mbm.mu.Lock()
	defer mbm.mu.Unlock()
	mbm.callbacks = append(mbm.callbacks, cb)
}

// GetCurrentUsage returns the current memory usage.
func (mbm *MemoryBudgetManager) GetCurrentUsage() uint64 {
	mbm.mu.RLock()
	defer mbm.mu.RUnlock()
	return mbm.currentUsage
}

// StatechartSizeEstimator estimates memory requirements for statechart operations.
type StatechartSizeEstimator struct {
	averageStateSize      uint64
	averageTransitionSize uint64
	averageEventSize      uint64
}

// NewStatechartSizeEstimator creates a new size estimator.
func NewStatechartSizeEstimator() *StatechartSizeEstimator {
	return &StatechartSizeEstimator{
		averageStateSize:      200, // bytes
		averageTransitionSize: 150, // bytes
		averageEventSize:      50,  // bytes
	}
}

// EstimateStatechartMemory estimates the memory required for a statechart.
func (sse *StatechartSizeEstimator) EstimateStatechartMemory(stateCount, transitionCount, eventCount int) uint64 {
	estimate := uint64(0)
	estimate += uint64(stateCount) * sse.averageStateSize
	estimate += uint64(transitionCount) * sse.averageTransitionSize
	estimate += uint64(eventCount) * sse.averageEventSize

	// Add overhead for internal structures (20%)
	estimate = uint64(float64(estimate) * 1.2)

	return estimate
}

// EstimateMachineMemory estimates memory for a machine instance.
func (sse *StatechartSizeEstimator) EstimateMachineMemory(statechart *sc.Statechart, historySize int) uint64 {
	// Base statechart size
	stateCount := countStates(statechart.RootState)
	estimate := sse.EstimateStatechartMemory(stateCount, len(statechart.Transitions), len(statechart.Events))

	// Configuration overhead
	estimate += uint64(stateCount) * 50 // State references in configuration

	// History overhead
	estimate += uint64(historySize) * 500 // Approximate step size

	// Context and runtime overhead
	estimate += 10 * 1024 // 10KB baseline

	return estimate
}

// MemoryPooledMachineFactory creates machines with pooled memory allocation.
type MemoryPooledMachineFactory struct {
	pool              *ObjectPool
	optimizer         *LargeStatechartMemoryOptimizer
	budgetManager     *MemoryBudgetManager
	sizeEstimator     *StatechartSizeEstimator
	preOptimizedCache map[string]*sc.Statechart
	cacheMu           sync.RWMutex
}

// NewMemoryPooledMachineFactory creates a new factory with memory pooling.
func NewMemoryPooledMachineFactory(maxMemory uint64) *MemoryPooledMachineFactory {
	return &MemoryPooledMachineFactory{
		pool:              NewObjectPool(),
		optimizer:         NewLargeStatechartMemoryOptimizer(),
		budgetManager:     NewMemoryBudgetManager(maxMemory),
		sizeEstimator:     NewStatechartSizeEstimator(),
		preOptimizedCache: make(map[string]*sc.Statechart),
	}
}

// CreateOptimizedMachine creates a memory-optimized machine instance.
func (mpmf *MemoryPooledMachineFactory) CreateOptimizedMachine(statechart *sc.Statechart, id string, historyLimit int) (*semantics.MachineWrapper, error) {
	// Estimate memory requirements
	stateCount := countStates(statechart.RootState)
	estimatedSize := mpmf.sizeEstimator.EstimateMachineMemory(statechart, historyLimit)

	// Check budget
	if ok, available := mpmf.budgetManager.CheckBudget(estimatedSize); !ok {
		return nil, fmt.Errorf("insufficient memory budget: need %d bytes, only %d available", estimatedSize, available)
	}

	// Check if we have a pre-optimized version
	cacheKey := fmt.Sprintf("%p", statechart) // Simple pointer-based cache key

	mpmf.cacheMu.RLock()
	optimized, exists := mpmf.preOptimizedCache[cacheKey]
	mpmf.cacheMu.RUnlock()

	if !exists {
		// Optimize the statechart
		optimized, _ = mpmf.optimizer.OptimizeLargeStatechart(statechart)

		// Cache if it's large enough
		if stateCount > 100 {
			mpmf.cacheMu.Lock()
			mpmf.preOptimizedCache[cacheKey] = optimized
			mpmf.cacheMu.Unlock()
		}
	}

	// Create the machine
	machine, err := semantics.NewMachine(semantics.NewStatechart(optimized), id, nil)
	if err != nil {
		return nil, err
	}

	// Allocate memory budget
	if err := mpmf.budgetManager.AllocateMemory(estimatedSize); err != nil {
		return nil, err
	}

	// Wrap with optimization
	wrapped := &OptimizedMachineWrapper{
		MachineWrapper: machine,
		pool:           mpmf.pool,
		historyLimit:   historyLimit,
		compactFreq:    max(10, historyLimit/10),
	}

	return wrapped.MachineWrapper, nil
}

// ReleaseMachine releases a machine and its memory.
func (mpmf *MemoryPooledMachineFactory) ReleaseMachine(machine *semantics.MachineWrapper, estimatedSize uint64) {
	// Release memory budget
	mpmf.budgetManager.ReleaseMemory(estimatedSize)

	// Machine cleanup would go here
}

// GetMemoryStats returns current memory statistics.
func (mpmf *MemoryPooledMachineFactory) GetMemoryStats() MemoryPoolStats {
	return MemoryPoolStats{
		CurrentUsage:  mpmf.budgetManager.GetCurrentUsage(),
		MaxMemory:     mpmf.budgetManager.maxMemory,
		CachedCharts:  len(mpmf.preOptimizedCache),
		Optimizations: mpmf.optimizer.stats,
	}
}

// MemoryPoolStats contains memory pool statistics.
type MemoryPoolStats struct {
	CurrentUsage  uint64
	MaxMemory     uint64
	CachedCharts  int
	Optimizations OptimizationStats
}

// Helper functions

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MemoryProfiledStatechart wraps a statechart with automatic memory profiling.
type MemoryProfiledStatechart struct {
	*semantics.Statechart
	profiler      *StatechartProfiler
	originalSize  int64
	optimizedSize int64
}

// NewMemoryProfiledStatechart creates a statechart with built-in profiling.
func NewMemoryProfiledStatechart(statechart *sc.Statechart, optimize bool) *MemoryProfiledStatechart {
	profiler := NewStatechartProfiler()
	wrapped := semantics.NewStatechart(statechart)

	estimator := NewLargeStatechartMemoryOptimizer()
	originalSize := estimator.estimateStatechartSize(statechart)

	optimizedSize := originalSize
	if optimize {
		optimized, report := estimator.OptimizeLargeStatechart(statechart)
		wrapped = semantics.NewStatechart(optimized)
		optimizedSize = report.OptimizedSize
	}

	return &MemoryProfiledStatechart{
		Statechart:    wrapped,
		profiler:      profiler,
		originalSize:  originalSize,
		optimizedSize: optimizedSize,
	}
}

// GetMemoryStats returns memory statistics for the statechart.
func (mps *MemoryProfiledStatechart) GetMemoryStats() StatechartMemoryStats {
	return StatechartMemoryStats{
		OriginalSize:     mps.originalSize,
		OptimizedSize:    mps.optimizedSize,
		ReductionPercent: float64(mps.originalSize-mps.optimizedSize) / float64(mps.originalSize) * 100,
	}
}

// StatechartMemoryStats contains memory statistics for a statechart.
type StatechartMemoryStats struct {
	OriginalSize     int64
	OptimizedSize    int64
	ReductionPercent float64
}

// LargeStatechartBenchmarkSuite provides benchmarks for large statechart operations.
type LargeStatechartBenchmarkSuite struct {
	sizes     []StatechartSize
	optimizer *LargeStatechartMemoryOptimizer
	profiler  *StatechartProfiler
}

// StatechartSize defines a statechart size for benchmarking.
type StatechartSize struct {
	Name            string
	StateCount      int
	TransitionCount int
	Depth           int
}

// NewLargeStatechartBenchmarkSuite creates a benchmark suite for large statecharts.
func NewLargeStatechartBenchmarkSuite() *LargeStatechartBenchmarkSuite {
	return &LargeStatechartBenchmarkSuite{
		sizes: []StatechartSize{
			{"Small", 10, 20, 3},
			{"Medium", 100, 200, 5},
			{"Large", 1000, 2000, 7},
			{"XLarge", 5000, 10000, 10},
			{"XXLarge", 10000, 20000, 12},
			{"Massive", 50000, 100000, 15},
		},
		optimizer: NewLargeStatechartMemoryOptimizer(),
		profiler:  NewStatechartProfiler(),
	}
}

// Note: The actual benchmark implementation should be in a _test.go file.
// This provides the infrastructure for benchmarking.

// createDeepStatechart creates a deep hierarchical statechart for testing.
func createDeepStatechart(targetStates, targetTransitions, maxDepth int) *sc.Statechart {
	builder := testing.NewStatechartBuilder()

	// Create a balanced tree structure
	statesPerLevel := make([]int, maxDepth)
	totalStates := 1 // root

	// Calculate states per level to reach target
	for level := 1; level < maxDepth && totalStates < targetStates; level++ {
		statesAtLevel := min(targetStates-totalStates, int(float64(targetStates-totalStates)/float64(maxDepth-level)))
		statesPerLevel[level] = statesAtLevel
		totalStates += statesAtLevel
	}

	// Build the state tree
	rootState := createStateTree("root", 0, maxDepth, statesPerLevel, &totalStates, targetStates)
	builder.WithRootState(rootState)

	// Add transitions
	allStates := collectAllStates(rootState)
	transitionsAdded := 0

	for i := 0; i < len(allStates)-1 && transitionsAdded < targetTransitions; i++ {
		fromState := allStates[i]
		toState := allStates[(i+1)%len(allStates)]

		transition := testing.NewTransitionBuilder(fmt.Sprintf("t_%d", transitionsAdded)).
			From(fromState.Label).
			To(toState.Label).
			OnEvent(fmt.Sprintf("event_%d", transitionsAdded%100)).
			Build()

		builder.WithTransition(transition)
		transitionsAdded++
	}

	return builder.Build()
}

// createStateTree recursively creates a state tree.
func createStateTree(prefix string, currentDepth, maxDepth int, statesPerLevel []int, totalStates *int, targetStates int) *sc.State {
	stateBuilder := testing.NewStateBuilder(prefix)

	if currentDepth < maxDepth-1 && *totalStates < targetStates && currentDepth+1 < len(statesPerLevel) {
		statesAtNextLevel := statesPerLevel[currentDepth+1]
		if statesAtNextLevel > 0 {
			stateBuilder.WithType(sc.StateTypeNormal)

			childrenPerNode := max(1, statesAtNextLevel/max(1, statesPerLevel[currentDepth]))
			children := make([]*sc.State, 0, childrenPerNode)

			for i := 0; i < childrenPerNode && *totalStates < targetStates; i++ {
				childPrefix := fmt.Sprintf("%s_%d", prefix, i)
				child := createStateTree(childPrefix, currentDepth+1, maxDepth, statesPerLevel, totalStates, targetStates)
				children = append(children, child)
				(*totalStates)++
			}

			if len(children) > 0 {
				children[0].IsInitial = true
				stateBuilder.WithChildren(children...)
			}
		}
	}

	return stateBuilder.Build()
}

// collectAllStates collects all states in a statechart.
func collectAllStates(state *sc.State) []*sc.State {
	if state == nil {
		return nil
	}

	states := []*sc.State{state}
	for _, child := range state.Children {
		states = append(states, collectAllStates(child)...)
	}

	return states
}

// MemoryEfficientStatechartCache provides a memory-efficient cache for statecharts.
type MemoryEfficientStatechartCache struct {
	maxSize       int
	maxMemory     uint64
	currentMemory uint64
	entries       map[string]*CacheEntry
	lru           *LRUList
	mu            sync.RWMutex
	estimator     *StatechartSizeEstimator
}

// CacheEntry represents a cached statechart.
type CacheEntry struct {
	key        string
	statechart *sc.Statechart
	size       uint64
	lastAccess time.Time
	hitCount   int64
	prev       *CacheEntry
	next       *CacheEntry
}

// LRUList manages least recently used entries.
type LRUList struct {
	head *CacheEntry
	tail *CacheEntry
}

// NewMemoryEfficientStatechartCache creates a new cache with memory limits.
func NewMemoryEfficientStatechartCache(maxSize int, maxMemory uint64) *MemoryEfficientStatechartCache {
	return &MemoryEfficientStatechartCache{
		maxSize:   maxSize,
		maxMemory: maxMemory,
		entries:   make(map[string]*CacheEntry),
		lru:       &LRUList{},
		estimator: NewStatechartSizeEstimator(),
	}
}

// Get retrieves a statechart from the cache.
func (cache *MemoryEfficientStatechartCache) Get(key string) (*sc.Statechart, bool) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	entry, exists := cache.entries[key]
	if !exists {
		return nil, false
	}

	// Update access time and move to front
	entry.lastAccess = time.Now()
	entry.hitCount++
	cache.lru.moveToFront(entry)

	return entry.statechart, true
}

// Put adds a statechart to the cache.
func (cache *MemoryEfficientStatechartCache) Put(key string, statechart *sc.Statechart) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	// Estimate size
	stateCount := countStates(statechart.RootState)
	size := cache.estimator.EstimateStatechartMemory(stateCount, len(statechart.Transitions), len(statechart.Events))

	// Check if we need to evict entries
	for (len(cache.entries) >= cache.maxSize || cache.currentMemory+size > cache.maxMemory) && cache.lru.tail != nil {
		cache.evictLRU()
	}

	// Add new entry
	entry := &CacheEntry{
		key:        key,
		statechart: statechart,
		size:       size,
		lastAccess: time.Now(),
	}

	cache.entries[key] = entry
	cache.lru.addToFront(entry)
	cache.currentMemory += size
}

// evictLRU removes the least recently used entry.
func (cache *MemoryEfficientStatechartCache) evictLRU() {
	if cache.lru.tail == nil {
		return
	}

	entry := cache.lru.tail
	cache.lru.remove(entry)
	delete(cache.entries, entry.key)
	cache.currentMemory -= entry.size
}

// moveToFront moves an entry to the front of the LRU list.
func (lru *LRUList) moveToFront(entry *CacheEntry) {
	lru.remove(entry)
	lru.addToFront(entry)
}

// remove removes an entry from the LRU list.
func (lru *LRUList) remove(entry *CacheEntry) {
	if entry.prev != nil {
		entry.prev.next = entry.next
	} else {
		lru.head = entry.next
	}

	if entry.next != nil {
		entry.next.prev = entry.prev
	} else {
		lru.tail = entry.prev
	}

	entry.prev = nil
	entry.next = nil
}

// addToFront adds an entry to the front of the LRU list.
func (lru *LRUList) addToFront(entry *CacheEntry) {
	entry.prev = nil
	entry.next = lru.head

	if lru.head != nil {
		lru.head.prev = entry
	}

	lru.head = entry

	if lru.tail == nil {
		lru.tail = entry
	}
}

// GetStats returns cache statistics.
func (cache *MemoryEfficientStatechartCache) GetStats() CacheStats {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	totalHits := int64(0)
	for _, entry := range cache.entries {
		totalHits += entry.hitCount
	}

	return CacheStats{
		Size:           len(cache.entries),
		MemoryUsage:    cache.currentMemory,
		MaxMemory:      cache.maxMemory,
		TotalHits:      totalHits,
		UtilizationPct: float64(cache.currentMemory) / float64(cache.maxMemory) * 100,
	}
}

// CacheStats contains cache statistics.
type CacheStats struct {
	Size           int
	MemoryUsage    uint64
	MaxMemory      uint64
	TotalHits      int64
	UtilizationPct float64
}
