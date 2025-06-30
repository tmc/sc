package profiling

import (
	"sync"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

// ObjectPool provides efficient object reuse for statechart operations.
type ObjectPool struct {
	configPool      sync.Pool
	contextPool     sync.Pool
	transitionPool  sync.Pool
	stateRefPool    sync.Pool
	stepPool        sync.Pool
	eventPool       sync.Pool
	actionPool      sync.Pool
}

// NewObjectPool creates a new object pool with optimized allocators.
func NewObjectPool() *ObjectPool {
	return &ObjectPool{
		configPool: sync.Pool{
			New: func() interface{} {
				return &sc.Configuration{
					States: make([]*sc.StateRef, 0, 8), // Pre-allocate for common case
				}
			},
		},
		contextPool: sync.Pool{
			New: func() interface{} {
				return &structpb.Struct{
					Fields: make(map[string]*structpb.Value, 16), // Pre-allocate common size
				}
			},
		},
		transitionPool: sync.Pool{
			New: func() interface{} {
				return &sc.Transition{
					From:    make([]string, 0, 2),
					To:      make([]string, 0, 2),
					Actions: make([]*sc.Action, 0, 2),
				}
			},
		},
		stateRefPool: sync.Pool{
			New: func() interface{} {
				return &sc.StateRef{}
			},
		},
		stepPool: sync.Pool{
			New: func() interface{} {
				return &sc.Step{
					Events:      make([]*sc.Event, 0, 1),
					Transitions: make([]*sc.Transition, 0, 4),
				}
			},
		},
		eventPool: sync.Pool{
			New: func() interface{} {
				return &sc.Event{}
			},
		},
		actionPool: sync.Pool{
			New: func() interface{} {
				return &sc.Action{}
			},
		},
	}
}

// GetConfiguration returns a pooled configuration object.
func (op *ObjectPool) GetConfiguration() *sc.Configuration {
	config := op.configPool.Get().(*sc.Configuration)
	// Reset the slice but keep capacity
	config.States = config.States[:0]
	return config
}

// PutConfiguration returns a configuration object to the pool.
func (op *ObjectPool) PutConfiguration(config *sc.Configuration) {
	if config != nil {
		op.configPool.Put(config)
	}
}

// GetContext returns a pooled context object.
func (op *ObjectPool) GetContext() *structpb.Struct {
	ctx := op.contextPool.Get().(*structpb.Struct)
	// Clear the map but keep capacity
	for k := range ctx.Fields {
		delete(ctx.Fields, k)
	}
	return ctx
}

// PutContext returns a context object to the pool.
func (op *ObjectPool) PutContext(ctx *structpb.Struct) {
	if ctx != nil {
		op.contextPool.Put(ctx)
	}
}

// GetTransition returns a pooled transition object.
func (op *ObjectPool) GetTransition() *sc.Transition {
	t := op.transitionPool.Get().(*sc.Transition)
	// Reset fields
	t.Label = ""
	t.Event = ""
	t.From = t.From[:0]
	t.To = t.To[:0]
	t.Actions = t.Actions[:0]
	t.Guard = nil
	return t
}

// PutTransition returns a transition object to the pool.
func (op *ObjectPool) PutTransition(transition *sc.Transition) {
	if transition != nil {
		op.transitionPool.Put(transition)
	}
}

// GetStateRef returns a pooled state reference object.
func (op *ObjectPool) GetStateRef() *sc.StateRef {
	ref := op.stateRefPool.Get().(*sc.StateRef)
	ref.Label = ""
	return ref
}

// PutStateRef returns a state reference object to the pool.
func (op *ObjectPool) PutStateRef(ref *sc.StateRef) {
	if ref != nil {
		op.stateRefPool.Put(ref)
	}
}

// GetStep returns a pooled step object.
func (op *ObjectPool) GetStep() *sc.Step {
	step := op.stepPool.Get().(*sc.Step)
	step.Events = step.Events[:0]
	step.Transitions = step.Transitions[:0]
	step.StartingConfiguration = nil
	step.ResultingConfiguration = nil
	step.Context = nil
	return step
}

// PutStep returns a step object to the pool.
func (op *ObjectPool) PutStep(step *sc.Step) {
	if step != nil {
		op.stepPool.Put(step)
	}
}

// GetEvent returns a pooled event object.
func (op *ObjectPool) GetEvent() *sc.Event {
	event := op.eventPool.Get().(*sc.Event)
	event.Label = ""
	event.Parameters = nil
	return event
}

// PutEvent returns an event object to the pool.
func (op *ObjectPool) PutEvent(event *sc.Event) {
	if event != nil {
		op.eventPool.Put(event)
	}
}

// GetAction returns a pooled action object.
func (op *ObjectPool) GetAction() *sc.Action {
	action := op.actionPool.Get().(*sc.Action)
	action.Label = ""
	return action
}

// PutAction returns an action object to the pool.
func (op *ObjectPool) PutAction(action *sc.Action) {
	if action != nil {
		op.actionPool.Put(action)
	}
}

// OptimizedMachineWrapper provides memory-optimized machine operations.
type OptimizedMachineWrapper struct {
	*semantics.MachineWrapper
	pool         *ObjectPool
	historyLimit int
	compactFreq  int
	stepCount    int
}

// NewOptimizedMachine creates a memory-optimized machine wrapper.
func NewOptimizedMachine(statechart *semantics.Statechart, id string, pool *ObjectPool) (*OptimizedMachineWrapper, error) {
	machine, err := semantics.NewMachine(statechart, id, nil)
	if err != nil {
		return nil, err
	}
	
	return &OptimizedMachineWrapper{
		MachineWrapper: machine,
		pool:           pool,
		historyLimit:   1000, // Limit step history to prevent unbounded growth
		compactFreq:    100,  // Compact history every 100 steps
	}, nil
}

// OptimizedStep performs a step with memory optimizations.
func (omw *OptimizedMachineWrapper) OptimizedStep(eventName string) (bool, error) {
	// Perform regular step
	stepped, err := omw.Step(eventName)
	
	if err == nil {
		omw.stepCount++
		
		// Periodic history compaction
		if omw.stepCount%omw.compactFreq == 0 {
			omw.compactHistory()
		}
	}
	
	return stepped, err
}

// compactHistory reduces memory usage by limiting step history.
func (omw *OptimizedMachineWrapper) compactHistory() {
	history := omw.GetStepHistory()
	
	if len(history) > omw.historyLimit {
		// Keep only the most recent steps
		keepCount := omw.historyLimit / 2
		compactedHistory := make([]*sc.Step, keepCount)
		copy(compactedHistory, history[len(history)-keepCount:])
		
		// Replace the history (note: this is a simplified approach)
		// In a real implementation, you might want to expose this functionality
		// through the MachineWrapper API
		omw.StepHistory = compactedHistory
	}
}

// SetHistoryLimit configures the maximum number of steps to retain.
func (omw *OptimizedMachineWrapper) SetHistoryLimit(limit int) {
	omw.historyLimit = limit
}

// SetCompactionFrequency configures how often to compact history.
func (omw *OptimizedMachineWrapper) SetCompactionFrequency(freq int) {
	omw.compactFreq = freq
}

// MemoryOptimizer provides strategies for reducing memory usage.
type MemoryOptimizer struct {
	pool             *ObjectPool
	compactionPolicy CompactionPolicy
	cachePolicy      CachePolicy
}

// CompactionPolicy defines when and how to compact data structures.
type CompactionPolicy struct {
	HistoryLimit     int           // Maximum items to keep in history
	CompactInterval  time.Duration // How often to perform compaction
	GCTriggerRatio   float64       // Trigger GC when memory usage exceeds this ratio
	EnableAutoGC     bool          // Automatically trigger GC
}

// CachePolicy defines caching behavior for computed values.
type CachePolicy struct {
	MaxCacheSize     int           // Maximum number of cached items
	TTL              time.Duration // Time to live for cached items
	EnableCache      bool          // Whether caching is enabled
}

// NewMemoryOptimizer creates a new memory optimizer.
func NewMemoryOptimizer() *MemoryOptimizer {
	return &MemoryOptimizer{
		pool: NewObjectPool(),
		compactionPolicy: CompactionPolicy{
			HistoryLimit:     1000,
			CompactInterval:  30 * time.Second,
			GCTriggerRatio:   0.8,
			EnableAutoGC:     true,
		},
		cachePolicy: CachePolicy{
			MaxCacheSize: 10000,
			TTL:          5 * time.Minute,
			EnableCache:  true,
		},
	}
}

// OptimizeStatechart applies memory optimizations to a statechart.
func (mo *MemoryOptimizer) OptimizeStatechart(statechart *sc.Statechart) *sc.Statechart {
	// Create optimized copy with interned strings and reduced allocations
	optimized := &sc.Statechart{
		RootState:   mo.optimizeState(statechart.RootState),
		Transitions: mo.optimizeTransitions(statechart.Transitions),
		Events:      mo.optimizeEvents(statechart.Events),
	}
	
	return optimized
}

// optimizeState optimizes a state structure for reduced memory usage.
func (mo *MemoryOptimizer) optimizeState(state *sc.State) *sc.State {
	if state == nil {
		return nil
	}
	
	optimized := &sc.State{
		Label:     state.Label, // String interning could be applied here
		Type:      state.Type,
		IsInitial: state.IsInitial,
		IsFinal:   state.IsFinal,
	}
	
	// Optimize children
	if len(state.Children) > 0 {
		optimized.Children = make([]*sc.State, len(state.Children))
		for i, child := range state.Children {
			optimized.Children[i] = mo.optimizeState(child)
		}
	}
	
	return optimized
}

// optimizeTransitions optimizes transition arrays for memory efficiency.
func (mo *MemoryOptimizer) optimizeTransitions(transitions []*sc.Transition) []*sc.Transition {
	if len(transitions) == 0 {
		return nil
	}
	
	// Pre-allocate with exact size
	optimized := make([]*sc.Transition, len(transitions))
	for i, t := range transitions {
		optimized[i] = &sc.Transition{
			Label:   t.Label,
			From:    mo.optimizeStringSlice(t.From),
			To:      mo.optimizeStringSlice(t.To),
			Event:   t.Event,
			Guard:   t.Guard,
			Actions: mo.optimizeActions(t.Actions),
		}
	}
	
	return optimized
}

// optimizeStringSlice reduces memory overhead of string slices.
func (mo *MemoryOptimizer) optimizeStringSlice(slice []string) []string {
	if len(slice) == 0 {
		return nil
	}
	
	// Create slice with exact capacity
	optimized := make([]string, len(slice))
	copy(optimized, slice)
	return optimized
}

// optimizeActions optimizes action arrays.
func (mo *MemoryOptimizer) optimizeActions(actions []*sc.Action) []*sc.Action {
	if len(actions) == 0 {
		return nil
	}
	
	optimized := make([]*sc.Action, len(actions))
	for i, action := range actions {
		optimized[i] = &sc.Action{
			Label: action.Label,
		}
	}
	
	return optimized
}

// optimizeEvents optimizes event arrays.
func (mo *MemoryOptimizer) optimizeEvents(events []*sc.Event) []*sc.Event {
	if len(events) == 0 {
		return nil
	}
	
	optimized := make([]*sc.Event, len(events))
	for i, event := range events {
		optimized[i] = &sc.Event{
			Label: event.Label,
			Parameters: event.Parameters,
		}
	}
	
	return optimized
}

// GetObjectPool returns the managed object pool.
func (mo *MemoryOptimizer) GetObjectPool() *ObjectPool {
	return mo.pool
}

// SetCompactionPolicy updates the compaction policy.
func (mo *MemoryOptimizer) SetCompactionPolicy(policy CompactionPolicy) {
	mo.compactionPolicy = policy
}

// SetCachePolicy updates the cache policy.
func (mo *MemoryOptimizer) SetCachePolicy(policy CachePolicy) {
	mo.cachePolicy = policy
}

// LargeStatechartOptimizer provides specialized optimizations for large statecharts.
type LargeStatechartOptimizer struct {
	*MemoryOptimizer
	indexCache    map[string]*StateIndex
	transitionMap map[string][]*sc.Transition
	mu            sync.RWMutex
}

// StateIndex provides fast lookups for state operations.
type StateIndex struct {
	StatesByLabel    map[string]*sc.State
	ParentMap        map[string]*sc.State
	DepthMap         map[string]int
	ChildrenMap      map[string][]*sc.State
	TransitionMap    map[string][]*sc.Transition
	LastUpdated      time.Time
}

// NewLargeStatechartOptimizer creates an optimizer for large statecharts.
func NewLargeStatechartOptimizer() *LargeStatechartOptimizer {
	return &LargeStatechartOptimizer{
		MemoryOptimizer: NewMemoryOptimizer(),
		indexCache:      make(map[string]*StateIndex),
		transitionMap:   make(map[string][]*sc.Transition),
	}
}

// BuildIndex creates optimized indexes for fast lookups.
func (lso *LargeStatechartOptimizer) BuildIndex(statechart *sc.Statechart) *StateIndex {
	lso.mu.Lock()
	defer lso.mu.Unlock()
	
	// Check if we have a cached index
	key := statechart.String() // This would need a proper hash in production
	if index, exists := lso.indexCache[key]; exists {
		if time.Since(index.LastUpdated) < 5*time.Minute {
			return index
		}
	}
	
	index := &StateIndex{
		StatesByLabel: make(map[string]*sc.State),
		ParentMap:     make(map[string]*sc.State),
		DepthMap:      make(map[string]int),
		ChildrenMap:   make(map[string][]*sc.State),
		TransitionMap: make(map[string][]*sc.Transition),
		LastUpdated:   time.Now(),
	}
	
	// Build state indexes
	lso.indexState(statechart.RootState, nil, 0, index)
	
	// Build transition indexes
	for _, transition := range statechart.Transitions {
		for _, fromState := range transition.From {
			index.TransitionMap[fromState] = append(index.TransitionMap[fromState], transition)
		}
	}
	
	lso.indexCache[key] = index
	return index
}

// indexState recursively builds indexes for a state and its children.
func (lso *LargeStatechartOptimizer) indexState(state *sc.State, parent *sc.State, depth int, index *StateIndex) {
	if state == nil {
		return
	}
	
	index.StatesByLabel[state.Label] = state
	index.ParentMap[state.Label] = parent
	index.DepthMap[state.Label] = depth
	
	if len(state.Children) > 0 {
		children := make([]*sc.State, len(state.Children))
		copy(children, state.Children)
		index.ChildrenMap[state.Label] = children
		
		for _, child := range state.Children {
			lso.indexState(child, state, depth+1, index)
		}
	}
}

// FastLookup provides O(1) lookups using the built index.
func (lso *LargeStatechartOptimizer) FastLookup(statechart *sc.Statechart, stateLabel string) (*sc.State, int, []*sc.State) {
	index := lso.BuildIndex(statechart)
	
	state := index.StatesByLabel[stateLabel]
	depth := index.DepthMap[stateLabel]
	children := index.ChildrenMap[stateLabel]
	
	return state, depth, children
}

// GetTransitionsForState returns transitions from a specific state using cached indexes.
func (lso *LargeStatechartOptimizer) GetTransitionsForState(statechart *sc.Statechart, stateLabel string) []*sc.Transition {
	index := lso.BuildIndex(statechart)
	return index.TransitionMap[stateLabel]
}

// MemoryEfficientMachine provides a memory-optimized machine implementation.
type MemoryEfficientMachine struct {
	statechart *sc.Statechart
	config     *sc.Configuration
	context    *structpb.Struct
	optimizer  *LargeStatechartOptimizer
	pool       *ObjectPool
	history    *RingBuffer
	mu         sync.RWMutex
}

// RingBuffer provides a fixed-size circular buffer for step history.
type RingBuffer struct {
	buffer []interface{}
	head   int
	tail   int
	size   int
	count  int
}

// NewRingBuffer creates a new ring buffer with the specified size.
func NewRingBuffer(size int) *RingBuffer {
	return &RingBuffer{
		buffer: make([]interface{}, size),
		size:   size,
	}
}

// Add adds an item to the ring buffer.
func (rb *RingBuffer) Add(item interface{}) {
	rb.buffer[rb.head] = item
	rb.head = (rb.head + 1) % rb.size
	
	if rb.count < rb.size {
		rb.count++
	} else {
		rb.tail = (rb.tail + 1) % rb.size
	}
}

// GetAll returns all items in the buffer.
func (rb *RingBuffer) GetAll() []interface{} {
	if rb.count == 0 {
		return nil
	}
	
	result := make([]interface{}, rb.count)
	for i := 0; i < rb.count; i++ {
		idx := (rb.tail + i) % rb.size
		result[i] = rb.buffer[idx]
	}
	
	return result
}

// Size returns the number of items in the buffer.
func (rb *RingBuffer) Size() int {
	return rb.count
}

// NewMemoryEfficientMachine creates a memory-efficient machine.
func NewMemoryEfficientMachine(statechart *sc.Statechart, historySize int) *MemoryEfficientMachine {
	optimizer := NewLargeStatechartOptimizer()
	
	return &MemoryEfficientMachine{
		statechart: optimizer.OptimizeStatechart(statechart),
		config:     &sc.Configuration{States: make([]*sc.StateRef, 0, 8)},
		context:    &structpb.Struct{Fields: make(map[string]*structpb.Value)},
		optimizer:  optimizer,
		pool:       optimizer.GetObjectPool(),
		history:    NewRingBuffer(historySize),
	}
}

// Step performs an optimized step operation.
func (mem *MemoryEfficientMachine) Step(eventName string) error {
	mem.mu.Lock()
	defer mem.mu.Unlock()
	
	// Use pooled objects for temporary allocations
	tempConfig := mem.pool.GetConfiguration()
	defer mem.pool.PutConfiguration(tempConfig)
	
	// Copy current configuration
	for _, state := range mem.config.States {
		ref := mem.pool.GetStateRef()
		ref.Label = state.Label
		tempConfig.States = append(tempConfig.States, ref)
	}
	
	// Process the step (simplified)
	// In a real implementation, this would use the optimized indexes
	// for fast transition lookups and state operations
	
	// Add to history using ring buffer
	step := mem.pool.GetStep()
	step.Events = append(step.Events, &sc.Event{Label: eventName})
	step.StartingConfiguration = tempConfig
	step.ResultingConfiguration = mem.config
	
	mem.history.Add(step)
	
	return nil
}

// GetHistory returns the step history.
func (mem *MemoryEfficientMachine) GetHistory() []*sc.Step {
	mem.mu.RLock()
	defer mem.mu.RUnlock()
	
	items := mem.history.GetAll()
	history := make([]*sc.Step, len(items))
	for i, item := range items {
		history[i] = item.(*sc.Step)
	}
	
	return history
}

// GetCurrentConfiguration returns the current configuration.
func (mem *MemoryEfficientMachine) GetCurrentConfiguration() *sc.Configuration {
	mem.mu.RLock()
	defer mem.mu.RUnlock()
	
	// Return a copy
	config := mem.pool.GetConfiguration()
	for _, state := range mem.config.States {
		ref := mem.pool.GetStateRef()
		ref.Label = state.Label
		config.States = append(config.States, ref)
	}
	
	return config
}

// MemoryMonitor provides real-time memory monitoring for statechart operations.
type MemoryMonitor struct {
	profiler   *StatechartProfiler
	thresholds map[string]uint64
	callbacks  map[string]func(MemorySnapshot)
	mu         sync.RWMutex
}

// NewMemoryMonitor creates a new memory monitor.
func NewMemoryMonitor() *MemoryMonitor {
	return &MemoryMonitor{
		profiler:   NewStatechartProfiler(),
		thresholds: make(map[string]uint64),
		callbacks:  make(map[string]func(MemorySnapshot)),
	}
}

// SetThreshold sets a memory threshold for monitoring.
func (mm *MemoryMonitor) SetThreshold(name string, bytes uint64, callback func(MemorySnapshot)) {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	
	mm.thresholds[name] = bytes
	mm.callbacks[name] = callback
}

// CheckThresholds checks if any thresholds have been exceeded.
func (mm *MemoryMonitor) CheckThresholds() {
	snapshot := mm.profiler.mp.TakeSnapshot("threshold_check")
	
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	
	for name, threshold := range mm.thresholds {
		if snapshot.Alloc > threshold {
			if callback, exists := mm.callbacks[name]; exists {
				go callback(snapshot) // Run callback asynchronously
			}
		}
	}
}

// StartMonitoring begins continuous memory monitoring.
func (mm *MemoryMonitor) StartMonitoring(interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		for range ticker.C {
			mm.CheckThresholds()
		}
	}()
}