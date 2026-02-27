// Package profiling provides memory profiling and optimization tools for statechart operations.
package profiling

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// MemorySnapshot represents a point-in-time memory usage measurement.
type MemorySnapshot struct {
	Timestamp   time.Time
	Alloc       uint64           // Current heap allocations in bytes
	TotalAlloc  uint64           // Cumulative heap allocations in bytes
	Sys         uint64           // Total system memory obtained from OS
	NumGC       uint32           // Number of completed GC cycles
	HeapObjects uint64           // Number of allocated heap objects
	StackInUse  uint64           // Stack memory in use
	MemStats    runtime.MemStats // Complete memory statistics
	Label       string           // Description of this snapshot
}

// MemoryProfiler tracks memory usage during statechart operations.
type MemoryProfiler struct {
	mu        sync.RWMutex
	snapshots []MemorySnapshot
	baseline  *MemorySnapshot
	enabled   bool
}

// NewMemoryProfiler creates a new memory profiler.
func NewMemoryProfiler() *MemoryProfiler {
	return &MemoryProfiler{
		snapshots: make([]MemorySnapshot, 0, 100),
		enabled:   true,
	}
}

// Enable turns on memory profiling.
func (mp *MemoryProfiler) Enable() {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	mp.enabled = true
}

// Disable turns off memory profiling.
func (mp *MemoryProfiler) Disable() {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	mp.enabled = false
}

// SetBaseline records the current memory state as a baseline for comparisons.
func (mp *MemoryProfiler) SetBaseline(label string) {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	if !mp.enabled {
		return
	}

	baseline := mp.takeSnapshotUnsafe(label)
	mp.baseline = &baseline
}

// TakeSnapshot records the current memory state.
func (mp *MemoryProfiler) TakeSnapshot(label string) MemorySnapshot {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	snapshot := mp.takeSnapshotUnsafe(label)

	if mp.enabled {
		mp.snapshots = append(mp.snapshots, snapshot)
	}

	return snapshot
}

// takeSnapshotUnsafe takes a memory snapshot without acquiring locks.
func (mp *MemoryProfiler) takeSnapshotUnsafe(label string) MemorySnapshot {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemorySnapshot{
		Timestamp:   time.Now(),
		Alloc:       m.Alloc,
		TotalAlloc:  m.TotalAlloc,
		Sys:         m.Sys,
		NumGC:       m.NumGC,
		HeapObjects: m.HeapObjects,
		StackInUse:  m.StackInuse,
		MemStats:    m,
		Label:       label,
	}
}

// GetSnapshots returns all recorded snapshots.
func (mp *MemoryProfiler) GetSnapshots() []MemorySnapshot {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	snapshots := make([]MemorySnapshot, len(mp.snapshots))
	copy(snapshots, mp.snapshots)
	return snapshots
}

// GetBaseline returns the baseline snapshot if set.
func (mp *MemoryProfiler) GetBaseline() *MemorySnapshot {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	if mp.baseline == nil {
		return nil
	}

	baseline := *mp.baseline
	return &baseline
}

// Clear removes all snapshots and baseline.
func (mp *MemoryProfiler) Clear() {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	mp.snapshots = mp.snapshots[:0]
	mp.baseline = nil
}

// MemoryDelta represents the difference between two memory snapshots.
type MemoryDelta struct {
	DeltaAlloc       int64
	DeltaTotalAlloc  int64
	DeltaSys         int64
	DeltaNumGC       int32
	DeltaHeapObjects int64
	DeltaStackInUse  int64
	Duration         time.Duration
	FromLabel        string
	ToLabel          string
}

// ComputeDelta calculates the memory usage difference between two snapshots.
func ComputeDelta(from, to MemorySnapshot) MemoryDelta {
	return MemoryDelta{
		DeltaAlloc:       int64(to.Alloc) - int64(from.Alloc),
		DeltaTotalAlloc:  int64(to.TotalAlloc) - int64(from.TotalAlloc),
		DeltaSys:         int64(to.Sys) - int64(from.Sys),
		DeltaNumGC:       int32(to.NumGC) - int32(from.NumGC),
		DeltaHeapObjects: int64(to.HeapObjects) - int64(from.HeapObjects),
		DeltaStackInUse:  int64(to.StackInUse) - int64(from.StackInUse),
		Duration:         to.Timestamp.Sub(from.Timestamp),
		FromLabel:        from.Label,
		ToLabel:          to.Label,
	}
}

// String provides a human-readable representation of the memory delta.
func (md MemoryDelta) String() string {
	return fmt.Sprintf("Memory Delta [%s -> %s]: Alloc: %+d bytes, Objects: %+d, GC: %+d, Duration: %v",
		md.FromLabel, md.ToLabel, md.DeltaAlloc, md.DeltaHeapObjects, md.DeltaNumGC, md.Duration)
}

// MemoryReport provides a comprehensive analysis of memory usage patterns.
type MemoryReport struct {
	StartTime       time.Time
	EndTime         time.Time
	TotalDuration   time.Duration
	PeakAlloc       uint64
	PeakAllocLabel  string
	TotalAllocated  uint64
	NumSnapshots    int
	GCCount         uint32
	MemoryLeaks     []DetectedLeak
	Recommendations []string
}

// MemoryLeakInfo represents a potential memory leak detection.
type MemoryLeakInfo struct {
	StartSnapshot MemorySnapshot
	EndSnapshot   MemorySnapshot
	LeakRate      float64 // bytes per second
	Severity      string  // "low", "medium", "high"
	Description   string
}

// GenerateReport creates a comprehensive memory analysis report.
func (mp *MemoryProfiler) GenerateReport() MemoryReport {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	if len(mp.snapshots) == 0 {
		return MemoryReport{
			Recommendations: []string{"No memory snapshots available for analysis"},
		}
	}

	start := mp.snapshots[0]
	end := mp.snapshots[len(mp.snapshots)-1]

	// Find peak allocation
	peakAlloc := uint64(0)
	peakLabel := ""
	for _, snapshot := range mp.snapshots {
		if snapshot.Alloc > peakAlloc {
			peakAlloc = snapshot.Alloc
			peakLabel = snapshot.Label
		}
	}

	report := MemoryReport{
		StartTime:       start.Timestamp,
		EndTime:         end.Timestamp,
		TotalDuration:   end.Timestamp.Sub(start.Timestamp),
		PeakAlloc:       peakAlloc,
		PeakAllocLabel:  peakLabel,
		TotalAllocated:  end.TotalAlloc - start.TotalAlloc,
		NumSnapshots:    len(mp.snapshots),
		GCCount:         end.NumGC - start.NumGC,
		MemoryLeaks:     mp.detectLeaksBasic(),
		Recommendations: mp.generateRecommendations(),
	}

	return report
}

// detectLeaksBasic analyzes snapshots to identify potential memory leaks.
func (mp *MemoryProfiler) detectLeaksBasic() []DetectedLeak {
	if len(mp.snapshots) < 3 {
		return nil
	}

	var leaks []DetectedLeak

	// Look for sustained memory growth patterns
	windowSize := 5
	if len(mp.snapshots) < windowSize {
		windowSize = len(mp.snapshots)
	}

	for i := 0; i <= len(mp.snapshots)-windowSize; i++ {
		start := mp.snapshots[i]
		end := mp.snapshots[i+windowSize-1]

		duration := end.Timestamp.Sub(start.Timestamp).Seconds()
		if duration <= 0 {
			continue
		}

		allocGrowth := int64(end.Alloc) - int64(start.Alloc)
		leakRate := float64(allocGrowth) / duration

		// Detect significant memory growth without GC reclaiming it
		if allocGrowth > 1024*1024 && leakRate > 1024 { // > 1MB growth at > 1KB/sec
			severityLevel := Low
			if leakRate > 1024*1024 { // > 1MB/sec
				severityLevel = High
			} else if leakRate > 100*1024 { // > 100KB/sec
				severityLevel = Medium
			}

			// Convert MemorySnapshot to MemorySample
			samples := []MemorySample{
				{
					Timestamp:      start.Timestamp,
					HeapAlloc:      start.Alloc,
					HeapObjects:    start.HeapObjects,
					StackInUse:     start.StackInUse,
					MemStats:       start.MemStats,
					GoroutineCount: runtime.NumGoroutine(), // Approximate
					Label:          start.Label,
				},
				{
					Timestamp:      end.Timestamp,
					HeapAlloc:      end.Alloc,
					HeapObjects:    end.HeapObjects,
					StackInUse:     end.StackInUse,
					MemStats:       end.MemStats,
					GoroutineCount: runtime.NumGoroutine(), // Approximate
					Label:          end.Label,
				},
			}

			leak := DetectedLeak{
				Type:        MemoryLeak,
				Severity:    severityLevel,
				StartTime:   start.Timestamp,
				EndTime:     end.Timestamp,
				GrowthRate:  leakRate,
				TotalGrowth: uint64(allocGrowth),
				Description: fmt.Sprintf("Sustained memory growth of %d bytes over %v", allocGrowth, end.Timestamp.Sub(start.Timestamp)),
				Suggestions: []string{"Check for unbounded data structures", "Review step history limits"},
				Samples:     samples,
			}
			leaks = append(leaks, leak)
		}
	}

	return leaks
}

// generateRecommendations provides optimization suggestions based on memory patterns.
func (mp *MemoryProfiler) generateRecommendations() []string {
	if len(mp.snapshots) < 2 {
		return []string{"Need more snapshots for meaningful recommendations"}
	}

	var recommendations []string

	start := mp.snapshots[0]
	end := mp.snapshots[len(mp.snapshots)-1]

	// Check for excessive allocations
	totalAlloc := end.TotalAlloc - start.TotalAlloc
	duration := end.Timestamp.Sub(start.Timestamp).Seconds()

	if duration > 0 {
		allocRate := float64(totalAlloc) / duration
		if allocRate > 50*1024*1024 { // > 50MB/sec
			recommendations = append(recommendations, "High allocation rate detected. Consider object pooling or reducing temporary allocations.")
		}
	}

	// Check GC pressure
	gcCount := end.NumGC - start.NumGC
	if duration > 0 && float64(gcCount)/duration > 10 { // > 10 GC/sec
		recommendations = append(recommendations, "High GC frequency. Consider reducing allocation pressure or tuning GC parameters.")
	}

	// Check for memory growth
	memGrowth := int64(end.Alloc) - int64(start.Alloc)
	if memGrowth > 100*1024*1024 { // > 100MB growth
		recommendations = append(recommendations, "Significant memory growth detected. Check for memory leaks or excessive caching.")
	}

	// Check object count growth
	objGrowth := int64(end.HeapObjects) - int64(start.HeapObjects)
	if objGrowth > 1000000 { // > 1M objects
		recommendations = append(recommendations, "Large increase in heap objects. Consider object reuse or more efficient data structures.")
	}

	if len(recommendations) == 0 {
		recommendations = append(recommendations, "Memory usage patterns appear normal.")
	}

	return recommendations
}

// StatechartProfiler provides specialized profiling for statechart operations.
type StatechartProfiler struct {
	mp      *MemoryProfiler
	enabled bool
}

// NewStatechartProfiler creates a new statechart-specific memory profiler.
func NewStatechartProfiler() *StatechartProfiler {
	return &StatechartProfiler{
		mp:      NewMemoryProfiler(),
		enabled: true,
	}
}

// ProfileValidation profiles memory usage during statechart validation.
func (sp *StatechartProfiler) ProfileValidation(statechart *semantics.Statechart) (func() MemoryDelta, error) {
	if !sp.enabled {
		return func() MemoryDelta { return MemoryDelta{} }, nil
	}

	before := sp.mp.TakeSnapshot("validation_start")

	return func() MemoryDelta {
		after := sp.mp.TakeSnapshot("validation_end")
		return ComputeDelta(before, after)
	}, nil
}

// ProfileMachineCreation profiles memory usage during machine creation.
func (sp *StatechartProfiler) ProfileMachineCreation(statechart *semantics.Statechart, id string) (func() MemoryDelta, *semantics.MachineWrapper, error) {
	if !sp.enabled {
		// Still create the machine, just don't profile
		machine, err := semantics.NewMachine(statechart, id, nil)
		return func() MemoryDelta { return MemoryDelta{} }, machine, err
	}

	before := sp.mp.TakeSnapshot("machine_creation_start")
	machine, err := semantics.NewMachine(statechart, id, nil)
	after := sp.mp.TakeSnapshot("machine_creation_end")

	delta := ComputeDelta(before, after)
	return func() MemoryDelta { return delta }, machine, err
}

// ProfileEventProcessing profiles memory usage during event processing.
func (sp *StatechartProfiler) ProfileEventProcessing(machine *semantics.MachineWrapper, eventName string) (func() MemoryDelta, bool, error) {
	if !sp.enabled {
		stepped, err := machine.Step(eventName)
		return func() MemoryDelta { return MemoryDelta{} }, stepped, err
	}

	before := sp.mp.TakeSnapshot(fmt.Sprintf("event_%s_start", eventName))
	stepped, err := machine.Step(eventName)
	after := sp.mp.TakeSnapshot(fmt.Sprintf("event_%s_end", eventName))

	delta := ComputeDelta(before, after)
	return func() MemoryDelta { return delta }, stepped, err
}

// ProfileStatechartLifecycle profiles the complete lifecycle of a statechart operation.
func (sp *StatechartProfiler) ProfileStatechartLifecycle(statechartFunc func() *sc.Statechart, operations []string) (*LifecycleProfile, error) {
	if !sp.enabled {
		return &LifecycleProfile{}, nil
	}

	profile := &LifecycleProfile{
		Operations: make(map[string]MemoryDelta),
		StartTime:  time.Now(),
	}

	// Baseline
	sp.mp.SetBaseline("lifecycle_baseline")

	// Create statechart
	before := sp.mp.TakeSnapshot("statechart_creation_start")
	statechart := semantics.NewStatechart(statechartFunc())
	after := sp.mp.TakeSnapshot("statechart_creation_end")
	profile.Operations["statechart_creation"] = ComputeDelta(before, after)

	// Validation
	before = sp.mp.TakeSnapshot("validation_start")
	err := statechart.Validate()
	after = sp.mp.TakeSnapshot("validation_end")
	profile.Operations["validation"] = ComputeDelta(before, after)

	if err != nil {
		return profile, fmt.Errorf("validation failed: %w", err)
	}

	// Machine creation
	before = sp.mp.TakeSnapshot("machine_creation_start")
	machine, err := semantics.NewMachine(statechart, "profile_machine", nil)
	after = sp.mp.TakeSnapshot("machine_creation_end")
	profile.Operations["machine_creation"] = ComputeDelta(before, after)

	if err != nil {
		return profile, fmt.Errorf("machine creation failed: %w", err)
	}

	// Start machine
	before = sp.mp.TakeSnapshot("machine_start_start")
	err = machine.Start()
	after = sp.mp.TakeSnapshot("machine_start_end")
	profile.Operations["machine_start"] = ComputeDelta(before, after)

	if err != nil {
		return profile, fmt.Errorf("machine start failed: %w", err)
	}

	// Process events
	for i, operation := range operations {
		label := fmt.Sprintf("operation_%d_%s", i, operation)
		before = sp.mp.TakeSnapshot(label + "_start")
		_, _ = machine.Step(operation) // Ignore errors for profiling
		after = sp.mp.TakeSnapshot(label + "_end")
		profile.Operations[label] = ComputeDelta(before, after)
	}

	// Stop machine
	before = sp.mp.TakeSnapshot("machine_stop_start")
	_ = machine.Stop()
	after = sp.mp.TakeSnapshot("machine_stop_end")
	profile.Operations["machine_stop"] = ComputeDelta(before, after)

	profile.EndTime = time.Now()
	profile.TotalDuration = profile.EndTime.Sub(profile.StartTime)

	return profile, nil
}

// LifecycleProfile represents memory usage throughout a statechart's lifecycle.
type LifecycleProfile struct {
	StartTime     time.Time
	EndTime       time.Time
	TotalDuration time.Duration
	Operations    map[string]MemoryDelta
}

// GetMemoryProfiler returns the underlying memory profiler.
func (sp *StatechartProfiler) GetMemoryProfiler() *MemoryProfiler {
	return sp.mp
}

// Enable turns on profiling.
func (sp *StatechartProfiler) Enable() {
	sp.enabled = true
	sp.mp.Enable()
}

// Disable turns off profiling.
func (sp *StatechartProfiler) Disable() {
	sp.enabled = false
	sp.mp.Disable()
}

// MemoryAnalyzer provides advanced memory analysis capabilities.
type MemoryAnalyzer struct {
	profiles []LifecycleProfile
	mu       sync.RWMutex
}

// NewMemoryAnalyzer creates a new memory analyzer.
func NewMemoryAnalyzer() *MemoryAnalyzer {
	return &MemoryAnalyzer{
		profiles: make([]LifecycleProfile, 0, 100),
	}
}

// AddProfile adds a lifecycle profile to the analyzer.
func (ma *MemoryAnalyzer) AddProfile(profile LifecycleProfile) {
	ma.mu.Lock()
	defer ma.mu.Unlock()
	ma.profiles = append(ma.profiles, profile)
}

// AnalyzePatterns analyzes memory usage patterns across multiple profiles.
func (ma *MemoryAnalyzer) AnalyzePatterns() MemoryPatternAnalysis {
	ma.mu.RLock()
	defer ma.mu.RUnlock()

	if len(ma.profiles) == 0 {
		return MemoryPatternAnalysis{
			Issues: []string{"No profiles available for analysis"},
		}
	}

	analysis := MemoryPatternAnalysis{
		ProfileCount: len(ma.profiles),
		Operations:   make(map[string]OperationStats),
	}

	// Aggregate statistics by operation type
	for _, profile := range ma.profiles {
		for opName, delta := range profile.Operations {
			stats := analysis.Operations[opName]
			stats.Count++
			stats.TotalAlloc += delta.DeltaAlloc
			stats.TotalObjects += delta.DeltaHeapObjects

			if delta.DeltaAlloc > stats.MaxAlloc {
				stats.MaxAlloc = delta.DeltaAlloc
			}
			if stats.MinAlloc == 0 || delta.DeltaAlloc < stats.MinAlloc {
				stats.MinAlloc = delta.DeltaAlloc
			}

			analysis.Operations[opName] = stats
		}
	}

	// Calculate averages and identify issues
	for opName, stats := range analysis.Operations {
		if stats.Count > 0 {
			stats.AvgAlloc = stats.TotalAlloc / int64(stats.Count)
			stats.AvgObjects = stats.TotalObjects / int64(stats.Count)
			analysis.Operations[opName] = stats
		}

		// Identify potential issues
		if stats.AvgAlloc > 10*1024*1024 { // > 10MB average
			analysis.Issues = append(analysis.Issues, fmt.Sprintf("Operation '%s' has high average allocation: %d bytes", opName, stats.AvgAlloc))
		}

		if stats.MaxAlloc > 100*1024*1024 { // > 100MB peak
			analysis.Issues = append(analysis.Issues, fmt.Sprintf("Operation '%s' has very high peak allocation: %d bytes", opName, stats.MaxAlloc))
		}
	}

	if len(analysis.Issues) == 0 {
		analysis.Issues = append(analysis.Issues, "No significant memory issues detected")
	}

	return analysis
}

// MemoryPatternAnalysis represents analysis results across multiple profiles.
type MemoryPatternAnalysis struct {
	ProfileCount int
	Operations   map[string]OperationStats
	Issues       []string
}

// OperationStats represents aggregated statistics for a specific operation.
type OperationStats struct {
	Count        int
	TotalAlloc   int64
	TotalObjects int64
	MaxAlloc     int64
	MinAlloc     int64
	AvgAlloc     int64
	AvgObjects   int64
}

// ProfiledOperation wraps an operation with automatic memory profiling.
type ProfiledOperation struct {
	profiler *StatechartProfiler
	name     string
}

// NewProfiledOperation creates a new profiled operation wrapper.
func NewProfiledOperation(profiler *StatechartProfiler, name string) *ProfiledOperation {
	return &ProfiledOperation{
		profiler: profiler,
		name:     name,
	}
}

// Execute runs an operation with memory profiling.
func (po *ProfiledOperation) Execute(operation func() error) (MemoryDelta, error) {
	before := po.profiler.mp.TakeSnapshot(po.name + "_start")
	err := operation()
	after := po.profiler.mp.TakeSnapshot(po.name + "_end")

	return ComputeDelta(before, after), err
}

// MemoryConstraint represents a memory usage constraint for operations.
type MemoryConstraint struct {
	MaxAllocation int64         // Maximum allocation in bytes
	MaxObjects    int64         // Maximum number of objects
	MaxDuration   time.Duration // Maximum duration for the operation
	Name          string        // Name of the constraint
}

// Check verifies if a memory delta satisfies the constraint.
func (mc MemoryConstraint) Check(delta MemoryDelta) error {
	alloc := delta.DeltaAlloc
	if delta.DeltaTotalAlloc > alloc {
		alloc = delta.DeltaTotalAlloc
	}
	if mc.MaxAllocation > 0 && alloc > mc.MaxAllocation {
		return fmt.Errorf("constraint '%s' violated: allocation %d exceeds limit %d", mc.Name, alloc, mc.MaxAllocation)
	}

	if mc.MaxObjects > 0 && delta.DeltaHeapObjects > mc.MaxObjects {
		return fmt.Errorf("constraint '%s' violated: object count %d exceeds limit %d", mc.Name, delta.DeltaHeapObjects, mc.MaxObjects)
	}

	if mc.MaxDuration > 0 && delta.Duration > mc.MaxDuration {
		return fmt.Errorf("constraint '%s' violated: duration %v exceeds limit %v", mc.Name, delta.Duration, mc.MaxDuration)
	}

	return nil
}

// ConstrainedProfiler combines profiling with constraint validation.
type ConstrainedProfiler struct {
	profiler    *StatechartProfiler
	constraints []MemoryConstraint
}

// NewConstrainedProfiler creates a profiler with memory constraints.
func NewConstrainedProfiler(constraints []MemoryConstraint) *ConstrainedProfiler {
	return &ConstrainedProfiler{
		profiler:    NewStatechartProfiler(),
		constraints: constraints,
	}
}

// ProfileWithConstraints profiles an operation and validates against constraints.
func (cp *ConstrainedProfiler) ProfileWithConstraints(name string, operation func() error) (MemoryDelta, error) {
	profOp := NewProfiledOperation(cp.profiler, name)
	delta, err := profOp.Execute(operation)

	if err != nil {
		return delta, err
	}

	// Check constraints
	for _, constraint := range cp.constraints {
		if constraintErr := constraint.Check(delta); constraintErr != nil {
			return delta, constraintErr
		}
	}

	return delta, nil
}

// BatchProfiler profiles multiple operations in sequence or parallel.
type BatchProfiler struct {
	profiler *StatechartProfiler
	results  []BatchResult
	mu       sync.Mutex
}

// BatchResult represents the result of a batch operation.
type BatchResult struct {
	Name     string
	Delta    MemoryDelta
	Error    error
	Duration time.Duration
}

// NewBatchProfiler creates a new batch profiler.
func NewBatchProfiler() *BatchProfiler {
	return &BatchProfiler{
		profiler: NewStatechartProfiler(),
		results:  make([]BatchResult, 0, 100),
	}
}

// ProfileSequential profiles operations run in sequence.
func (bp *BatchProfiler) ProfileSequential(operations map[string]func() error) []BatchResult {
	results := make([]BatchResult, 0, len(operations))

	for name, operation := range operations {
		start := time.Now()
		profOp := NewProfiledOperation(bp.profiler, name)
		delta, err := profOp.Execute(operation)
		duration := time.Since(start)

		result := BatchResult{
			Name:     name,
			Delta:    delta,
			Error:    err,
			Duration: duration,
		}
		results = append(results, result)
	}

	bp.mu.Lock()
	bp.results = append(bp.results, results...)
	bp.mu.Unlock()

	return results
}

// ProfileParallel profiles operations run in parallel.
func (bp *BatchProfiler) ProfileParallel(operations map[string]func() error) []BatchResult {
	var wg sync.WaitGroup
	resultChan := make(chan BatchResult, len(operations))

	for name, operation := range operations {
		wg.Add(1)
		go func(opName string, op func() error) {
			defer wg.Done()

			start := time.Now()
			profOp := NewProfiledOperation(bp.profiler, opName)
			delta, err := profOp.Execute(op)
			duration := time.Since(start)

			resultChan <- BatchResult{
				Name:     opName,
				Delta:    delta,
				Error:    err,
				Duration: duration,
			}
		}(name, operation)
	}

	wg.Wait()
	close(resultChan)

	results := make([]BatchResult, 0, len(operations))
	for result := range resultChan {
		results = append(results, result)
	}

	bp.mu.Lock()
	bp.results = append(bp.results, results...)
	bp.mu.Unlock()

	return results
}

// GetAllResults returns all batch results.
func (bp *BatchProfiler) GetAllResults() []BatchResult {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	results := make([]BatchResult, len(bp.results))
	copy(results, bp.results)
	return results
}

// ContinuousProfiler performs continuous memory monitoring.
type ContinuousProfiler struct {
	profiler *StatechartProfiler
	interval time.Duration
	ctx      context.Context
	cancel   context.CancelFunc
	running  bool
	mu       sync.RWMutex
}

// NewContinuousProfiler creates a continuous profiler.
func NewContinuousProfiler(interval time.Duration) *ContinuousProfiler {
	return &ContinuousProfiler{
		profiler: NewStatechartProfiler(),
		interval: interval,
	}
}

// Start begins continuous profiling.
func (cp *ContinuousProfiler) Start() {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if cp.running {
		return
	}

	cp.ctx, cp.cancel = context.WithCancel(context.Background())
	cp.running = true

	go cp.run()
}

// Stop ends continuous profiling.
func (cp *ContinuousProfiler) Stop() {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if !cp.running {
		return
	}

	cp.cancel()
	cp.running = false
}

// run performs the continuous profiling loop.
func (cp *ContinuousProfiler) run() {
	ticker := time.NewTicker(cp.interval)
	defer ticker.Stop()

	for {
		select {
		case <-cp.ctx.Done():
			return
		case <-ticker.C:
			cp.profiler.mp.TakeSnapshot("continuous_sample")
		}
	}
}

// GetProfiler returns the underlying profiler.
func (cp *ContinuousProfiler) GetProfiler() *StatechartProfiler {
	return cp.profiler
}

// IsRunning returns whether continuous profiling is active.
func (cp *ContinuousProfiler) IsRunning() bool {
	cp.mu.RLock()
	defer cp.mu.RUnlock()
	return cp.running
}
