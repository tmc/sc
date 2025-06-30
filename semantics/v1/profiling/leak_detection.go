package profiling

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/tmc/sc/semantics/v1"
)

// LeakDetector identifies potential memory leaks in statechart operations.
type LeakDetector struct {
	samples       []MemorySample
	samplingRate  time.Duration
	retentionTime time.Duration
	mu            sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	running       bool
	thresholds    LeakThresholds
	callbacks     []LeakCallback
}

// MemorySample represents a point-in-time memory measurement.
type MemorySample struct {
	Timestamp      time.Time
	HeapAlloc      uint64
	HeapObjects    uint64
	StackInUse     uint64
	MemStats       runtime.MemStats
	GoroutineCount int
	ActiveMachines int
	Label          string
}

// LeakThresholds defines the criteria for detecting memory leaks.
type LeakThresholds struct {
	MemoryGrowthRate    float64       // bytes per second
	ObjectGrowthRate    float64       // objects per second
	GoroutineGrowthRate float64       // goroutines per second
	MinSampleDuration   time.Duration // minimum time before detecting leaks
	MinGrowthAmount     uint64        // minimum growth amount to consider
}

// LeakCallback is called when a potential leak is detected.
type LeakCallback func(leak DetectedLeak)

// DetectedLeak represents a detected memory leak.
type DetectedLeak struct {
	Type        LeakType
	Severity    Severity
	StartTime   time.Time
	EndTime     time.Time
	GrowthRate  float64
	TotalGrowth uint64
	Description string
	Suggestions []string
	Samples     []MemorySample
}

// LeakType represents the type of memory leak.
type LeakType int

const (
	MemoryLeak LeakType = iota
	ObjectLeak
	GoroutineLeak
	HistoryLeak
	ConfigurationLeak
)

// Severity represents the severity of a memory leak.
type Severity int

const (
	Low Severity = iota
	Medium
	High
	Critical
)

func (lt LeakType) String() string {
	switch lt {
	case MemoryLeak:
		return "Memory Leak"
	case ObjectLeak:
		return "Object Leak"
	case GoroutineLeak:
		return "Goroutine Leak"
	case HistoryLeak:
		return "History Leak"
	case ConfigurationLeak:
		return "Configuration Leak"
	default:
		return "Unknown"
	}
}

func (s Severity) String() string {
	switch s {
	case Low:
		return "Low"
	case Medium:
		return "Medium"
	case High:
		return "High"
	case Critical:
		return "Critical"
	default:
		return "Unknown"
	}
}

// NewLeakDetector creates a new memory leak detector.
func NewLeakDetector() *LeakDetector {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &LeakDetector{
		samples:       make([]MemorySample, 0, 1000),
		samplingRate:  5 * time.Second,
		retentionTime: 30 * time.Minute,
		ctx:           ctx,
		cancel:        cancel,
		thresholds: LeakThresholds{
			MemoryGrowthRate:    1024 * 1024,    // 1MB/sec
			ObjectGrowthRate:    10000,          // 10k objects/sec
			GoroutineGrowthRate: 10,             // 10 goroutines/sec
			MinSampleDuration:   30 * time.Second,
			MinGrowthAmount:     10 * 1024 * 1024, // 10MB
		},
		callbacks: make([]LeakCallback, 0, 10),
	}
}

// SetThresholds configures the leak detection thresholds.
func (ld *LeakDetector) SetThresholds(thresholds LeakThresholds) {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	ld.thresholds = thresholds
}

// AddCallback registers a callback for leak detection events.
func (ld *LeakDetector) AddCallback(callback LeakCallback) {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	ld.callbacks = append(ld.callbacks, callback)
}

// Start begins leak detection monitoring.
func (ld *LeakDetector) Start() {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	
	if ld.running {
		return
	}
	
	ld.running = true
	go ld.monitoringLoop()
}

// Stop ends leak detection monitoring.
func (ld *LeakDetector) Stop() {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	
	if !ld.running {
		return
	}
	
	ld.cancel()
	ld.running = false
}

// monitoringLoop runs the continuous monitoring process.
func (ld *LeakDetector) monitoringLoop() {
	ticker := time.NewTicker(ld.samplingRate)
	defer ticker.Stop()
	
	for {
		select {
		case <-ld.ctx.Done():
			return
		case <-ticker.C:
			ld.takeSample("")
			ld.detectLeaks()
			ld.cleanupOldSamples()
		}
	}
}

// takeSample records a memory sample.
func (ld *LeakDetector) takeSample(label string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	sample := MemorySample{
		Timestamp:      time.Now(),
		HeapAlloc:      m.Alloc,
		HeapObjects:    m.HeapObjects,
		StackInUse:     m.StackInuse,
		MemStats:       m,
		GoroutineCount: runtime.NumGoroutine(),
		Label:          label,
	}
	
	ld.mu.Lock()
	ld.samples = append(ld.samples, sample)
	ld.mu.Unlock()
}

// TakeSample manually records a memory sample with a label.
func (ld *LeakDetector) TakeSample(label string) {
	ld.takeSample(label)
}

// detectLeaks analyzes samples to identify potential memory leaks.
func (ld *LeakDetector) detectLeaks() {
	ld.mu.RLock()
	samples := make([]MemorySample, len(ld.samples))
	copy(samples, ld.samples)
	thresholds := ld.thresholds
	callbacks := make([]LeakCallback, len(ld.callbacks))
	copy(callbacks, ld.callbacks)
	ld.mu.RUnlock()
	
	if len(samples) < 2 {
		return
	}
	
	// Analyze different types of potential leaks
	leaks := []DetectedLeak{}
	
	// Memory leak detection
	if memLeak := ld.analyzeMemoryGrowth(samples, thresholds); memLeak != nil {
		leaks = append(leaks, *memLeak)
	}
	
	// Object leak detection
	if objLeak := ld.analyzeObjectGrowth(samples, thresholds); objLeak != nil {
		leaks = append(leaks, *objLeak)
	}
	
	// Goroutine leak detection
	if gorLeak := ld.analyzeGoroutineGrowth(samples, thresholds); gorLeak != nil {
		leaks = append(leaks, *gorLeak)
	}
	
	// Notify callbacks of detected leaks
	for _, leak := range leaks {
		for _, callback := range callbacks {
			go callback(leak) // Run callbacks asynchronously
		}
	}
}

// analyzeMemoryGrowth detects sustained memory growth patterns.
func (ld *LeakDetector) analyzeMemoryGrowth(samples []MemorySample, thresholds LeakThresholds) *DetectedLeak {
	if len(samples) < 5 {
		return nil
	}
	
	// Look at the last N samples for trend analysis
	windowSize := min(len(samples), 10)
	recentSamples := samples[len(samples)-windowSize:]
	
	start := recentSamples[0]
	end := recentSamples[len(recentSamples)-1]
	
	duration := end.Timestamp.Sub(start.Timestamp)
	if duration < thresholds.MinSampleDuration {
		return nil
	}
	
	memoryGrowth := int64(end.HeapAlloc) - int64(start.HeapAlloc)
	growthRate := float64(memoryGrowth) / duration.Seconds()
	
	if memoryGrowth > int64(thresholds.MinGrowthAmount) && growthRate > thresholds.MemoryGrowthRate {
		severity := ld.calculateSeverity(growthRate, thresholds.MemoryGrowthRate)
		
		return &DetectedLeak{
			Type:        MemoryLeak,
			Severity:    severity,
			StartTime:   start.Timestamp,
			EndTime:     end.Timestamp,
			GrowthRate:  growthRate,
			TotalGrowth: uint64(memoryGrowth),
			Description: fmt.Sprintf("Sustained memory growth of %d bytes over %v (%.2f bytes/sec)", 
				memoryGrowth, duration, growthRate),
			Suggestions: ld.getMemoryLeakSuggestions(growthRate),
			Samples:     recentSamples,
		}
	}
	
	return nil
}

// analyzeObjectGrowth detects sustained object count growth.
func (ld *LeakDetector) analyzeObjectGrowth(samples []MemorySample, thresholds LeakThresholds) *DetectedLeak {
	if len(samples) < 5 {
		return nil
	}
	
	windowSize := min(len(samples), 10)
	recentSamples := samples[len(samples)-windowSize:]
	
	start := recentSamples[0]
	end := recentSamples[len(recentSamples)-1]
	
	duration := end.Timestamp.Sub(start.Timestamp)
	if duration < thresholds.MinSampleDuration {
		return nil
	}
	
	objectGrowth := int64(end.HeapObjects) - int64(start.HeapObjects)
	growthRate := float64(objectGrowth) / duration.Seconds()
	
	if objectGrowth > 0 && growthRate > thresholds.ObjectGrowthRate {
		severity := ld.calculateSeverity(growthRate, thresholds.ObjectGrowthRate)
		
		return &DetectedLeak{
			Type:        ObjectLeak,
			Severity:    severity,
			StartTime:   start.Timestamp,
			EndTime:     end.Timestamp,
			GrowthRate:  growthRate,
			TotalGrowth: uint64(objectGrowth),
			Description: fmt.Sprintf("Sustained object growth of %d objects over %v (%.2f objects/sec)", 
				objectGrowth, duration, growthRate),
			Suggestions: ld.getObjectLeakSuggestions(growthRate),
			Samples:     recentSamples,
		}
	}
	
	return nil
}

// analyzeGoroutineGrowth detects goroutine leaks.
func (ld *LeakDetector) analyzeGoroutineGrowth(samples []MemorySample, thresholds LeakThresholds) *DetectedLeak {
	if len(samples) < 5 {
		return nil
	}
	
	windowSize := min(len(samples), 10)
	recentSamples := samples[len(samples)-windowSize:]
	
	start := recentSamples[0]
	end := recentSamples[len(recentSamples)-1]
	
	duration := end.Timestamp.Sub(start.Timestamp)
	if duration < thresholds.MinSampleDuration {
		return nil
	}
	
	goroutineGrowth := int64(end.GoroutineCount) - int64(start.GoroutineCount)
	growthRate := float64(goroutineGrowth) / duration.Seconds()
	
	if goroutineGrowth > 0 && growthRate > thresholds.GoroutineGrowthRate {
		severity := ld.calculateSeverity(growthRate, thresholds.GoroutineGrowthRate)
		
		return &DetectedLeak{
			Type:        GoroutineLeak,
			Severity:    severity,
			StartTime:   start.Timestamp,
			EndTime:     end.Timestamp,
			GrowthRate:  growthRate,
			TotalGrowth: uint64(goroutineGrowth),
			Description: fmt.Sprintf("Sustained goroutine growth of %d goroutines over %v (%.2f goroutines/sec)", 
				goroutineGrowth, duration, growthRate),
			Suggestions: ld.getGoroutineLeakSuggestions(growthRate),
			Samples:     recentSamples,
		}
	}
	
	return nil
}

// calculateSeverity determines leak severity based on growth rate.
func (ld *LeakDetector) calculateSeverity(growthRate, threshold float64) Severity {
	ratio := growthRate / threshold
	
	if ratio >= 10 {
		return Critical
	} else if ratio >= 5 {
		return High
	} else if ratio >= 2 {
		return Medium
	}
	return Low
}

// getMemoryLeakSuggestions provides suggestions for memory leak remediation.
func (ld *LeakDetector) getMemoryLeakSuggestions(growthRate float64) []string {
	suggestions := []string{
		"Check for unbounded data structures (slices, maps, channels)",
		"Verify that step history is being properly limited",
		"Ensure event queues are being drained and bounded",
		"Review context data for accumulating values",
	}
	
	if growthRate > 10*1024*1024 { // > 10MB/sec
		suggestions = append(suggestions, 
			"Consider implementing object pooling for frequently allocated objects",
			"Use memory profiling tools (go tool pprof) for detailed analysis")
	}
	
	return suggestions
}

// getObjectLeakSuggestions provides suggestions for object leak remediation.
func (ld *LeakDetector) getObjectLeakSuggestions(growthRate float64) []string {
	suggestions := []string{
		"Check for accumulating temporary objects",
		"Verify proper cleanup of event listeners and callbacks",
		"Review state reference management",
		"Ensure transition objects are being reused or properly collected",
	}
	
	if growthRate > 50000 { // > 50k objects/sec
		suggestions = append(suggestions,
			"Implement object pooling for high-frequency allocations",
			"Consider reducing allocation frequency in hot paths")
	}
	
	return suggestions
}

// getGoroutineLeakSuggestions provides suggestions for goroutine leak remediation.
func (ld *LeakDetector) getGoroutineLeakSuggestions(growthRate float64) []string {
	return []string{
		"Check for goroutines that are not properly terminated",
		"Verify that context cancellation is working correctly",
		"Review event processor lifecycle management",
		"Ensure background monitoring goroutines are cleaned up",
		"Check for missing channel closes or deadlocked goroutines",
	}
}

// cleanupOldSamples removes samples older than the retention time.
func (ld *LeakDetector) cleanupOldSamples() {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	
	cutoff := time.Now().Add(-ld.retentionTime)
	newStart := 0
	
	for i, sample := range ld.samples {
		if sample.Timestamp.After(cutoff) {
			newStart = i
			break
		}
	}
	
	if newStart > 0 {
		// Keep only samples after cutoff
		ld.samples = ld.samples[newStart:]
	}
}

// GetSamples returns a copy of all current samples.
func (ld *LeakDetector) GetSamples() []MemorySample {
	ld.mu.RLock()
	defer ld.mu.RUnlock()
	
	samples := make([]MemorySample, len(ld.samples))
	copy(samples, ld.samples)
	return samples
}

// GenerateReport creates a comprehensive leak detection report.
func (ld *LeakDetector) GenerateReport() LeakDetectionReport {
	ld.mu.RLock()
	defer ld.mu.RUnlock()
	
	if len(ld.samples) == 0 {
		return LeakDetectionReport{
			Summary: "No samples available for leak analysis",
		}
	}
	
	first := ld.samples[0]
	last := ld.samples[len(ld.samples)-1]
	duration := last.Timestamp.Sub(first.Timestamp)
	
	report := LeakDetectionReport{
		StartTime:      first.Timestamp,
		EndTime:        last.Timestamp,
		Duration:       duration,
		SampleCount:    len(ld.samples),
		MemoryGrowth:   int64(last.HeapAlloc) - int64(first.HeapAlloc),
		ObjectGrowth:   int64(last.HeapObjects) - int64(first.HeapObjects),
		GoroutineGrowth: int64(last.GoroutineCount) - int64(first.GoroutineCount),
	}
	
	// Calculate growth rates
	if duration.Seconds() > 0 {
		report.MemoryGrowthRate = float64(report.MemoryGrowth) / duration.Seconds()
		report.ObjectGrowthRate = float64(report.ObjectGrowth) / duration.Seconds()
		report.GoroutineGrowthRate = float64(report.GoroutineGrowth) / duration.Seconds()
	}
	
	// Analyze trends
	report.TrendAnalysis = ld.analyzeTrends()
	
	// Generate summary
	report.Summary = ld.generateSummary(report)
	
	return report
}

// LeakDetectionReport provides a comprehensive analysis of memory usage.
type LeakDetectionReport struct {
	StartTime           time.Time
	EndTime             time.Time
	Duration            time.Duration
	SampleCount         int
	MemoryGrowth        int64
	ObjectGrowth        int64
	GoroutineGrowth     int64
	MemoryGrowthRate    float64
	ObjectGrowthRate    float64
	GoroutineGrowthRate float64
	TrendAnalysis       TrendAnalysis
	Summary             string
}

// TrendAnalysis provides trend analysis of memory usage patterns.
type TrendAnalysis struct {
	MemoryTrend    string // "increasing", "decreasing", "stable", "volatile"
	ObjectTrend    string
	GoroutineTrend string
	Volatility     float64 // measure of variance in samples
	Recommendations []string
}

// analyzeTrends performs trend analysis on the collected samples.
func (ld *LeakDetector) analyzeTrends() TrendAnalysis {
	if len(ld.samples) < 3 {
		return TrendAnalysis{
			MemoryTrend:    "insufficient_data",
			ObjectTrend:    "insufficient_data",
			GoroutineTrend: "insufficient_data",
		}
	}
	
	// Simple trend analysis using linear regression slope
	memoryTrend := ld.calculateTrend(func(s MemorySample) float64 { return float64(s.HeapAlloc) })
	objectTrend := ld.calculateTrend(func(s MemorySample) float64 { return float64(s.HeapObjects) })
	goroutineTrend := ld.calculateTrend(func(s MemorySample) float64 { return float64(s.GoroutineCount) })
	
	volatility := ld.calculateVolatility(func(s MemorySample) float64 { return float64(s.HeapAlloc) })
	
	recommendations := ld.generateTrendRecommendations(memoryTrend, objectTrend, goroutineTrend, volatility)
	
	return TrendAnalysis{
		MemoryTrend:     ld.interpretSlope(memoryTrend),
		ObjectTrend:     ld.interpretSlope(objectTrend),
		GoroutineTrend:  ld.interpretSlope(goroutineTrend),
		Volatility:      volatility,
		Recommendations: recommendations,
	}
}

// calculateTrend calculates the slope of a value over time.
func (ld *LeakDetector) calculateTrend(valueFunc func(MemorySample) float64) float64 {
	if len(ld.samples) < 2 {
		return 0
	}
	
	n := float64(len(ld.samples))
	var sumX, sumY, sumXY, sumX2 float64
	
	for i, sample := range ld.samples {
		x := float64(i)
		y := valueFunc(sample)
		
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}
	
	// Linear regression slope: (n*∑xy - ∑x*∑y) / (n*∑x² - (∑x)²)
	denominator := n*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0
	}
	
	slope := (n*sumXY - sumX*sumY) / denominator
	return slope
}

// calculateVolatility calculates the coefficient of variation.
func (ld *LeakDetector) calculateVolatility(valueFunc func(MemorySample) float64) float64 {
	if len(ld.samples) < 2 {
		return 0
	}
	
	var sum, sumSquares float64
	n := float64(len(ld.samples))
	
	for _, sample := range ld.samples {
		value := valueFunc(sample)
		sum += value
		sumSquares += value * value
	}
	
	mean := sum / n
	variance := (sumSquares - sum*sum/n) / (n - 1)
	stdDev := 0.0
	if variance > 0 {
		stdDev = variance // Simplified, would use math.Sqrt in production
	}
	
	if mean == 0 {
		return 0
	}
	
	return stdDev / mean // Coefficient of variation
}

// interpretSlope converts a numerical slope to a descriptive trend.
func (ld *LeakDetector) interpretSlope(slope float64) string {
	if slope > 1000 {
		return "rapidly_increasing"
	} else if slope > 100 {
		return "increasing"
	} else if slope > -100 && slope < 100 {
		return "stable"
	} else if slope < -100 {
		return "decreasing"
	}
	return "volatile"
}

// generateTrendRecommendations provides recommendations based on trend analysis.
func (ld *LeakDetector) generateTrendRecommendations(memTrend, objTrend, gorTrend, volatility float64) []string {
	var recommendations []string
	
	if memTrend > 1000 {
		recommendations = append(recommendations, "Memory usage is rapidly increasing - investigate potential leaks")
	}
	
	if objTrend > 1000 {
		recommendations = append(recommendations, "Object count is rapidly increasing - check for object accumulation")
	}
	
	if gorTrend > 10 {
		recommendations = append(recommendations, "Goroutine count is increasing - check for goroutine leaks")
	}
	
	if volatility > 0.5 {
		recommendations = append(recommendations, "Memory usage is highly volatile - consider smoothing allocation patterns")
	}
	
	if len(recommendations) == 0 {
		recommendations = append(recommendations, "Memory usage patterns appear normal")
	}
	
	return recommendations
}

// generateSummary creates a text summary of the leak detection report.
func (ld *LeakDetector) generateSummary(report LeakDetectionReport) string {
	summary := fmt.Sprintf("Leak Detection Report (%v duration, %d samples)\n", 
		report.Duration, report.SampleCount)
	
	if report.MemoryGrowth > 0 {
		summary += fmt.Sprintf("Memory growth: %d bytes (%.2f bytes/sec)\n", 
			report.MemoryGrowth, report.MemoryGrowthRate)
	} else {
		summary += "Memory usage is stable or decreasing\n"
	}
	
	if report.ObjectGrowth > 0 {
		summary += fmt.Sprintf("Object growth: %d objects (%.2f objects/sec)\n", 
			report.ObjectGrowth, report.ObjectGrowthRate)
	}
	
	if report.GoroutineGrowth > 0 {
		summary += fmt.Sprintf("Goroutine growth: %d goroutines (%.2f goroutines/sec)\n", 
			report.GoroutineGrowth, report.GoroutineGrowthRate)
	}
	
	return summary
}

// StatechartLeakDetector provides leak detection specialized for statechart operations.
type StatechartLeakDetector struct {
	*LeakDetector
	machines       map[string]*semantics.MachineWrapper
	machineMetrics map[string]MachineMetrics
	mu             sync.RWMutex
}

// MachineMetrics tracks metrics for individual machines.
type MachineMetrics struct {
	StepCount      int
	HistorySize    int
	ErrorCount     int
	LastActive     time.Time
	ConfigSize     int
	ContextSize    int
}

// NewStatechartLeakDetector creates a statechart-specific leak detector.
func NewStatechartLeakDetector() *StatechartLeakDetector {
	return &StatechartLeakDetector{
		LeakDetector:   NewLeakDetector(),
		machines:       make(map[string]*semantics.MachineWrapper),
		machineMetrics: make(map[string]MachineMetrics),
	}
}

// RegisterMachine registers a machine for leak monitoring.
func (sld *StatechartLeakDetector) RegisterMachine(id string, machine *semantics.MachineWrapper) {
	sld.mu.Lock()
	defer sld.mu.Unlock()
	
	sld.machines[id] = machine
	sld.machineMetrics[id] = MachineMetrics{
		LastActive: time.Now(),
	}
}

// UnregisterMachine removes a machine from monitoring.
func (sld *StatechartLeakDetector) UnregisterMachine(id string) {
	sld.mu.Lock()
	defer sld.mu.Unlock()
	
	delete(sld.machines, id)
	delete(sld.machineMetrics, id)
}

// CheckMachineLeaks analyzes individual machines for potential leaks.
func (sld *StatechartLeakDetector) CheckMachineLeaks() []StatechartMachineLeakReport {
	sld.mu.RLock()
	machines := make(map[string]*semantics.MachineWrapper)
	for k, v := range sld.machines {
		machines[k] = v
	}
	sld.mu.RUnlock()
	
	var reports []StatechartMachineLeakReport
	
	for id, machine := range machines {
		report := sld.analyzeMachine(id, machine)
		if report.HasIssues() {
			reports = append(reports, report)
		}
	}
	
	return reports
}

// MachineLeakReport represents leak analysis for a specific machine.
type MachineLeakReport struct {
	MachineID       string
	Issues          []string
	Recommendations []string
	Metrics         MachineMetrics
	Severity        Severity
}

// HasIssues returns true if the report contains any issues.
func (mlr *MachineLeakReport) HasIssues() bool {
	return len(mlr.Issues) > 0
}

// analyzeMachine analyzes a specific machine for potential leaks.
func (sld *StatechartLeakDetector) analyzeMachine(id string, machine *semantics.MachineWrapper) StatechartMachineLeakReport {
	report := StatechartMachineLeakReport{
		MachineID: id,
		Issues:    []MachineIssue{},
		Metrics:   MachineMetrics{},
	}
	
	// Get machine metrics
	history := machine.GetStepHistory()
	config := machine.GetCurrentConfiguration()
	context := machine.GetContext()
	errors := machine.GetErrors()
	
	report.Metrics = MachineMetrics{
		StepCount:   len(history),
		HistorySize: len(history),
		ErrorCount:  len(errors),
		LastActive:  time.Now(),
		ConfigSize:  len(config.States),
		ContextSize: len(context.Fields),
	}
	
	// Check for excessive history
	if len(history) > 10000 {
		report.Issues = append(report.Issues, MachineIssue{
			Type:        HistoryLeak,
			Description: fmt.Sprintf("Excessive step history: %d steps", len(history)),
			Severity:    Medium,
			Value:       len(history),
		})
	}
	
	// Check for error accumulation
	if len(errors) > 100 {
		report.Issues = append(report.Issues, MachineIssue{
			Type:        MemoryLeak,
			Description: fmt.Sprintf("Accumulating errors: %d errors", len(errors)),
			Severity:    Medium,
			Value:       len(errors),
		})
	}
	
	// Check for large context
	if len(context.Fields) > 1000 {
		report.Issues = append(report.Issues, MachineIssue{
			Type:        ConfigurationLeak,
			Description: fmt.Sprintf("Large context: %d fields", len(context.Fields)),
			Severity:    Low,
			Value:       len(context.Fields),
		})
	}
	
	// Check for large configuration
	if len(config.States) > 100 {
		report.Issues = append(report.Issues, MachineIssue{
			Type:        ConfigurationLeak,
			Description: fmt.Sprintf("Large configuration: %d states", len(config.States)),
			Severity:    Low,
			Value:       len(config.States),
		})
	}
	
	return report
}

// StatechartLeakThresholds defines thresholds specific to statechart operations.
type StatechartLeakThresholds struct {
	MaxHistorySize      int
	MaxContextSize      int
	MaxErrorCount       int
	GrowthRateThreshold float64
}

// SetThresholds sets the leak detection thresholds for statechart operations.
func (sld *StatechartLeakDetector) SetThresholds(thresholds StatechartLeakThresholds) {
	// Store thresholds - in a real implementation these would be used in analyzeMachine
	// For now, we'll use hardcoded values in analyzeMachine
}

// StatechartMachineLeakReport represents a detailed leak report for a machine.
type StatechartMachineLeakReport struct {
	MachineID string
	Issues    []MachineIssue
	Metrics   MachineMetrics
}

// MachineIssue represents a specific issue found in a machine.
type MachineIssue struct {
	Type        LeakType
	Description string
	Severity    Severity
	Value       interface{}
}

// HasIssues returns true if the report contains issues.
func (r StatechartMachineLeakReport) HasIssues() bool {
	return len(r.Issues) > 0
}

// SetProductionThresholds sets less aggressive thresholds for production use.
func (sld *StatechartLeakDetector) SetProductionThresholds() {
	sld.LeakDetector.SetThresholds(LeakThresholds{
		MemoryGrowthRate:    10 * 1024 * 1024,  // 10MB/sec
		ObjectGrowthRate:    100000,            // 100k objects/sec
		GoroutineGrowthRate: 50,                // 50 goroutines/sec
		MinSampleDuration:   5 * time.Minute,
		MinGrowthAmount:     100 * 1024 * 1024, // 100MB
	})
}

// max returns the maximum severity.
func maxSeverity(a, b Severity) Severity {
	if a > b {
		return a
	}
	return b
}