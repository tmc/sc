package profiling

import (
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestObjectPool(t *testing.T) {
	pool := NewObjectPool()
	
	// Test configuration pooling
	config1 := pool.GetConfiguration()
	if config1 == nil {
		t.Fatal("Expected configuration from pool")
	}
	if len(config1.States) != 0 {
		t.Error("Expected empty states slice")
	}
	
	// Add some states and return to pool
	config1.States = append(config1.States, &sc.StateRef{Label: "test"})
	pool.PutConfiguration(config1)
	
	// Get another configuration - should be reset
	config2 := pool.GetConfiguration()
	if len(config2.States) != 0 {
		t.Error("Expected reset states slice")
	}
	
	// Test context pooling
	ctx1 := pool.GetContext()
	if ctx1 == nil {
		t.Fatal("Expected context from pool")
	}
	if len(ctx1.Fields) != 0 {
		t.Error("Expected empty fields map")
	}
	
	// Add field and return to pool
	ctx1.Fields["test"] = &structpb.Value{}
	pool.PutContext(ctx1)
	
	// Get another context - should be reset
	ctx2 := pool.GetContext()
	if len(ctx2.Fields) != 0 {
		t.Error("Expected reset fields map")
	}
	
	// Test transition pooling
	trans1 := pool.GetTransition()
	if trans1 == nil {
		t.Fatal("Expected transition from pool")
	}
	
	trans1.Label = "test_transition"
	trans1.From = append(trans1.From, "state1")
	pool.PutTransition(trans1)
	
	trans2 := pool.GetTransition()
	if trans2.Label != "" {
		t.Error("Expected reset label")
	}
	if len(trans2.From) != 0 {
		t.Error("Expected reset from slice")
	}
}

func TestOptimizedMachineWrapper(t *testing.T) {
	pool := NewObjectPool()
	statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
	
	machine, err := NewOptimizedMachine(statechart, "test_machine", pool)
	if err != nil {
		t.Fatalf("Failed to create optimized machine: %v", err)
	}
	
	// Test configuration
	machine.SetHistoryLimit(100)
	machine.SetCompactionFrequency(10)
	
	// Start and run some steps
	err = machine.Start()
	if err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	
	// Run enough steps to trigger compaction
	for i := 0; i < 15; i++ {
		_, err = machine.OptimizedStep("go")
		if err != nil {
			t.Errorf("Step %d failed: %v", i, err)
		}
	}
	
	history := machine.GetStepHistory()
	if len(history) > 100 {
		t.Errorf("History should be limited to 100, got %d", len(history))
	}
	
	machine.Stop()
}

func TestMemoryOptimizer(t *testing.T) {
	optimizer := NewMemoryOptimizer()
	
	// Test statechart optimization
	original := testutil.CreateLargeStatechart(10, 20)
	optimized := optimizer.OptimizeStatechart(original)
	
	if optimized == nil {
		t.Fatal("Expected optimized statechart")
	}
	
	// Basic structure should be preserved
	if original.RootState.Label != optimized.RootState.Label {
		t.Error("Root state label should be preserved")
	}
	
	if len(original.Transitions) != len(optimized.Transitions) {
		t.Error("Number of transitions should be preserved")
	}
	
	if len(original.Events) != len(optimized.Events) {
		t.Error("Number of events should be preserved")
	}
	
	// Test object pool access
	pool := optimizer.GetObjectPool()
	if pool == nil {
		t.Error("Expected object pool to be available")
	}
	
	// Test policy configuration
	compactionPolicy := CompactionPolicy{
		HistoryLimit:     500,
		CompactInterval:  time.Minute,
		GCTriggerRatio:   0.9,
		EnableAutoGC:     false,
	}
	optimizer.SetCompactionPolicy(compactionPolicy)
	
	cachePolicy := CachePolicy{
		MaxCacheSize: 5000,
		TTL:          10 * time.Minute,
		EnableCache:  false,
	}
	optimizer.SetCachePolicy(cachePolicy)
}

func TestLargeStatechartOptimizer(t *testing.T) {
	optimizer := NewLargeStatechartOptimizer()
	statechart := testutil.CreateLargeStatechart(20, 40)
	
	// Test index building
	index := optimizer.BuildIndex(statechart)
	if index == nil {
		t.Fatal("Expected index to be built")
	}
	
	// Test fast lookup
	rootState, depth, children := optimizer.FastLookup(statechart, "__root__")
	if rootState == nil {
		t.Error("Expected to find root state")
	}
	if depth != 0 {
		t.Errorf("Expected root depth 0, got %d", depth)
	}
	if len(children) == 0 {
		t.Error("Expected root to have children")
	}
	
	// Test transition lookup
	transitions := optimizer.GetTransitionsForState(statechart, "State0")
	// Should find transitions that have State0 as a source
	if transitions == nil {
		t.Error("Expected transitions to be returned (even if empty)")
	}
	
	// Test cached index (should return same instance)
	index2 := optimizer.BuildIndex(statechart)
	if index != index2 {
		t.Error("Expected cached index to be returned")
	}
}

func TestRingBuffer(t *testing.T) {
	buffer := NewRingBuffer(3)
	
	if buffer.Size() != 0 {
		t.Error("Expected empty buffer")
	}
	
	// Add items
	buffer.Add("item1")
	buffer.Add("item2")
	buffer.Add("item3")
	
	if buffer.Size() != 3 {
		t.Errorf("Expected size 3, got %d", buffer.Size())
	}
	
	items := buffer.GetAll()
	if len(items) != 3 {
		t.Errorf("Expected 3 items, got %d", len(items))
	}
	
	// Add one more item (should wrap around)
	buffer.Add("item4")
	
	if buffer.Size() != 3 {
		t.Errorf("Expected size to remain 3, got %d", buffer.Size())
	}
	
	items = buffer.GetAll()
	if len(items) != 3 {
		t.Errorf("Expected 3 items after wrap, got %d", len(items))
	}
	
	// The first item should be "item2" now (item1 was overwritten)
	if items[0] != "item2" {
		t.Errorf("Expected first item to be 'item2', got %v", items[0])
	}
}

func TestMemoryEfficientMachine(t *testing.T) {
	statechart := testutil.CreateSimpleStatechart()
	machine := NewMemoryEfficientMachine(statechart, 5) // Small history size
	
	// Test initial state
	config := machine.GetCurrentConfiguration()
	if config == nil {
		t.Fatal("Expected initial configuration")
	}
	
	// Test step execution
	for i := 0; i < 10; i++ {
		err := machine.Step("go")
		if err != nil {
			t.Errorf("Step %d failed: %v", i, err)
		}
	}
	
	// Check history is limited
	history := machine.GetHistory()
	if len(history) > 5 {
		t.Errorf("Expected history to be limited to 5, got %d", len(history))
	}
}

func TestMemoryMonitor(t *testing.T) {
	monitor := NewMemoryMonitor()
	
	monitor.SetThreshold("test_threshold", 1024, func(snapshot MemorySnapshot) {
		// Callback for threshold hit
	})
	
	// Check thresholds (may or may not trigger depending on current memory usage)
	monitor.CheckThresholds()
	
	// Start monitoring for a short time
	monitor.StartMonitoring(10 * time.Millisecond)
	time.Sleep(50 * time.Millisecond)
	
	// The test passes if no panics occur
	// In a real scenario, you'd allocate memory to trigger thresholds
}

// Benchmark tests for optimizations
func BenchmarkObjectPoolVsAllocation(b *testing.B) {
	pool := NewObjectPool()
	
	b.Run("DirectAllocation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			config := &sc.Configuration{
				States: make([]*sc.StateRef, 0, 8),
			}
			_ = config
		}
	})
	
	b.Run("ObjectPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			config := pool.GetConfiguration()
			pool.PutConfiguration(config)
		}
	})
}

func BenchmarkOptimizedVsRegularMachine(b *testing.B) {
	pool := NewObjectPool()
	statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
	
	b.Run("RegularMachine", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			machine, _ := semantics.NewMachine(statechart, "regular", nil)
			machine.Start()
			machine.Step("go")
			machine.Stop()
		}
	})
	
	b.Run("OptimizedMachine", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			machine, _ := NewOptimizedMachine(statechart, "optimized", pool)
			machine.Start()
			machine.OptimizedStep("go")
			machine.Stop()
		}
	})
}

func BenchmarkRingBufferVsSlice(b *testing.B) {
	const historySize = 100
	
	b.Run("SliceAppend", func(b *testing.B) {
		var history []string
		for i := 0; i < b.N; i++ {
			history = append(history, "item")
			if len(history) > historySize {
				// Simulate compaction
				copy(history, history[len(history)-historySize/2:])
				history = history[:historySize/2]
			}
		}
	})
	
	b.Run("RingBuffer", func(b *testing.B) {
		buffer := NewRingBuffer(historySize)
		for i := 0; i < b.N; i++ {
			buffer.Add("item")
		}
	})
}

func BenchmarkStatechartOptimization(b *testing.B) {
	optimizer := NewMemoryOptimizer()
	original := testutil.CreateLargeStatechart(50, 100)
	
	b.Run("Unoptimized", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sc := semantics.NewStatechart(original)
			_ = sc.Validate()
		}
	})
	
	b.Run("Optimized", func(b *testing.B) {
		optimized := optimizer.OptimizeStatechart(original)
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			sc := semantics.NewStatechart(optimized)
			_ = sc.Validate()
		}
	})
}

func BenchmarkLargeStatechartIndexing(b *testing.B) {
	optimizer := NewLargeStatechartOptimizer()
	statechart := testutil.CreateLargeStatechart(100, 200)
	
	b.Run("WithoutIndex", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// Simulate looking up states without index
			for _, state := range statechart.RootState.Children {
				_ = state.Label
			}
		}
	})
	
	b.Run("WithIndex", func(b *testing.B) {
		index := optimizer.BuildIndex(statechart)
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			for label := range index.StatesByLabel {
				_ = index.StatesByLabel[label]
			}
		}
	})
}

func BenchmarkMemoryEfficientMachine(b *testing.B) {
	statechart := testutil.CreateSimpleStatechart()
	
	b.Run("RegularMachine", func(b *testing.B) {
		sc := semantics.NewStatechart(statechart)
		machine, _ := semantics.NewMachine(sc, "regular", nil)
		machine.Start()
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			machine.Step("go")
		}
	})
	
	b.Run("MemoryEfficientMachine", func(b *testing.B) {
		machine := NewMemoryEfficientMachine(statechart, 100)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			machine.Step("go")
		}
	})
}