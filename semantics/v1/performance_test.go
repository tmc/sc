package semantics

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/sc"
	testutil "github.com/tmc/sc/testing"
	"google.golang.org/protobuf/types/known/structpb"
)

// BenchmarkStatechartValidation measures statechart validation performance
func BenchmarkStatechartValidation(b *testing.B) {
	tests := []struct {
		name string
		fn   func() *sc.Statechart
	}{
		{"Simple", testutil.CreateSimpleStatechart},
		{"Hierarchical", testutil.CreateHierarchicalStatechart},
		{"Orthogonal", testutil.CreateOrthogonalStatechart},
		{"Large", func() *sc.Statechart { return testutil.CreateLargeStatechart(50, 100) }},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			statechart := NewStatechart(tt.fn())
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				if err := statechart.Validate(); err != nil {
					b.Fatalf("Validation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkInitialConfiguration measures initial configuration computation
func BenchmarkInitialConfiguration(b *testing.B) {
	tests := []struct {
		name string
		fn   func() *sc.Statechart
	}{
		{"Simple", testutil.CreateSimpleStatechart},
		{"Hierarchical", testutil.CreateHierarchicalStatechart},
		{"Orthogonal", testutil.CreateOrthogonalStatechart},
		{"Large", func() *sc.Statechart { return testutil.CreateLargeStatechart(50, 100) }},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			statechart := NewStatechart(tt.fn())
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				_, err := statechart.InitialConfiguration()
				if err != nil {
					b.Fatalf("InitialConfiguration failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkMachineStep measures machine step execution performance
func BenchmarkMachineStep(b *testing.B) {
	statechart := testutil.CreateSimpleStatechart()
	machine, err := NewMachine(NewStatechart(statechart), "bench-machine", nil)
	if err != nil {
		b.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		b.Fatalf("Failed to start machine: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Alternate between two events to trigger transitions
		event := "event1"
		if i%2 == 0 {
			event = "event2"
		}
		
		_, err := machine.Step(event)
		if err != nil {
			// Some steps might not trigger transitions, which is fine
			continue
		}
	}
}

// BenchmarkEventProcessing measures event processing performance
func BenchmarkEventProcessing(b *testing.B) {
	machine := &sc.Machine{
		Id:    "event-bench",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{Fields: make(map[string]*structpb.Value)},
		Statechart: testutil.CreateSimpleStatechart(),
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "idle"}},
		},
	}

	processor := NewEventProcessor(machine)
	processor.Start()
	defer processor.Stop()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		processor.SendEvent("test_event", nil)
	}

	// Allow events to be processed
	time.Sleep(10 * time.Millisecond)
}

// BenchmarkEventQueue measures event queue performance
func BenchmarkEventQueue(b *testing.B) {
	queue := NewEventQueue()
	defer queue.Close()

	event := ProcessedEvent{
		Event:     &sc.Event{Label: "bench_event"},
		Priority:  PriorityNormal,
		Timestamp: time.Now(),
		ID:        "bench_id",
	}

	b.Run("Enqueue", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			queue.Enqueue(event)
		}
	})

	b.Run("Dequeue", func(b *testing.B) {
		ctx := context.Background()
		
		// Pre-fill queue
		for i := 0; i < b.N; i++ {
			queue.Enqueue(event)
		}
		
		b.ResetTimer()
		b.ReportAllocs()
		
		for i := 0; i < b.N; i++ {
			_, ok := queue.Dequeue(ctx)
			if !ok {
				break
			}
		}
	})

	b.Run("ConcurrentEnqueueDequeue", func(b *testing.B) {
		ctx := context.Background()
		
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				// Enqueue and dequeue in the same goroutine
				queue.Enqueue(event)
				queue.Dequeue(ctx)
			}
		})
	})
}

// BenchmarkTransitionExecution measures transition execution performance
func BenchmarkTransitionExecution(b *testing.B) {
	statechart := NewStatechart(testutil.CreateHierarchicalStatechart())
	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: "idle"}},
	}
	context := &structpb.Struct{Fields: make(map[string]*structpb.Value)}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		result, err := statechart.ExecuteTransitions(config, context, "start")
		if err != nil {
			b.Fatalf("ExecuteTransitions failed: %v", err)
		}
		config = result.NewConfig
		context = result.NewContext
	}
}

// BenchmarkMemoryUsage measures memory usage patterns
func BenchmarkMemoryUsage(b *testing.B) {
	b.Run("SmallStatechart", func(b *testing.B) {
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			statechart := NewStatechart(testutil.CreateSimpleStatechart())
			machine, _ := NewMachine(statechart, fmt.Sprintf("machine-%d", i), nil)
			machine.Start()
			machine.Stop()
		}

		runtime.GC()
		runtime.ReadMemStats(&m2)
		
		b.ReportMetric(float64(m2.Alloc-m1.Alloc)/float64(b.N), "bytes/op")
	})

	b.Run("LargeStatechart", func(b *testing.B) {
		var m1, m2 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m1)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			statechart := NewStatechart(testutil.CreateLargeStatechart(100, 200))
			machine, _ := NewMachine(statechart, fmt.Sprintf("machine-%d", i), nil)
			machine.Start()
			machine.Stop()
		}

		runtime.GC()
		runtime.ReadMemStats(&m2)
		
		b.ReportMetric(float64(m2.Alloc-m1.Alloc)/float64(b.N), "bytes/op")
	})
}

// BenchmarkConcurrentMachines measures performance with multiple machines
func BenchmarkConcurrentMachines(b *testing.B) {
	numMachines := []int{1, 10, 100, 1000}
	
	for _, n := range numMachines {
		b.Run(fmt.Sprintf("Machines_%d", n), func(b *testing.B) {
			machines := make([]*MachineWrapper, n)
			
			// Setup
			for i := 0; i < n; i++ {
				statechart := NewStatechart(testutil.CreateSimpleStatechart())
				machine, err := NewMachine(statechart, fmt.Sprintf("machine-%d", i), nil)
				if err != nil {
					b.Fatalf("Failed to create machine %d: %v", i, err)
				}
				machines[i] = machine
				machine.Start()
			}
			
			b.ResetTimer()
			b.ReportAllocs()
			
			// Benchmark concurrent step execution
			b.RunParallel(func(pb *testing.PB) {
				machineIdx := 0
				for pb.Next() {
					machine := machines[machineIdx%n]
					machine.Step("test_event")
					machineIdx++
				}
			})
			
			// Cleanup
			for _, machine := range machines {
				machine.Stop()
			}
		})
	}
}

// BenchmarkLargeStatechartOperations measures performance on large statecharts
func BenchmarkLargeStatechartOperations(b *testing.B) {
	sizes := []struct {
		name   string
		states int
		trans  int
	}{
		{"Small_10_20", 10, 20},
		{"Medium_50_100", 50, 100},
		{"Large_100_500", 100, 500},
		{"XLarge_500_1000", 500, 1000},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			statechart := NewStatechart(testutil.CreateLargeStatechart(size.states, size.trans))
			
			b.Run("Validation", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					statechart.Validate()
				}
			})
			
			b.Run("InitialConfig", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					statechart.InitialConfiguration()
				}
			})
			
			b.Run("FindTransitions", func(b *testing.B) {
				config := &sc.Configuration{
					States: []*sc.StateRef{{Label: "state_0"}},
				}
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					statechart.FindEnabledTransitions(config, nil, "event_0")
				}
			})
		})
	}
}

// BenchmarkEventTracing measures event tracing overhead
func BenchmarkEventTracing(b *testing.B) {
	machine := &sc.Machine{
		Id:    "trace-bench",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{Fields: make(map[string]*structpb.Value)},
		Statechart: testutil.CreateSimpleStatechart(),
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "idle"}},
		},
	}

	b.Run("WithoutTracing", func(b *testing.B) {
		processor := NewEventProcessor(machine)
		processor.Start()
		defer processor.Stop()
		
		b.ResetTimer()
		b.ReportAllocs()
		
		for i := 0; i < b.N; i++ {
			processor.SendEvent("test_event", nil)
		}
	})

	b.Run("WithTracing", func(b *testing.B) {
		processor := NewEventProcessor(machine)
		processor.EnableTracing()
		processor.Start()
		defer processor.Stop()
		
		b.ResetTimer()
		b.ReportAllocs()
		
		for i := 0; i < b.N; i++ {
			processor.SendEvent("test_event", nil)
		}
		
		// Clear trace periodically to prevent unbounded growth in benchmark
		if b.N > 1000 {
			processor.ClearTrace()
		}
	})
}

// BenchmarkFilterProcessing measures event filter performance
func BenchmarkFilterProcessing(b *testing.B) {
	machine := &sc.Machine{
		Id:    "filter-bench",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{Fields: make(map[string]*structpb.Value)},
		Statechart: testutil.CreateSimpleStatechart(),
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "idle"}},
		},
	}

	processor := NewEventProcessor(machine)
	
	// Add multiple filters
	for i := 0; i < 10; i++ {
		filter := NewConditionalFilter(fmt.Sprintf("filter_%d", i), 
			func(event ProcessedEvent, machine *sc.Machine) bool {
				return len(event.Event.Label) > 0
			})
		processor.AddFilter(filter)
	}
	
	processor.Start()
	defer processor.Stop()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		processor.SendEvent("filtered_event", nil)
	}
}