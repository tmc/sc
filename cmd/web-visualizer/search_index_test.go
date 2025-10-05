package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestSearchIndexBasicOperations(t *testing.T) {
	// Create temporary database
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_search.db")

	idx, err := NewSearchIndex(dbPath)
	if err != nil {
		t.Fatalf("Failed to create search index: %v", err)
	}
	defer idx.Close()

	// Create a test machine
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "Idle", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "Active", Type: sc.StateTypeBasic},
			},
		},
	}

	machine, err := semantics.NewMachine(
		semantics.NewStatechart(chart),
		"test-machine-1",
		&structpb.Struct{},
	)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Index the machine
	tags := []string{"test", "demo"}
	metadata := map[string]string{"project": "test", "env": "dev"}

	if err := idx.IndexMachine("test-machine-1", machine, tags, metadata); err != nil {
		t.Fatalf("Failed to index machine: %v", err)
	}

	// Search for the machine
	results, err := idx.Search("test", 10, 0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected at least one search result")
	}

	if results[0].ID != "test-machine-1" {
		t.Errorf("Expected machine ID 'test-machine-1', got '%s'", results[0].ID)
	}

	// Delete the machine
	if err := idx.DeleteMachine("test-machine-1"); err != nil {
		t.Fatalf("Failed to delete machine: %v", err)
	}

	// Verify deletion
	results, err = idx.Search("test", 10, 0)
	if err != nil {
		t.Fatalf("Search after delete failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected no results after delete, got %d", len(results))
	}
}

func TestSearchIndexPerformance(t *testing.T) {
	// Create temporary database
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_perf.db")

	idx, err := NewSearchIndex(dbPath)
	if err != nil {
		t.Fatalf("Failed to create search index: %v", err)
	}
	defer idx.Close()

	// Create test statechart template
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "StateA", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "StateB", Type: sc.StateTypeBasic},
				{Label: "StateC", Type: sc.StateTypeBasic},
			},
		},
	}

	// Index 1000 machines
	numMachines := 1000
	t.Logf("Indexing %d machines...", numMachines)

	startIndex := time.Now()
	for i := 0; i < numMachines; i++ {
		machineID := fmt.Sprintf("machine-%d", i)

		machine, err := semantics.NewMachine(
			semantics.NewStatechart(chart),
			machineID,
			&structpb.Struct{},
		)
		if err != nil {
			t.Fatalf("Failed to create machine %d: %v", i, err)
		}

		tags := []string{
			fmt.Sprintf("tag-%d", i%10),
			fmt.Sprintf("category-%d", i%5),
		}
		metadata := map[string]string{
			"index": fmt.Sprintf("%d", i),
			"group": fmt.Sprintf("group-%d", i%20),
		}

		if err := idx.IndexMachine(machineID, machine, tags, metadata); err != nil {
			t.Fatalf("Failed to index machine %d: %v", i, err)
		}
	}
	indexDuration := time.Since(startIndex)
	t.Logf("Indexed %d machines in %v (%.2f ms per machine)",
		numMachines, indexDuration, float64(indexDuration.Microseconds())/float64(numMachines)/1000.0)

	// Perform searches and measure latency
	searchTerms := []string{
		"machine-500",
		"tag-5",
		"StateA",
		"group-10",
		"category-3",
	}

	for _, term := range searchTerms {
		start := time.Now()
		results, err := idx.Search(term, 10, 0)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("Search for '%s' failed: %v", term, err)
		}

		t.Logf("Search for '%s': %d results in %v (%.2f ms)",
			term, len(results), elapsed, float64(elapsed.Microseconds())/1000.0)

		// Validate <50ms latency requirement
		if elapsed.Milliseconds() > 50 {
			t.Errorf("Search latency %.2f ms exceeds 50ms target for term '%s'",
				float64(elapsed.Microseconds())/1000.0, term)
		}
	}

	// Test prefix search performance
	t.Log("\nTesting prefix search performance...")
	prefixes := []string{"machine-1", "machine-5", "machine-9"}

	for _, prefix := range prefixes {
		start := time.Now()
		results, err := idx.SearchByPrefix(prefix, 10)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("Prefix search for '%s' failed: %v", prefix, err)
		}

		t.Logf("Prefix search for '%s': %d results in %v (%.2f ms)",
			prefix, len(results), elapsed, float64(elapsed.Microseconds())/1000.0)

		if elapsed.Milliseconds() > 50 {
			t.Errorf("Prefix search latency %.2f ms exceeds 50ms target for prefix '%s'",
				float64(elapsed.Microseconds())/1000.0, prefix)
		}
	}

	// Test count operation
	start := time.Now()
	count, err := idx.Count()
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Count failed: %v", err)
	}

	if count != numMachines {
		t.Errorf("Expected count of %d, got %d", numMachines, count)
	}

	t.Logf("Count operation: %d machines in %v (%.2f ms)",
		count, elapsed, float64(elapsed.Microseconds())/1000.0)
}

func TestSearchIndexRanking(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_ranking.db")

	idx, err := NewSearchIndex(dbPath)
	if err != nil {
		t.Fatalf("Failed to create search index: %v", err)
	}
	defer idx.Close()

	// Create machines with different relevance
	machines := []struct {
		id       string
		tags     []string
		metadata map[string]string
	}{
		{
			id:   "payment-processor",
			tags: []string{"payment", "critical", "production"},
			metadata: map[string]string{
				"description": "payment processing state machine",
			},
		},
		{
			id:   "payment-validator",
			tags: []string{"payment", "validation"},
			metadata: map[string]string{
				"description": "validates payment data",
			},
		},
		{
			id:   "order-processor",
			tags: []string{"orders", "processing"},
			metadata: map[string]string{
				"description": "order processing workflow",
			},
		},
	}

	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeBasic,
		},
	}

	for _, m := range machines {
		machine, err := semantics.NewMachine(
			semantics.NewStatechart(chart),
			m.id,
			&structpb.Struct{},
		)
		if err != nil {
			t.Fatalf("Failed to create machine %s: %v", m.id, err)
		}

		if err := idx.IndexMachine(m.id, machine, m.tags, m.metadata); err != nil {
			t.Fatalf("Failed to index machine %s: %v", m.id, err)
		}
	}

	// Search for "payment" - should rank payment-processor higher
	results, err := idx.Search("payment", 10, 0)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) < 2 {
		t.Fatalf("Expected at least 2 results for 'payment', got %d", len(results))
	}

	t.Logf("Search results for 'payment':")
	for i, r := range results {
		t.Logf("  %d. %s (rank: %.2f, snippet: %s)", i+1, r.ID, r.Rank, r.Snippet)
	}

	// Verify ranking (lower BM25 score = better rank in SQLite FTS5)
	if results[0].Rank > results[1].Rank {
		t.Errorf("Expected first result to have better (lower) rank than second")
	}
}

func BenchmarkSearchIndex(b *testing.B) {
	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench.db")

	idx, err := NewSearchIndex(dbPath)
	if err != nil {
		b.Fatalf("Failed to create search index: %v", err)
	}
	defer idx.Close()

	// Index sample machines
	chart := &sc.Statechart{
		RootState: &sc.State{Label: "__root__", Type: sc.StateTypeBasic},
	}

	for i := 0; i < 100; i++ {
		machineID := fmt.Sprintf("bench-machine-%d", i)
		machine, _ := semantics.NewMachine(
			semantics.NewStatechart(chart),
			machineID,
			&structpb.Struct{},
		)
		idx.IndexMachine(machineID, machine, []string{"bench"}, nil)
	}

	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := idx.Search("bench", 10, 0)
			if err != nil {
				b.Fatalf("Search failed: %v", err)
			}
		}
	})

	b.Run("SearchByPrefix", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := idx.SearchByPrefix("bench-machine", 10)
			if err != nil {
				b.Fatalf("Prefix search failed: %v", err)
			}
		}
	})

	b.Run("Count", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := idx.Count()
			if err != nil {
				b.Fatalf("Count failed: %v", err)
			}
		}
	})
}

func TestMain(m *testing.M) {
	// Ensure test cleanup
	code := m.Run()
	os.Exit(code)
}
