package main

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/tmc/sc/semantics/v1"
)

// Ensure SQLite is compiled with FTS5 support via build tags
// Build with: go build -tags "fts5"

// SearchIndex provides FTS5-based search indexing for machines.
type SearchIndex struct {
	db    *sql.DB
	mutex sync.RWMutex
}

// SearchResult represents a search result with relevance ranking.
type SearchResult struct {
	ID            string
	Rank          float64
	Snippet       string
	Configuration []string
	State         string
	Timestamp     time.Time
}

// NewSearchIndex creates a new FTS5 search index.
func NewSearchIndex(dbPath string) (*SearchIndex, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	idx := &SearchIndex{
		db: db,
	}

	if err := idx.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return idx, nil
}

// initSchema creates the FTS5 virtual table and supporting tables.
func (idx *SearchIndex) initSchema() error {
	schema := `
	-- Main metadata table for machines
	CREATE TABLE IF NOT EXISTS machines (
		id TEXT PRIMARY KEY,
		state TEXT NOT NULL,
		configuration TEXT NOT NULL,
		context_json TEXT,
		tags_json TEXT,
		metadata_json TEXT,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL
	);

	-- FTS5 virtual table for full-text search
	CREATE VIRTUAL TABLE IF NOT EXISTS machines_fts USING fts5(
		id UNINDEXED,
		machine_id,
		state,
		configuration,
		tags,
		metadata,
		content='machines',
		content_rowid='rowid',
		tokenize='porter unicode61'
	);

	-- Triggers to keep FTS5 index in sync
	CREATE TRIGGER IF NOT EXISTS machines_ai AFTER INSERT ON machines BEGIN
		INSERT INTO machines_fts(rowid, id, machine_id, state, configuration, tags, metadata)
		VALUES (new.rowid, new.id, new.id, new.state, new.configuration, new.tags_json, new.metadata_json);
	END;

	CREATE TRIGGER IF NOT EXISTS machines_ad AFTER DELETE ON machines BEGIN
		DELETE FROM machines_fts WHERE rowid = old.rowid;
	END;

	CREATE TRIGGER IF NOT EXISTS machines_au AFTER UPDATE ON machines BEGIN
		UPDATE machines_fts SET
			machine_id = new.id,
			state = new.state,
			configuration = new.configuration,
			tags = new.tags_json,
			metadata = new.metadata_json
		WHERE rowid = old.rowid;
	END;

	-- Index for efficient timestamp-based queries
	CREATE INDEX IF NOT EXISTS idx_machines_updated_at ON machines(updated_at);
	CREATE INDEX IF NOT EXISTS idx_machines_created_at ON machines(created_at);
	`

	if _, err := idx.db.Exec(schema); err != nil {
		return fmt.Errorf("failed to create schema: %w", err)
	}

	return nil
}

// IndexMachine adds or updates a machine in the search index.
func (idx *SearchIndex) IndexMachine(
	id string,
	machine *semantics.MachineWrapper,
	tags []string,
	metadata map[string]string,
) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	configuration := strings.Join(getStateLabels(machine.Configuration.States), " ")
	tagsStr := strings.Join(tags, " ")
	metadataStr := ""
	if len(metadata) > 0 {
		parts := make([]string, 0, len(metadata))
		for k, v := range metadata {
			parts = append(parts, k+": "+v)
		}
		metadataStr = strings.Join(parts, " ")
	}

	now := time.Now().Unix()

	// Use INSERT OR REPLACE to handle both insert and update
	query := `
		INSERT OR REPLACE INTO machines (
			id, state, configuration, context_json, tags_json, metadata_json, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?,
			COALESCE((SELECT created_at FROM machines WHERE id = ?), ?),
			?
		)
	`

	_, err := idx.db.Exec(
		query,
		id,
		machine.State.String(),
		configuration,
		"{}",           // context_json placeholder
		tagsStr,        // tags_json
		metadataStr,    // metadata_json
		id,             // for COALESCE
		now,            // default created_at
		now,            // updated_at
	)

	if err != nil {
		return fmt.Errorf("failed to index machine: %w", err)
	}

	return nil
}

// DeleteMachine removes a machine from the search index.
func (idx *SearchIndex) DeleteMachine(id string) error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	_, err := idx.db.Exec("DELETE FROM machines WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("failed to delete from index: %w", err)
	}

	return nil
}

// Search performs a full-text search using FTS5.
func (idx *SearchIndex) Search(query string, limit, offset int) ([]SearchResult, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	if limit <= 0 {
		limit = 10
	}

	// Escape query for FTS5 - wrap in quotes to treat as phrase
	ftsQuery := `"` + strings.ReplaceAll(query, `"`, `""`) + `"`

	// Use FTS5 MATCH for full-text search with BM25 ranking
	sqlQuery := `
		SELECT
			m.id,
			m.state,
			m.configuration,
			m.updated_at,
			bm25(machines_fts) as rank
		FROM machines m
		JOIN machines_fts ON machines_fts.rowid = m.rowid
		WHERE machines_fts MATCH ?
		ORDER BY rank
		LIMIT ? OFFSET ?
	`

	rows, err := idx.db.Query(sqlQuery, ftsQuery, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var r SearchResult
		var updatedAt int64
		var configStr string

		if err := rows.Scan(&r.ID, &r.State, &configStr, &updatedAt, &r.Rank); err != nil {
			continue
		}

		r.Configuration = strings.Fields(configStr)
		r.Timestamp = time.Unix(updatedAt, 0)
		r.Snippet = configStr // Use configuration as snippet for now
		results = append(results, r)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating results: %w", err)
	}

	return results, nil
}

// SearchByPrefix finds machines by ID prefix (for autocomplete).
func (idx *SearchIndex) SearchByPrefix(prefix string, limit int) ([]string, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	if limit <= 0 {
		limit = 10
	}

	query := `
		SELECT id FROM machines
		WHERE id LIKE ? || '%'
		ORDER BY id
		LIMIT ?
	`

	rows, err := idx.db.Query(query, prefix, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search by prefix: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			continue
		}
		ids = append(ids, id)
	}

	return ids, nil
}

// Count returns the total number of indexed machines.
func (idx *SearchIndex) Count() (int, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	var count int
	err := idx.db.QueryRow("SELECT COUNT(*) FROM machines").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to count machines: %w", err)
	}

	return count, nil
}

// GetStats returns search index statistics.
func (idx *SearchIndex) GetStats() (map[string]interface{}, error) {
	idx.mutex.RLock()
	defer idx.mutex.RUnlock()

	stats := make(map[string]interface{})

	var count int
	if err := idx.db.QueryRow("SELECT COUNT(*) FROM machines").Scan(&count); err != nil {
		return nil, fmt.Errorf("failed to get count: %w", err)
	}
	stats["total_machines"] = count

	// Get FTS5 table statistics
	var ftsStats string
	err := idx.db.QueryRow("SELECT * FROM machines_fts WHERE machines_fts = 'optimize'").Scan(&ftsStats)
	if err == nil {
		stats["fts_status"] = ftsStats
	}

	return stats, nil
}

// Optimize rebuilds the FTS5 index for better performance.
func (idx *SearchIndex) Optimize() error {
	idx.mutex.Lock()
	defer idx.mutex.Unlock()

	_, err := idx.db.Exec("INSERT INTO machines_fts(machines_fts) VALUES('optimize')")
	if err != nil {
		return fmt.Errorf("failed to optimize index: %w", err)
	}

	return nil
}

// Close closes the search index database connection.
func (idx *SearchIndex) Close() error {
	return idx.db.Close()
}
