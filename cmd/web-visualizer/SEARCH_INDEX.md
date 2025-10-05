# SQLite FTS5 Search Index

## Overview

The GraphQL server now includes a SQLite-based FTS5 (Full-Text Search 5) index that replaces the O(n) filesystem scan with indexed search, achieving <50ms latency for all search operations.

## Architecture

### Components

1. **SearchIndex** (`search_index.go`)
   - FTS5-based full-text search engine
   - Thread-safe concurrent access with RWMutex
   - Automatic index maintenance via SQLite triggers
   - BM25 ranking algorithm for relevance scoring

2. **Database Schema**
   - `machines` table: Core metadata storage
   - `machines_fts` virtual table: FTS5 index
   - Automatic triggers: Keep FTS5 in sync on INSERT/UPDATE/DELETE

3. **Integration** (`main.go`)
   - Search index initialized on server startup
   - Automatic indexing on machine create/update
   - Automatic cleanup on machine delete
   - Graceful fallback to linear scan if index unavailable

## Performance

Benchmark results from `TestSearchIndexPerformance` (1000 machines indexed):

| Operation | Latency | Target | Status |
|-----------|---------|--------|--------|
| FTS Search | 0.11-0.37ms | <50ms | ✅ Pass |
| Prefix Search | 0.03-0.05ms | <50ms | ✅ Pass |
| Count | 0.02ms | <50ms | ✅ Pass |
| Index Rate | 0.45ms/machine | - | - |

All operations are **well under the 50ms latency target**.

## Usage

### Build Requirements

The SQLite library must be compiled with FTS5 support:

```bash
go build -tags "fts5" ./cmd/web-visualizer
go test -tags "fts5" ./cmd/web-visualizer
```

### API

The search endpoint automatically uses the FTS5 index when available:

```bash
POST /api/v1/machines/search
{
  "query": "payment processor",
  "limit": 10,
  "offset": 0
}
```

Response includes search metrics:

```json
{
  "machines": [...],
  "total": 1000,
  "offset": 0,
  "limit": 10,
  "query_time": "142µs"
}
```

The response header `X-Search-Time-Ms` provides latency in milliseconds.

### Features

1. **Full-Text Search**
   - Searches across machine ID, state, configuration, tags, and metadata
   - BM25 relevance ranking
   - Porter stemming with Unicode support
   - Phrase search with automatic escaping

2. **Prefix Search**
   - Fast autocomplete for machine IDs
   - O(log n) lookup performance

3. **Index Maintenance**
   - Automatic indexing on machine create/update
   - Automatic cleanup on delete
   - SQLite triggers ensure consistency

4. **Graceful Degradation**
   - Falls back to linear scan if index unavailable
   - Logs errors but continues serving requests

## Implementation Details

### FTS5 Configuration

```sql
CREATE VIRTUAL TABLE machines_fts USING fts5(
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
```

- `porter`: Porter stemming algorithm for better matching
- `unicode61`: Full Unicode support
- `content='machines'`: External content table (saves space)

### Query Escaping

Queries are automatically escaped for FTS5 by wrapping in quotes:

```go
ftsQuery := `"` + strings.ReplaceAll(query, `"`, `""`) + `"`
```

This treats hyphens and special characters as literals.

### BM25 Ranking

Results are ranked using SQLite's built-in BM25 algorithm:

```sql
ORDER BY bm25(machines_fts)
```

Lower scores indicate better matches.

## Testing

Run the comprehensive test suite:

```bash
go test -tags fts5 -v -run TestSearchIndexPerformance
```

Tests include:
- Basic CRUD operations
- Performance benchmarks (1000 machines)
- Relevance ranking validation
- Concurrent access safety

## Database Location

The search index database is stored at:

```
$DATA_DIR/search.db
```

Default: `./data/search.db`

## Optimization

To optimize the FTS5 index (optional):

```go
searchIndex.Optimize()
```

This merges FTS5 segments for better query performance.

## Statistics

Get index statistics:

```go
stats, err := searchIndex.GetStats()
// stats["total_machines"] = count
// stats["fts_status"] = optimization status
```

## Migration Notes

- Existing deployments will automatically build the index on first startup
- No manual migration required
- Old linear scan remains as fallback for compatibility

## References

- [SQLite FTS5 Documentation](https://www.sqlite.org/fts5.html)
- [BM25 Ranking](https://en.wikipedia.org/wiki/Okapi_BM25)
- Implementation: `cmd/web-visualizer/search_index.go`
- Tests: `cmd/web-visualizer/search_index_test.go`
