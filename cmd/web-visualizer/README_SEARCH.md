# Search Index Quick Start

## Building

Build with FTS5 support:

```bash
go build -tags fts5 -o web-visualizer .
```

## Running

```bash
./web-visualizer
```

The server will:
1. Create `./data/search.db` for the search index
2. Initialize FTS5 tables and triggers
3. Start listening on http://localhost:8080

## Testing Search

### Create a machine

```bash
curl -X POST http://localhost:8080/api/v1/machines \
  -H "Content-Type: application/json" \
  -d '{
    "id": "payment-processor",
    "statechart": {
      "root_state": {
        "label": "__root__",
        "type": 1,
        "children": [
          {"label": "idle", "type": 0, "is_initial": true},
          {"label": "processing", "type": 0}
        ]
      }
    },
    "tags": ["payment", "critical"],
    "metadata": {"env": "production", "team": "payments"}
  }'
```

### Search

```bash
# Search for "payment"
curl -X POST http://localhost:8080/api/v1/machines/search \
  -H "Content-Type: application/json" \
  -d '{"query": "payment", "limit": 10}'

# Check response header for latency
curl -X POST http://localhost:8080/api/v1/machines/search \
  -H "Content-Type: application/json" \
  -d '{"query": "critical"}' \
  -i | grep X-Search-Time
```

## Performance

Expected latencies (tested with 1000 machines):

- Full-text search: **0.1-0.4ms**
- Prefix search: **0.03-0.05ms**
- Count operations: **0.02ms**

All operations are **<50ms** (target achieved).

## Troubleshooting

### "no such module: fts5"

Build with the `fts5` tag:

```bash
go build -tags fts5
go test -tags fts5
```

### Search not working

Check server logs for:
- Index initialization errors
- Search fallback messages

The server falls back to linear scan if the index fails.

## Development

Run tests:

```bash
go test -tags fts5 -v -run TestSearchIndex
```

Run performance benchmarks:

```bash
go test -tags fts5 -bench=BenchmarkSearchIndex -benchmem
```

See `SEARCH_INDEX.md` for detailed documentation.
