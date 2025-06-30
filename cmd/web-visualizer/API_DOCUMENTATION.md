# Enhanced Statechart Web Visualizer API Documentation

This document describes the RESTful API and WebSocket endpoints for the enhanced statechart web visualizer backend.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

Currently, no authentication is required. In production, consider implementing API keys or OAuth.

## Machine Management

### Create Machine

**POST** `/machines`

Creates a new statechart machine instance.

**Request Body:**
```json
{
  "id": "string (required)",
  "statechart": {
    "root_state": {
      "label": "root",
      "type": "NORMAL",
      "children": [...]
    },
    "transitions": [...]
  },
  "context": {
    "key": "value"
  },
  "tags": ["tag1", "tag2"],
  "metadata": {
    "description": "Machine description",
    "author": "User name"
  }
}
```

**Response (201):**
```json
{
  "id": "machine-1",
  "configuration": ["state1", "state2"],
  "context": {},
  "tags": ["tag1"],
  "metadata": {},
  "created_at": "2023-12-01T10:00:00Z"
}
```

### List Machines

**GET** `/machines`

Returns a list of all machine instances.

**Response (200):**
```json
{
  "machines": [
    {
      "id": "machine-1",
      "state": "RUNNING",
      "configuration": ["state1"],
      "last_updated": "2023-12-01T10:00:00Z"
    }
  ],
  "total": 1
}
```

### Get Machine

**GET** `/machines/{id}`

Returns detailed information about a specific machine.

**Response (200):**
```json
{
  "id": "machine-1",
  "state": "RUNNING",
  "configuration": ["state1"],
  "context": {},
  "statechart": {...},
  "step_history": [...],
  "errors": [],
  "last_updated": "2023-12-01T10:00:00Z"
}
```

### Update Machine

**PUT** `/machines/{id}`

Updates an existing machine's statechart or context.

**Request Body:**
```json
{
  "statechart": {...},
  "context": {...},
  "tags": [...],
  "metadata": {...}
}
```

### Delete Machine

**DELETE** `/machines/{id}`

Deletes a machine instance.

**Response (204):** No content

### Search Machines

**POST** `/machines/search`

Search for machines based on query parameters.

**Request Body:**
```json
{
  "query": "search text",
  "tags": ["tag1"],
  "limit": 10,
  "offset": 0
}
```

## Machine Operations

### Process Event

**POST** `/machines/{id}/events`

Processes a single event on the machine.

**Request Body:**
```json
{
  "event": "button_click",
  "data": {
    "optional": "payload"
  }
}
```

**Response (200):**
```json
{
  "id": "machine-1",
  "configuration": ["new_state"],
  "context": {},
  "last_step": {...},
  "transitioned": true,
  "timestamp": "2023-12-01T10:00:00Z"
}
```

### Process Batch Events

**POST** `/machines/{id}/events/batch`

Processes multiple events in sequence.

**Request Body:**
```json
{
  "events": [
    {"event": "event1"},
    {"event": "event2", "data": {...}}
  ]
}
```

### Reset Machine

**POST** `/machines/{id}/reset`

Resets the machine to its initial state.

**Response (200):**
```json
{
  "id": "machine-1",
  "configuration": ["initial_state"],
  "context": {},
  "reset_at": "2023-12-01T10:00:00Z"
}
```

### Validate Machine

**GET** `/machines/{id}/validate`

Validates the machine's current state and statechart.

**Response (200):**
```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Machine is currently stopped"]
}
```

## Export Endpoints

### Export as JSON

**GET** `/machines/{id}/export/json`

Exports the machine as a JSON file.

**Response:** File download with machine data.

### Export as XState

**GET** `/machines/{id}/export/xstate`

Exports the machine in XState-compatible format.

**Response:** JSON file compatible with XState library.

### Export as SVG

**GET** `/machines/{id}/export/svg`

Exports a visual representation of the statechart as SVG.

**Response:** SVG file with statechart diagram.

## Examples and Templates

### Get Examples

**GET** `/examples`

Returns predefined example statecharts.

**Response (200):**
```json
{
  "hierarchical": {...},
  "orthogonal": {...},
  "compound": {...}
}
```

## Server Management

### Health Check

**GET** `/health`

Returns server health status.

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00Z",
  "active_machines": 5,
  "active_clients": 2,
  "uptime": "2h30m45s"
}
```

### Metrics

**GET** `/metrics`

Returns server performance metrics.

**Response (200):**
```json
{
  "request_count": 1234,
  "connection_count": 56,
  "error_count": 7,
  "active_machines": 5,
  "active_clients": 2,
  "uptime": "2h30m45s",
  "timestamp": "2023-12-01T10:00:00Z"
}
```

## WebSocket API

### Connection

**WebSocket** `/ws`

Establishes a WebSocket connection for real-time updates.

### Message Types

#### Welcome Message (Server → Client)
```json
{
  "type": "welcome",
  "client_id": "client_1234567890",
  "data": {
    "connected_clients": 3
  },
  "timestamp": "2023-12-01T10:00:00Z"
}
```

#### Subscribe to Machine (Client → Server)
```json
{
  "type": "subscribe",
  "machine_id": "machine-1",
  "data": "machine-1"
}
```

#### Machine State Update (Server → Client)
```json
{
  "type": "machine_state",
  "machine_id": "machine-1",
  "data": {
    "configuration": ["state1"],
    "context": {},
    "state": "RUNNING"
  },
  "timestamp": "2023-12-01T10:00:00Z"
}
```

#### Event Processed (Server → Client)
```json
{
  "type": "event_processed",
  "machine_id": "machine-1",
  "data": {
    "event": "button_click",
    "configuration": ["new_state"],
    "context": {},
    "last_step": {...}
  },
  "timestamp": "2023-12-01T10:00:00Z"
}
```

#### Machine Updated (Server → Client)
```json
{
  "type": "machine_updated",
  "machine_id": "machine-1",
  "data": {
    "action": "created|updated|deleted|reset",
    "configuration": ["state1"],
    "context": {},
    "state": "RUNNING"
  },
  "timestamp": "2023-12-01T10:00:00Z"
}
```

#### Ping/Pong (Bidirectional)
```json
{
  "type": "ping|pong",
  "timestamp": "2023-12-01T10:00:00Z"
}
```

## Error Responses

All endpoints return standardized error responses:

### 400 Bad Request
```json
{
  "error": "Invalid request: missing required field 'id'"
}
```

### 404 Not Found
```json
{
  "error": "Machine not found"
}
```

### 409 Conflict
```json
{
  "error": "Machine with this ID already exists"
}
```

### 500 Internal Server Error
```json
{
  "error": "Failed to process event: invalid state transition"
}
```

## Data Persistence

The server automatically persists machine states to the file system in the configured data directory (default: `./data`). Each machine is stored as a JSON file named `{machine-id}.json`.

### Backup and Restore

The persistence layer supports automatic backups and cleanup of old data. Backups are created in `{data-dir}/backups/` with timestamps.

## Rate Limiting

Currently not implemented. In production, consider implementing rate limiting based on:
- Requests per minute per IP
- WebSocket connections per IP
- Machine creation limits per user

## Security Considerations

1. **Input Validation**: All input is validated before processing
2. **CORS**: Currently allows all origins for development
3. **WebSocket Origin**: Currently allows all origins for development
4. **File System**: Machine data is stored in plain JSON files

## Performance

The server is designed to handle:
- 10+ concurrent users editing machines
- <100ms response times for API calls
- Real-time WebSocket updates with minimal latency
- Memory-efficient machine storage and retrieval

## Development Notes

- API versioning is implemented via `/api/v1` prefix
- All timestamps are in ISO 8601 format
- The server gracefully handles machine validation errors
- WebSocket connections are cleaned up automatically on disconnect
- Comprehensive logging is available for debugging

## Example Usage

### Creating and Using a Machine

1. **Create a machine:**
   ```bash
   curl -X POST http://localhost:8080/api/v1/machines \
     -H "Content-Type: application/json" \
     -d '{"id": "test-machine", "statechart": {...}}'
   ```

2. **Process an event:**
   ```bash
   curl -X POST http://localhost:8080/api/v1/machines/test-machine/events \
     -H "Content-Type: application/json" \
     -d '{"event": "start"}'
   ```

3. **Export the machine:**
   ```bash
   curl http://localhost:8080/api/v1/machines/test-machine/export/json \
     -o test-machine.json
   ```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  // Subscribe to machine updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    machine_id: 'test-machine',
    data: 'test-machine'
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```

This enhanced backend provides a robust foundation for professional-grade statechart editing and collaboration features.