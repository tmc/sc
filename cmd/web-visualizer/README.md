# Statechart Web Visualizer & Simulator

A web-based tool for visualizing and simulating statecharts built with the SC library.

## Features

- **Interactive Visualization**: D3.js-based visualization of statechart hierarchies
- **Real-time Simulation**: Step-by-step execution of statechart transitions
- **Event Injection**: Send events to trigger state transitions
- **State Highlighting**: Visual indication of currently active states
- **Simulation History**: Track and display the execution history
- **Example Statecharts**: Pre-loaded examples (hierarchical and orthogonal)
- **REST API**: Complete API for statechart management and simulation
- **WebSocket Support**: Real-time updates (foundation for future enhancements)

## Architecture

### Backend (Go)
- **Web Server**: Gorilla Mux-based HTTP server
- **REST API**: Endpoints for statechart management and simulation
- **Machine Management**: Create and manage statechart machine instances
- **Event Processing**: Process events and execute transitions
- **WebSocket**: Real-time communication channel

### Frontend (JavaScript)
- **D3.js Visualization**: Interactive hierarchical statechart rendering
- **Simulation Controls**: Play, pause, step, and reset functionality
- **Event Interface**: Manual event injection and predefined event buttons
- **State Display**: Current configuration and context visualization
- **History Tracking**: Visual history of simulation steps

## Quick Start

1. **Build the application**:
   ```bash
   cd cmd/web-visualizer
   go build -o web-visualizer .
   ```

2. **Run the server**:
   ```bash
   ./web-visualizer
   ```
   
   The server will start on port 8080 by default. If port 8080 is in use, specify a different port:
   ```bash
   PORT=8081 ./web-visualizer
   ```

3. **Open the web interface**:
   - Navigate to `http://localhost:8080` (or your specified port)
   - Select an example statechart from the dropdown
   - Click "Load Example" to begin

## Usage

### Loading Examples
1. Select "Hierarchical Statechart" or "Orthogonal Statechart" from the dropdown
2. Click "Load Example" to create a machine instance
3. The visualization will appear on the left panel

### Simulating State Transitions
1. **Using Event Buttons**: Click any of the available event buttons (right panel)
2. **Manual Event Entry**: Type an event name in the input field and click "Send Event"
3. **Available Events**: Each example displays its available events as clickable buttons

### Monitoring Simulation
- **Current States**: Active states are highlighted in green in the visualization
- **State List**: Current active states are shown in the "Current Configuration" panel
- **History**: Each step is recorded in the "Simulation History" panel
- **Context**: Machine context is displayed in JSON format

### Controls
- **Reset**: Reset the machine to its initial state
- **Event Input**: Manually type event names to send
- **Event Buttons**: Quick access to predefined events

## API Endpoints

### Examples
- `GET /api/examples` - Get all available example statecharts

### Machine Management
- `POST /api/machines` - Create a new machine instance
- `GET /api/machines/{id}` - Get machine state and configuration
- `POST /api/machines/{id}/events` - Send an event to the machine
- `POST /api/machines/{id}/reset` - Reset machine to initial state

### WebSocket
- `WS /ws` - WebSocket endpoint for real-time updates

## Example Statecharts

### Hierarchical Statechart (Alarm System)
- **States**: Off, On (Idle, Armed (Monitoring, Triggered))
- **Events**: POWER_ON, POWER_OFF, ARM, DISARM, MOTION_DETECTED, RESET
- **Demonstrates**: State hierarchy, default state selection, cross-level transitions

### Orthogonal Statechart (Media Player)
- **States**: PlaybackControl (PlaybackState (Playing, Paused, Stopped), VolumeControl (Normal, Muted))
- **Events**: PLAY, PAUSE, STOP, MUTE, UNMUTE
- **Demonstrates**: Parallel/orthogonal regions, concurrent state configurations

## API Request Examples

### Create a Machine
```bash
curl -X POST http://localhost:8080/api/machines \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-machine-1",
    "statechart": {...},
    "context": {}
  }'
```

### Send an Event
```bash
curl -X POST http://localhost:8080/api/machines/test-machine-1/events \
  -H "Content-Type: application/json" \
  -d '{"event": "POWER_ON"}'
```

### Get Machine State
```bash
curl http://localhost:8080/api/machines/test-machine-1
```

## File Structure

```
cmd/web-visualizer/
├── main.go           # Go web server with REST API
├── go.mod            # Go module dependencies
├── README.md         # This documentation
└── web/              # Frontend static files
    ├── index.html    # Main HTML interface
    ├── styles.css    # CSS styling
    └── visualizer.js # JavaScript visualization and API client
```

## Development

### Adding New Examples
1. Create the statechart using the semantics package
2. Add it to the `GetExamples` handler in `main.go`
3. Restart the server to see the new example

### Extending the API
- Add new endpoints to the Gorilla Mux router in `main.go`
- Follow the existing pattern for request/response handling
- Update the frontend JavaScript to use new endpoints

### Customizing Visualization
- Modify `visualizer.js` to change the D3.js rendering
- Update `styles.css` for visual styling changes
- The visualization uses a hierarchical tree layout by default

## Troubleshooting

### Port Already in Use
If port 8080 is in use, start the server with a different port:
```bash
PORT=8081 ./web-visualizer
```

### Build Errors
Ensure you have the latest dependencies:
```bash
go mod tidy
```

### Visualization Not Appearing
- Check browser console for JavaScript errors
- Ensure the server is running and accessible
- Verify the example statechart loaded successfully

## Future Enhancements

- **Statechart Editor**: Visual editor for creating custom statecharts
- **Export Functionality**: Export simulation traces and visualizations
- **Advanced Visualization**: More sophisticated layout algorithms
- **Real-time Collaboration**: Multi-user simulation sessions
- **Integration**: Connect to external event sources
- **Testing**: Automated test generation from statecharts