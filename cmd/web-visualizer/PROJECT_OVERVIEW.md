# Statechart Web Visualization Project Overview

## Project Summary

This project implements a comprehensive web-based visualizer and simulator for statecharts, built on top of the existing SC (Statecharts) library. It provides an interactive interface for creating, visualizing, and simulating statechart execution in real-time.

## Key Components

### 1. Backend Server (Go)

**File**: `main.go`

The backend is a Go HTTP server that provides:

- **REST API endpoints** for statechart management
- **Machine lifecycle management** (create, start, stop, reset)
- **Event processing** with step-by-step execution
- **WebSocket support** for real-time updates
- **Example statechart serving** (hierarchical and orthogonal patterns)

**Key API Endpoints**:
- `GET /api/examples` - Retrieve available example statecharts
- `POST /api/machines` - Create new machine instances
- `GET /api/machines/{id}` - Get machine state and configuration
- `POST /api/machines/{id}/events` - Send events to trigger transitions
- `POST /api/machines/{id}/reset` - Reset machine to initial state
- `WS /ws` - WebSocket for real-time communication

### 2. Frontend Application (JavaScript/HTML/CSS)

**Files**: `web/index.html`, `web/visualizer.js`, `web/styles.css`

The frontend provides:

- **Interactive D3.js visualization** of statechart hierarchies
- **Real-time state highlighting** showing active configurations
- **Event injection interface** with manual input and quick-access buttons
- **Simulation controls** (load, reset, step-through)
- **History tracking** displaying execution trace
- **Context monitoring** showing machine state variables
- **Responsive design** working across different screen sizes

**Visualization Features**:
- Hierarchical tree layout of states
- Visual distinction between state types (basic, parallel, orthogonal)
- Transition arrows with event labels
- Active state highlighting in real-time
- Interactive event buttons for quick testing

### 3. Integration Layer

The project seamlessly integrates with the existing SC library:

- **Semantics Package**: Uses `semantics/v1` for machine execution
- **Example Integration**: Leverages existing hierarchical and orthogonal examples
- **Protocol Buffer Types**: Works with the established SC data models
- **Validation**: Uses built-in statechart validation mechanisms

## Technical Architecture

### Data Flow

1. **Initialization**: Frontend loads available examples from backend
2. **Machine Creation**: User selects example, frontend creates machine via API
3. **Visualization**: D3.js renders statechart structure and initial configuration
4. **Event Processing**: User triggers events, backend processes with SC semantics
5. **State Updates**: Backend returns new configuration, frontend updates visualization
6. **History Tracking**: Each step recorded and displayed in simulation history

### State Management

- **Backend**: Maintains machine instances in memory with thread-safe access
- **Frontend**: Tracks current machine state and synchronizes with backend
- **Real-time Updates**: WebSocket foundation for future live collaboration features

### Error Handling

- **Backend**: Comprehensive error handling with HTTP status codes
- **Frontend**: User-friendly error messages and fallback states
- **Validation**: Input validation at both API and UI levels

## Example Statecharts

### 1. Hierarchical Statechart (Alarm System)
- **Purpose**: Demonstrates state hierarchy and cross-level transitions
- **States**: Off → On (Idle ↔ Armed (Monitoring ↔ Triggered))
- **Events**: POWER_ON, POWER_OFF, ARM, DISARM, MOTION_DETECTED, RESET
- **Features**: Default state selection, hierarchical entry/exit

### 2. Orthogonal Statechart (Media Player)
- **Purpose**: Shows concurrent state regions (AND-decomposition)
- **States**: PlaybackControl with parallel regions for playback and volume
- **Events**: PLAY, PAUSE, STOP, MUTE, UNMUTE
- **Features**: Independent concurrent state transitions

## Development Workflow

### Setup and Build
```bash
cd cmd/web-visualizer
go mod tidy
go build -o web-visualizer .
```

### Running the Application
```bash
./web-visualizer                    # Default port 8080
PORT=8081 ./web-visualizer          # Custom port
```

### Accessing the Interface
- Web UI: `http://localhost:8080`
- API Base: `http://localhost:8080/api`
- WebSocket: `ws://localhost:8080/ws`

## Key Features Implemented

### ✅ Core Functionality
- Interactive statechart visualization
- Real-time simulation with step execution
- Event injection and processing
- State configuration tracking
- Simulation history and replay
- Multiple example statechart patterns

### ✅ User Experience
- Intuitive web interface
- Responsive design for different screen sizes
- Visual feedback for all interactions
- Error handling and user guidance
- Quick-access event buttons

### ✅ Technical Excellence
- Clean separation of frontend/backend concerns
- RESTful API design
- Integration with existing SC library semantics
- Thread-safe machine management
- Comprehensive documentation

## Future Enhancement Opportunities

### Near-term Extensions
- **Visual Editor**: Drag-and-drop statechart creation
- **Export Functionality**: Save simulation traces and configurations
- **Advanced Layouts**: Alternative visualization algorithms
- **Custom Examples**: User-defined statechart upload

### Advanced Features
- **Real-time Collaboration**: Multi-user simulation sessions
- **External Integration**: Connect to message queues or event streams
- **Testing Framework**: Automated test case generation
- **Performance Analytics**: Execution profiling and optimization

## Project Impact

This web visualizer serves multiple purposes:

1. **Educational Tool**: Helps users understand statechart concepts visually
2. **Development Aid**: Debugging and testing statechart-based systems
3. **Demonstration Platform**: Showcasing SC library capabilities
4. **Research Foundation**: Base for advanced statechart tooling

The project successfully bridges the gap between the formal SC library implementation and practical usability, making statechart technology more accessible to developers and researchers.

## Dependencies

### Backend Dependencies
- `github.com/gorilla/mux` - HTTP routing
- `github.com/gorilla/websocket` - WebSocket support
- `github.com/tmc/sc` - Core statechart library
- `google.golang.org/protobuf` - Protocol buffer support

### Frontend Dependencies
- **D3.js v7** - Data visualization library (CDN)
- **Modern Web APIs** - Fetch, WebSocket, ES6+ features
- **No build process required** - Pure JavaScript implementation

## File Structure Summary

```
cmd/web-visualizer/
├── main.go                 # Backend server implementation
├── go.mod                  # Go module configuration
├── README.md               # User documentation
├── PROJECT_OVERVIEW.md     # This technical overview
└── web/                    # Frontend static files
    ├── index.html          # Main application interface
    ├── styles.css          # Visual styling and responsive design
    └── visualizer.js       # Core visualization and API client logic
```

This project represents a successful integration of formal statechart semantics with modern web technologies, creating a practical tool for understanding and working with statechart-based systems.