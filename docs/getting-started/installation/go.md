# Go Installation Guide

This guide will help you install and set up the Statecharts library for Go development.

## Prerequisites

- **Go 1.19 or later** - [Download Go](https://golang.org/dl/)
- **Protocol Buffers compiler** (for development) - [Install protoc](https://grpc.io/docs/protoc-installation/)

## Quick Installation

### Option 1: Using go get (Recommended)

```bash
go get github.com/tmc/sc
```

### Option 2: Using go mod

Add to your `go.mod`:

```go
require github.com/tmc/sc v0.1.0
```

Then run:

```bash
go mod tidy
```

## Verify Installation

Create a simple test file to verify everything works:

**main.go**
```go
package main

import (
    "fmt"
    "github.com/tmc/sc"
    pb "github.com/tmc/sc/gen/statecharts/v1"
)

func main() {
    // Create a simple statechart
    statechart := &pb.Statechart{
        RootState: &pb.State{
            Label: "__root__",
            Type:  pb.StateType_STATE_TYPE_NORMAL,
            Children: []*pb.State{
                {
                    Label:     "off",
                    Type:      pb.StateType_STATE_TYPE_BASIC,
                    IsInitial: true,
                },
                {
                    Label: "on",
                    Type:  pb.StateType_STATE_TYPE_BASIC,
                },
            },
        },
        Transitions: []*pb.Transition{
            {
                Label: "power_on",
                From:  []string{"off"},
                To:    []string{"on"},
                Event: "POWER_ON",
            },
            {
                Label: "power_off", 
                From:  []string{"on"},
                To:    []string{"off"},
                Event: "POWER_OFF",
            },
        },
        Events: []*pb.Event{
            {Label: "POWER_ON"},
            {Label: "POWER_OFF"},
        },
    }

    fmt.Printf("Created statechart with %d states\n", 
        len(statechart.RootState.Children))
    fmt.Printf("Transitions: %d\n", len(statechart.Transitions))
    fmt.Printf("Events: %d\n", len(statechart.Events))
}
```

Run the test:

```bash
go run main.go
```

Expected output:
```
Created statechart with 2 states
Transitions: 2
Events: 2
```

## Package Structure

The Go implementation provides several packages:

```
github.com/tmc/sc/
├── gen/statecharts/v1/          # Generated Protocol Buffer types
├── semantics/v1/                # Statechart execution semantics
├── validation/v1/               # Validation rules and functions
└── statecharts/v1/              # High-level API and utilities
```

### Core Imports

```go
import (
    // Protocol Buffer generated types
    pb "github.com/tmc/sc/gen/statecharts/v1"
    
    // Semantic operations
    "github.com/tmc/sc/semantics/v1"
    
    // Validation
    "github.com/tmc/sc/validation/v1"
)
```

## Development Setup

If you plan to contribute or work with the source code:

### 1. Clone the Repository

```bash
git clone https://github.com/tmc/sc.git
cd sc
```

### 2. Install Development Tools

```bash
# Install Protocol Buffer compiler
# macOS
brew install protobuf

# Linux (Ubuntu/Debian)
sudo apt-get install protobuf-compiler

# Windows
# Download from https://github.com/protocolbuffers/protobuf/releases
```

### 3. Install Go Tools

```bash
cd proto
make tools
```

### 4. Generate Code

```bash
make generate
```

### 5. Run Tests

```bash
go test ./...
```

## IDE Setup

### VS Code

Recommended extensions:
- **Go** - Official Go extension
- **Protocol Buffer** - Syntax highlighting for .proto files
- **Statecharts** - (if available) Visual editing support

### GoLand/IntelliJ

Built-in Go support with Protocol Buffer plugin.

## Common Issues

### Issue: "package not found"

**Solution**: Ensure you're using Go modules:

```bash
go mod init your-project
go get github.com/tmc/sc
```

### Issue: Protocol Buffer errors

**Solution**: Make sure protoc is installed and accessible:

```bash
protoc --version
```

Should show version 3.15+ or later.

### Issue: Import cycle

**Solution**: The library uses internal packages. Import only the public APIs:

```go
// ✅ Correct
import pb "github.com/tmc/sc/gen/statecharts/v1"

// ❌ Avoid internal packages  
import "github.com/tmc/sc/internal/..."
```

## Performance Considerations

### Memory Usage

The library uses Protocol Buffers which are efficient but not zero-copy. For high-performance applications:

- Reuse statechart instances
- Pool frequently allocated objects
- Use streaming APIs for large datasets

### CPU Usage

- Validation is computationally expensive - cache results when possible
- Consider pre-compiling statecharts for production use
- Use profiling tools (`go tool pprof`) to identify bottlenecks

## Next Steps

Now that you have Go set up:

1. **[Build your first statechart](../first-statechart.md)** - Hands-on tutorial
2. **[Learn basic concepts](../basic-concepts.md)** - Understand the fundamentals  
3. **[Explore examples](../../examples/simple/)** - See practical applications
4. **[Read API documentation](../../api/go/)** - Detailed reference

## Examples Repository

Check out the examples directory for working code:

```bash
cd $GOPATH/src/github.com/tmc/sc
ls semantics/v1/examples/
```

Available examples:
- Simple state machine
- Hierarchical states
- Orthogonal (concurrent) states
- History states
- Real-world applications

---

Ready to build your first statechart? → [First Statechart Tutorial](../first-statechart.md)