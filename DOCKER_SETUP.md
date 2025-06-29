# Docker Development Environment Setup

This guide explains how to use the Docker-based development environment for the Statechart Library project.

## Quick Start

1. **One-Command Setup**:
   ```bash
   make -f Makefile.docker docker-up
   ```

   This single command will:
   - Build the Docker development image with all dependencies
   - Start all necessary services
   - Set up the development environment

2. **Access the Development Shell**:
   ```bash
   make -f Makefile.docker docker-shell
   ```

## Architecture Overview

The Docker environment includes:

- **Main Development Container** (`dev`): Full development environment with Go, Python, Node.js, Rust, and all tools
- **Web Visualizer** (`web-visualizer`): Hot-reload development server for the web UI
- **Python SDK Server** (`python-sdk`): FastAPI/Flask development server
- **Rust SDK Development** (`rust-sdk`): Cargo watch for automatic testing
- **PostgreSQL** (`postgres`): Database for persistence (optional)
- **Redis** (`redis`): Caching and session storage (optional)
- **Documentation Server** (`docs`): Nginx server for documentation

## Service URLs

After running `docker-compose up`, services are available at:

- **Web Visualizer**: http://localhost:8080 (main), http://localhost:8081 (hot-reload)
- **Python FastAPI**: http://localhost:8001 (with docs at /docs)
- **Python Flask**: http://localhost:5001
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Documentation**: http://localhost:8090
- **Delve Debugger**: localhost:2345
- **Node.js Debug**: localhost:9229

## Common Commands

### Docker Management

```bash
# Build the development image
make -f Makefile.docker docker-build

# Start all services
make -f Makefile.docker docker-up

# Stop all services
make -f Makefile.docker docker-down

# View logs
make -f Makefile.docker docker-logs

# Clean up everything
make -f Makefile.docker docker-clean
```

### Development Tasks

```bash
# Open development shell
make -f Makefile.docker docker-shell

# Run tests in Docker
make -f Makefile.docker docker-test

# Generate protobuf code
make -f Makefile.docker docker-generate

# Start web visualizer with hot-reload
make -f Makefile.docker docker-web

# Start Python SDK server
make -f Makefile.docker docker-python

# Start Rust SDK development
make -f Makefile.docker docker-rust
```

## Development Workflow

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/tmc/sc.git
cd sc

# Start the Docker environment
make -f Makefile.docker docker-up

# Enter the development container
make -f Makefile.docker docker-shell
```

### 2. Working with Go Code

Inside the development container:

```bash
# Generate protobuf code
make generate

# Run tests
go test ./...

# Run specific package tests
go test ./semantics/v1/... -v

# Run with coverage
go test -cover ./...
```

### 3. Working with Web Visualizer

The web visualizer supports hot-reload:

```bash
# In a separate terminal, start the web visualizer
make -f Makefile.docker docker-web

# Edit files in cmd/web-visualizer/web/
# Changes will be reflected immediately
```

### 4. Working with Python SDK

```bash
# Start the Python SDK server
make -f Makefile.docker docker-python

# Access the API documentation
open http://localhost:8001/docs

# Run Python tests (inside container)
cd /workspace/sdks/python
pytest
```

### 5. Working with Rust SDK

```bash
# Start Rust development with auto-testing
make -f Makefile.docker docker-rust

# Or manually in the container
cd /workspace/sdks/rust
cargo test
cargo run --example simple_statechart
```

## Advanced Usage

### Using Docker Compose Override

Create a `docker-compose.override.yml` file for local customizations:

```yaml
version: '3.8'

services:
  dev:
    environment:
      - MY_CUSTOM_VAR=value
    volumes:
      - ~/.gitconfig:/root/.gitconfig:ro
```

### Debugging

#### Go Debugging with Delve

1. In the container, start your application with Delve:
   ```bash
   dlv debug --headless --listen=:2345 --api-version=2 --accept-multiclient
   ```

2. Connect your IDE to `localhost:2345`

#### Node.js Debugging

1. Start Node.js with inspect flag:
   ```bash
   node --inspect=0.0.0.0:9229 your-script.js
   ```

2. Connect Chrome DevTools or your IDE to `localhost:9229`

### Performance Optimization

The Docker setup includes several optimizations:

1. **Volume Caching**: Uses `:cached` flag for better macOS performance
2. **Persistent Caches**: Go modules, pip, npm, and cargo caches are persisted
3. **Build Cache**: Docker layer caching speeds up rebuilds

## Troubleshooting

### Container Won't Start

```bash
# Check logs
make -f Makefile.docker docker-logs

# Rebuild from scratch
make -f Makefile.docker docker-rebuild
```

### Permission Issues

If you encounter permission issues with generated files:

```bash
# Inside the container
chmod -R 777 /workspace/.cache
```

### Port Conflicts

If ports are already in use, either:

1. Stop conflicting services
2. Modify port mappings in `docker-compose.yml`
3. Use `docker-compose.override.yml` to change ports

### Slow Performance on macOS

The `:cached` volume flag is already applied. For additional performance:

1. Increase Docker Desktop resources (CPU, Memory)
2. Consider using Docker Desktop's VirtioFS option
3. Exclude large directories from mounting

## Cross-Platform Notes

### Windows

- Ensure Docker Desktop is installed with WSL2 backend
- Use Git Bash or WSL2 terminal for commands
- Line endings: Configure Git to use LF endings

### Linux

- Ensure your user is in the `docker` group
- No special configuration needed

### macOS

- Docker Desktop required
- Performance may vary; adjust resources as needed

## Extending the Environment

### Adding New Services

1. Edit `docker-compose.yml` to add your service
2. Update `Makefile.docker` with convenience commands
3. Document the service in this file

### Customizing the Development Image

1. Modify `Dockerfile.dev` to add tools/dependencies
2. Rebuild: `make -f Makefile.docker docker-build`
3. Restart services: `make -f Makefile.docker docker-up`

## Best Practices

1. **Regular Cleanup**: Run `make -f Makefile.docker docker-clean` periodically
2. **Use Override Files**: Keep local customizations in `docker-compose.override.yml`
3. **Commit Generated Files**: Don't commit generated protobuf files
4. **Volume Management**: Be careful with volume permissions

## Support

If you encounter issues:

1. Check the logs: `make -f Makefile.docker docker-logs`
2. Ensure Docker Desktop is up to date
3. Try a full rebuild: `make -f Makefile.docker docker-rebuild`
4. Open an issue on GitHub with details