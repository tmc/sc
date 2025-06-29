#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Statechart Development Environment${NC}"
echo -e "${BLUE}=====================================>${NC}"

# Check if this is the first run
if [ ! -f /workspace/.docker-initialized ]; then
    echo -e "${YELLOW}First run detected. Setting up environment...${NC}"
    
    # Install Go dependencies
    echo -e "${GREEN}Installing Go dependencies...${NC}"
    cd /workspace
    go mod download
    
    # Generate protobuf code
    echo -e "${GREEN}Generating protobuf code...${NC}"
    make generate || echo "Proto generation failed, continuing..."
    
    # Install Python SDK dependencies
    if [ -d "/workspace/sdks/python" ]; then
        echo -e "${GREEN}Installing Python SDK dependencies...${NC}"
        cd /workspace/sdks/python
        pip install -e .[dev,web] || echo "Python setup failed, continuing..."
    fi
    
    # Install Node.js dependencies for web visualizer
    if [ -d "/workspace/cmd/web-visualizer" ]; then
        echo -e "${GREEN}Installing Node.js dependencies...${NC}"
        cd /workspace/cmd/web-visualizer
        npm install || echo "Node.js setup failed, continuing..."
    fi
    
    # Build Rust SDK
    if [ -d "/workspace/sdks/rust" ]; then
        echo -e "${GREEN}Building Rust SDK...${NC}"
        cd /workspace/sdks/rust
        cargo build || echo "Rust build failed, continuing..."
    fi
    
    # Mark as initialized
    touch /workspace/.docker-initialized
    
    echo -e "${GREEN}✅ Environment setup complete!${NC}"
fi

# Set up Git config (if not already set)
if [ ! -f ~/.gitconfig ]; then
    git config --global --add safe.directory /workspace
    git config --global user.email "dev@statechart.local"
    git config --global user.name "Statechart Developer"
fi

# Display helpful information
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo -e "  ${GREEN}make generate${NC}         - Generate protobuf code"
echo -e "  ${GREEN}go test ./...${NC}         - Run all Go tests"
echo -e "  ${GREEN}make build-sdk-rust${NC}   - Build Rust SDK"
echo -e "  ${GREEN}make test-sdk-rust${NC}    - Test Rust SDK"
echo -e "  ${GREEN}cd cmd/web-visualizer && go run .${NC} - Run web visualizer"
echo ""
echo -e "${BLUE}Port mappings:${NC}"
echo -e "  ${GREEN}8080${NC} - Web visualizer"
echo -e "  ${GREEN}8081${NC} - Web visualizer (hot-reload)"
echo -e "  ${GREEN}5000${NC} - Python Flask"
echo -e "  ${GREEN}8000${NC} - Python FastAPI"
echo -e "  ${GREEN}5432${NC} - PostgreSQL"
echo -e "  ${GREEN}6379${NC} - Redis"
echo -e "  ${GREEN}2345${NC} - Delve debugger"
echo ""

# Execute the passed command or start bash
exec "$@"