#!/bin/bash
# Convenience script for Docker development

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}Statechart Docker Development Script${NC}"
    echo -e "${BLUE}====================================${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup       - Initial setup (build and start all services)"
    echo "  start       - Start all services"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  shell       - Open development shell"
    echo "  test        - Run all tests"
    echo "  generate    - Generate protobuf code"
    echo "  logs        - Show logs from all services"
    echo "  clean       - Clean up all Docker resources"
    echo "  status      - Show status of all services"
    echo "  web         - Start web visualizer only"
    echo "  python      - Start Python SDK server only"
    echo "  rust        - Start Rust SDK development only"
    echo ""
}

# Function to run Docker Compose commands
dc() {
    docker-compose "$@"
}

# Main command handling
case "$1" in
    setup)
        echo -e "${GREEN}🚀 Setting up Docker development environment...${NC}"
        check_docker
        
        # Copy example files if they don't exist
        if [ ! -f .env ]; then
            cp .env.example .env
            echo -e "${YELLOW}📝 Created .env file from .env.example${NC}"
        fi
        
        if [ ! -f docker-compose.override.yml ]; then
            cp docker-compose.override.yml.example docker-compose.override.yml
            echo -e "${YELLOW}📝 Created docker-compose.override.yml${NC}"
        fi
        
        # Build and start
        echo -e "${GREEN}🔨 Building Docker images...${NC}"
        dc build --no-cache
        
        echo -e "${GREEN}🚀 Starting services...${NC}"
        dc up -d
        
        # Wait for services to be ready
        echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
        sleep 5
        
        # Show status
        dc ps
        
        echo -e "${GREEN}✅ Setup complete!${NC}"
        echo -e "${BLUE}Run '$0 shell' to enter the development container.${NC}"
        ;;
        
    start)
        echo -e "${GREEN}▶️  Starting services...${NC}"
        check_docker
        dc up -d
        dc ps
        ;;
        
    stop)
        echo -e "${YELLOW}⏸️  Stopping services...${NC}"
        check_docker
        dc down
        ;;
        
    restart)
        echo -e "${YELLOW}🔄 Restarting services...${NC}"
        check_docker
        dc restart
        dc ps
        ;;
        
    shell)
        echo -e "${GREEN}🐚 Opening development shell...${NC}"
        check_docker
        dc exec dev bash
        ;;
        
    test)
        echo -e "${GREEN}🧪 Running tests...${NC}"
        check_docker
        dc exec dev bash -c "cd /workspace && go test ./..."
        ;;
        
    generate)
        echo -e "${GREEN}🔧 Generating protobuf code...${NC}"
        check_docker
        dc exec dev bash -c "cd /workspace && make generate"
        ;;
        
    logs)
        echo -e "${BLUE}📋 Showing logs (Ctrl+C to exit)...${NC}"
        check_docker
        dc logs -f
        ;;
        
    clean)
        echo -e "${RED}🧹 Cleaning up Docker resources...${NC}"
        check_docker
        read -p "Are you sure? This will remove all containers and volumes. (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            dc down -v --remove-orphans
            docker system prune -f
            echo -e "${GREEN}✅ Cleanup complete!${NC}"
        else
            echo -e "${YELLOW}❌ Cleanup cancelled.${NC}"
        fi
        ;;
        
    status)
        echo -e "${BLUE}📊 Service status:${NC}"
        check_docker
        dc ps
        ;;
        
    web)
        echo -e "${GREEN}🌐 Starting web visualizer...${NC}"
        check_docker
        dc up -d web-visualizer
        echo -e "${BLUE}Web visualizer available at http://localhost:8081${NC}"
        ;;
        
    python)
        echo -e "${GREEN}🐍 Starting Python SDK server...${NC}"
        check_docker
        dc up -d python-sdk
        echo -e "${BLUE}FastAPI server available at http://localhost:8001${NC}"
        echo -e "${BLUE}API docs available at http://localhost:8001/docs${NC}"
        ;;
        
    rust)
        echo -e "${GREEN}🦀 Starting Rust SDK development...${NC}"
        check_docker
        dc up -d rust-sdk
        echo -e "${BLUE}Rust SDK is running with cargo watch${NC}"
        ;;
        
    *)
        show_usage
        exit 1
        ;;
esac