#!/bin/bash
# Test script for Docker environment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 Testing Docker Development Environment${NC}"
echo -e "${BLUE}=======================================${NC}"

# Test 1: Docker is running
echo -e "${YELLOW}Test 1: Checking Docker...${NC}"
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is running${NC}"
else
    echo -e "${RED}❌ Docker is not running${NC}"
    exit 1
fi

# Test 2: Docker Compose is available
echo -e "${YELLOW}Test 2: Checking Docker Compose...${NC}"
if docker-compose --version > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Compose is available${NC}"
else
    echo -e "${RED}❌ Docker Compose is not available${NC}"
    exit 1
fi

# Test 3: Build development image
echo -e "${YELLOW}Test 3: Building development image...${NC}"
if docker-compose build dev --no-cache; then
    echo -e "${GREEN}✅ Development image built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build development image${NC}"
    exit 1
fi

# Test 4: Start services
echo -e "${YELLOW}Test 4: Starting services...${NC}"
if docker-compose up -d; then
    echo -e "${GREEN}✅ Services started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 10

# Test 5: Check service health
echo -e "${YELLOW}Test 5: Checking service health...${NC}"
services_healthy=true

# Check each service
services=("dev" "postgres" "redis")
for service in "${services[@]}"; do
    if docker-compose ps "$service" | grep -q "Up"; then
        echo -e "${GREEN}✅ $service is running${NC}"
    else
        echo -e "${RED}❌ $service is not running${NC}"
        services_healthy=false
    fi
done

if [ "$services_healthy" = false ]; then
    echo -e "${RED}❌ Some services are not healthy${NC}"
    exit 1
fi

# Test 6: Test Go environment
echo -e "${YELLOW}Test 6: Testing Go environment...${NC}"
if docker-compose exec -T dev go version; then
    echo -e "${GREEN}✅ Go environment is working${NC}"
else
    echo -e "${RED}❌ Go environment failed${NC}"
    exit 1
fi

# Test 7: Test Python environment
echo -e "${YELLOW}Test 7: Testing Python environment...${NC}"
if docker-compose exec -T dev python3 --version; then
    echo -e "${GREEN}✅ Python environment is working${NC}"
else
    echo -e "${RED}❌ Python environment failed${NC}"
    exit 1
fi

# Test 8: Test Node.js environment
echo -e "${YELLOW}Test 8: Testing Node.js environment...${NC}"
if docker-compose exec -T dev node --version; then
    echo -e "${GREEN}✅ Node.js environment is working${NC}"
else
    echo -e "${RED}❌ Node.js environment failed${NC}"
    exit 1
fi

# Test 9: Test Rust environment
echo -e "${YELLOW}Test 9: Testing Rust environment...${NC}"
if docker-compose exec -T dev rustc --version; then
    echo -e "${GREEN}✅ Rust environment is working${NC}"
else
    echo -e "${RED}❌ Rust environment failed${NC}"
    exit 1
fi

# Test 10: Test protobuf generation
echo -e "${YELLOW}Test 10: Testing protobuf generation...${NC}"
if docker-compose exec -T dev bash -c "cd /workspace && make generate"; then
    echo -e "${GREEN}✅ Protobuf generation is working${NC}"
else
    echo -e "${RED}❌ Protobuf generation failed${NC}"
    exit 1
fi

# Test 11: Test Go tests
echo -e "${YELLOW}Test 11: Running Go tests...${NC}"
if docker-compose exec -T dev bash -c "cd /workspace && go test ./semantics/v1 -timeout 30s"; then
    echo -e "${GREEN}✅ Go tests passed${NC}"
else
    echo -e "${RED}❌ Go tests failed${NC}"
    exit 1
fi

# Test 12: Test database connectivity
echo -e "${YELLOW}Test 12: Testing database connectivity...${NC}"
if docker-compose exec -T postgres pg_isready -U statechart; then
    echo -e "${GREEN}✅ Database is accessible${NC}"
else
    echo -e "${RED}❌ Database is not accessible${NC}"
    exit 1
fi

# Test 13: Test Redis connectivity
echo -e "${YELLOW}Test 13: Testing Redis connectivity...${NC}"
if docker-compose exec -T redis redis-cli ping; then
    echo -e "${GREEN}✅ Redis is accessible${NC}"
else
    echo -e "${RED}❌ Redis is not accessible${NC}"
    exit 1
fi

# Test 14: Test port accessibility
echo -e "${YELLOW}Test 14: Testing port accessibility...${NC}"
ports=("8080" "5432" "6379")
for port in "${ports[@]}"; do
    if nc -z localhost "$port" 2>/dev/null; then
        echo -e "${GREEN}✅ Port $port is accessible${NC}"
    else
        echo -e "${RED}❌ Port $port is not accessible${NC}"
        exit 1
    fi
done

# Cleanup
echo -e "${YELLOW}Cleaning up test environment...${NC}"
docker-compose down

echo -e "${GREEN}🎉 All tests passed! Docker environment is ready for development.${NC}"
echo -e "${BLUE}To get started:${NC}"
echo -e "  ${GREEN}./scripts/docker-dev.sh setup${NC}    - Full setup"
echo -e "  ${GREEN}./scripts/docker-dev.sh shell${NC}    - Enter development shell"
echo -e "  ${GREEN}make -f Makefile.docker help${NC}     - See all available commands"