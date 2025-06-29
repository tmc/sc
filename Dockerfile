# Multi-stage Dockerfile for statechart library development environment
# This creates a comprehensive development environment with all required tools

FROM ubuntu:22.04 as base

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    gnupg \
    lsb-release \
    unzip \
    vim \
    nano \
    htop \
    jq \
    tree \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Go 1.24 (matching project toolchain)
ENV GO_VERSION=1.24.3
RUN wget -O go.tar.gz "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
    && tar -C /usr/local -xzf go.tar.gz \
    && rm go.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GOPROXY="https://proxy.golang.org,direct"
ENV GOSUMDB="sum.golang.org"

# Install Node.js 20 LTS (for web visualizer)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Rust (for Rust SDK)
ENV RUST_VERSION=1.75.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION \
    && echo 'source ~/.cargo/env' >> ~/.bashrc

ENV PATH="/root/.cargo/bin:${PATH}"

# Install Protocol Buffers compiler
ENV PROTOC_VERSION=25.1
RUN wget -O protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
    && unzip protoc.zip -d /usr/local \
    && rm protoc.zip \
    && chmod +x /usr/local/bin/protoc

# Install Go development tools
RUN go install github.com/bufbuild/buf/cmd/buf@latest \
    && go install github.com/tmc/protoc-gen-apidocs@latest \
    && go install github.com/gorilla/mux@latest \
    && go install github.com/gorilla/websocket@latest

# Install Rust protobuf tools
RUN cargo install protoc-gen-prost protoc-gen-tonic

# Install global Node.js development tools
RUN npm install -g \
    @playwright/test \
    jest \
    eslint \
    prettier \
    rollup \
    concurrently \
    nodemon

# Install Python development tools
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install \
    grpcio-tools \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    mypy \
    ruff \
    flask \
    fastapi \
    uvicorn

# Create workspace directory
WORKDIR /workspace

# Copy project files (will be overridden by volume mount in development)
COPY . .

# Install project-specific Go dependencies
RUN go mod download

# Install project-specific Node.js dependencies for web visualizer
WORKDIR /workspace/cmd/web-visualizer
RUN if [ -f package.json ]; then npm install; fi

# Install project-specific Python dependencies
WORKDIR /workspace/sdks/python
RUN if [ -f pyproject.toml ]; then pip3 install -e .[dev,web]; fi

# Install Rust dependencies
WORKDIR /workspace/sdks/rust
RUN if [ -f Cargo.toml ]; then cargo build; fi

# Set working directory back to root
WORKDIR /workspace

# Create development user (optional, for better file permissions)
RUN useradd -m -u 1000 -G sudo -s /bin/bash developer \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create directories for development
RUN mkdir -p /workspace/data \
    && mkdir -p /workspace/logs \
    && mkdir -p /workspace/tmp \
    && chown -R developer:developer /workspace

# Development stage
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    strace \
    tcpdump \
    net-tools \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Install additional Go development tools
RUN go install github.com/go-delve/delve/cmd/dlv@latest \
    && go install golang.org/x/tools/cmd/goimports@latest \
    && go install golang.org/x/tools/cmd/godoc@latest \
    && go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Install Playwright browsers for e2e testing
RUN npx playwright install --with-deps

# Set environment variables for development
ENV NODE_ENV=development
ENV GO_ENV=development
ENV RUST_LOG=debug
ENV PYTHONPATH=/workspace/sdks/python

# Expose ports for development services
EXPOSE 8080    # Web visualizer
EXPOSE 3000    # Development server (if needed)
EXPOSE 5000    # Python Flask apps
EXPOSE 8000    # Python FastAPI apps
EXPOSE 40000   # Go Delve debugger
EXPOSE 9229    # Node.js inspector

# Create entrypoint script for development
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Generate protocol buffer code
echo "Generating protocol buffer code..."
cd /workspace && make generate

# Check if we should run a specific service
case "${SERVICE:-all}" in
    "web-visualizer")
        echo "Starting web visualizer..."
        cd /workspace/cmd/web-visualizer
        if [ "${NODE_ENV}" = "development" ]; then
            npm run dev
        else
            go run .
        fi
        ;;
    "python-dev")
        echo "Starting Python development environment..."
        cd /workspace/sdks/python
        python3 -m pytest tests/ -v
        ;;
    "rust-dev")
        echo "Starting Rust development environment..."
        cd /workspace/sdks/rust
        cargo test
        ;;
    "tests")
        echo "Running all tests..."
        cd /workspace
        go test ./...
        cd sdks/rust && cargo test
        cd ../python && python3 -m pytest tests/ -v
        cd ../../cmd/web-visualizer && npm test
        ;;
    *)
        echo "Starting interactive development shell..."
        echo "Available commands:"
        echo "  make generate     - Generate protocol buffer code"
        echo "  go test ./...     - Run Go tests"
        echo "  cd cmd/web-visualizer && npm run dev - Start web visualizer in dev mode"
        echo "  cd sdks/rust && cargo test - Run Rust tests"
        echo "  cd sdks/python && python3 -m pytest tests/ -v - Run Python tests"
        exec /bin/bash
        ;;
esac
EOF

RUN chmod +x /entrypoint.sh

# Switch to development user
USER developer

ENTRYPOINT ["/entrypoint.sh"]