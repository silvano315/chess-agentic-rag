#!/bin/bash

# ============================================
# Ollama Setup Script for macOS
# ============================================
# 
# This script installs Ollama and downloads required models
# for the Chess Agentic RAG project.
#
# Usage: bash scripts/setup_ollama.sh
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is designed for macOS only."
    log_info "For other platforms, visit: https://ollama.ai/download"
    exit 1
fi

log_info "Starting Ollama setup for Chess Agentic RAG..."
echo ""

# ============================================
# Step 1: Check if Homebrew is installed
# ============================================
log_info "Checking for Homebrew..."

if ! command -v brew &> /dev/null; then
    log_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    
    log_success "Homebrew installed successfully"
else
    log_success "Homebrew is already installed"
fi

echo ""

# ============================================
# Step 2: Install Ollama
# ============================================
log_info "Checking for Ollama..."

if command -v ollama &> /dev/null; then
    log_success "Ollama is already installed"
    OLLAMA_VERSION=$(ollama --version | head -n 1)
    log_info "Current version: $OLLAMA_VERSION"
    
    # Ask if user wants to update
    read -p "Do you want to update Ollama? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Updating Ollama..."
        brew upgrade ollama
        log_success "Ollama updated"
    fi
else
    log_info "Installing Ollama via Homebrew..."
    brew install ollama
    log_success "Ollama installed successfully"
fi

echo ""

# ============================================
# Step 3: Start Ollama service
# ============================================
log_info "Starting Ollama service..."

# Check if Ollama is already running
if pgrep -x "ollama" > /dev/null; then
    log_success "Ollama service is already running"
else
    log_info "Starting Ollama in the background..."
    
    # Start Ollama as a background service
    brew services start ollama
    
    # Wait a bit for the service to start
    sleep 3
    
    # Verify it's running
    if pgrep -x "ollama" > /dev/null; then
        log_success "Ollama service started successfully"
    else
        log_warning "Could not verify Ollama service status"
        log_info "You may need to start it manually: ollama serve"
    fi
fi

echo ""

# ============================================
# Step 4: Download required models
# ============================================
log_info "Downloading required models for Chess Agentic RAG..."
echo ""

# Array of models to download
declare -a models=(
    "deepseek-r1:1.5b|Primary reasoning model (small, fast)"
    "qwen2.5:7b|Fallback model (more capable)"
    "nomic-embed-text|Embedding model for semantic search"
)

# Function to pull model
pull_model() {
    local model_name=$1
    local description=$2
    
    log_info "Checking model: $model_name"
    log_info "Description: $description"
    
    # Check if model is already present
    if ollama list | grep -q "$model_name"; then
        log_success "Model '$model_name' is already available"
        
        # Ask if user wants to update
        read -p "Update this model? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Pulling latest version of '$model_name'..."
            ollama pull "$model_name"
            log_success "Model '$model_name' updated"
        fi
    else
        log_info "Pulling model '$model_name' (this may take a while)..."
        ollama pull "$model_name"
        log_success "Model '$model_name' downloaded successfully"
    fi
    
    echo ""
}

# Download each model
for model_info in "${models[@]}"; do
    IFS='|' read -r model_name description <<< "$model_info"
    pull_model "$model_name" "$description"
done

# ============================================
# Step 5: Verify installation
# ============================================
log_info "Verifying installation..."
echo ""

# Test Ollama connection
log_info "Testing Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log_success "Ollama API is responding"
else
    log_error "Cannot connect to Ollama API"
    log_info "Try running: ollama serve"
    exit 1
fi

# List available models
log_info "Available models:"
ollama list

echo ""

# ============================================
# Step 6: Test model
# ============================================
log_info "Testing primary model (deepseek-r1:1.5b)..."
echo ""

TEST_PROMPT="What is the Italian Opening in chess? Answer in one sentence."

log_info "Running test query: '$TEST_PROMPT'"
RESPONSE=$(ollama run deepseek-r1:1.5b "$TEST_PROMPT" 2>/dev/null | head -n 5)

if [ -n "$RESPONSE" ]; then
    log_success "Model test successful!"
    echo ""
    echo "Response:"
    echo "$RESPONSE"
else
    log_error "Model test failed"
    exit 1
fi

echo ""

# ============================================
# Final Summary
# ============================================
log_success "=========================================="
log_success "Ollama Setup Complete! ✅"
log_success "=========================================="
echo ""
log_info "Summary:"
echo "  ✅ Ollama installed and running"
echo "  ✅ Models downloaded:"
echo "     - deepseek-r1:1.5b (primary reasoning)"
echo "     - qwen2.5:7b (fallback)"
echo "     - nomic-embed-text (embeddings)"
echo "  ✅ API accessible at http://localhost:11434"
echo ""
log_info "Next steps:"
echo "  1. Copy .env.example to .env: cp .env.example .env"
echo "  2. Test connection: uv run python tests/unit/test_ollama_connection.py"
echo "  3. Start building: Follow docs/milestones/M1_DATA_PIPELINE.md"
echo ""
log_info "Useful commands:"
echo "  - List models:        ollama list"
echo "  - Test model:         ollama run deepseek-r1:1.5b"
echo "  - Stop service:       brew services stop ollama"
echo "  - Start service:      brew services start ollama"
echo "  - View logs:          tail -f ~/.ollama/logs/server.log"
echo ""
log_success "Happy coding! 🚀"