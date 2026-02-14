## Milestone 0: Environment Setup ✅

**Status**: In Progress  
**Timeline**: Week 1  
**Goal**: Set up development environment, install Ollama, verify all tools work

---

### 📋 Objectives

1. ✅ Initialize project repository with proper structure
2. ✅ Configure package management with `uv`
3. ✅ Install and configure Ollama on Mac
4. ✅ Download required LLM models
5. ✅ Verify Ollama connection and basic functionality
6. ✅ Set up testing infrastructure
7. ✅ Configure code quality tools (black, ruff, mypy)

---

### 🎯 Deliverables

#### Configuration Files
- ✅ `pyproject.toml` - Project dependencies and tool configuration
- ✅ `.env.example` - Environment variable template
- ✅ `.gitignore` - Git ignore rules
- ✅ `.python-version` - Python version specification
- ✅ `README.md` - Project documentation

#### Source Code
- ✅ `src/chess_agentic_rag/core/config.py` - Configuration management
- ✅ `src/chess_agentic_rag/core/exceptions.py` - Custom exception hierarchy
- ✅ `src/chess_agentic_rag/llm/ollama_client.py` - Ollama client wrapper

#### Tests
- ✅ `tests/conftest.py` - Pytest configuration and fixtures
- ✅ `tests/unit/test_ollama_connection.py` - Ollama connection tests

#### Helpers
- ✅ `helpers/setup_ollama.sh` - Ollama installation script for Mac

#### Documentation
- ✅ `docs/PROJECT_OVERVIEW.md` - Master project reference
- ✅ `docs/milestones/M0_SETUP.md` - This file

---

### 🚀 Setup Instructions

#### 1. Prerequisites

Ensure you have:
- macOS (script is Mac-specific)
- Homebrew (will be installed if missing)
- Python 3.11+
- `uv` package manager

Install `uv` if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone and Setup Repository

```bash
# Clone repository
git clone https://github.com/silvano315/chess-agentic-rag.git
cd chess-agentic-rag

# Install dependencies
uv sync
uv sync --extra dev

# Setup project in src
uv pip install -e .

# Copy environment template
cp .env.example .env
```

#### 3. Install Ollama and Models

```bash
# Run setup script (handles installation + model download)
bash scripts/setup_ollama.sh
```

This script will:
- Install Ollama via Homebrew (if needed)
- Start Ollama service
- Pull required models:
  - `qwen2.5:7b`(primary reasoning model)
  - `deepseek-r1:1.5b` (fallback model)
  - `nomic-embed-text` (embeddings)

#### 4. Verify Setup

**Option A: Run test script directly**
```bash
uv run python tests/unit/test_ollama_connection.py
```

**Option B: Run pytest**
```bash
uv run pytest tests/unit/test_ollama_connection.py -v
```

#### 5. Verify Code Quality Tools

```bash
# Format code
uv run black src tests

# Lint
uv run ruff check src tests

# Type check
uv run mypy src
```

---

### ✅ Acceptance Criteria

- [x] Repository structure matches specification
- [x] All configuration files present and valid
- [x] `uv sync` completes without errors
- [x] Ollama service running (check: `ollama list`)
- [x] All required models downloaded
- [x] Ollama client can connect and generate text
- [x] Embeddings generation works
- [x] Test suite runs successfully
- [x] Code quality tools (black, ruff, mypy) configured

---

### 🧪 Testing

#### Manual Tests

1. **Ollama Service**:
   ```bash
   # Check service status
   brew services list | grep ollama
   
   # List models
   ollama list
   
   # Test model interactively
   ollama run qwen2.5:7b "What is chess?"
   ```

2. **Python Environment**:
   ```bash
   # Check Python version
   python --version  # Should be 3.11+
   
   # Verify imports
   uv run python -c "from src.llm.ollama_client import OllamaClient; print('✅ Imports work')"
   ```

3. **Dependencies**:
   ```bash
   # List installed packages
   uv pip list
   
   # Check for critical packages
   uv pip show ollama llama-index chromadb fastapi
   ```

#### Automated Tests

```bash
# Run all M0 tests
uv run pytest tests/unit/test_ollama_connection.py -v -m requires_ollama

# Run quick test (without Ollama dependency)
uv run pytest tests/unit/test_ollama_connection.py::TestOllamaConnection::test_client_initialization

# Run with coverage
uv run pytest tests/unit/test_ollama_connection.py --cov=src.llm --cov-report=term
```

---

### 🐛 Troubleshooting

#### Ollama Not Responding

**Symptom**: `OllamaClient` health check fails

**Solutions**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama manually
ollama serve

# Or via Homebrew services
brew services start ollama

# Check logs
tail -f ~/.ollama/logs/server.log
```

#### Model Not Found

**Symptom**: "Model not found" errors

**Solutions**:
```bash
# List installed models
ollama list

# Pull missing model
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Or run setup script again
bash scripts/setup_ollama.sh
```

---

### 📚 Key Learnings & Notes

#### Ollama Models

- **DeepSeek-R1:1.5b**: Fast, lightweight, good for reasoning (~1GB)
- **Qwen2.5:7b**: More capable, used for complex queries (~4GB)
- **nomic-embed-text**: Embedding model, 768 dimensions (~274MB)

Model sizes are approximate and may vary.

#### uv vs pip/poetry

We chose `uv` because:
- ✅ Faster dependency resolution
- ✅ Better lockfile management
- ✅ Simpler command structure
- ✅ Built-in virtual environment handling

Always use `uv run` to execute scripts to ensure proper environment.

---

**Last Updated**: February 14, 2026  
**Completed By**: In Progress  
**Review Status**: Pending