# Chess Agentic RAG

An intelligent agentic RAG (Retrieval-Augmented Generation) system for chess knowledge, running entirely locally using Ollama.

The following sections are just an example of what I would like to implement. Everything is a working process.

## Features

- **Agentic Intelligence**: ReAct pattern with multi-step reasoning
- **Knowledge Base**: Chess theory, openings, historical games, strategies
- **Tool Integration**: ELO fetcher, PGN parser, game search, Stockfish (optional)
- **Local First**: Runs 100% locally with Ollama
- **REST API**: FastAPI with WebSocket support for streaming

## Tech Stack

- **LLM Backend**: Ollama (DeepSeek, Qwen)
- **RAG Framework**: LlamaIndex
- **Vector DB**: ChromaDB
- **API**: FastAPI
- **Tools**: python-chess, requests, stockfish (optional)

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/)

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/silvano315/chess-agentic-rag.git
   cd chess-agentic-rag
```

2. Install dependencies:
```bash
   uv sync
```

3. Setup Ollama:
```bash
   bash scripts/setup_ollama.sh
```

4. Configure environment:
```bash
   cp .env.example .env
   # Edit .env with your settings
```

5. Run the API:
```bash
   uv run uvicorn src.api.main:app --reload
```

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.

## Development Roadmap

- [x] M0: Setup environment
- [ ] M1: Data pipeline
- [ ] M2: Vector store + Simple RAG
- [ ] M3: Tool implementation
- [ ] M4: Agentic orchestrator
- [ ] M5: Memory system
- [ ] M6: FastAPI interface
- [ ] M7: Advanced features

See [docs/milestones/](docs/milestones/) for detailed milestone documentation.

## Testing
```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file.
