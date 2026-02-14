# chess-agentic-rag
This is a RAG system for chess lovers like me. It's a first attempt for something bigger, I hope :) 


## Project structure

chess-agentic-rag/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # GitHub Actions: tests, linting
│       └── release.yml               # Optional: release automation
│
├── docs/
│   ├── PROJECT_OVERVIEW.md          # 🎯 Il documento master che genereremo
│   ├── ARCHITECTURE.md              # Decisioni architetturali
│   ├── API_REFERENCE.md             # API documentation
│   └── milestones/
│       ├── M0_SETUP.md
│       ├── M1_DATA_PIPELINE.md
│       ├── M2_VECTOR_STORE.md
│       ├── M3_TOOLS.md
│       ├── M4_ORCHESTRATOR.md
│       ├── M5_MEMORY.md
│       ├── M6_API.md
│       └── M7_ADVANCED.md
│
├── data/
│   ├── raw/                          # Original data sources
│   │   ├── wikipedia/
│   │   ├── lichess_pgn/
│   │   ├── articles/
│   │   └── books/
│   ├── processed/                    # Cleaned and chunked data
│   │   ├── chunks/
│   │   └── metadata/
│   └── vector_store/                 # Chroma persistence
│       └── chroma_db/
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                         # Core domain models
│   │   ├── __init__.py
│   │   ├── models.py                # Pydantic models (Query, Document, etc.)
│   │   ├── config.py                # Configuration management
│   │   └── exceptions.py            # Custom exceptions
│   │
│   ├── data/                         # Data pipeline
│   │   ├── __init__.py
│   │   ├── loaders/                 # Data source loaders
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract base loader
│   │   │   ├── wikipedia_loader.py
│   │   │   ├── pgn_loader.py
│   │   │   └── pdf_loader.py
│   │   ├── processors/              # Data processors
│   │   │   ├── __init__.py
│   │   │   ├── text_chunker.py
│   │   │   ├── pgn_processor.py
│   │   │   └── metadata_extractor.py
│   │   └── pipeline.py              # Main data pipeline orchestrator
│   │
│   ├── retrieval/                    # RAG retrieval layer
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Vector DB abstraction
│   │   ├── embeddings.py            # Embedding models
│   │   ├── query_engine.py          # LlamaIndex query engine wrapper
│   │   └── hybrid_search.py         # Hybrid semantic + keyword (M7)
│   │
│   ├── tools/                        # Agent tools
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseTool abstract class
│   │   ├── registry.py              # Tool registry
│   │   ├── elo_fetcher.py           # ELO rating fetcher
│   │   ├── pgn_parser.py            # PGN parser tool
│   │   ├── game_search.py           # Game search tool
│   │   └── stockfish.py             # Stockfish integration (M7)
│   │
│   ├── agent/                        # Agentic orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # Main agent orchestrator (ReAct)
│   │   ├── planner.py               # Query planning and decomposition
│   │   ├── executor.py              # Action executor
│   │   └── prompts.py               # System prompts and templates
│   │
│   ├── memory/                       # Memory management
│   │   ├── __init__.py
│   │   ├── conversation.py          # Conversation history
│   │   ├── working_memory.py        # Agent working memory
│   │   └── storage.py               # Persistent memory (optional)
│   │
│   ├── llm/                          # LLM backend
│   │   ├── __init__.py
│   │   ├── ollama_client.py         # Ollama client wrapper
│   │   ├── model_manager.py         # Model selection and management
│   │   └── function_calling.py      # Function calling utilities
│   │
│   ├── api/                          # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── query.py            # Query endpoints
│   │   │   ├── tools.py            # Tool management endpoints
│   │   │   └── admin.py            # Admin endpoints (re-index, etc.)
│   │   ├── models/                  # API request/response models
│   │   │   ├── __init__.py
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── dependencies.py          # FastAPI dependencies
│   │
│   ├── evaluation/                   # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py               # Faithfulness, relevancy, etc.
│   │   ├── evaluator.py             # Evaluation runner
│   │   └── test_queries.py          # Test query sets
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── logging.py               # Logging configuration
│       ├── validators.py            # Input validation
│       └── helpers.py               # Misc helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   │
│   ├── unit/                         # Unit tests
│   │   ├── __init__.py
│   │   ├── test_data_loaders.py
│   │   ├── test_processors.py
│   │   ├── test_tools.py
│   │   ├── test_memory.py
│   │   └── test_llm_client.py
│   │
│   ├── integration/                  # Integration tests
│   │   ├── __init__.py
│   │   ├── test_data_pipeline.py
│   │   ├── test_retrieval.py
│   │   ├── test_agent.py
│   │   └── test_api.py
│   │
│   └── fixtures/                     # Test data
│       ├── sample_pgn.txt
│       ├── sample_article.txt
│       └── test_queries.json
│
├── helpers/                         # Utility scripts
│   ├── setup_ollama.sh              # Ollama installation + model download
│   ├── download_data.py             # Data collection script
│   ├── index_documents.py           # Index creation script
│   ├── evaluate_rag.py              # Run evaluation
│   └── benchmark.py                 # Performance benchmarking
│
├── notebooks/                        # Jupyter notebooks
│   ├── 00_environment_test.ipynb    # Test environment setup
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_testing.ipynb
│   ├── 03_tool_testing.ipynb
│   └── 04_agent_testing.ipynb
│
├── results/                          # Evaluation results
│   ├── evaluations/
│   └── benchmarks/
│
├── .env.example                      # Environment variables template
├── .gitignore
├── .python-version                   # Python version for pyenv/uv
├── pyproject.toml                    # Project dependencies (uv)
├── uv.lock                          # Lock file
├── README.md                         # Main project README
├── CONTRIBUTING.md                   # Contribution guidelines
└── LICENSE                           # MIT or your choice