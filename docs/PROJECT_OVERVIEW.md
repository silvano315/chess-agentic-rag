# Chess Agentic RAG - Project Overview

**Version**: 0.1.0  
**Last Updated**: February 14, 2026  
**Status**: In Development - Milestone 0

---

## 📋 Table of Contents

1. [Project Vision](#project-vision)
2. [Core Objectives](#core-objectives)
3. [Architecture Overview](#architecture-overview)
4. [Tech Stack & Rationale](#tech-stack--rationale)
5. [Development Philosophy](#development-philosophy)
6. [Milestone Roadmap](#milestone-roadmap)
7. [Module Specifications](#module-specifications)
8. [Data Strategy](#data-strategy)
9. [Testing Strategy](#testing-strategy)
10. [API Design](#api-design)
11. [Prompt Engineering Guidelines](#prompt-engineering-guidelines)
12. [Code Standards](#code-standards)

---

## 🎯 Project Vision

Build a **locally-running agentic RAG (Retrieval-Augmented Generation) system** specialized in chess knowledge that can:

- Answer complex chess questions using a knowledge base
- Decide autonomously when to retrieve information vs. use tools
- Execute multi-step reasoning (ReAct pattern)
- Provide insights on openings, tactics, historical games, and player analysis
- Run 100% locally using Ollama (privacy-first, no API costs)

---

## 🎯 Core Objectives

### 1. RAG System Capabilities

**Knowledge Base Coverage:**
- Official FIDE chess rules
- Opening theory (mainlines, variations, sidelines)
- Strategic and tactical patterns
- Historical games and annotations
- Famous player analysis
- Online materials (books, PDFs, PGN collections, articles)

**Target Knowledge Base Size:** ~1,000 documents initially, scalable to 10k+

### 2. Agentic Intelligence

**Decision-Making:**
- Autonomous choice between retrieval and tool usage
- Multi-step reasoning with intermediate state tracking
- Query decomposition for complex questions
- Self-correction and error recovery

**Reasoning Pattern:** ReAct (Reasoning + Acting)

### 3. Tool Ecosystem

**Priority Tools (M3):**
- **ELO Fetcher**: Retrieve player ratings from Lichess/Chess.com APIs
- **PGN Parser**: Parse and analyze game notation using python-chess
- **Game Search**: Search historical games by player, opening, year, result

**Future Tools (M7):**
- **Stockfish Integration**: Position evaluation and best move analysis
- **Opening Database**: Query specific opening lines
- **Endgame Tablebase**: Perfect play lookup for endgame positions

### 4. Local-First Architecture

- No external LLM API dependencies
- All models run via Ollama
- Data stored locally (ChromaDB)
- Optional external API calls only for data enrichment (ELO, games)

---

## 🏗️ Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│  • CLI (notebooks, scripts)                                  │
│  • REST API (FastAPI)                                        │
│  • WebSocket (streaming responses)                           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   AGENTIC ORCHESTRATOR                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Planner    │→ │   Executor   │→ │  Synthesizer │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  • Query understanding and decomposition                     │
│  • Decision making (retrieve vs. tool vs. parametric)        │
│  • Multi-step reasoning loop (ReAct pattern)                 │
│  • State management and memory integration                   │
└────────────┬────────────────┬────────────────┬──────────────┘
             │                │                │
   ┌─────────▼─────┐  ┌───────▼──────┐  ┌─────▼──────┐
   │   RETRIEVAL   │  │ TOOL REGISTRY │  │   MEMORY   │
   │    ENGINE     │  │               │  │   SYSTEM   │
   └───────┬───────┘  └───────┬───────┘  └─────┬──────┘
           │                  │                 │
   ┌───────▼────────┐  ┌──────▼────────┐  ┌────▼──────┐
   │  Vector Store  │  │  Tool Plugins │  │ Conv. Mem │
   │   (Chroma)     │  │  • ELO Fetch  │  │ Work. Mem │
   │                │  │  • PGN Parse  │  │           │
   │  • Semantic    │  │  • Game Search│  │           │
   │  • Metadata    │  │  • Stockfish  │  │           │
   │  • Hybrid      │  └───────────────┘  └───────────┘
   └────────────────┘
           │
   ┌───────▼────────┐
   │  LLM BACKEND   │
   │   (Ollama)     │
   │                │
   │  • DeepSeek-R1 │
   │  • Qwen 2.5    │
   │  • Embeddings  │
   └────────────────┘
```

### Architecture Principles

1. **Modularity**: Each component is independently testable and replaceable
2. **Separation of Concerns**: Clear boundaries between retrieval, reasoning, and execution
3. **Plugin Architecture**: Tools can be added/removed without core changes
4. **Type Safety**: Full type hints and Pydantic validation throughout
5. **Testability**: Unit and integration tests for every component

---

## 🛠️ Tech Stack & Rationale

### Core Technologies

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| **Python 3.11+** | Language | Modern type hints, performance improvements, asyncio |
| **uv** | Package Manager | Fast, reliable, modern alternative to pip/poetry |
| **Ollama** | LLM Backend | Local inference, model management, OpenAI-compatible API |
| **LlamaIndex** | RAG Framework | Best-in-class for retrieval, strong Ollama integration |
| **ChromaDB** | Vector Store | Embedded, zero-config, good for <100k docs |
| **FastAPI** | API Framework | Modern async, auto docs, WebSocket support |
| **Pydantic v2** | Validation | Type-safe configs and models |
| **python-chess** | Chess Library | Industry standard for PGN parsing and game analysis |

### LLM Models (via Ollama)

**Primary Models:**
- **deepseek-r1:1.5b**: Fast reasoning, good for orchestration
- **qwen2.5:7b**: Strong general knowledge, fallback model
- **nomic-embed-text**: Embedding model for semantic search

**Model Selection Strategy:**
- Use DeepSeek for agentic reasoning (lighter, faster)
- Use Qwen for complex synthesis (more capable)
- Swap models dynamically based on task

### Development Tools

| Tool | Purpose |
|------|---------|
| **pytest** | Testing framework |
| **black** | Code formatting |
| **ruff** | Linting and import sorting |
| **mypy** | Static type checking |
| **loguru** | Structured logging |
| **rich** | CLI formatting and progress bars |

---

## 💡 Development Philosophy

### 1. Iterative and Incremental

- Each milestone delivers a working system
- No big-bang integration
- Test early and often

### 2. Data-First Approach

**Rationale**: You cannot build a good RAG without good data.

**Order of Operations:**
1. Collect and validate data sources
2. Design chunking strategy
3. Build retrieval pipeline
4. Add intelligence layer
5. Optimize and refine

### 3. Type Safety and Validation

**All code must include:**
- Type hints on all function signatures
- Pydantic models for data validation
- Mypy compliance (no `type: ignore` without justification)

**Example:**
```python
from pydantic import BaseModel, Field

class ChessQuery(BaseModel):
    """User query with metadata."""
    
    query: str = Field(..., min_length=1, max_length=1000)
    context: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=20)
```

### 4. Testability

**Testing Pyramid:**
```
        /\
       /  \      E2E Tests (few, critical paths)
      /────\
     / Intg \    Integration Tests (component interactions)
    /────────\
   /   Unit   \  Unit Tests (majority, fast, isolated)
  /────────────\
```

**Coverage Target:** >80% for core modules

### 5. Documentation as Code

- Docstrings for all public functions (Google style)
- README per milestone
- Inline comments for complex logic only
- Type hints are documentation

---

## 🗓️ Milestone Roadmap

### Timeline: 3 Months (12 weeks)

```
Month 1: Foundation & Simple RAG
├─ Week 1:  M0 - Environment Setup
├─ Week 2:  M1 - Data Pipeline (part 1)
├─ Week 3:  M1 - Data Pipeline (part 2)
└─ Week 4:  M2 - Vector Store & Simple RAG

Month 2: Tools & Agentic Intelligence
├─ Week 5:  M3 - Tool Implementation
├─ Week 6:  M3 - Tool Registry & Testing
├─ Week 7:  M4 - Orchestrator (ReAct pattern)
└─ Week 8:  M5 - Memory System

Month 3: API & Advanced Features
├─ Week 9:  M6 - FastAPI Implementation
├─ Week 10: M6 - WebSocket & Streaming
├─ Week 11: M7 - Advanced Features (Hybrid Search, Stockfish)
└─ Week 12: M7 - Optimization & Documentation
```

---

## 📊 Module Specifications

### Module 1: Core (`src/core/`)

**Purpose:** Domain models, configuration, shared utilities

**Key Files:**
- `models.py`: Pydantic models for all data structures
- `config.py`: Environment-based configuration management
- `exceptions.py`: Custom exception hierarchy

**Example Model:**
```python
from enum import Enum
from pydantic import BaseModel

class DocumentType(str, Enum):
    ARTICLE = "article"
    PGN = "pgn"
    BOOK = "book"
    ANNOTATION = "annotation"

class Document(BaseModel):
    """Represents a document in the knowledge base."""
    
    id: str
    content: str
    doc_type: DocumentType
    metadata: dict[str, Any]
    embedding: list[float] | None = None
```

---

### Module 2: Data Pipeline (`src/data/`)

**Purpose:** Load, process, and prepare data for indexing

**Architecture:**
```
Loaders (raw data → structured)
    ↓
Processors (structured → chunks + metadata)
    ↓
Pipeline (orchestrates full flow)
```

**Key Components:**

#### 2.1 Loaders (`src/data/loaders/`)

Abstract base loader:
```python
from abc import ABC, abstractmethod
from typing import Iterator

class BaseLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self) -> Iterator[Document]:
        """Load documents from source."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate data source accessibility."""
        pass
```

Implementations:
- `WikipediaLoader`: Scrape chess articles
- `PGNLoader`: Parse PGN game files
- `PDFLoader`: Extract text from chess books

#### 2.2 Processors (`src/data/processors/`)

**Text Chunker:**
- Strategy: Semantic chunking with overlap
- Chunk size: 500-800 tokens
- Overlap: 100 tokens
- Preserve paragraph boundaries

**PGN Processor:**
- Parse games using python-chess
- Extract: players, ELO, opening, moves, result, annotations
- Each game = atomic document
- Metadata-rich for filtering

**Metadata Extractor:**
- Extract structured metadata from all document types
- Standardize format for vector store

#### 2.3 Pipeline (`src/data/pipeline.py`)

**Responsibilities:**
- Orchestrate load → process → validate flow
- Handle errors gracefully (skip bad documents, log)
- Generate manifest file with statistics
- Checkpointing for long-running jobs

---

### Module 3: Retrieval (`src/retrieval/`)

**Purpose:** Vector storage, embedding, and semantic search

**Key Components:**

#### 3.1 Vector Store (`src/retrieval/vector_store.py`)

**Abstraction layer over ChromaDB:**
```python
from typing import Protocol

class VectorStore(Protocol):
    """Protocol for vector store implementations."""
    
    def add_documents(self, documents: list[Document]) -> None: ...
    def query(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[Document]: ...
    def delete(self, doc_ids: list[str]) -> None: ...
    def count(self) -> int: ...
```

**ChromaDB Implementation:**
- Persistent storage in `data/vector_store/chroma_db/`
- Collections per document type (optional optimization)
- Metadata filtering support

#### 3.2 Embeddings (`src/retrieval/embeddings.py`)

**Ollama Embedding Wrapper:**
- Model: `nomic-embed-text` (768 dimensions)
- Caching layer for repeated queries
- Batch processing for indexing

#### 3.3 Query Engine (`src/retrieval/query_engine.py`)

**LlamaIndex Wrapper:**
```python
from llama_index.core import VectorStoreIndex, QueryBundle

class QueryEngine:
    """LlamaIndex-powered query engine."""
    
    def __init__(self, index: VectorStoreIndex, llm: Ollama):
        self.index = index
        self.llm = llm
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: MetadataFilters | None = None
    ) -> QueryResponse:
        """Execute semantic search with LLM synthesis."""
        pass
```

**Features:**
- Semantic search with reranking
- Metadata filtering (e.g., only openings, only games by Kasparov)
- Response synthesis with citations

---

### Module 4: Tools (`src/tools/`)

**Purpose:** Extensible plugin system for external capabilities

**Architecture:**

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ToolInput(BaseModel):
    """Base class for tool inputs."""
    pass

class ToolOutput(BaseModel):
    """Base class for tool outputs."""
    pass

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    name: str
    description: str
    
    @abstractmethod
    def execute(self, input: ToolInput) -> ToolOutput:
        """Execute the tool with given input."""
        pass
    
    def get_schema(self) -> dict:
        """Return JSON schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": input.model_json_schema()
        }
```

#### Tool Implementations

**1. ELO Fetcher (`elo_fetcher.py`)**
```python
class ELOFetcherInput(ToolInput):
    player_name: str
    platform: Literal["lichess", "chess.com"] = "lichess"
    date: str | None = None  # YYYY-MM-DD format

class ELOFetcherOutput(ToolOutput):
    player_name: str
    elo: int
    rank: int | None
    games_played: int
    platform: str

class ELOFetcherTool(BaseTool):
    name = "fetch_player_elo"
    description = "Fetch current or historical ELO rating for a chess player"
    
    def execute(self, input: ELOFetcherInput) -> ELOFetcherOutput:
        # Call Lichess/Chess.com API
        pass
```

**2. PGN Parser (`pgn_parser.py`)**
```python
class PGNParserInput(ToolInput):
    pgn_string: str | None = None
    game_id: str | None = None  # For database lookup

class PGNParserOutput(ToolOutput):
    white: str
    black: str
    elo_white: int | None
    elo_black: int | None
    opening: str
    result: str
    moves: list[str]
    annotations: list[str]

class PGNParserTool(BaseTool):
    name = "parse_pgn"
    description = "Parse PGN notation and extract game information"
    
    def execute(self, input: PGNParserInput) -> PGNParserOutput:
        # Use python-chess library
        pass
```

**3. Game Search (`game_search.py`)**
```python
class GameSearchInput(ToolInput):
    player: str | None = None
    opening: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    min_elo: int = 2000
    limit: int = 10

class GameSearchOutput(ToolOutput):
    games: list[dict]  # List of game metadata
    total_found: int

class GameSearchTool(BaseTool):
    name = "search_games"
    description = "Search historical games by filters"
    
    def execute(self, input: GameSearchInput) -> GameSearchOutput:
        # Query Lichess API + local indexed PGNs
        pass
```

#### Tool Registry (`registry.py`)

```python
class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> BaseTool | None:
        """Retrieve tool by name."""
        return self._tools.get(name)
    
    def get_all_schemas(self) -> list[dict]:
        """Get all tool schemas for LLM function calling."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def execute(self, tool_name: str, input: dict) -> ToolOutput:
        """Execute a tool by name with dict input."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool.execute(tool.input_model(**input))
```

---

### Module 5: Agent (`src/agent/`)

**Purpose:** Agentic orchestration with ReAct pattern

**ReAct Pattern Overview:**
```
Thought → Action → Observation → Thought → Action → ... → Answer
```

#### 5.1 Orchestrator (`orchestrator.py`)

**Main Agent Loop:**
```python
class ChessAgentOrchestrator:
    """Main agentic orchestrator using ReAct pattern."""
    
    def __init__(
        self,
        llm: Ollama,
        query_engine: QueryEngine,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
        max_iterations: int = 5
    ):
        self.llm = llm
        self.query_engine = query_engine
        self.tools = tool_registry
        self.memory = memory
        self.max_iterations = max_iterations
    
    def process_query(self, query: str) -> AgentResponse:
        """Process a user query with multi-step reasoning."""
        
        # Initialize state
        state = AgentState(query=query)
        
        for iteration in range(self.max_iterations):
            # 1. Thought: Reason about next action
            thought = self._reason(state)
            state.add_thought(thought)
            
            # 2. Action: Decide what to do
            action = self._decide_action(thought, state)
            
            if action.type == "final_answer":
                break
            
            # 3. Observation: Execute and observe result
            observation = self._execute_action(action)
            state.add_observation(observation)
        
        # 4. Synthesize final answer
        return self._synthesize(state)
```

#### 5.2 Planner (`planner.py`)

**Query Decomposition:**
```python
class QueryPlanner:
    """Breaks down complex queries into sub-tasks."""
    
    def create_plan(self, query: str) -> Plan:
        """Decompose query into executable steps."""
        # Use LLM to analyze query and create plan
        pass

class Plan(BaseModel):
    """Execution plan for a query."""
    
    steps: list[PlanStep]
    dependencies: dict[str, list[str]]  # Step dependencies

class PlanStep(BaseModel):
    """Single step in execution plan."""
    
    step_id: str
    action_type: Literal["retrieve", "tool", "reason"]
    description: str
    parameters: dict[str, Any]
```

#### 5.3 Executor (`executor.py`)

**Action Execution:**
```python
class ActionExecutor:
    """Executes actions decided by the orchestrator."""
    
    def execute(self, action: Action, context: dict) -> ActionResult:
        """Execute a single action."""
        if action.type == "retrieve":
            return self._execute_retrieval(action)
        elif action.type == "tool":
            return self._execute_tool(action)
        elif action.type == "reason":
            return self._execute_reasoning(action)
```

#### 5.4 Prompts (`prompts.py`)

**System Prompts for Agent:**

```python
CHESS_AGENT_SYSTEM_PROMPT = """
You are an expert chess AI assistant with access to:

1. **Knowledge Base**: Comprehensive chess theory, openings, tactics, historical games
2. **Tools**:
   - fetch_player_elo: Get current/historical player ratings
   - parse_pgn: Analyze chess game notation
   - search_games: Find historical games by filters

**Your Capabilities:**
- Answer questions about chess theory, openings, tactics, endgames
- Analyze specific games and positions
- Compare players and their styles
- Recommend openings and strategies

**Reasoning Process (ReAct Pattern):**
For each query, follow this pattern:

Thought: Analyze what information is needed
Action: Decide to either:
  - Retrieve from knowledge base
  - Call a tool
  - Answer directly from parametric knowledge
Observation: Review the results
... (repeat if needed)
Answer: Synthesize final response

**Guidelines:**
- Always explain your reasoning before acting
- Use tools when you need current data (ELO ratings, specific games)
- Retrieve from knowledge base for theory and historical context
- Be precise with chess notation
- Cite sources when using retrieved information
"""
```

---

### Module 6: Memory (`src/memory/`)

**Purpose:** Maintain conversation context and working state

#### 6.1 Conversation Memory (`conversation.py`)

```python
class Message(BaseModel):
    """Single message in conversation."""
    
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

class ConversationMemory:
    """Manages conversation history with summarization."""
    
    def __init__(self, max_messages: int = 10, llm: Ollama | None = None):
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.llm = llm
    
    def add_message(self, role: str, content: str) -> None:
        """Add message and manage memory limits."""
        self.messages.append(Message(role=role, content=content, timestamp=datetime.now()))
        
        if len(self.messages) > self.max_messages:
            self._summarize_old_messages()
    
    def get_context(self) -> str:
        """Get formatted conversation context for LLM."""
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])
    
    def _summarize_old_messages(self) -> None:
        """Summarize older messages to maintain context window."""
        # Use LLM to create summary of old messages
        pass
```

#### 6.2 Working Memory (`working_memory.py`)

```python
class WorkingMemory:
    """Scratch space for agent reasoning."""
    
    def __init__(self):
        self.facts: dict[str, Any] = {}
        self.intermediate_results: list[dict] = []
        self.retrieved_documents: list[Document] = []
        self.tool_outputs: list[ToolOutput] = []
    
    def add_fact(self, key: str, value: Any) -> None:
        """Store a fact discovered during reasoning."""
        self.facts[key] = value
    
    def get_context_summary(self) -> str:
        """Get summary of working memory for next reasoning step."""
        pass
```

---

### Module 7: LLM Backend (`src/llm/`)

**Purpose:** Abstraction layer for Ollama interactions

#### 7.1 Ollama Client (`ollama_client.py`)

```python
import ollama
from typing import Iterator

class OllamaClient:
    """Wrapper for Ollama API with error handling and retries."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=base_url)
    
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate text completion."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["response"]
    
    def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
        temperature: float = 0.7
    ) -> dict:
        """Chat completion with optional function calling."""
        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=functions,
            options={"temperature": temperature}
        )
        return response
    
    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7
    ) -> Iterator[str]:
        """Stream chat completion."""
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": temperature}
        )
        for chunk in stream:
            yield chunk["message"]["content"]
```

#### 7.2 Function Calling (`function_calling.py`)

```python
def parse_function_call(response: dict) -> FunctionCall | None:
    """Parse function call from LLM response."""
    if "tool_calls" in response.get("message", {}):
        tool_call = response["message"]["tool_calls"][0]
        return FunctionCall(
            name=tool_call["function"]["name"],
            arguments=json.loads(tool_call["function"]["arguments"])
        )
    return None

class FunctionCall(BaseModel):
    """Represents a function call from LLM."""
    
    name: str
    arguments: dict[str, Any]
```

---

### Module 8: API (`src/api/`)

**Purpose:** FastAPI REST and WebSocket interface

#### 8.1 Main App (`main.py`)

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Chess Agentic RAG API",
    description="Intelligent chess assistant with RAG and tool calling",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from src.api.routes import query, tools, admin

app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(tools.router, prefix="/tools", tags=["Tools"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}
```

#### 8.2 Query Routes (`routes/query.py`)

```python
from fastapi import APIRouter, Depends
from src.api.models.requests import QueryRequest
from src.api.models.responses import QueryResponse

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    orchestrator: ChessAgentOrchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """Execute a query against the chess knowledge base."""
    result = orchestrator.process_query(request.query)
    return QueryResponse(
        query=request.query,
        answer=result.answer,
        sources=result.sources,
        reasoning_steps=result.reasoning_steps,
        tools_used=result.tools_used
    )

@router.websocket("/ws")
async def websocket_query(websocket: WebSocket):
    """WebSocket endpoint for streaming responses."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Stream response chunks
            async for chunk in orchestrator.stream_query(data):
                await websocket.send_text(chunk)
    except WebSocketDisconnect:
        pass
```

---

## 📚 Data Strategy

### Data Sources (Priority Order)

#### 1. Wikipedia Chess Articles (Week 2)
- **Target**: ~200 articles
- **Topics**: Major openings, famous players, historical tournaments, chess theory
- **Scraping**: Use `beautifulsoup4` with rate limiting and caching
- **Quality**: High (curated, accurate)

#### 2. Lichess Elite Database (Week 2)
- **Target**: 500-1000 games (ELO >2500)
- **Format**: PGN files
- **Source**: [Lichess Elite Database](https://database.nikonoel.fr/)
- **Metadata**: Players, ELO, opening, date, event, result, annotations

#### 3. Chess.com Articles (Week 3)
- **Target**: ~300 articles
- **Topics**: Strategy, tactics, opening guides
- **Access**: Ethical scraping with robots.txt compliance
- **Quality**: High (expert-written)

#### 4. Open Source Chess Books (Week 3 - Optional)
- **Target**: 2-3 books
- **Format**: PDF
- **Candidates**:
  - "My System" by Aron Nimzowitsch (public domain)
  - "Lasker's Manual of Chess" (public domain)
  - Modern opening theory PDFs (if available)

### Chunking Strategy

**Text Documents (Articles, Books):**
```python
CHUNK_SIZE = 600  # tokens
OVERLAP = 100     # tokens
STRATEGY = "semantic"  # Preserve paragraphs

# Example metadata
{
    "source": "wikipedia",
    "title": "Sicilian Defense",
    "topic": "opening",
    "url": "https://...",
    "chunk_id": "sicilian_001"
}
```

**PGN Games:**
```python
# Each game = single document
{
    "white": "Magnus Carlsen",
    "black": "Fabiano Caruana",
    "elo_white": 2882,
    "elo_black": 2835,
    "opening": "Sicilian Najdorf",
    "eco": "B90",
    "result": "1/2-1/2",
    "year": 2023,
    "event": "World Championship",
    "moves": "1. e4 c5 2. Nf3 ...",
    "annotations": ["!!", "Brilliant move", ...]
}
```

### Validation Pipeline

```python
def validate_document(doc: Document) -> bool:
    """Validate document quality before indexing."""
    checks = [
        len(doc.content) > 50,  # Minimum length
        doc.metadata.get("source") is not None,
        not contains_corrupted_chars(doc.content),
        is_chess_relevant(doc.content)  # Basic keyword check
    ]
    return all(checks)
```

---

## 🧪 Testing Strategy

### Testing Pyramid

```
E2E Tests (5%)
├─ Full API request → response
└─ Multi-tool agent queries

Integration Tests (25%)
├─ Data pipeline: load → process → index
├─ RAG: query → retrieve → synthesize
├─ Agent: query → plan → execute → answer
└─ API: endpoint → orchestrator → response

Unit Tests (70%)
├─ Data loaders
├─ Text processors
├─ Tool implementations
├─ Memory management
├─ LLM client wrappers
└─ Utility functions
```

### Test Coverage Targets

| Module | Unit Test Coverage | Integration Test Coverage |
|--------|-------------------|--------------------------|
| `core` | 90%+ | N/A |
| `data` | 85%+ | 80%+ |
| `retrieval` | 85%+ | 90%+ |
| `tools` | 90%+ | 85%+ |
| `agent` | 75%+ | 85%+ |
| `memory` | 85%+ | 70%+ |
| `llm` | 80%+ | 75%+ |
| `api` | 70%+ | 90%+ |

### Example Test Structure

**Unit Test Example (`tests/unit/test_tools.py`):**
```python
import pytest
from src.tools.elo_fetcher import ELOFetcherTool, ELOFetcherInput

@pytest.fixture
def elo_tool():
    return ELOFetcherTool()

def test_elo_fetcher_valid_player(elo_tool, mock_lichess_api):
    """Test ELO fetcher with valid player name."""
    input_data = ELOFetcherInput(player_name="MagnusCarlsen", platform="lichess")
    result = elo_tool.execute(input_data)
    
    assert result.elo > 2000
    assert result.player_name == "MagnusCarlsen"
    assert result.platform == "lichess"

def test_elo_fetcher_invalid_player(elo_tool, mock_lichess_api):
    """Test ELO fetcher with non-existent player."""
    input_data = ELOFetcherInput(player_name="NonExistentPlayer123", platform="lichess")
    
    with pytest.raises(PlayerNotFoundError):
        elo_tool.execute(input_data)
```

**Integration Test Example (`tests/integration/test_agent.py`):**
```python
import pytest
from src.agent.orchestrator import ChessAgentOrchestrator

@pytest.fixture
def agent(test_llm, test_rag, test_tools, test_memory):
    return ChessAgentOrchestrator(
        llm=test_llm,
        query_engine=test_rag,
        tool_registry=test_tools,
        memory=test_memory
    )

def test_agent_multi_step_reasoning(agent):
    """Test agent can handle multi-step query requiring RAG + tool."""
    query = "What is Magnus Carlsen's current ELO and what opening does he play most?"
    
    response = agent.process_query(query)
    
    # Should use ELO fetcher tool
    assert "fetch_player_elo" in response.tools_used
    
    # Should retrieve from knowledge base
    assert len(response.sources) > 0
    
    # Should have reasoning steps
    assert len(response.reasoning_steps) >= 2
    
    # Should have valid answer
    assert "Carlsen" in response.answer
    assert any(opening in response.answer for opening in ["Sicilian", "Ruy Lopez"])
```

### Test Fixtures (`tests/conftest.py`)

```python
import pytest
from src.llm.ollama_client import OllamaClient
from src.retrieval.query_engine import QueryEngine
from src.tools.registry import ToolRegistry

@pytest.fixture(scope="session")
def test_llm():
    """Mock or real Ollama client for testing."""
    return OllamaClient(model="deepseek-r1:1.5b")

@pytest.fixture(scope="session")
def test_rag(test_vector_store, test_llm):
    """Test RAG query engine with small dataset."""
    return QueryEngine(index=test_vector_store, llm=test_llm)

@pytest.fixture
def test_tools():
    """Registry with mock tools for testing."""
    registry = ToolRegistry()
    # Register mock tools
    return registry

@pytest.fixture
def sample_pgn():
    """Sample PGN for testing."""
    return '''
    [Event "World Championship"]
    [White "Carlsen, Magnus"]
    [Black "Caruana, Fabiano"]
    [Result "1/2-1/2"]
    
    1. e4 c5 2. Nf3 ...
    '''
```

---

## 🌐 API Design

### REST Endpoints

#### Query Endpoints
- `POST /query` - Execute query
- `POST /query/stream` - Execute query with streaming response
- `GET /query/history` - Get conversation history

#### Tool Endpoints
- `GET /tools` - List available tools
- `GET /tools/{tool_name}` - Get tool schema
- `POST /tools/{tool_name}/execute` - Execute tool directly

#### Admin Endpoints
- `POST /admin/index/refresh` - Re-index documents
- `GET /admin/stats` - Get system statistics
- `GET /admin/health` - Detailed health check

### Request/Response Models

**Query Request:**
```json
{
  "query": "Explain the Najdorf variation of the Sicilian Defense",
  "max_results": 5,
  "include_reasoning": true,
  "filters": {
    "document_type": "article",
    "topic": "opening"
  }
}
```

**Query Response:**
```json
{
  "query": "Explain the Najdorf variation...",
  "answer": "The Najdorf variation is one of the sharpest...",
  "sources": [
    {
      "title": "Sicilian Defense Overview",
      "snippet": "...named after GM Miguel Najdorf...",
      "url": "https://...",
      "relevance_score": 0.92
    }
  ],
  "reasoning_steps": [
    {
      "step": 1,
      "thought": "User asks about specific opening variation",
      "action": "retrieve_from_knowledge_base",
      "observation": "Found 5 relevant documents about Najdorf"
    }
  ],
  "tools_used": [],
  "response_time_ms": 1234
}
```

### WebSocket Protocol

**Client → Server:**
```json
{
  "type": "query",
  "query": "Compare Kasparov vs Carlsen",
  "stream": true
}
```

**Server → Client (streaming):**
```json
{"type": "thought", "content": "Analyzing query..."}
{"type": "action", "action_name": "fetch_player_elo", "arguments": {...}}
{"type": "observation", "content": "Retrieved ELO data..."}
{"type": "chunk", "content": "Kasparov's peak ELO was..."}
{"type": "chunk", "content": "while Carlsen's current..."}
{"type": "done", "final_answer": "..."}
```

---

## 🎨 Prompt Engineering Guidelines

### System Prompt Structure

**Components:**
1. **Role Definition**: Who the AI is
2. **Capabilities**: What it can do
3. **Available Tools**: Tool descriptions and when to use them
4. **Reasoning Pattern**: How to approach queries (ReAct)
5. **Output Format**: Expected response structure
6. **Guidelines**: Best practices and constraints

### Few-Shot Examples

**Include in system prompt:**
```
Example 1:
User: "What is Magnus Carlsen's current rating?"
Thought: User wants current data, need to use tool
Action: fetch_player_elo(player_name="MagnusCarlsen", platform="lichess")
Observation: ELO=2830, rank=1
Answer: Magnus Carlsen's current rating is 2830 on Lichess, where he is ranked #1.

Example 2:
User: "Explain the Queen's Gambit"
Thought: This is theory question, check knowledge base
Action: retrieve_from_knowledge_base(query="Queen's Gambit opening theory")
Observation: Found detailed explanation from Wikipedia and chess articles
Answer: The Queen's Gambit is a chess opening that starts with 1.d4 d5 2.c4...
```

### Function Calling Schemas

**Tool schemas should be:**
- Clear and descriptive
- Include examples in descriptions
- Specify required vs. optional parameters
- Include validation constraints

**Example:**
```json
{
  "name": "fetch_player_elo",
  "description": "Fetches the current or historical ELO rating for a chess player from Lichess or Chess.com. Use this when user asks about player ratings, rankings, or strength.",
  "parameters": {
    "type": "object",
    "properties": {
      "player_name": {
        "type": "string",
        "description": "Player's username on the platform (e.g., 'MagnusCarlsen')"
      },
      "platform": {
        "type": "string",
        "enum": ["lichess", "chess.com"],
        "description": "Chess platform to query. Default: lichess"
      },
      "date": {
        "type": "string",
        "description": "Optional: Historical date in YYYY-MM-DD format. If not provided, returns current rating."
      }
    },
    "required": ["player_name"]
  }
}
```

---

## 📏 Code Standards

### General Principles

1. **Type Everything**: Full type hints on all functions
2. **Validate Everything**: Use Pydantic for all data structures
3. **Document Public APIs**: Docstrings in Google style
4. **Test Everything**: Unit + integration tests for all modules
5. **Log Appropriately**: Use structured logging (loguru)
6. **Handle Errors Gracefully**: Custom exceptions with clear messages

### Code Style

**Formatting:**
- **Formatter**: black (line length: 100)
- **Import sorting**: ruff
- **Linting**: ruff (E, F, I, N, W, UP rules)
- **Type checking**: mypy (strict mode)

**Naming Conventions:**
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Type aliases: `PascalCase` with `Type` suffix (e.g., `DocumentType`)

### Docstring Format (Google Style)

```python
def process_query(
    query: str,
    filters: dict[str, Any] | None = None,
    max_results: int = 5
) -> QueryResponse:
    """Process a user query with optional filtering.
    
    This function orchestrates the full RAG pipeline including retrieval,
    reranking, and synthesis.
    
    Args:
        query: The user's natural language query.
        filters: Optional metadata filters to apply during retrieval.
            Keys should match document metadata fields.
        max_results: Maximum number of documents to retrieve. Must be
            between 1 and 20.
    
    Returns:
        QueryResponse containing the answer, sources, and metadata.
    
    Raises:
        ValueError: If max_results is out of valid range.
        RetrievalError: If document retrieval fails.
    
    Example:
        >>> response = process_query(
        ...     "Explain the Najdorf",
        ...     filters={"doc_type": "article"},
        ...     max_results=3
        ... )
        >>> print(response.answer)
    """
    pass
```

### Error Handling

**Custom Exception Hierarchy:**
```python
# src/core/exceptions.py

class ChessRAGException(Exception):
    """Base exception for all chess RAG errors."""
    pass

class DataPipelineError(ChessRAGException):
    """Errors during data loading or processing."""
    pass

class RetrievalError(ChessRAGException):
    """Errors during document retrieval."""
    pass

class ToolExecutionError(ChessRAGException):
    """Errors during tool execution."""
    pass

class AgentError(ChessRAGException):
    """Errors in agent orchestration."""
    pass
```

**Usage:**
```python
try:
    documents = loader.load()
except DataPipelineError as e:
    logger.error(f"Failed to load documents: {e}")
    raise
```

### Logging Standards

**Use loguru with structured logging:**
```python
from loguru import logger

logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)

# Usage
logger.info("Starting document indexing", num_docs=len(documents))
logger.debug("Query parameters", query=query, filters=filters)
logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
```

---

## 🔄 Development Workflow

### Git Workflow

**Branch Strategy:**
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature branches
- `bugfix/*`: Bug fixes
- `milestone/*`: Milestone-specific work

**Commit Messages:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: feat, fix, docs, style, refactor, test, chore

**Example:**
```
feat(tools): add ELO fetcher with Lichess API integration

- Implement ELOFetcherTool with Lichess API client
- Add comprehensive error handling for API failures
- Include unit tests with mocked API responses
- Add integration test with real API (marked as slow)

Closes #23
```

### PR Checklist

Before merging:
- [ ] All tests pass (`pytest`)
- [ ] Code formatted (`black src tests`)
- [ ] Linting clean (`ruff check src tests`)
- [ ] Type checking passes (`mypy src`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Milestone README updated (if applicable)

---

## 📊 Success Metrics

### Milestone Acceptance Criteria

**M2 (Simple RAG):**
- ✅ Retrieval accuracy: 15/20 test queries correct
- ✅ Average query time: <2 seconds
- ✅ Faithfulness score: >0.8 (LlamaIndex metric)

**M4 (Agent):**
- ✅ Tool selection accuracy: 8/10 correct choices
- ✅ No infinite loops in 20 test scenarios
- ✅ Multi-step completion: 7/10 queries with 2+ steps

**M6 (API):**
- ✅ API response time p95: <5 seconds
- ✅ WebSocket stability: >5 min continuous connection
- ✅ Error rate: <1% on happy path

### Quality Metrics

**Code Quality:**
- Test coverage: >80%
- Mypy compliance: 100%
- Ruff violations: 0
- Documentation coverage: 100% for public APIs

**RAG Quality:**
- Faithfulness: Answer supported by retrieved docs
- Relevancy: Retrieved docs relevant to query
- Latency: Time from query to response

---

## 🚀 Deployment Considerations (Future)

### Optimization Opportunities

1. **Embedding Caching**: Cache embeddings for common queries
2. **Model Quantization**: Use quantized models for faster inference
3. **Batch Processing**: Batch document indexing
4. **Connection Pooling**: Pool Ollama connections

### Monitoring (M7 - Optional)

- **Metrics**: Prometheus + Grafana
- **Tracing**: OpenTelemetry
- **Logging**: Centralized with loguru
- **Alerting**: Error rate thresholds

---

## 📚 References

### Key Technologies Documentation

- **LlamaIndex**: https://docs.llamaindex.ai/
- **Ollama**: https://ollama.ai/
- **ChromaDB**: https://docs.trychroma.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **python-chess**: https://python-chess.readthedocs.io/

### Academic Papers

- **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- **Tool Use**: "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)

### Chess Resources

- **Lichess API**: https://lichess.org/api
- **Chess.com API**: https://www.chess.com/news/view/published-data-api
- **PGN Standard**: http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm

---

## 🎯 Next Steps (Post-Document)

After this document is created:

1. **Review**: Ensure alignment with project vision
2. **Setup**: Initialize repository with structure
3. **M0.3**: Begin Ollama setup and testing
4. **Iterate**: Refine as we learn during implementation

---

**Document Version**: 1.0  
**Maintained By**: Project Team  
**Review Cycle**: Updated at each milestone completion