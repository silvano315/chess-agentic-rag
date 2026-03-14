# Milestone 1: Data Pipeline

**Status**: In Progress  
**Timeline**: Week 2-4  
**Goal**: Build data collection, processing, and preparation pipeline for chess knowledge

---

## 📋 Objectives

1. ✅ Implement data loaders for multiple sources
2. ✅ Build text processing and chunking system
3. ✅ Create PGN parsing and metadata extraction
4. ✅ Develop data validation pipeline
5. ✅ Generate data manifest and statistics

---

## 🎯 Deliverables

### Phase 1: Data Loaders (Week 2)
- [ ] `src/data/loaders/base.py` - Abstract base loader
- [ ] `src/data/loaders/wikipedia_loader.py` - Wikipedia scraper
- [ ] `src/data/loaders/lichess_loader.py` - Lichess PGN downloader
- [ ] `src/data/loaders/pdf_loader.py` - PDF text extractor (optional)
- [ ] Unit tests for all loaders
- [ ] Download script: `scripts/download_data.py`

### Phase 2: Data Processing (Week 3)
- [ ] `src/data/processors/text_chunker.py` - Semantic chunking
- [ ] `src/data/processors/pgn_processor.py` - PGN parser
- [ ] `src/data/processors/metadata_extractor.py` - Metadata extraction
- [ ] Unit tests for all processors
- [ ] Integration tests for processing pipeline

### Phase 3: Pipeline & Validation (Week 4)
- [ ] `src/data/pipeline.py` - Main pipeline orchestrator
- [ ] Data validation logic
- [ ] Manifest generator
- [ ] Validation script: `scripts/validate_data.py`
- [ ] Full integration test
- [ ] Data exploration notebook

---

## 📊 Data Sources

### Priority 1: Wikipedia (Week 2)
**Target**: ~200 articles  
**Topics**: Openings, players, tournaments, theory  
**Format**: HTML → Markdown/Text  
**Tools**: `requests`, `beautifulsoup4`

**Example articles**:
- https://en.wikipedia.org/wiki/Sicilian_Defence
- https://en.wikipedia.org/wiki/Magnus_Carlsen
- https://en.wikipedia.org/wiki/World_Chess_Championship

### Priority 2: Lichess Elite Database (Week 2-3)
**Target**: 500-1000 games (ELO >2500)  
**Source**: https://database.lichess.org/ or Lichess API  
**Format**: PGN files  
**Tools**: `requests`, `python-chess`

### Priority 3: Chess.com Articles (Week 3 - Optional)
**Target**: ~300 articles  
**Topics**: Strategy, tactics, opening guides  
**Format**: HTML → Text  
**Tools**: `requests`, `beautifulsoup4`

### Priority 4: PDF Books (Week 4 - Optional)
**Target**: 2-3 books  
**Candidates**: Public domain chess books  
**Format**: PDF → Text  
**Tools**: `pypdf` or `pdfplumber`

---

## 🔧 Technical Specifications

### Chunking Strategy

**For Text Documents (Articles, Books)**:
```python
CHUNK_SIZE = 600  # tokens
OVERLAP = 100     # tokens
STRATEGY = "semantic"  # Preserve paragraphs

# Metadata structure
{
    "source": "wikipedia",
    "title": "Sicilian Defense",
    "topic": "opening",
    "url": "https://...",
    "chunk_id": "sicilian_001",
    "chunk_index": 0,
    "total_chunks": 5
}
```

**For PGN Games**:
```python
# Each game = atomic document (no chunking)
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
    "moves": "1. e4 c5 2. Nf3...",
    "move_count": 45,
    "annotations": ["!!", "Brilliant move"]
}
```

### Data Validation Rules

```python
def validate_document(doc: Document) -> bool:
    """Validate document quality before processing."""
    checks = [
        len(doc.content) > 50,  # Minimum length
        doc.metadata.get("source") is not None,
        not contains_corrupted_chars(doc.content),
        is_chess_relevant(doc.content),  # Basic keyword check
        has_required_metadata(doc)
    ]
    return all(checks)
```

---

## 📝 Implementation Plan

### Week 2: Loaders

**Day 1-2: Base Architecture**
- Create `base.py` with abstract loader
- Set up loader interface
- Add error handling patterns
- Write base loader tests

**Day 3-4: Wikipedia Loader**
- Implement Wikipedia scraper
- Handle rate limiting
- Extract clean text
- Add metadata extraction
- Write tests

**Day 5-7: Lichess Loader**
- Implement Lichess API client
- Download PGN files
- Parse game metadata
- Handle pagination
- Write tests

**Deliverable**: ~200 Wikipedia articles + ~500 PGN games in `data/raw/`

### Week 3: Processors

**Day 1-2: Text Chunker**
- Implement semantic chunking
- Add overlap logic
- Preserve paragraph boundaries
- Handle edge cases
- Write tests

**Day 3-4: PGN Processor**
- Parse PGN using python-chess
- Extract all metadata
- Handle annotations
- Validate game structure
- Write tests

**Day 5-7: Metadata Extractor**
- Standardize metadata format
- Extract topics/tags
- Add derived metadata
- Write tests

**Deliverable**: Processed chunks in `data/processed/chunks/`

### Week 4: Pipeline & Validation

**Day 1-3: Pipeline Orchestrator**
- Implement main pipeline
- Add load → process → validate flow
- Handle errors gracefully
- Add checkpointing
- Add progress reporting

**Day 4-5: Validation & Manifest**
- Implement validators
- Generate statistics
- Create manifest file
- Add quality reports

**Day 6-7: Integration & Testing**
- Full integration tests
- Data exploration notebook
- Documentation
- Bug fixes

**Deliverable**: Complete pipeline that produces validated, processed data

---

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] Can download Wikipedia articles by topic
- [ ] Can download Lichess games by filters
- [ ] Text chunking produces 500-800 token chunks
- [ ] PGN parsing extracts all required metadata
- [ ] Pipeline runs end-to-end without errors
- [ ] Validation catches corrupted/invalid data
- [ ] Manifest file generated with statistics

### Quality Requirements
- [ ] Unit test coverage >80% for loaders
- [ ] Unit test coverage >80% for processors
- [ ] Integration test covers full pipeline
- [ ] All code passes type checking (mypy)
- [ ] All code formatted (black) and linted (ruff)
- [ ] Comprehensive error handling and logging

### Data Quality
- [ ] >90% of documents pass validation
- [ ] Chunks maintain semantic coherence
- [ ] PGN games parse without errors
- [ ] Metadata complete for all documents
- [ ] No duplicate documents

---

## 🧪 Testing Strategy

### Unit Tests

```bash
# Test individual loaders
uv run pytest tests/unit/test_loaders.py -v

# Test individual processors
uv run pytest tests/unit/test_processors.py -v

# Test with mocked data
uv run pytest tests/unit -m "not integration"
```

### Integration Tests

```bash
# Test full pipeline with real data
uv run pytest tests/integration/test_data_pipeline.py -v --slow

# Test with sample data
uv run pytest tests/integration -k "sample"
```

### Manual Testing

```bash
# Download sample data
uv run python scripts/download_data.py --source wikipedia --limit 10

# Validate downloaded data
uv run python scripts/validate_data.py --input data/raw/

# Run full pipeline
uv run python -m src.data.pipeline --config config.yaml
```

---

## 📊 Expected Outputs

### Data Manifest (data/processed/manifest.json)

```json
{
  "version": "1.0",
  "created_at": "2026-03-15T10:30:00Z",
  "sources": {
    "wikipedia": {
      "documents": 200,
      "chunks": 1500,
      "avg_chunk_size": 650,
      "topics": ["openings", "players", "tournaments"]
    },
    "lichess": {
      "games": 500,
      "avg_elo": 2650,
      "date_range": ["2020-01-01", "2024-12-31"],
      "openings": ["Sicilian", "Ruy Lopez", "Queen's Gambit"]
    }
  },
  "statistics": {
    "total_documents": 700,
    "total_chunks": 2000,
    "validation_pass_rate": 0.95,
    "avg_metadata_completeness": 0.92
  }
}
```

### Directory Structure After M1

```
data/
├── raw/
│   ├── wikipedia/
│   │   ├── Sicilian_Defence.html
│   │   ├── Magnus_Carlsen.html
│   │   └── ...
│   ├── lichess_pgn/
│   │   ├── elite_2023_01.pgn
│   │   ├── elite_2023_02.pgn
│   │   └── ...
│   └── manifest_raw.json
├── processed/
│   ├── chunks/
│   │   ├── wikipedia_chunks.json
│   │   └── lichess_games.json
│   ├── metadata/
│   │   └── metadata_index.json
│   └── manifest.json
```

---

## 🐛 Common Issues & Solutions

### Issue: Rate Limiting (Wikipedia)
**Solution**: Add delays between requests, implement exponential backoff

### Issue: Large PGN Files
**Solution**: Stream processing, process in batches

### Issue: Corrupted/Invalid Data
**Solution**: Robust error handling, skip invalid entries, log warnings

### Issue: Inconsistent Metadata
**Solution**: Standardization layer, default values, validation

---

## 🔗 Related Documentation

- [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) - Module 2: Data Pipeline section
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
- [Lichess API](https://lichess.org/api)
- [python-chess Documentation](https://python-chess.readthedocs.io/)

---

## ⏭️ Next Milestone

Once M1 is complete, proceed to:

**Milestone 2: Vector Store & Simple RAG**
- ChromaDB setup
- LlamaIndex integration
- Embedding generation
- Query engine implementation
- RAG evaluation

---

**Last Updated**: March 15, 2026  
**Status**: In Progress  
**Current Phase**: Phase 1 - Data Loaders