# Changelog

All notable changes to the NVIDIA RAG Agent project will be documented in this file.

## [Unreleased]

### Added
- **Docker Support**: Added `Dockerfile` and `docker-compose.yml` for containerized deployment
- **Health Check Script**: New `health_check.py` utility to validate environment, dependencies, and API connectivity
- **Enhanced Circular Import Handling**: `sarvam_ask_agent()` now accepts optional `retrieve_fn` parameter to avoid circular imports
- **Type Hints**: Improved type annotations across codebase for better IDE support
- **Version Pinning**: All dependencies in `requirements.txt` now pinned for reproducibility
- **Comprehensive Test Suite**: Fixed test files to run without requiring actual API keys
  - `tests/test_chunking.py` - Tests for PDF chunking logic
  - `tests/test_embeddings.py` - Tests for embedding similarity calculations
  - `tests/test_retry_logic.py` - Tests for exponential backoff retry mechanism

### Changed
- **requirements.txt**: Pinned specific versions for all dependencies
- **Test Files**: Updated to use dummy API keys during import, allowing tests to run in CI/CD environments
- **Error Handling**: Improved floating-point tolerance in cosine similarity tests

### Fixed
- **Circular Import Risk**: Resolved potential circular dependency between `rag_agent.py` and `sarvam_rag.py`
- **Test Failures**: Fixed floating-point precision issues in embedding tests
- **Missing Dependency**: Confirmed `pymupdf4llm` is included in requirements.txt

### Documentation
- Added `.gitignore` with comprehensive Python/Docker exclusions
- Created `CHANGELOG.md` for tracking changes
- Enhanced docstrings with parameter descriptions

---

## [1.0.0] - Initial Release
- NVIDIA RAG Pipeline with multi-modal support
- Sarvam AI multilingual integration
- Production-ready features (retry logic, checkpointing, atomic saves)
- CLI interface with flexible options
- Streamlit UI support
