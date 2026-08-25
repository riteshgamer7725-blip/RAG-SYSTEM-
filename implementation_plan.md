# RAG Pipeline Architecture Audit & Fix Plan

Full audit of your pipeline against the NVIDIA RAG checklist. Each item is graded ✅ (pass), ⚠️ (partial), or ❌ (missing/broken).

---

## 1. Architecture & Data Flow

### Component Map

| Component | Checklist | Your Pipeline | Status |
|-----------|-----------|---------------|--------|
| **Embedding Model** | NVIDIA NIM endpoint | `nvidia/llama-3.2-nv-embedqa-1b-v2` | ⚠️ Works but small (1B). Upgrade path: `nv-embedqa-e5-v5` (1024-dim) |
| **Vector Store** | Milvus / FAISS / pgvector | In-memory `np.ndarray` + `.npy` file | ⚠️ Works for MVP, no ANN index — O(n) brute-force |
| **Retriever** | Dense / Sparse / Hybrid | Dense-only (cosine similarity) | ⚠️ No BM25 or hybrid — misses keyword matches |
| **Reranker** | `nvidia/nv-rerankqa-mistral-4b-v3` | **Missing entirely** | ❌ Critical for accuracy |
| **LLM Generator** | NVIDIA NIM endpoint | `meta/llama-3.3-70b-instruct` | ✅ Good choice |
| **Vision Model** | For image-heavy PDFs | `meta/llama-3.2-90b-vision-instruct` | ✅ Good |
| **Orchestrator** | Manages the pipeline flow | `main()` + `ask_agent()` | ✅ Functional |
| **Chunker** | Semantic / overlap-based | Naive `\n\n` split, no overlap | ❌ Loses context at boundaries |

### Data Flow Audit

```
Current Flow:
PDF files → pymupdf4llm (markdown) → split on "\n\n" → embed → np.array store → cosine_similarity → top-K → prompt → LLM → response

Problems identified:
  1. Chunker: No overlap → context lost at chunk boundaries
  2. No reranking → noisy top-K results hurt answer quality  
  3. No FAISS index → brute-force search doesn't scale past ~10K chunks
  4. Dense-only retrieval → misses exact keyword matches
```

---

## 2. Issues Found & Fixes

### Issue A — ❌ No Reranking Step (CRITICAL)

**Problem:** After dense retrieval, the top-K chunks go straight to the LLM. Without reranking, irrelevant chunks pollute the context window and degrade answer quality.

**Fix:** Add `nvidia/nv-rerankqa-mistral-4b-v3` reranking via the `/v1/ranking` endpoint. Retrieve top-20 candidates, rerank to top-5.

**New CLI flag:** `--rerank` (enabled by default, `--no-rerank` to skip)

---

### Issue B — ❌ Naive Chunking (No Overlap)

**Problem:** Splitting on `\n\n` with no overlap means information at paragraph boundaries is split across two chunks and neither chunk has full context. The 50-char minimum also discards meaningful short paragraphs (tables, lists).

**Fix:** Replace with sliding-window chunker: `--chunk-size 1200` and `--chunk-overlap 200` characters. Keeps context continuity.

---

### Issue C — ⚠️ Brute-Force Vector Search

**Problem:** `cosine_similarity([query], all_embeddings)` is O(n). Fine for <5K chunks but becomes slow at 50K+.

**Fix:** Add optional FAISS index for ANN (Approximate Nearest Neighbor) search. Falls back to numpy for small datasets.

---

### Issue D — ⚠️ Dense-Only Retrieval

**Problem:** Pure embedding search misses exact keyword matches (e.g., searching for "Section 4.2" or a specific product code).

**Fix:** Not adding full hybrid/BM25 now (would require a new dependency like `rank_bm25`), but the reranker compensates significantly since it uses cross-attention that catches keyword matches the embedding missed.

---

### Issue E — ⚠️ Bug 4 Still Pending: write_output .jsonl

**Problem:** `write_output()` falls into the `.txt` branch for `.jsonl` files (no `--append`), corrupting the format.

**Fix:** Add explicit `.jsonl` branch.

---

## 3. Proposed Changes

### [MODIFY] [rag_agent.py](file:///d:/LLM%20RAG%20MODEL/rag_agent.py)

1. **Add reranking** — new `RERANK_MODEL` constant, `rerank_chunks()` function, `/v1/ranking` API call via `requests`
2. **Add `--rerank` / `--no-rerank` CLI flag** and `--rerank-top-n`
3. **Improve chunking** — replace `\n\n` split with sliding-window overlap chunker, add `--chunk-size` and `--chunk-overlap` flags
4. **Fix write_output .jsonl** — add explicit `.jsonl` branch
5. **Wire reranking into `ask_agent()`** — retrieve top-20 → rerank → use top-K
6. **Add `requests` import** for the reranking API (non-OpenAI endpoint)
7. **Update banner** to show rerank status

---

## 4. Verification Plan

### Automated
- `python -c "import py_compile; py_compile.compile('rag_agent.py', doraise=True)"`
- `python rag_agent.py --help` to verify new CLI flags

### Manual  
- Run with a test PDF and `--rerank` to verify reranking works
- Compare answer quality with `--no-rerank` vs `--rerank`
- Test `--output file.jsonl` without `--append`
