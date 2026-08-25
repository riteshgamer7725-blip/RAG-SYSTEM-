# Sarvam AI Integration — Task Tracker

- [x] Create `sarvam_rag.py` with all bugs fixed
- [x] Create `.env.example` template
- [x] Validate syntax & imports
- [x] Write analysis artifact
- [x] Integrate into `rag_agent.py`:
  - [x] Add `--llm` CLI flag (`nvidia` | `sarvam` | `auto`)
  - [x] Add `--sarvam-model` CLI flag (`sarvam-30b` | `sarvam-105b`)
  - [x] Add auto language detection for routing (Unicode script detection)
  - [x] Add fallback logic (Sarvam fails → retry with NVIDIA)
  - [x] Refactor `ask_agent()` into `_retrieve_chunks()` + `_generate_nvidia()` + `_generate_sarvam()`
  - [x] Update `print_banner()` to show LLM backend
  - [x] Wire `llm_backend` + `sarvam_model` through one-shot and interactive modes
- [x] Final validation:
  - [x] Syntax check — both files pass `py_compile`
  - [x] Import chain — no circular dependency issues
  - [x] Language detection — Hindi ✓, Bengali ✓, Mixed ✓, English ✓
  - [x] `ask_agent()` signature — 8 params including new `llm_backend` and `sarvam_model`
- [x] Write walkthrough
