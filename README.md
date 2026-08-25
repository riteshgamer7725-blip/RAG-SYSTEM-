# LLM RAG Model

A robust, production-ready Retrieval-Augmented Generation (RAG) agent that allows you to chat with your PDF documents using NVIDIA AI models and Sarvam AI (for multilingual support).

## Features

- **NVIDIA AI Integration**: Uses `llama-3.2-nv-embedqa-1b-v2` for embeddings and `llama-3.3-70b-instruct` for generation.
- **Vision Support**: Extracts and describes diagrams/images from PDFs using `llama-3.2-90b-vision-instruct`.
- **Reranking**: Improves accuracy using NVIDIA's `nv-rerankqa-mistral-4b-v3`.
- **Multilingual Support**: Auto-detects Indian languages and routes queries to Sarvam AI (`sarvam-30b`/`sarvam-105b`).
- **Resilient Pipeline**: Checkpointing, atomic file saves, exponential-backoff retries, and local caching via MD5 fingerprinting.
- **Full CLI Interface**: Run one-shot queries, save outputs to `.json`/`.txt`/`.jsonl`, or use interactive chat mode.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "LLM RAG MODEL"
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your `NVIDIA_API_KEY` and optionally `SARVAM_API_KEY`.

## Usage

Place your `.pdf` files into the `my_docs/` folder.

**Interactive Chat Mode:**
```bash
python rag_agent.py
```

**One-shot Query:**
```bash
python rag_agent.py --query "What is the revenue in 2023?" --top-k 5
```

**Fast Mode (Skip reranking and image descriptions):**
```bash
python rag_agent.py --query "Summarize the document" --no-rerank --no-images
```

**Save Results to Log:**
```bash
python rag_agent.py --query "What is the key takeaway?" --output results.jsonl --append
```

## Cache Wiping
If you want to forcefully rebuild the PDF index from scratch, use the `--force-reindex` flag.
