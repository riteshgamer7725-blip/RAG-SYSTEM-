# ─── Docker Support for NVIDIA RAG Agent ─────────────────────
FROM python:3.14-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt

# Copy application code
COPY rag_agent.py sarvam_rag.py health_check.py ./
COPY my_docs/ ./my_docs/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose Streamlit default port
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "rag_agent.py", "--query", "What is this document about?"]
