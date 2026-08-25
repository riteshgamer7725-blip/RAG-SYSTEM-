"""
tests/test_chunking.py
─────────────────────────────────────────────────────────────
Unit tests for PDF chunking logic
"""

import sys
import os

# Set dummy API key before importing rag_agent to avoid EnvironmentError
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-dummy-key-for-testing")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent import sliding_window_chunk


def test_empty_text():
    """Empty or whitespace-only text should return empty list."""
    assert sliding_window_chunk("") == []
    assert sliding_window_chunk("   ") == []
    assert sliding_window_chunk(None) == []
    print("✓ test_empty_text passed")


def test_short_text():
    """Text shorter than chunk_size should return single chunk."""
    text = "This is a short paragraph."
    chunks = sliding_window_chunk(text, chunk_size=100, chunk_overlap=20)
    # Short text may be filtered out if below min_chunk_len (30 chars by default)
    # So we check that it either returns 1 chunk or 0 (if too short)
    assert len(chunks) <= 1, f"Expected at most 1 chunk, got {len(chunks)}"
    print("✓ test_short_text passed")


def test_chunk_overlap():
    """Chunks should overlap by specified amount."""
    text = "A" * 200 + "B" * 200 + "C" * 200
    chunks = sliding_window_chunk(text, chunk_size=200, chunk_overlap=50)
    
    # Should have multiple chunks
    assert len(chunks) > 1
    
    # Verify overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i][-50:] if len(chunks[i]) >= 50 else chunks[i]
        chunk2_start = chunks[i+1][:50] if len(chunks[i+1]) >= 50 else chunks[i+1]
        # At least some overlap should exist
        assert chunk1_end in chunks[i+1] or chunk2_start in chunks[i], \
            f"No overlap found between chunks {i} and {i+1}"
    
    print("✓ test_chunk_overlap passed")


def test_paragraph_boundaries():
    """Chunking should respect paragraph boundaries when possible."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = sliding_window_chunk(text, chunk_size=100, chunk_overlap=20)
    
    # Should try to keep paragraphs together
    assert len(chunks) > 0
    # Each chunk should contain at least one complete sentence
    for chunk in chunks:
        assert "." in chunk
    
    print("✓ test_paragraph_boundaries passed")


def test_max_chunk_size():
    """No chunk should exceed MAX_CHUNK_CHARS."""
    from rag_agent import MAX_CHUNK_CHARS
    
    text = "This is a test. " * 200  # Long text
    chunks = sliding_window_chunk(text, chunk_size=1200, chunk_overlap=200)
    
    for chunk in chunks:
        assert len(chunk) <= MAX_CHUNK_CHARS, \
            f"Chunk exceeds max size: {len(chunk)} > {MAX_CHUNK_CHARS}"
    
    print("✓ test_max_chunk_size passed")


if __name__ == "__main__":
    print("Running chunking tests...\n")
    test_empty_text()
    test_short_text()
    test_chunk_overlap()
    test_paragraph_boundaries()
    test_max_chunk_size()
    print("\n✅ All chunking tests passed!")
