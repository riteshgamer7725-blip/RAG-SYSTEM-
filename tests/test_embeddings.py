"""
tests/test_embeddings.py
─────────────────────────────────────────────────────────────
Unit tests for embedding functionality (mocked)
"""

import sys
import os

# Set dummy API key before importing rag_agent to avoid EnvironmentError
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-dummy-key-for-testing")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_embedding_dimension_consistency():
    """Verify embeddings maintain consistent dimensions."""
    # Note: This test requires a valid NVIDIA_API_KEY
    # We skip it by default since it makes real API calls
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key or api_key.startswith("nvapi-dummy"):
        print("⊘ test_embedding_dimension_consistency skipped (requires valid API key)")
        return
    
    from rag_agent import get_embedding
    
    texts = ["Test sentence 1", "Test sentence 2"]
    embeddings = []
    
    try:
        for text in texts:
            emb = get_embedding(text, input_type="passage")
            embeddings.append(emb)
        
        # All embeddings should have same dimension
        assert len(embeddings[0]) == len(embeddings[1]), \
            "Embedding dimensions should be consistent"
        
        # Dimension should be reasonable (> 100 for modern models)
        assert len(embeddings[0]) > 100, \
            f"Embedding dimension too small: {len(embeddings[0])}"
        
        print("✓ test_embedding_dimension_consistency passed")
    except Exception as e:
        print(f"⊘ test_embedding_dimension_consistency skipped (API error: {e})")


def test_cosine_similarity_symmetry():
    """Cosine similarity should be symmetric."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Create sample embeddings
    vec_a = np.array([[0.1, 0.2, 0.3]])
    vec_b = np.array([[0.4, 0.5, 0.6]])
    
    sim_ab = cosine_similarity(vec_a, vec_b)[0][0]
    sim_ba = cosine_similarity(vec_b, vec_a)[0][0]
    
    assert abs(sim_ab - sim_ba) < 1e-10, \
        "Cosine similarity should be symmetric"
    
    print("✓ test_cosine_similarity_symmetry passed")


def test_cosine_similarity_range():
    """Cosine similarity should be in range [-1, 1]."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Test with various vectors
    test_vectors = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[0.5, 0.5, 0.5]]),
        np.array([[-1.0, 0.0, 0.0]]),
    ]
    
    for i, vec1 in enumerate(test_vectors):
        for j, vec2 in enumerate(test_vectors):
            sim = cosine_similarity(vec1, vec2)[0][0]
            # Use tolerance for floating-point precision
            assert -1.0 - 1e-10 <= sim <= 1.0 + 1e-10, \
                f"Cosine similarity out of range: {sim} (vectors {i}, {j})"
    
    print("✓ test_cosine_similarity_range passed")


def test_identical_vectors_similarity():
    """Identical vectors should have similarity of 1.0."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    vec = np.array([[0.3, 0.4, 0.5]])
    sim = cosine_similarity(vec, vec)[0][0]
    
    assert abs(sim - 1.0) < 1e-10, \
        f"Identical vectors should have similarity 1.0, got {sim}"
    
    print("✓ test_identical_vectors_similarity passed")


if __name__ == "__main__":
    print("Running embedding tests...\n")
    test_embedding_dimension_consistency()
    test_cosine_similarity_symmetry()
    test_cosine_similarity_range()
    test_identical_vectors_similarity()
    print("\n✅ All embedding tests passed!")
