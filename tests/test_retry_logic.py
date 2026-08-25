"""
tests/test_retry_logic.py
─────────────────────────────────────────────────────────────
Unit tests for retry logic with exponential backoff
"""

import sys
import os
import time

# Set dummy API key before importing rag_agent to avoid EnvironmentError
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-dummy-key-for-testing")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent import with_retry


def test_successful_call_no_retry():
    """Successful calls should not trigger retries."""
    call_count = 0
    
    def success_fn():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = with_retry(success_fn, max_retries=4)
    assert result == "success"
    assert call_count == 1, f"Expected 1 call, got {call_count}"
    print("✓ test_successful_call_no_retry passed")


def test_retry_on_exception():
    """Should retry on transient errors."""
    call_count = 0
    
    def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TimeoutError("Temporary timeout")  # This is retryable
        return "success"
    
    result = with_retry(fail_then_succeed, max_retries=4, base_delay=0.1)
    assert result == "success"
    assert call_count == 3, f"Expected 3 calls, got {call_count}"
    print("✓ test_retry_on_exception passed")


def test_max_retries_exceeded():
    """Should raise after max_retries attempts."""
    call_count = 0
    
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise TimeoutError("Persistent timeout")  # This is retryable
    
    try:
        with_retry(always_fail, max_retries=3, base_delay=0.1)
        assert False, "Should have raised an exception"
    except TimeoutError:
        assert call_count == 3, f"Expected 3 calls, got {call_count}"
        print("✓ test_max_retries_exceeded passed")


def test_non_retryable_error():
    """Non-retryable errors should not trigger retries."""
    call_count = 0
    
    def non_retryable_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid input")  # Not retryable
    
    try:
        with_retry(non_retryable_error, max_retries=4, base_delay=0.1)
        assert False, "Should have raised an exception"
    except ValueError:
        assert call_count == 1, f"Expected 1 call, got {call_count}"
        print("✓ test_non_retryable_error passed")


def test_exponential_backoff_timing():
    """Verify delays follow exponential backoff pattern."""
    call_times = []
    
    def fail_three_times():
        call_times.append(time.time())
        if len(call_times) < 3:
            raise TimeoutError("Timed out")
        return "success"
    
    start = time.time()
    result = with_retry(fail_three_times, max_retries=4, base_delay=0.2)
    
    assert result == "success"
    assert len(call_times) == 3
    
    # Check delays between calls (with tolerance)
    if len(call_times) >= 2:
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # Second delay should be roughly 2x first delay (exponential)
        assert delay2 > delay1 * 1.5, "Delay should increase exponentially"
    
    print("✓ test_exponential_backoff_timing passed")


if __name__ == "__main__":
    print("Running retry logic tests...\n")
    test_successful_call_no_retry()
    test_retry_on_exception()
    test_max_retries_exceeded()
    test_non_retryable_error()
    test_exponential_backoff_timing()
    print("\n✅ All retry logic tests passed!")
