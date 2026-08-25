# NVIDIA RAG Agent - Health Check Script
"""
health_check.py
─────────────────────────────────────────────────────────────
Quick health check for the RAG agent components

Usage:
    python health_check.py [--api-check] [--env-check] [--all]

This script validates:
  - Environment variables (NVIDIA_API_KEY, SARVAM_API_KEY)
  - API connectivity (optional, requires valid keys)
  - Module imports
  - PDF directory accessibility
"""

import argparse
import os
import sys
from pathlib import Path


def check_env_vars() -> bool:
    """Check if required environment variables are set."""
    print("Checking environment variables...")
    
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    sarvam_key = os.environ.get("SARVAM_API_KEY")
    
    # Load from .env if available
    if not nvidia_key or not sarvam_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            nvidia_key = os.environ.get("NVIDIA_API_KEY")
            sarvam_key = os.environ.get("SARVAM_API_KEY")
        except ImportError:
            pass
    
    nvidia_ok = bool(nvidia_key and nvidia_key.startswith("nvapi-"))
    sarvam_ok = bool(sarvam_key and sarvam_key.strip())
    
    print(f"  NVIDIA_API_KEY: {'✓ Set' if nvidia_ok else '✗ Missing/Invalid'}")
    print(f"  SARVAM_API_KEY: {'✓ Set' if sarvam_ok else '✗ Missing'}")
    
    return nvidia_ok or sarvam_ok


def check_imports() -> bool:
    """Check if all required modules can be imported."""
    print("\nChecking module imports...")
    
    required_modules = [
        "numpy",
        "requests",
        "sklearn",
        "tqdm",
        "structlog",
        "openai",
        "dotenv",
    ]
    
    all_ok = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"  {module}: ✓")
        except ImportError as e:
            print(f"  {module}: ✗ ({e})")
            all_ok = False
    
    # Check optional modules
    optional_modules = ["PyMuPDF", "pymupdf4llm", "streamlit"]
    for module in optional_modules:
        try:
            __import__(module.replace("-", "_"))
            print(f"  {module}: ✓ (optional)")
        except ImportError:
            print(f"  {module}: ⊘ (not installed, optional)")
    
    return all_ok


def check_pdf_dir(path: str = "./my_docs") -> bool:
    """Check if PDF directory exists and is accessible."""
    print(f"\nChecking PDF directory: {path}")
    
    pdf_path = Path(path).resolve()
    
    if not pdf_path.exists():
        print(f"  ✗ Directory does not exist: {pdf_path}")
        print(f"  Hint: Create it with: mkdir -p {pdf_path}")
        return False
    
    if not pdf_path.is_dir():
        print(f"  ✗ Path exists but is not a directory: {pdf_path}")
        return False
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"  ✓ Directory exists: {pdf_path}")
    print(f"  Found {len(pdf_files)} PDF file(s)")
    
    if not pdf_files:
        print(f"  ⊘ Warning: No PDF files found in {pdf_path}")
    
    return True


def check_nvidia_api() -> bool:
    """Test NVIDIA API connectivity."""
    print("\nTesting NVIDIA API connectivity...")
    
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("  ⊘ Skipping (NVIDIA_API_KEY not set)")
        return True
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=10.0,
        )
        
        # Try a minimal embedding request
        response = client.embeddings.create(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            input=["test"],
        )
        print(f"  ✓ NVIDIA API connected successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ NVIDIA API test failed: {e}")
        return False


def check_sarvam_api() -> bool:
    """Test Sarvam API connectivity."""
    print("\nTesting Sarvam API connectivity...")
    
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print("  ⊘ Skipping (SARVAM_API_KEY not set)")
        return True
    
    try:
        from sarvam_rag import SarvamRAG
        rag = SarvamRAG(model="sarvam-30b")
        
        if rag.health_check():
            print(f"  ✓ Sarvam API connected successfully")
            return True
        else:
            print(f"  ✗ Sarvam API health check failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Sarvam API test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Health check for NVIDIA RAG Agent"
    )
    parser.add_argument(
        "--api-check",
        action="store_true",
        help="Run API connectivity tests (requires valid API keys)"
    )
    parser.add_argument(
        "--env-check",
        action="store_true",
        help="Only check environment variables"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./my_docs",
        help="Path to PDF directory"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks including API tests"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NVIDIA RAG Agent - Health Check")
    print("=" * 60)
    
    all_passed = True
    
    # Always check imports
    if not check_imports():
        all_passed = False
    
    # Always check env vars
    if not check_env_vars():
        all_passed = False
    
    # Always check PDF directory
    if not check_pdf_dir(args.pdf_dir):
        all_passed = False
    
    # API checks (only if requested)
    run_api_checks = args.api_check or args.all
    if run_api_checks:
        if not check_nvidia_api():
            all_passed = False
        if not check_sarvam_api():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed or were skipped")
        print("   Review the output above for details")
        return 0  # Return 0 even on failures (warnings only)


if __name__ == "__main__":
    sys.exit(main())
