# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in this project (e.g., API key exposure,
unsafe file handling, or a dependency vulnerability), please report it privately
rather than opening a public issue.

- **Preferred:** Open a [GitHub Security Advisory](../../security/advisories/new)
  for this repository.
- **Alternative:** Email the maintainer directly (add your contact email here).

Please include:
- A description of the issue and its potential impact
- Steps to reproduce
- Any relevant logs or code snippets (redact API keys and personal data)

We aim to acknowledge reports within 5 business days.

## Scope Notes

- This project sends document text chunks to third-party APIs (NVIDIA NIM,
  Sarvam AI) for embedding, vision, and generation. Report any issue related to
  how data is transmitted or cached locally (`.rag_cache/`) as a security issue.
- API keys are read from environment variables / `.env` and are never intended
  to be logged or committed. If you find a code path that logs a key, please
  report it.
