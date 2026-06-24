"""
tests/conftest.py — shared pytest fixtures
"""
import os
import pytest

# Ensure fake keys are set before any import
os.environ.setdefault("OPENAI_API_KEY",          "test-fake-key")
os.environ.setdefault("ALLOWED_API_KEYS",         "test-key-good")
os.environ.setdefault("RATE_LIMIT_REQUESTS",      "5")
os.environ.setdefault("RATE_LIMIT_WINDOW_SECONDS","3600")