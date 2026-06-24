"""
tests/test_config.py
Smoke tests for config loading.
"""
import os
import pytest


def test_config_loads_with_env(monkeypatch):
    """Config should load cleanly when env vars are set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("MAX_STEPS", "8")

    import importlib
    import agent.config as cfg
    importlib.reload(cfg)

    assert cfg.OPENAI_API_KEY == "sk-test-fake-key"
    assert cfg.OPENAI_MODEL == "gpt-4o-mini"
    assert cfg.MAX_STEPS == 8



def test_ddgs_importable():
    """ddgs must be importable — no API key needed."""
    from ddgs import DDGS  # noqa: F401


def test_defaults():
    """Default values should match config.py defaults."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")

    import importlib
    import agent.config as cfg
    importlib.reload(cfg)

    # MAX_STEPS default is 6 (set in Session 5)
    assert cfg.MAX_STEPS == int(os.getenv("MAX_STEPS", "6"))
    assert cfg.MAX_SEARCH_RESULTS == int(os.getenv("MAX_SEARCH_RESULTS", "5"))