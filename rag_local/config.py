"""
rag_local.config

Goal
----
Centralize configuration for:
- model selection
- system prompt
- RAG settings (top_k, chunk_size, overlap, etc.)

Design principles
-----------------
- Single source of truth: callers import get_config()
- Defaults work out of the box (important for onboarding + CI)
- Optional overrides via a local JSON file (simple, no extra deps)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.local.json"


@dataclass(frozen=True)
class AppConfig:
    # LLM settings
    model: str = "llama3.1:8b"   # change to whatever your team standardizes on
    system_prompt: str = (
        "You are an NLP tutor. Explain concepts clearly, step-by-step, "
        "and ask brief clarifying questions when needed."
    )

    # Chat settings
    max_history_turns: int = 12

    # RAG settings (used later)
    rag_enabled: bool = False
    top_k: int = 5
    chunk_size: int = 800
    overlap: int = 200


def get_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Load config with this precedence:
    1) JSON file (config_path or DEFAULT_CONFIG_PATH) if it exists
    2) Environment variables (optional overrides)
    3) Dataclass defaults
    """
    base = AppConfig()

    cfg_file = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    overrides: Dict[str, Any] = {}
    if cfg_file.exists():
        overrides.update(json.loads(cfg_file.read_text(encoding="utf-8")))

    # Optional env overrides (useful in CI)
    overrides.setdefault("model", os.getenv("RAG_MODEL", base.model))
    overrides.setdefault("rag_enabled", _env_bool("RAG_ENABLED", base.rag_enabled))

    # Build final config
    return AppConfig(
        model=str(overrides.get("model", base.model)),
        system_prompt=str(overrides.get("system_prompt", base.system_prompt)),
        max_history_turns=int(overrides.get("max_history_turns", base.max_history_turns)),
        rag_enabled=bool(overrides.get("rag_enabled", base.rag_enabled)),
        top_k=int(overrides.get("top_k", base.top_k)),
        chunk_size=int(overrides.get("chunk_size", base.chunk_size)),
        overlap=int(overrides.get("overlap", base.overlap)),
    )


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}
