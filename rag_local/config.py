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
    # Ollama host
    ollama_host: str = "http://localhost:11434"

    # LLM chat model
    model: str = "llama3.1:8b"
    system_prompt: str = (
        "You are an NLP tutor. Explain concepts clearly, step-by-step, "
        "and ask brief clarifying questions when needed."
    )

    # Chat
    max_history_turns: int = 12

    # RAG
    rag_enabled: bool = False
    top_k: int = 5
    chunk_size: int = 800
    overlap: int = 200
    max_context_chars: int = 6000

    # Data + index paths
    data_dir: str = "rag_local/Data"
    index_path: str = "rag_local/Data/.index/local_index.json"

    # Embeddings
    embed_backend: str = "ollama"          # "ollama" or "hash"
    embed_model: str = "nomic-embed-text"  # used for Ollama embeddings


def get_config(config_path: Optional[str | Path] = None) -> AppConfig:
    base = AppConfig()

    cfg_file = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    overrides: Dict[str, Any] = {}
    if cfg_file.exists():
        overrides.update(json.loads(cfg_file.read_text(encoding="utf-8")))

    # Env overrides (CI-friendly)
    overrides.setdefault("ollama_host", os.getenv("OLLAMA_HOST", base.ollama_host))
    overrides.setdefault("model", os.getenv("RAG_MODEL", base.model))
    overrides.setdefault("rag_enabled", _env_bool("RAG_ENABLED", base.rag_enabled))
    overrides.setdefault("embed_backend", os.getenv("EMBED_BACKEND", base.embed_backend))
    overrides.setdefault("embed_model", os.getenv("EMBED_MODEL", base.embed_model))

    return AppConfig(
        ollama_host=str(overrides.get("ollama_host", base.ollama_host)),
        model=str(overrides.get("model", base.model)),
        system_prompt=str(overrides.get("system_prompt", base.system_prompt)),
        max_history_turns=int(overrides.get("max_history_turns", base.max_history_turns)),
        rag_enabled=bool(overrides.get("rag_enabled", base.rag_enabled)),
        top_k=int(overrides.get("top_k", base.top_k)),
        chunk_size=int(overrides.get("chunk_size", base.chunk_size)),
        overlap=int(overrides.get("overlap", base.overlap)),
        max_context_chars=int(overrides.get("max_context_chars", base.max_context_chars)),
        data_dir=str(overrides.get("data_dir", base.data_dir)),
        index_path=str(overrides.get("index_path", base.index_path)),
        embed_backend=str(overrides.get("embed_backend", base.embed_backend)),
        embed_model=str(overrides.get("embed_model", base.embed_model)),
    )


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}