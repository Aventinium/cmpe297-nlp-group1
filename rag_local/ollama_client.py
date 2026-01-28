# rag_local/ollama_client.py
from __future__ import annotations

import os
import requests
from typing import Optional, Dict, Any


# Read from env so Docker can override localhost properly.
# In Docker on Windows/Mac, use:
#   http://host.docker.internal:11434/api/generate
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Model can also be overridden from env for flexibility
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def generate(prompt: str, model: Optional[str] = None, timeout_s: int = 120) -> str:
    """
    Minimal Ollama generate client.
    Uses /api/generate (non-streaming) for simplicity.
    """
    payload: Dict[str, Any] = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    # Ollama returns {"response": "...", ...}
    return (data.get("response") or "").strip()
