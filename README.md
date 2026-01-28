CMPE297 Group 1 — Local CLI Chatbot (Ollama)

This repository implements a minimal, fully local command-line chatbot
using Ollama as the LLM backend.

This project satisfies the CMPE297 requirement of a working chatbot
(no specialization or RAG required for the MVP).
RAG-related modules exist in the codebase but are not required to run the chatbot.


WHAT THIS PROJECT DOES
- Runs a local chatbot in the terminal
- Uses Ollama for inference (no cloud APIs)
- Supports multi-turn conversation
- Packaged as a Python wheel for clean installation


PREREQUISITES (REQUIRED BEFORE RUNNING)

1) Ollama must be installed and running
Install Ollama using the official installer for your OS.

Verify Ollama is running:
    curl http://localhost:11434

If this fails, start Ollama and try again.


2) Pull the model used by this project
This project defaults to llama3.1:8b.

Pull it once:
    ollama pull llama3.1:8b

Verify:
    ollama list


3) Python 3.10 or newer
Verify Python version:
    python --version


QUICK START (RECOMMENDED PATH: WHEEL INSTALL)

These steps assume a completely clean Python environment.


1) Clone the repository
    git clone <REPO_URL>
    cd cmpe297-nlp-group1


2) Create and activate a virtual environment

Windows (PowerShell):
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

macOS / Linux:
    python -m venv .venv
    source .venv/bin/activate

Upgrade pip:
    python -m pip install --upgrade pip


3) Install build tools and build the wheel
    python -m pip install --upgrade build
    python -m build

This creates a wheel file in the dist/ directory.


4) Install the wheel
    python -m pip install dist/*.whl


5) Run the chatbot
After installation, a CLI command is available:

    cmpe297-chat

Type messages to chat.
Exit with exit, quit, or Ctrl+C.


ALTERNATE RUN (DEVELOPMENT MODE, NO WHEEL)

If you prefer to run directly from source:

    cd rag_local
    python chat.py


TROUBLESHOOTING

Ollama connection error
- Ensure Ollama is running
- Confirm it is listening on http://localhost:11434
- Try:
    ollama list


Model not found
Pull the model:
    ollama pull llama3.1:8b


cmpe297-chat command not found
- Ensure the virtual environment is active
- Verify installation:
    pip show cmpe297-chatbot

Reinstall the wheel if needed:
    python -m pip install --force-reinstall dist/*.whl


PROJECT STRUCTURE (MVP RELEVANT)

rag_local/
    chat.py            - CLI chatbot entrypoint
    ollama_client.py   - Minimal Ollama API wrapper
    __init__.py

Additional modules exist for future RAG functionality but are not required
for the chatbot MVP.


NOTES
- No cloud services are used
- No API keys are required
- All inference runs locally via Ollama
