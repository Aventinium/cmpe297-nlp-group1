1) What this project is
-----------------------
This repo provides:
- An MVP command-line chatbot (baseline requirement: “working chatbot”)
- A planned path to a Retrieval-Augmented Generation (RAG) system + evaluation (later sprints)

You run the chatbot locally from your terminal.

2) Quick Start (fastest path)
-----------------------------
A) Clone:
   git clone git@github.com:JCarter19999/cmpe297-nlp-group1.git
   cd cmpe297-nlp-group1

B) Create + activate an environment (Conda recommended on Windows):
   conda create -n cmpe297-chatbot python=3.10 -y
   conda activate cmpe297-chatbot

C) Install:
   python -m pip install --upgrade pip
   pip install -e .

D) Install Ollama + pull a model (one-time):
   - Install Ollama (must be running)
   - Pull the model:
       ollama pull llama3.1:8b

E) Run:
   cmpe297-chat

3) Step-by-step (detailed)
--------------------------

3.1 Clone the repository
------------------------
1) Open a terminal (PowerShell on Windows).
2) Go to a folder where you keep projects:
   Example (Windows):
     cd C:\\Users\\YOURNAME\\Desktop
   Example (Mac/Linux):
     cd ~/Desktop

3) Clone the repo:
   git clone git@github.com:JCarter19999/cmpe297-nlp-group1.git

4) Enter the repo:
   cd cmpe297-nlp-group1

3.2 Create a Python environment
-------------------------------

Option A: Conda (recommended on Windows)
1) Create:
   conda create -n cmpe297-chatbot python=3.10 -y
2) Activate:
   conda activate cmpe297-chatbot

Option B: venv (works everywhere)
1) Create:
   python -m venv .venv
2) Activate:

   Windows (PowerShell):
     .\\.venv\\Scripts\\Activate.ps1

   Windows (cmd):
     .\\.venv\\Scripts\\activate.bat

   Mac/Linux:
     source .venv/bin/activate

3.3 Install the project
-----------------------
From the repo root (the folder that contains rag_local/):

1) Upgrade pip:
   python -m pip install --upgrade pip

2) Install the project in editable mode:
   pip install -e .

Editable mode means you can edit code and re-run without reinstalling.

3.4 Install and run Ollama (one-time)
-------------------------------------
This chatbot uses Ollama as the local model runner.

1) Install Ollama (and ensure it is running).
2) Pull the model:
   ollama pull llama3.1:8b

If your team chooses a different model later, update the README and/or config.

3.5 Run the chatbot
-------------------
From the repo root:

Option A (recommended):
  cmpe297-chat

Option B (debug / fallback):
  python -m rag_local.chat

How to use it:
- Type a message and press Enter
- To quit: type 'exit' or press Ctrl+C

4) Installing RAG + Evaluation dependencies (later sprints)
-----------------------------------------------------------
For the baseline chatbot, you only need the default install (pip install -e .).

When you start RAG work:
  pip install -e ".[rag]"

When you start evaluation work:
  pip install -e ".[eval]"

If you want everything:
  pip install -e ".[rag,eval,dev]"

5) Common issues (quick fixes)
------------------------------

Issue: “cmpe297-chat is not recognized”
- Make sure your environment is activated
- Make sure you ran:
    pip install -e .
- If still stuck, run:
    python -m rag_local.chat

Issue: “Ollama connection error” / “Ollama not running”
- Start Ollama
- Confirm the service is up:
    ollama list
- Then re-run:
    cmpe297-chat

Issue: “ModuleNotFoundError …”
- Make sure you are running from the repo root (same folder as rag_local/)
- Make sure the file is committed and pushed if teammates are cloning it:
    git status
    git add <missing_file>
    git commit -m "Add missing file"
    git push
- Reinstall editable:
    pip install -e .

Issue: “ImportError: cannot import name …”
- Usually means the function name in an import does not match the actual function in the file.
- Open the referenced file and confirm the function exists.