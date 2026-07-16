# Novel Agent v2

Novel Agent v2 is a local-first novel writing assistant with a structured RAG memory layer. It is designed around a Control Agent that receives writing requirements, calls an OpenAI-compatible writer model, extracts structured story facts, stores them in local SQLite memory databases, and recalls relevant continuity context for later chapters or revisions.

This repository is the code-only public copy. It intentionally does not include private writing data, generated memory databases, model weights, logs, caches, or API keys.

## What This Project Contains

```text
.
├── control_agent.py              # Main interactive entry point
├── write_agent.py                # Writer-model adapter and draft generation logic
├── chapter_library.py            # Chapter draft/archive storage helpers
├── memory_library.py             # Memory browsing and maintenance service
├── material_library.py           # Story material/reference library service
├── material_extractor.py         # Qwen-based material extraction adapter
├── control_schemas.py            # Control Agent request/response schemas
├── agent_schema.py               # Shared agent message schema
├── progress_display.py           # Console progress display
├── memory_maintenance.py         # Memory audit/maintenance CLI
├── pipeline_smoke_test.py        # End-to-end smoke test
├── partition_index_smoke_test.py # Entity partition/index smoke test
├── rag/                          # Core RAG package
├── tests/                        # Unit tests
├── requirements.txt              # Standard Python dependencies
└── requirements-qwen35.txt       # Optional newest Transformers dependency for Qwen3.5
```

The `rag/` package contains the memory system: SQLite stores, retrieval, indexing, conflict records, audit scanning, patch application, snapshots, message protocol, and Qwen-based memory judgment.

## Requirements

- Python 3.10 or newer. Python 3.11/3.12 is recommended.
- An OpenAI-compatible chat/completions endpoint for the writer model.
- Optional but recommended: a local Qwen model for structured memory judgment.
- Optional GPU: CUDA is recommended for Qwen3.5-4B local inference. CPU can work for smaller models but will be slow.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want to run Qwen3.5-4B locally, install the extra Transformers dependency:

```powershell
pip install -r requirements-qwen35.txt
```

Install PyTorch according to your platform. For example, use the official PyTorch selector for the exact CUDA or CPU command:

```text
https://pytorch.org/get-started/locally/
```

For a CPU-only quick setup, this is usually enough:

```powershell
pip install torch
```

## Environment Variables

Create a local `.env` file in the project root. Do not commit it.

```env
LLM_API_KEY=your_writer_model_api_key
LLM_MODEL_ID=your_writer_model_name
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MAX_OUTPUT_TOKENS=8192

# Optional Qwen settings
QWEN_HF_MODEL_ID=Qwen/Qwen3.5-4B
QWEN_LOCAL_MODEL_PATH=models/Qwen3.5-4B
QWEN_HF_CACHE=models/.hf-cache
QWEN_DEVICE=auto

# Optional, only needed for private/gated Hugging Face models
HF_TOKEN=
```

`LLM_BASE_URL` must point to an OpenAI-compatible endpoint. The code accepts either a base URL ending in `/v1` or a full `/v1/chat/completions` endpoint.

## Model Files

The public repository should not include model weights. Use one of these options locally.

### Option A: Let the Code Download Qwen

Leave `QWEN_LOCAL_MODEL_PATH` empty or point it at a future local folder. When the Qwen client starts, it will try to find the model locally first, then download from Hugging Face using `QWEN_HF_MODEL_ID`.

Example:

```env
QWEN_HF_MODEL_ID=Qwen/Qwen3.5-4B
QWEN_HF_CACHE=models/.hf-cache
QWEN_DEVICE=auto
```

### Option B: Download Manually

Download the model from Hugging Face and place it under:

```text
models/Qwen3.5-4B/
```

The folder should contain files such as:

```text
models/Qwen3.5-4B/config.json
models/Qwen3.5-4B/tokenizer.json
models/Qwen3.5-4B/model.safetensors.index.json
models/Qwen3.5-4B/model-*.safetensors
```

Then set:

```env
QWEN_LOCAL_MODEL_PATH=models/Qwen3.5-4B
```

You can also use a smaller model for local testing:

```env
QWEN_HF_MODEL_ID=Qwen/Qwen3-0.6B
QWEN_LOCAL_MODEL_PATH=models/Qwen3-0.6B
```

## Data Directory

Runtime writing data is stored in `rag_data/` by default. This directory is generated locally and should not be committed.

Expected layout after running the system:

```text
rag_data/
└── <book_id>/
    ├── documents/
    ├── canon_memory/
    ├── chapter_memory/
    ├── relation_hook_memory/
    ├── state_timeline_memory/
    ├── index/
    ├── conflicts/
    └── materials/
```

Each memory store uses SQLite files. If you want a separate local data location, pass `--data-dir`:

```powershell
python control_agent.py --data-dir rag_data
```

## Running the Project

Start the interactive Control Agent:

```powershell
python control_agent.py
```

Use a custom data folder:

```powershell
python control_agent.py --data-dir rag_data
```

Run the memory maintenance tool directly. This script is mainly an internal maintenance/testing entry point; normal interactive use should ask `control_agent.py` to organize the memory library from inside the conversation.

```powershell
python memory_maintenance.py --allow-direct-test --data-dir rag_data --book-id your_book_id
```

Run the end-to-end smoke test:

```powershell
python pipeline_smoke_test.py --data-dir rag_smoke_data
```

Run the partition/index smoke test:

```powershell
python partition_index_smoke_test.py
```

Run unit tests:

```powershell
python -m unittest discover -s tests
```

## Typical Local Setup

After cloning the repository, a practical local workspace usually looks like this:

```text
Novel_Agentv2/
├── .env
├── models/
│   └── Qwen3.5-4B/
├── rag_data/
│   └── your_book_id/
├── rag/
├── tests/
├── control_agent.py
├── write_agent.py
└── requirements.txt
```

## Notes

- The writer model is accessed through `langchain-openai` and an OpenAI-compatible API.
- The memory judge can use local Qwen inference through Hugging Face Transformers.
- The system stores long-term continuity information locally in SQLite, so it can be used without uploading private story data to a separate database service.
