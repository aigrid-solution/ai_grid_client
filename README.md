# AI Grid Client

Python client examples for the **AI Grid** API (`app.ai-grid.io`). Uses the OpenAI-compatible interface for chat, embeddings, vision/OCR, tool calling, and realtime audio.

---

## Overview

This repo is a **categorized collection of scripts** that demonstrate:

- **Chat** — single and streaming completions (Qwen, OSS).
- **RAG** — minimal retrieval-augmented generation, agentic RAG, and graph RAG pipelines.
- **LangGraph** — stateful multi-node workflows with conditional edges.
- **OCR** — single image, batch, and long-document (PDF) OCR.
- **Tools** — ReAct-style tool-calling agents (LlamaIndex + OpenAILike).
- **Embedding** — text embeddings (e.g. GTE-Qwen2).
- **Audio** — Voxtral realtime transcription over WebSocket.

All scripts read configuration from a root `.env` file (API key, base URL, model names). Paths and endpoints are not printed in logs.

---

## Project layout (by category)

Scripts are grouped by **class/category** in subfolders:

```
client/
├── .env                    # API key, BASE_URL, model names (do not commit)
├── requirements.txt
├── README.md
├── images/                 # Optional: sample images for OCR
│
├── chat/                   # Chat completions
│   ├── qwen.py             # Qwen model, single reply
│   ├── oss.py              # OSS model, single reply
│   └── stream_chat.py      # Streaming tokens (OSS or STREAM_CHAT_MODEL)
│
├── langgraph/              # LangGraph workflows
│   └── langraph_oss.py     # Multi-node graph: generate → optional refine
│
├── rag/                    # Retrieval-augmented generation
│   ├── rag_lite.py         # Minimal: embed → retrieve top-k → generate
│   ├── agentic_rag.py      # Agent: retrieve ↔ generate loop (LangGraph)
│   └── graph_rag.py        # Pipeline: rewrite_query → retrieve → generate
│
├── ocr/                    # Vision / OCR
│   ├── ocr.py              # Single-image OCR (IMAGE_FOLDER or IMAGE_PATH)
│   ├── ocr_batch.py        # Concurrent batch OCR
│   └── ocr_doc.py          # Long PDF: pages → windows → sequential OCR
│
├── tools/                  # Tool calling / agents
│   └── qwen_tool_caller.py # ReActAgent + FunctionTool (LlamaIndex, OpenAILike)
│
├── embedding/              # Embeddings
│   └── embeding_qwen_alibaba.py  # GTE-Qwen2 (or EMBEDDING_MODEL from .env)
│
└── audio/                  # Realtime audio
    └── voxtral.py          # Voxtral WebSocket transcription (AUDIO_PATH)
```

---

## Requirements

- Python 3.10+
- [openai](https://pypi.org/project/openai/) — OpenAI Python client
- [python-dotenv](https://pypi.org/project/python-dotenv/) — load `.env`
- [langgraph](https://pypi.org/project/langgraph/), [langchain-core](https://pypi.org/project/langchain-core/), [langchain-openai](https://pypi.org/project/langchain-openai/) — for `langgraph/` and `rag/` (agentic, graph)
- [llama-index](https://pypi.org/project/llama-index/) + [llama-index-llms-openai-like](https://pypi.org/project/llama-index-llms-openai-like/) — for `tools/qwen_tool_caller.py`
- [pdf2image](https://pypi.org/project/pdf2image/), [Pillow](https://pypi.org/project/Pillow/) — for `ocr/ocr_doc.py` (long-doc OCR)
- [librosa](https://pypi.org/project/librosa/), [websockets](https://pypi.org/project/websockets/) — for `audio/voxtral.py`

```bash
pip install -r requirements.txt
# or selectively:
pip install openai python-dotenv
pip install langgraph langchain-core langchain-openai   # langgraph + rag
pip install llama-index llama-index-llms-openai-like   # tools
pip install pdf2image Pillow   # ocr_doc
pip install librosa websockets   # voxtral
```

---

## Setup

1. **Use this repo** (clone or copy).

2. **Create a `.env` file** at the **project root** (`client/`) with at least:

   ```env
   BASE_URL=http://app.ai-grid.io:4000/v1
   AI_GRID_KEY=your_api_key_here
   QWEN_MODEL=Qwen3-30B-A3B-Thinking
   OSS_MODEL=gpt-oss-120b
   EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-7B-instruct
   OCR_MODEL=deepseek-ocr
   IMAGE_FOLDER=/path/to/ali/client/images
   OCR_BATCH_CONCURRENCY=5
   AI_GRID_BASE_URL=http://app.ai-grid.io:4000/v1
   AI_GRID_TOOL_MODEL=Qwen3-30B-A3B-Thinking
   ```

   For **voxtral** (e.g. local `voxtral-live` container):

   ```env
   AUDIO_PATH=/path/to/audio.mp3
   VOXTRAL_BASE_URL=http://localhost:8000/v1
   VOXTRAL_MODEL=mistralai/Voxtral-Mini-4B-Realtime-2602
   ```

   Do not commit `.env` (add to `.gitignore`).

3. **`images/`** — Optional. Put sample images here for OCR; `ocr.py` and `ocr_batch.py` use `IMAGE_FOLDER` (e.g. `images/`). Results are written as `.txt` next to each image.

---

## Scripts by category

| Category   | Script | Description |
|-----------|--------|-------------|
| **chat**  | `chat/qwen.py` | Chat with **Qwen**; single "Hello!" and reply. |
| **chat**  | `chat/oss.py` | Chat with **OSS** model; single reply. |
| **chat**  | `chat/stream_chat.py` | **Streaming** chat; tokens printed as they arrive. Uses `OSS_MODEL` or `STREAM_CHAT_MODEL`. |
| **langgraph** | `langgraph/langraph_oss.py` | **LangGraph** example: generate → (if long) refine → end. Uses `BASE_URL`, `AI_GRID_KEY`, `OSS_MODEL`. |
| **rag**    | `rag/rag_lite.py` | **Minimal RAG**: embed corpus + query → top-k → generate. Edit `DOCS` for your corpus. |
| **rag**    | `rag/agentic_rag.py` | **Agentic RAG**: LangGraph retrieve ↔ generate; state holds messages, query, retrieved_docs. |
| **rag**    | `rag/graph_rag.py` | **Graph RAG**: rewrite_query → retrieve → generate. Query expansion for better recall. |
| **ocr**    | `ocr/ocr.py` | **Single-image OCR** (v2). Uses `IMAGE_FOLDER` or `IMAGE_PATH`; writes `.txt` next to image. |
| **ocr**    | `ocr/ocr_batch.py` | **Batch OCR** (v2). Concurrent; set `OCR_BATCH_CONCURRENCY` in `.env`. |
| **ocr**    | `ocr/ocr_doc.py` | **Long-doc OCR**: PDF → pages → vertical windows → sequential OCR. One `.txt` per page + full doc. |
| **tools**  | `tools/qwen_tool_caller.py` | **Tool calling**: ReActAgent, FunctionTool, OpenAILike. Needs server tool-call support (e.g. vLLM `--enable-auto-tool-choice`). |
| **embedding** | `embedding/embeding_qwen_alibaba.py` | **Embeddings** (e.g. GTE-Qwen2). Prints vector length for input text. |
| **audio**  | `audio/voxtral.py` | **Voxtral** transcription: `AUDIO_PATH` from `.env`; REST or realtime WebSocket. |

All scripts load `.env` from the **project root**, so you can run them from the root or from inside a category folder.

---

## Usage

From the **project root** (`client/`), with `.env` present:

```bash
# Chat
python chat/qwen.py
python chat/oss.py
python chat/stream_chat.py

# LangGraph
python langgraph/langraph_oss.py

# RAG
python rag/rag_lite.py
python rag/agentic_rag.py
python rag/graph_rag.py

# OCR
python ocr/ocr.py
python ocr/ocr_batch.py
python ocr/ocr_doc.py path/to/doc.pdf -o path/to/output_dir

# Tools
python tools/qwen_tool_caller.py

# Embedding
python embedding/embeding_qwen_alibaba.py

# Audio
python audio/voxtral.py
```

**Notes:**

- **stream_chat.py** — Set `STREAM_CHAT_MODEL` in `.env` to override `OSS_MODEL`.
- **rag_lite.py** — Edit the `DOCS` list in the script; optional `k` in `retrieve()`.
- **agentic_rag.py** / **graph_rag.py** — Same `.env`; edit `DOCS` to change the corpus.
- **langraph_oss.py** — Same OSS endpoint as `chat/oss.py`. You can call `run("your question")` or `run_stream("your question")` from code.
- **qwen_tool_caller.py** — Add tools with `FunctionTool.from_defaults(fn=your_function)`. Endpoint must support tool calling.
- **voxtral.py** — Set `AUDIO_PATH` in `.env`. Run `python audio/voxtral.py` for REST or with `realtime` for WebSocket.
- **ocr.py / ocr_batch.py** — Use `IMAGE_FOLDER` in `.env` (e.g. `images/`). OCR v2 results are written as `.txt` next to each image.
- **ocr_doc.py** — Optional `.env`: `OCR_DPI` (default 200), `OCR_WINDOWS` (default 3). Requires `poppler-utils` (e.g. `apt install poppler-utils`).

**ocr_doc.py — install and run:**

```bash
pip install pdf2image Pillow
# Debian/Ubuntu: apt install poppler-utils

python ocr/ocr_doc.py path/to/document.pdf
python ocr/ocr_doc.py path/to/document.pdf -o ./ocr_out --dpi 200 --windows 3
```

---

## Environment reference

| Variable | Used by | Description |
|----------|---------|-------------|
| `BASE_URL` | chat, rag, ocr, embedding, langgraph | OpenAI-compatible API base (e.g. `http://app.ai-grid.io:4000/v1`) |
| `AI_GRID_KEY` | all | API key |
| `QWEN_MODEL` | chat/qwen, rag (fallback) | Chat model name for Qwen |
| `OSS_MODEL` | chat/oss, stream_chat, rag, langraph | Chat model name for OSS |
| `STREAM_CHAT_MODEL` | chat/stream_chat | Override for streaming chat model |
| `EMBEDDING_MODEL` | rag, embedding | Embedding model name |
| `OCR_MODEL` | ocr/* | Vision/OCR model name |
| `IMAGE_FOLDER` | ocr/ocr, ocr_batch | Folder of images to process |
| `IMAGE_PATH` | ocr/ocr | Single image path (overrides folder) |
| `OCR_BATCH_CONCURRENCY` | ocr/ocr_batch | Max concurrent OCR requests |
| `OCR_DPI` | ocr/ocr_doc | DPI for PDF→image (default 200) |
| `OCR_WINDOWS` | ocr/ocr_doc | Vertical windows per page (default 3) |
| `AI_GRID_BASE_URL` | tools/qwen_tool_caller | Base URL for tool-call endpoint |
| `AI_GRID_TOOL_MODEL` | tools/qwen_tool_caller | Model for tool calling |
| `AUDIO_PATH` | audio/voxtral | Input audio file |
| `VOXTRAL_BASE_URL` | audio/voxtral | Voxtral API base (optional) |
| `VOXTRAL_MODEL` | audio/voxtral | Voxtral model (optional) |

---

## License

Use according to your organization’s terms for the AI Grid API and the underlying models.
