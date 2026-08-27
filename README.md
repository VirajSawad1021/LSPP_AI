# LSPP RAG Chatbot: Research Papers Assistant

## 📝 Submission Summary
- **Documents Used:** Curated arXiv AI/RL research papers: *Attention Is All You Need* (Vaswani et al.), *Direct Preference Optimization* (Rafailov et al.), and *Proximal Policy Optimization* (Schulman et al.), plus user upload support for any custom PDF.
- **Levels Reached:** Completed **Levels 1, 2, 3, 4, 5, and Level 6 (Bonus Full-Stack App)**.
- **What Surprised Me:** Standard text extractors (`pypdf`) completely mangle multi-column benchmark tables (like Table 2 BLEU scores in the Attention paper) into out-of-order text tokens, causing standard RAG to fail or hallucinate; converting detected table structures into explicit Markdown tables via `pdfplumber` dramatically improved retrieval accuracy and numerical precision for tabular question answering.

---

## 🌟 Implemented Levels Overview

- 🎯 **Level 1: Bring Your Own PDF**
  - Ingests uploaded PDFs or pre-downloaded arXiv papers.
  - Configurable chunk size, overlap, and top-$k$ retrieval.
  - Polite refusal (`"I don't know based on the provided document."`) when asked out-of-domain questions (e.g. World Cup results).
- 🥈 **Level 2: Handle Messy PDFs (Tables & Columns)**
  - Dual extraction pipelines: Standard `pypdf` vs Enhanced `pdfplumber`.
  - Converts table bounding boxes into structured Markdown tables with column alignments.
  - Includes a dedicated side-by-side table parser inspector tool.
- 🥉 **Level 3: Implement Streaming**
  - Token-by-token streaming generator (`ask_stream()`) delivering smooth ChatGPT/Gemini-like real-time output.
  - Server-Sent Events (SSE) streaming in FastAPI and generator streaming in Gradio.
- 🏅 **Level 4: Show the Citations**
  - Tracks chunk provenance metadata (document filename, exact 1-indexed page number, and text excerpt).
  - Appends structured citation footers and renders interactive citation badges.
- 🧠 **Level 5: Conversational Memory & Pronoun Resolution**
  - Multi-turn conversational history.
  - Context-aware query reformulation that resolves pronouns (`"he"`, `"she"`, `"it"`, `"its formula"`) into self-contained search queries before vector lookup.
- 💎 **Level 6: Full-Stack Real App (Bonus)**
  - **Backend**: FastAPI REST & SSE streaming server (`/api/ask`, `/api/ask/stream`, `/api/upload`, `/api/select_sample`, `/api/compare_parsers`, `/health`).
  - **Frontend**: Sleek dark-mode Web UI with live streaming, parameter tuning, citation badges, and table inspector.
  - **Gradio App**: Standalone Gradio interface (`app_gradio.py`).

---

## 🚀 Quick Start Guide

### 1. Environment Setup
The project runs with Python 3.12 and a self-contained virtual environment:

```bash
cd /home/one-point/lab/LSPP_RAG
source .venv/bin/activate
```

Ensure `GEMINI_API_KEY` is present in `.env`:
```bash
GEMINI_API_KEY=AIzaSy...
```

### 2. Run Automated Verification Suite (Tests Levels 1–6)
```bash
python test_levels.py
```

### 3. Launch FastAPI + Web Frontend (Level 6)
```bash
python app_fastapi.py
```
Open your browser and navigate to: `http://localhost:8000`

### 4. Launch Standalone Gradio App
```bash
python app_gradio.py
```
Open your browser and navigate to: `http://localhost:7860`

---

## 📁 Repository Structure

```
LSPP_RAG/
├── rag_engine.py          # Core RAG pipeline (Levels 1-5)
├── app_fastapi.py         # FastAPI REST & SSE Backend (Level 6)
├── app_gradio.py          # Gradio Interactive Application
├── test_levels.py         # Automated Level 1-6 Test Suite
├── sample_docs/           # arXiv AI & RL Research Papers
│   ├── 1706.03762_attention_is_all_you_need.pdf
│   ├── 2305.18290_direct_preference_optimization.pdf
│   └── 1707.06347_proximal_policy_optimization.pdf
├── frontend/              # Modern Single-Page Application
│   ├── index.html
│   ├── style.css
│   └── app.js
├── README.md              # Project write-up & evaluation notes
└── requirements.txt       # Project dependencies
```
