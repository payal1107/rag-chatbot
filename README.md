# RAG Chatbot

## Overview

Ask a plain chatbot about a document and it will often answer confidently even
when the document says nothing about it. This project fixes that. Upload a
document, ask anything, and the app tells you whether the answer came from your
document or from the model's general knowledge.

## How It Works

Built with LangChain. The document is split into chunks, embedded, and stored in
a FAISS index. A question retrieves the closest chunks and passes them to the
LLM as context.

The fallback is what makes it different. Two checks decide whether the answer is
really in the document:

1. **Distance cutoff** — chunks too far from the question are dropped before the
   LLM sees them.
2. **LLM self-check** — the rest go to the LLM with a strict instruction: answer
   only from this context, otherwise reply `NOT_FOUND`.

The second check matters because the first isn't enough. Ask a resume about
expected salary and the retriever still returns chunks, since salary is
topically close to everything else on a resume. Only reading them reveals the
answer isn't there.

If either check fires, the question goes to the LLM without context and the
answer is marked as general knowledge.

## How to Use

1. Visit [RAG Chatbot](https://rag-chatbot-7oirtdo5lnjegmw5hkptxx.streamlit.app)
   — it sleeps when idle, so the first load takes about 30 seconds
2. Upload a PDF, TXT, MD or DOCX in the sidebar
3. Click **Process documents**
4. Ask your question and check the badge on the answer

## Features

* **Source-backed answers** — expand any answer to see the exact chunks it used
* **Honest fallback** — a warning badge instead of a made-up answer
* **Retrieval debug panel** — every candidate chunk with its distance score and
  why a fallback happened
* **Tunable cutoff** — a sidebar slider adjusts strictness without touching code
* **Local embeddings** — document text stays on the machine apart from the few
  chunks sent with each question

## Example Output

| Question | Result |
|---|---|
| "How many days of sick leave?" | ✅ Answered from your document |
| "What is the capital of France?" | ⚠️ Not in the document — answered by the LLM |
| "What is the annual bonus percentage?" | ⚠️ Falls back, even though the topic is close — the case `NOT_FOUND` exists for |

## Configuration

At the top of `app.py`:

| Setting | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 400 | Larger chunks hold more context but blur the embedding |
| `CHUNK_OVERLAP` | 80 | Stops an answer splitting across a boundary |
| `TOP_K` | 6 | Chunks retrieved per question |
| `DISTANCE_CUTOFF` | 1.60 | Lower is stricter; also on the sidebar slider |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Any current Groq model |

## Tech Stack

Streamlit · LangChain (LCEL) · FAISS · all-MiniLM-L6-v2 embeddings · Groq

## Known Limitations
* Works best on plain text PDFs. Multi-column layouts and tables get broken up
  during extraction, which hurts answer quality. PyMuPDF would handle these better.
* No OCR, so scanned PDFs will not work
* Chat history isn't used for retrieval, so follow-ups won't resolve
* Semantic search only; hybrid search with a re-ranker would do better on exact
  terms and numbers
* The FAISS index is in memory and is lost on restart
* `DISTANCE_CUTOFF` was tuned by hand; a labelled question set would set it properly
