# RAG Chatbot — LangChain + Streamlit
Live Demo: https://rag-chatbot-7oirtdo5lnjegmw5hkptxx.streamlit.app
Upload a document, ask questions about it. If the answer is not in the
document, the app **flags it** and falls back to the LLM's general knowledge
instead of making something up.

## Setup

1. Create and activate a virtual environment

   Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   Mac / Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Get a free API key from https://console.groq.com and create a file named
   `.env` in this folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
   (Or skip this and paste the key into the sidebar at runtime.)

4. Run the app
   ```
   streamlit run app.py
   ```

5. Browser opens at http://localhost:8501 — upload a PDF/TXT/DOCX in the
   sidebar, click **Process documents**, then start asking questions.

## How the "not available" fallback works

Two filters decide whether the answer really lives in the document:

1. **Distance cutoff.** FAISS returns an L2 distance for each retrieved chunk
   (smaller = more similar). Chunks above `DISTANCE_CUTOFF` are dropped. If
   nothing survives, we skip RAG entirely and save an LLM call.
2. **LLM self-check.** The surviving chunks go to the LLM with a strict
   instruction: answer only from this context, and if the answer isn't there,
   reply with exactly `NOT_FOUND`. This catches the case where a chunk is
   topically similar but doesn't actually contain the answer.

If either filter fires, the question is re-sent to the LLM without context and
the UI shows a ⚠️ badge so the user knows the answer did not come from their
document.

## Tuning

Everything lives in the config block at the top of `app.py`:

| Setting | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 800 | Bigger = more context per chunk, fewer chunks |
| `CHUNK_OVERLAP` | 120 | Prevents answers being split across a chunk boundary |
| `TOP_K` | 4 | How many chunks are retrieved per question |
| `DISTANCE_CUTOFF` | 1.15 | Lower = stricter, more fallbacks. Raise if valid questions are wrongly falling back |
| `LLM_MODEL` | llama-3.3-70b-versatile | Any Groq model |

## Demo checklist

- Ask something clearly **in** the document → green "Answered from your document"
  badge, expandable retrieved chunks.
- Ask something clearly **outside** it (e.g. "who won the 2011 cricket world cup?")
  → orange fallback badge.
- Ask something **related but absent** (e.g. salary if the doc is a resume with
  no salary) → should also fall back. This is the case the `NOT_FOUND` check
  exists for; screenshot this one, it's the most interesting.

## Stack

- **UI** — Streamlit
- **Orchestration** — LangChain (LCEL chains)
- **Embeddings** — `all-MiniLM-L6-v2`, runs locally, free
- **Vector store** — FAISS, in-memory
- **LLM** — Llama 3.3 70B via Groq, free tier
