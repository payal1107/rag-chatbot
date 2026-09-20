"""
RAG Chatbot with LangChain + Streamlit
--------------------------------------
Upload a document -> ask questions -> answers come from the document.
If the answer is NOT in the document, the app flags it and falls back to the
plain LLM (general knowledge) instead of hallucinating from context.

Run:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"      # free on Groq
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
# FAISS returns L2 distance -> SMALLER is better. Above this we treat the
# retrieved chunks as "probably irrelevant" and skip RAG entirely.
DISTANCE_CUTOFF = 1.60
NOT_FOUND_TOKEN = "NOT_FOUND"

# Questions about the document as a whole. Ranking chunks by similarity is
# meaningless here -- "what is this about" is not close to any one passage --
# so these bypass the distance cutoff and use the top chunks directly.
OVERVIEW_HINTS = (
    "what is this", "what's this", "what is inside", "what's inside",
    "what does this", "what is the document", "this document about",
    "this pdf about", "this file about", "summarise", "summarize",
    "summary", "overview", "main points", "key points", "tell me about this",
    "what is it about", "describe this", "gist",
)


def is_overview_question(q):
    low = q.lower()
    return any(h in low for h in OVERVIEW_HINTS)
    
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Loading / indexing
# --------------------------------------------------------------------------
def load_document(uploaded_file):
    """Save the upload to a temp file and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:  # .txt, .md
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    for d in docs:
        d.metadata["source_file"] = uploaded_file.name
    return docs


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Loaded once per session -- downloading the model takes a few seconds."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(all_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        return None, 0
    vs = FAISS.from_documents(chunks, get_embeddings())
    return vs, len(chunks)


# --------------------------------------------------------------------------
# LLM + chains
# --------------------------------------------------------------------------
def get_llm(api_key):
    return ChatGroq(model=LLM_MODEL, temperature=0.2, api_key=api_key)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "The CONTEXT below is a set of excerpts from a document the user "
     "uploaded. You can read it; they can see it too. Answer their question "
     "using only these excerpts.\n"
     "Rules:\n"
     "1. If the question asks what the document is about, or asks for a "
     "summary, overview or topic, describe what the excerpts contain. This "
     "is always answerable -- never refuse it.\n"
     "2. If it asks for a specific fact and the excerpts contain it, state "
     "it clearly and concisely.\n"
     f"3. If it asks for a specific fact the excerpts do NOT contain, reply "
     f"with exactly {NOT_FOUND_TOKEN} and nothing else.\n"
     "4. Never say you cannot see or open the file -- the excerpts are the "
     "file. Never fill gaps from outside knowledge.\n\n"
     "CONTEXT:\n{context}"),
    ("human", "{question}"),
])

# Overview questions get their own prompt with no NOT_FOUND escape hatch.
# "What is this document about" is always answerable from any excerpts, so
# offering the model a way to refuse only invites it to refuse.
OVERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "The CONTEXT below is a set of excerpts from a document the user "
     "uploaded. Describe what the document covers, based only on these "
     "excerpts.\n"
     "The excerpts may be fragmentary or out of order. Work with whatever is "
     "there and describe the subject matter and the main topics you can see. "
     "Do not ask the user for more information, do not say you cannot see the "
     "file, and do not refuse -- the excerpts are the file.\n\n"
     "CONTEXT:\n{context}"),
    ("human", "{question}"),
])

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. The user uploaded a document, but it does "
     "not contain what they asked for, so answer from your own general "
     "knowledge instead. The interface has already told them the answer is "
     "not from their document, so do not repeat that, and never claim you "
     "cannot see or open their file. Be concise, and say plainly if you are "
     "unsure or if the question cannot be answered without the document."),
    ("human", "{question}"),
])


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        tag = f"[chunk {i}" + (f", page {page + 1}" if page is not None else "") + "]"
        parts.append(f"{tag}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, vectorstore, llm, cutoff):
    """
    Returns: (answer_text, mode, sources, debug)
      mode  = "document"  -> answered from the uploaded document
      mode  = "fallback"  -> not in document, answered by the LLM
      debug = every retrieved chunk with its distance and whether it passed,
              plus which stage caused a fallback
    """
    hits = []
    debug = {"scored": [], "reason": "no document indexed"}

    if vectorstore is not None:
        scored = vectorstore.similarity_search_with_score(question, k=TOP_K)
        # Stage 1 filter: cheap distance cutoff, saves an LLM call
        if is_overview_question(question):
            hits = [doc for doc, _ in scored]
        else:
            hits = [doc for doc, dist in scored if dist <= cutoff]
        debug["scored"] = [{
            "distance": float(dist),
            "passed": bool(dist <= cutoff),
            "file": doc.metadata.get("source_file", "document"),
            "page": (doc.metadata["page"] + 1) if doc.metadata.get("page") is not None else None,
            "text": doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
        } for doc, dist in scored]
        debug["reason"] = ("every chunk was above the distance cutoff"
                           if not hits else "")

    if hits and is_overview_question(question):
        # No second filter here -- an overview is always answerable
        chain = OVERVIEW_PROMPT | llm | StrOutputParser()
        result = chain.invoke({
            "context": format_context(hits),
            "question": question,
        }).strip()
        return result, "document", hits, debug

    if hits:
        # Stage 2 filter: let the LLM decide if the context really answers it
        chain = RAG_PROMPT | llm | StrOutputParser()
        result = chain.invoke({
            "context": format_context(hits),
            "question": question,
        }).strip()

        if NOT_FOUND_TOKEN not in result.upper():
            return result, "document", hits, debug
        debug["reason"] = "chunks passed the cutoff, but the LLM said NOT_FOUND"

    # Fallback -- either nothing retrieved, or the LLM said NOT_FOUND
    chain = FALLBACK_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip()
    return result, "fallback", [], debug


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("indexed_files", [])
st.session_state.setdefault("chunk_count", 0)


# --------------------------------------------------------------------------
# Sidebar -- upload + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password",
                                help="Get a free key at console.groq.com")

    uploaded = st.file_uploader(
        "Upload document(s)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    if st.button("🔍 Process documents", use_container_width=True,
                 disabled=not uploaded):
        with st.spinner("Reading, chunking and embedding..."):
            all_docs = []
            for f in uploaded:
                all_docs.extend(load_document(f))
            vs, n = build_vectorstore(all_docs)

        if vs is None:
            st.error("No readable text found. Is the PDF a scanned image?")
        else:
            st.session_state.vectorstore = vs
            st.session_state.chunk_count = n
            st.session_state.indexed_files = [f.name for f in uploaded]
            st.success(f"Indexed {n} chunks from {len(uploaded)} file(s).")

    if st.session_state.indexed_files:
        st.markdown("**Indexed:**")
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")
        st.caption(f"{st.session_state.chunk_count} chunks in the vector store")

    st.divider()
    st.subheader("⚙️ Retrieval")
    cutoff = st.slider(
        "Distance cutoff",
        min_value=0.5, max_value=2.0, value=DISTANCE_CUTOFF, step=0.05,
        help="FAISS L2 distance. Lower = stricter, more fallbacks. "
             "Chunks further than this are dropped before the LLM sees them.",
    )
    show_debug = st.checkbox("Show retrieval debug", value=True)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Main -- chat
# --------------------------------------------------------------------------
st.title("RAG Chatbot")
st.caption(
    "Ask about your document. If the answer isn't in it, the app says so "
    "and falls back to the LLM's general knowledge."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("mode") == "document":
                st.success("📄 Answered from your document", icon="✅")
            elif msg.get("mode") == "fallback":
                st.warning(
                    "⚠️ Not found in the document — answered by the LLM "
                    "(general knowledge)",
                    icon="🌐",
                )
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("Show retrieved chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}** — {src['file']}"
                                + (f", page {src['page']}" if src["page"] else ""))
                    st.caption(src["text"])

question = st.chat_input("Ask something about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Add your Groq API key in the sidebar first.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, mode, hits, debug = answer_question(
                    question, st.session_state.vectorstore, get_llm(api_key),
                    cutoff,
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        if mode == "document":
            st.success("📄 Answered from your document", icon="✅")
        else:
            st.warning(
                "⚠️ Not found in the document — answered by the LLM "
                "(general knowledge)",
                icon="🌐",
            )
        st.markdown(answer)

        sources = [{
            "file": d.metadata.get("source_file", "document"),
            "page": (d.metadata["page"] + 1) if d.metadata.get("page") is not None else None,
            "text": d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""),
        } for d in hits]

        if sources:
            with st.expander("Show retrieved chunks"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}** — {src['file']}"
                                + (f", page {src['page']}" if src["page"] else ""))
                    st.caption(src["text"])

        if show_debug and debug["scored"]:
            with st.expander("🔬 Retrieval debug — all candidates and distances"):
                if debug["reason"]:
                    st.info(f"Fell back because: {debug['reason']}")
                st.caption(f"Cutoff in use: {cutoff:.2f} — lower distance is a closer match.")
                for i, s in enumerate(debug["scored"], 1):
                    mark = "✅ passed" if s["passed"] else "❌ dropped"
                    st.markdown(
                        f"**{i}. distance {s['distance']:.3f}** — {mark} — {s['file']}"
                        + (f", page {s['page']}" if s["page"] else "")
                    )
                    st.caption(s["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "mode": mode,
        "sources": sources,
    })
