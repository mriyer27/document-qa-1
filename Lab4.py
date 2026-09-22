# --- sqlite3 compatibility fix for Streamlit Community Cloud ---
# chromadb requires sqlite3 >= 3.35, but Streamlit Cloud's default Linux
# environment ships an older system sqlite3 that doesn't meet this
# requirement, causing chromadb to fail on import once deployed (it can
# still work fine locally, since local machines often have a newer sqlite3).
# The fix swaps in the pysqlite3-binary package (a modern, bundled sqlite3
# build) in place of the standard library's sqlite3 module before chromadb
# is imported. This snippet, and the reasoning behind it, was worked out
# with the help of Claude (Anthropic).
# Reference: https://stackoverflow.com/questions/76958817
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # not needed locally on most machines, only on Streamlit Cloud

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from openai import OpenAI

# ---------------------------------------------------------------------------
# Part A: Build (or reuse) the ChromaDB vector database
# ---------------------------------------------------------------------------

st.title("Lab 4 - Course Info Chatbot (RAG)")
st.write(
    "Ask a question about the course materials. This chatbot retrieves relevant "
    "passages from the course PDFs and uses them to ground its answers."
)

# Put the 7 course PDFs in a folder named Lab4_PDFs next to this script.
PDF_FOLDER = "Lab4_PDFs"

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


def build_vector_db():
    """
    Read every PDF in PDF_FOLDER, embed it with OpenAI's embedding model,
    and store it in a Chroma collection keyed by filename.
    """
    chroma_client = chromadb.Client()  # in-memory; rebuilt once per session
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small",
    )
    collection = chroma_client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=embedding_fn,
    )

    for filename in os.listdir(PDF_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_FOLDER, filename)
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            continue
        collection.add(
            documents=[text],
            ids=[filename],
            metadatas=[{"filename": filename}],
        )

    return collection


# Only build the vector DB once per session — this is what keeps embedding
# costs down, since OpenAI charges per token embedded.
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building the course vector database (first run only)..."):
        st.session_state.Lab4_VectorDB = build_vector_db()

collection = st.session_state.Lab4_VectorDB

# ---------------------------------------------------------------------------
# (Part A, step 3 — temporary test code)
# Uncomment this block once to sanity-check retrieval, then comment it back
# out / delete it before submitting, per the assignment's Part B step 4.
# ---------------------------------------------------------------------------
# test_query = "Generative AI"
# test_results = collection.query(query_texts=[test_query], n_results=3)
# st.write(f"Top 3 documents for test query '{test_query}':")
# st.write(test_results["ids"][0])

# ---------------------------------------------------------------------------
# Part B: RAG-augmented chatbot
# ---------------------------------------------------------------------------


def retrieve_context(query: str, k: int = 3) -> str:
    """Fetch the k most relevant documents and format them as context text."""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    parts = [
        f"[Source: {meta.get('filename', 'unknown')}]\n{doc}"
        for doc, meta in zip(docs, metas)
    ]
    return "\n\n".join(parts)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about the course")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = retrieve_context(user_input)

    system_prompt = (
        "You are a helpful course assistant. Use the reference material below, "
        "retrieved from the course documents, to answer the user's question. "
        "If you use this material, say so clearly (e.g., 'Based on the course "
        "materials...'). If the material doesn't help answer the question, say "
        "so and rely on your own general knowledge instead, making that clear too.\n\n"
        f"Reference material:\n{context}"
    )

    messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-6:]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})