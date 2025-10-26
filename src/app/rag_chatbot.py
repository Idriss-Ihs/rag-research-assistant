"""
rag_chatbot.py
---------------
Streamlit RAG chat interface.
"""

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))
import streamlit as st
from src.features.generator import generate_answer
from src.features.retriever import FaissRetriever, load_config

st.set_page_config(page_title="RAG Research Assistant", layout="wide")

st.title("📚 RAG Research Assistant")
st.markdown(
    "Ask questions grounded in your paper collection — with real context retrieval and cited snippets."
)

cfg = load_config()
retriever = FaissRetriever(cfg)

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:", placeholder="e.g. What are the main findings on transformer efficiency?")
if query:
    with st.spinner("Thinking..."):
        answer, refs = generate_answer(query, retriever)
    st.session_state.history.append({"query": query, "answer": answer, "refs": refs})

# Show chat history
for turn in reversed(st.session_state.history):
    st.markdown(f"### ❓ {turn['query']}")
    st.markdown(turn["answer"])
    with st.expander("📎 Sources"):
        for r in turn["refs"]:
            st.markdown(f"- **{r['source']}** (chunk {r['chunk_id']})")
    st.markdown("---")
