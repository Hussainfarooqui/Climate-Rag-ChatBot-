import streamlit as st
import os
from src.rag import RAGPipeline

# Page config
st.set_page_config(page_title="Climate-Rag-ChatBot", layout="centered")

# Main heading with an icon (using an emoji for "steam"/climate theme)
st.markdown(
    "<h1 style='text-align:center;'>🌿 Climate-Rag-ChatBot by Muhammad Hussain Ahmed Farooqui 🌬️</h1>",
    unsafe_allow_html=True
)

# Check Groq API
if "GROQ_API_KEY" not in os.environ or not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not set. Add it in Streamlit Secrets or locally.")
    st.stop()

# Initialize pipeline
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(db_path="vectors.db")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: file uploader
with st.sidebar:
    st.header("Upload & Index")
    uploaded = st.file_uploader(
        "Upload documents (pdf, docx, pptx, txt, csv, html)", 
        accept_multiple_files=True
    )
    if st.button("📂 Ingest & Build Index"):
        if uploaded:
            with st.spinner("Indexing..."):
                count = st.session_state.pipeline.ingest_files(uploaded)
                st.success(f"Indexed {count} chunks ✅")
        else:
            st.warning("Upload at least one document ⚠️")
    
    # Reset chat history button
    if st.button("🔄 Reset Chat"):
        st.session_state.history = []
        st.success("Chat history cleared!")

# Chat input
query = st.text_input("💬 Ask a question about the uploaded documents:")
if st.button("Send") and query:
    with st.spinner("Searching & generating answer..."):
        res = st.session_state.pipeline.query(query)
        # Add icons to messages
        st.session_state.history.append({
            "q": f"❓ {query}",
            "a": f"🤖 {res['answer']}",
            "sources": res["sources"]
        })

# Display chat history (most recent on top)
for msg in reversed(st.session_state.history):
    st.markdown(f"**{msg['q']}**")
    st.markdown(f"{msg['a']}")
    if msg["sources"]:
        st.markdown("**Sources:**")
        for s in msg["sources"]:
            st.markdown(f"- 📝 {s['metadata']['source']} (chunk {s['metadata']['chunk_index']}, score {s['score']:.3f})")
    st.markdown("---")
