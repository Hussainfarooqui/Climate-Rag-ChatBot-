import streamlit as st
import os
from src.rag import RAGPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate RAG ChatBot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS – Claude/ChatGPT style ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Layout shell ── */
.app-shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
    background: #0f0f0f;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2a2a2a !important;
    min-width: 280px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.2rem !important;
}

/* Sidebar header */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.8rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #2a2a2a;
}
.sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.sidebar-logo-text {
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f0;
    line-height: 1.2;
}
.sidebar-logo-sub {
    font-size: 11px;
    color: #666;
    font-weight: 400;
}

/* Sidebar section labels */
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555;
    margin: 1.4rem 0 0.6rem;
}

/* Streamlit widgets inside sidebar */
section[data-testid="stSidebar"] .stFileUploader {
    background: #1e1e1e !important;
    border: 1px dashed #333 !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: #1e1e1e !important;
    color: #d0d0d0 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #252525 !important;
    border-color: #3a3a3a !important;
    color: #f0f0f0 !important;
}

/* Primary ingest button */
section[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: #166534 !important;
    border-color: #15803d !important;
    color: #dcfce7 !important;
}
section[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: #15803d !important;
}

/* ── Main chat area ── */
.main-chat {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #141414;
    position: relative;
    height: 100vh;
}

/* Top nav bar */
.chat-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 2rem;
    border-bottom: 1px solid #222;
    background: #141414;
    position: sticky; top: 0; z-index: 10;
}
.chat-topbar-title {
    font-size: 15px;
    font-weight: 600;
    color: #e0e0e0;
}
.chat-topbar-badge {
    font-size: 11px;
    background: #1c2e1e;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 3px 10px;
    font-weight: 500;
}

/* Messages scroll area */
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 2rem 0;
    scroll-behavior: smooth;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: #444;
    padding: 4rem 2rem;
    text-align: center;
}
.empty-state-icon {
    font-size: 48px;
    opacity: 0.6;
}
.empty-state-title {
    font-size: 22px;
    font-weight: 600;
    color: #555;
}
.empty-state-sub {
    font-size: 14px;
    color: #3a3a3a;
    max-width: 340px;
    line-height: 1.6;
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    padding: 0.3rem 2rem;
    gap: 12px;
    max-width: 820px;
    margin: 0 auto;
    width: 100%;
    box-sizing: border-box;
}
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }

/* Avatar */
.msg-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user {
    background: #1d4ed8;
    color: #bfdbfe;
    font-weight: 600;
    font-size: 12px;
}
.avatar-bot {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    font-size: 16px;
}

/* Bubble */
.msg-bubble {
    max-width: 72%;
    padding: 0.75rem 1.1rem;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.65;
    word-break: break-word;
}
.bubble-user {
    background: #1e40af;
    color: #dbeafe;
    border-bottom-right-radius: 4px;
}
.bubble-assistant {
    background: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #2a2a2a;
    border-bottom-left-radius: 4px;
}

/* Sources expander */
.sources-toggle {
    margin-top: 8px;
    cursor: pointer;
    font-size: 12px;
    color: #4ade80;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    user-select: none;
}
.sources-toggle:hover { color: #86efac; }

.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #1a2e1a;
    border: 1px solid #166534;
    border-radius: 6px;
    padding: 4px 9px;
    margin: 4px 4px 0 0;
    font-size: 11.5px;
    color: #86efac;
}
.source-score {
    background: #14532d;
    color: #4ade80;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 10px;
}

/* ── Input bar ── */
.input-bar-wrap {
    padding: 1.2rem 2rem 1.5rem;
    background: #141414;
    border-top: 1px solid #1e1e1e;
    max-width: 820px;
    margin: 0 auto;
    width: 100%;
    box-sizing: border-box;
}

section.main .stTextInput > div > div > input {
    background: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    padding: 0.75rem 1rem !important;
    caret-color: #4ade80 !important;
    box-shadow: none !important;
    transition: border-color 0.15s;
}
section.main .stTextInput > div > div > input:focus {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.12) !important;
}
section.main .stTextInput > div > div > input::placeholder { color: #444 !important; }

/* Send button */
section.main > div > div > div > div > div:last-child .stButton > button {
    background: #16a34a !important;
    color: #f0fdf4 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 0.72rem 1.4rem !important;
    transition: background 0.15s;
    cursor: pointer;
}
section.main > div > div > div > div > div:last-child .stButton > button:hover {
    background: #15803d !important;
}

/* Thinking spinner */
.thinking-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.4rem 2rem;
    max-width: 820px;
    margin: 0 auto;
}
.thinking-dots span {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    animation: bounce 1.2s infinite ease-in-out;
    margin: 0 2px;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
    40% { transform: translateY(-6px); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ── Guard: API key ──────────────────────────────────────────────────────────────
if "GROQ_API_KEY" not in os.environ or not os.environ.get("GROQ_API_KEY"):
    st.error("🔑 **GROQ_API_KEY** not set. Add it in Streamlit Secrets or your local `.env`.")
    st.stop()


# ── Session state ───────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(db_path="vectors.db")
if "history" not in st.session_state:
    st.session_state.history = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = {}
if "thinking" not in st.session_state:
    st.session_state.thinking = False


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🌿</div>
      <div>
        <div class="sidebar-logo-text">Climate RAG</div>
        <div class="sidebar-logo-sub">Powered by Groq + RAG</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">📂 Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "csv", "html"],
        label_visibility="collapsed",
    )
    if st.button("⬆ Ingest & Build Index"):
        if uploaded:
            with st.spinner("Indexing documents…"):
                count = st.session_state.pipeline.ingest_files(uploaded)
            st.success(f"Indexed {count} chunks ✅")
        else:
            st.warning("Upload at least one document first.")

    st.markdown('<div class="sidebar-label">💬 Chat</div>', unsafe_allow_html=True)
    if st.button("🗑 Clear conversation"):
        st.session_state.history = []
        st.session_state.show_sources = {}
        st.rerun()

    # Stats
    if st.session_state.history:
        st.markdown('<div class="sidebar-label">📊 Session stats</div>', unsafe_allow_html=True)
        st.caption(f"**{len(st.session_state.history)}** exchanges in this session")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#333;'>Built by Muhammad Hussain Ahmed Farooqui</div>",
        unsafe_allow_html=True,
    )


# ── Main area ────────────────────────────────────────────────────────────────────
# Top bar
st.markdown("""
<div class="chat-topbar">
  <span class="chat-topbar-title">Climate Knowledge Assistant</span>
  <span class="chat-topbar-badge">🟢 Online</span>
</div>
""", unsafe_allow_html=True)


# ── Render chat history ──────────────────────────────────────────────────────────
def render_messages():
    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">🌍</div>
          <div class="empty-state-title">Ask me anything about climate</div>
          <div class="empty-state-sub">Upload documents in the sidebar, then type your question below to get answers with source citations.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for i, msg in enumerate(st.session_state.history):
        # User bubble
        st.markdown(f"""
        <div class="msg-row user">
          <div class="msg-bubble bubble-user">{msg['q']}</div>
          <div class="msg-avatar avatar-user">You</div>
        </div>
        """, unsafe_allow_html=True)

        # Assistant bubble
        st.markdown(f"""
        <div class="msg-row assistant">
          <div class="msg-avatar avatar-bot">🌿</div>
          <div>
            <div class="msg-bubble bubble-assistant">{msg['a']}</div>
        """, unsafe_allow_html=True)

        # Sources as collapsible chips
        if msg.get("sources"):
            key = f"src_{i}"
            show = st.session_state.show_sources.get(key, False)
            toggle_label = f"▾ {len(msg['sources'])} sources" if show else f"▸ {len(msg['sources'])} sources"
            if st.button(toggle_label, key=f"btn_{key}"):
                st.session_state.show_sources[key] = not show
                st.rerun()
            if show:
                chips = ""
                for s in msg["sources"]:
                    src = s["metadata"].get("source", "doc")
                    chunk = s["metadata"].get("chunk_index", "?")
                    score = s.get("score", 0)
                    chips += f'<span class="source-chip">📄 {src} · chunk {chunk} <span class="source-score">{score:.2f}</span></span>'
                st.markdown(f'<div style="padding: 0 0 4px 0">{chips}</div>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Thinking indicator
    if st.session_state.thinking:
        st.markdown("""
        <div class="thinking-row">
          <div class="msg-avatar avatar-bot">🌿</div>
          <div class="thinking-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)


render_messages()

# ── Input bar ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

col_input, col_btn = st.columns([6, 1])
with col_input:
    query = st.text_input(
        "message",
        placeholder="Ask a question about the uploaded documents…",
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    send = st.button("Send ↑", use_container_width=True)

# ── Handle send ──────────────────────────────────────────────────────────────────
if (send or query) and query.strip():
    st.session_state.thinking = True
    st.rerun()

if st.session_state.thinking and st.session_state.get("query_input", "").strip():
    user_q = st.session_state.query_input.strip()
    try:
        res = st.session_state.pipeline.query(user_q)
        st.session_state.history.append({
            "q": user_q,
            "a": res["answer"],
            "sources": res.get("sources", []),
        })
    except Exception as e:
        st.session_state.history.append({
            "q": user_q,
            "a": f"⚠️ Error: {e}",
            "sources": [],
        })
    finally:
        st.session_state.thinking = False
    st.rerun()