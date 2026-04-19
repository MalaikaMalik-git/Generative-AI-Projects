"""
app.py — RAG Tutor — Professional Redesign v3
Fixes:
  1. Example query click → auto-fills AND auto-submits the prompt
  2. Sidebar collapse/expand: custom JS toggle, clean button
  3. Floating ☰ FAB appears when sidebar is closed so it can be reopened
  4. All Streamlit layout edge-cases handled
"""
from __future__ import annotations
import os
from datetime import datetime
import streamlit as st

from rag.config import CHROMA_DIR, EMBEDDING_MODEL_NAME, FIXED_COLLECTION, RECURSIVE_COLLECTION
from rag.embedder import Embedder
from rag.memory import ConversationMemory
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vector_store import VectorStore

st.set_page_config(
    page_title="RAG Tutor",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "messages":       [],
        "memory":         ConversationMemory(),
        "pipeline":       None,
        "strategy":       "recursive",
        "top_k":          3,
        "show_chunks":    True,
        "pending_query":  "",
        "last_output":    None,
        "submit_pending": False,   # True → auto-submit on next render
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


@st.cache_resource
def _emb():
    return Embedder(EMBEDDING_MODEL_NAME)


def _pipe(strategy: str) -> RAGPipeline:
    cur = st.session_state.pipeline
    if cur and getattr(cur, "_strategy", None) == strategy:
        return cur
    col = FIXED_COLLECTION if strategy == "fixed" else RECURSIVE_COLLECTION
    p = RAGPipeline(
        retriever=Retriever(embedder=_emb(), vector_store=VectorStore(str(CHROMA_DIR), col)),
        memory=st.session_state.memory,
    )
    p._strategy = strategy
    st.session_state.pipeline = p
    return p


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #F7F8FC;
  --surface:   #FFFFFF;
  --surface2:  #F0F2F8;
  --surface3:  #E8EBF4;
  --border:    #E2E6F0;
  --border2:   #CDD3E8;
  --navy:      #0F1B3C;
  --navy2:     #1A2D5A;
  --navy3:     #2A4080;
  --accent:    #2563EB;
  --accent-lt: #EFF4FF;
  --accent-bd: #BFCFFE;
  --green:     #059669;
  --green-lt:  #ECFDF5;
  --green-bd:  #A7F3D0;
  --amber:     #D97706;
  --amber-lt:  #FFFBEB;
  --amber-bd:  #FDE68A;
  --red:       #DC2626;
  --red-lt:    #FEF2F2;
  --red-bd:    #FECACA;
  --violet:    #7C3AED;
  --violet-lt: #F5F3FF;
  --violet-bd: #DDD6FE;
  --text:      #111827;
  --text2:     #4B5563;
  --text3:     #9CA3AF;
  --text4:     #D1D5DB;
  --mono:      'JetBrains Mono', monospace;
  --sans:      'Sora', system-ui, sans-serif;
  --radius:    10px;
  --radius-sm: 6px;
  --shadow:    0 1px 3px rgba(15,27,60,0.08), 0 1px 2px rgba(15,27,60,0.04);
  --shadow-md: 0 4px 12px rgba(15,27,60,0.10), 0 2px 4px rgba(15,27,60,0.06);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: var(--sans) !important;
  color: var(--text) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stToolbar"]  { display: none !important; }

/* Hide the default sidebar collapse arrow/button — we replace it with JS */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

.block-container {
  padding: 2rem 2.5rem 7rem !important;
  max-width: 840px !important;
  margin: 0 auto !important;
}

/* ── SIDEBAR ─────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--navy) !important;
  border-right: none !important;
  box-shadow: 2px 0 20px rgba(15,27,60,0.22) !important;
  min-width: 280px !important;
  max-width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] section > div { padding: 0 !important; }
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
[data-testid="stSidebar"] label {
  color: #8FA4CC !important; font-size: 11px !important;
  font-weight: 500 !important; letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
}

/* Floating ☰ open-sidebar button — hidden by default, shown via JS */
#sb-open-fab {
  position: fixed;
  top: 16px; left: 14px; z-index: 99999;
  width: 38px; height: 38px;
  background: var(--navy);
  border: none; border-radius: 9px;
  display: none;           /* JS toggles this */
  align-items: center; justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(15,27,60,0.28);
  color: #AABFE0; font-size: 16px;
  transition: background 0.18s, transform 0.12s;
}
#sb-open-fab:hover { background: var(--navy2); color: #fff; transform: scale(1.06); }

/* ── Sidebar header ── */
.sb-header {
  padding: 22px 22px 18px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 2px;
}
.sb-logo-row { display: flex; align-items: center; gap: 12px; }
.sb-icon {
  width: 36px; height: 36px; border-radius: 9px;
  background: var(--accent); display: flex;
  align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(37,99,235,0.45);
}
.sb-title {
  font-size: 15px; font-weight: 600; color: #fff;
  letter-spacing: -0.01em; line-height: 1.2;
}
.sb-tagline {
  font-family: var(--mono) !important;
  font-size: 10px; color: #4A6490; margin-top: 2px; letter-spacing: 0.06em;
}

.sb-label {
  font-family: var(--mono) !important;
  font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase;
  color: #4A6490; padding: 18px 22px 8px; display: block;
}
.sb-divider { height: 1px; background: rgba(255,255,255,0.07); margin: 4px 0; }

/* ── Sidebar controls ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: var(--radius-sm) !important;
  color: #C8D8F0 !important; font-size: 13px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { color: #4A6490 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"]  { padding: 0 22px !important; }
[data-testid="stSidebar"] [data-testid="stToggle"]  { padding: 0 22px !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] { padding: 0 22px !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
  font-family: var(--mono) !important; font-size: 11px !important; color: #7EB3FF !important;
}

/* ── Memory stats ── */
.stat-row { display: flex; gap: 8px; padding: 4px 22px 14px; }
.stat-box {
  flex: 1; background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius-sm); padding: 10px 8px; text-align: center;
}
.stat-val {
  font-family: var(--mono) !important; font-size: 18px; font-weight: 500;
  color: #fff; line-height: 1; margin-bottom: 4px;
}
.stat-val.active { color: #7EB3FF; }
.stat-lbl {
  font-family: var(--mono) !important; font-size: 9px;
  letter-spacing: 0.08em; text-transform: uppercase; color: #4A6490;
}

/* ── Summary ── */
.sb-summary {
  margin: 0 22px 14px;
  background: rgba(37,99,235,0.10); border: 1px solid rgba(37,99,235,0.20);
  border-radius: var(--radius-sm); padding: 10px 12px;
}
.sb-summary-label {
  font-family: var(--mono) !important; font-size: 9px;
  letter-spacing: 0.10em; text-transform: uppercase; color: #7EB3FF; margin-bottom: 5px;
}
.sb-summary-text { font-size: 11px; color: #8FA4CC; line-height: 1.65; }

/* ── Topic chips ── */
.topic-row { display: flex; flex-wrap: wrap; gap: 5px; padding: 0 22px 14px; }
.topic-chip {
  font-family: var(--mono) !important; font-size: 10px; color: #6B84B0;
  background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 4px; padding: 3px 8px;
}

/* ── ALL sidebar Streamlit buttons: flat left-aligned list rows ── */
[data-testid="stSidebar"] [data-testid="stButton"] button {
  width: 100% !important; margin: 0 !important;
  background: transparent !important;
  border: none !important; border-radius: 0 !important;
  border-bottom: 1px solid rgba(255,255,255,0.05) !important;
  color: #8FA4CC !important; font-size: 12.5px !important;
  font-family: var(--sans) !important; font-weight: 400 !important;
  text-align: left !important; padding: 11px 22px !important;
  line-height: 1.6 !important; white-space: normal !important;
  word-break: break-word !important;
  min-height: unset !important; height: auto !important;
  cursor: pointer !important; box-shadow: none !important;
  display: block !important;
  transition: background 0.12s, color 0.12s !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
  background: rgba(37,99,235,0.16) !important; color: #C0D8FF !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:active {
  background: rgba(37,99,235,0.26) !important; color: #D8EAFF !important;
}

/* ── Clear button: pill style override ── */
[data-testid="stSidebar"] .clear-btn [data-testid="stButton"] button {
  width: calc(100% - 44px) !important; margin: 0 22px !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: var(--radius-sm) !important;
  color: #6B84B0 !important; font-size: 12px !important;
  text-align: center !important; padding: 8px 14px !important;
  white-space: nowrap !important;
}
[data-testid="stSidebar"] .clear-btn [data-testid="stButton"] button:hover {
  background: rgba(220,38,38,0.14) !important;
  border-color: rgba(220,38,38,0.28) !important; color: #FCA5A5 !important;
}

/* ══════════════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════════════ */
.page-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 0 20px 0; border-bottom: 2px solid var(--border);
  margin-bottom: 28px; flex-wrap: wrap; gap: 12px;
}
.ph-left { display: flex; align-items: center; gap: 10px; }
.ph-status {
  width: 8px; height: 8px; border-radius: 50%; background: var(--green);
  box-shadow: 0 0 0 3px var(--green-lt);
}
.ph-title { font-size: 15px; font-weight: 600; color: var(--navy); letter-spacing: -0.01em; }
.ph-sep   { width: 1px; height: 14px; background: var(--border2); }
.ph-sub   { font-size: 13px; color: var(--text3); }
.badge-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.badge {
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  padding: 4px 10px; border-radius: 20px; border: 1px solid; letter-spacing: 0.02em;
}
.b-green  { color: var(--green);  border-color: var(--green-bd);  background: var(--green-lt); }
.b-red    { color: var(--red);    border-color: var(--red-bd);    background: var(--red-lt); }
.b-accent { color: var(--accent); border-color: var(--accent-bd); background: var(--accent-lt); }
.b-muted  { color: var(--text3);  border-color: var(--border2);   background: var(--surface2); }
.b-violet { color: var(--violet); border-color: var(--violet-bd); background: var(--violet-lt); }
.b-amber  { color: var(--amber);  border-color: var(--amber-bd);  background: var(--amber-lt); }

/* ══════════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════════ */
.empty-state { text-align: center; padding: 48px 24px 36px; }
.empty-icon {
  width: 64px; height: 64px; margin: 0 auto 20px;
  background: var(--accent-lt); border-radius: 18px;
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; border: 1px solid var(--accent-bd);
}
.empty-title { font-size: 22px; font-weight: 600; color: var(--navy); letter-spacing: -0.02em; margin-bottom: 10px; }
.empty-desc  { font-size: 14px; color: var(--text2); line-height: 1.75; max-width: 420px; margin: 0 auto 28px; }
.feature-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.feature-chip  {
  font-size: 12px; color: var(--text2); font-weight: 500;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 6px 14px;
}

/* ══════════════════════════════════════════════
   MESSAGES
══════════════════════════════════════════════ */
.msg-user-wrap { display: flex; justify-content: flex-end; margin-bottom: 22px; }
.msg-user-inner { max-width: 72%; }
.msg-user-meta {
  display: flex; align-items: center; justify-content: flex-end;
  gap: 7px; margin-bottom: 5px;
}
.msg-label {
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  letter-spacing: 0.06em; text-transform: uppercase; padding: 2px 8px; border-radius: 3px;
}
.lbl-user { color: var(--accent); background: var(--accent-lt); border: 1px solid var(--accent-bd); }
.lbl-ai   { color: var(--navy);   background: var(--surface2);  border: 1px solid var(--border); }
.msg-time { font-family: var(--mono); font-size: 10px; color: var(--text4); }
.msg-user-bubble {
  background: var(--navy); color: #EFF3FF;
  border-radius: 12px 2px 12px 12px; padding: 12px 16px;
  font-size: 14px; line-height: 1.65; box-shadow: var(--shadow-md);
}
.msg-ai-wrap { margin-bottom: 22px; }
.msg-ai-meta { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }
.msg-ai-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 2px 12px 12px 12px; box-shadow: var(--shadow); overflow: hidden;
}
.msg-ai-body { padding: 14px 18px; font-size: 14px; line-height: 1.78; color: var(--text); }
.msg-ai-body p { margin: 0 0 10px; }
.msg-ai-body p:last-child { margin-bottom: 0; }
.msg-ai-body strong { color: var(--navy); font-weight: 600; }
.msg-ai-body code {
  font-family: var(--mono); font-size: 12px;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 4px; padding: 1px 6px; color: var(--accent);
}
.msg-ai-body pre {
  background: var(--navy); color: #C8D8F0; border-radius: var(--radius);
  padding: 14px 16px; overflow-x: auto; margin: 12px 0;
  font-family: var(--mono); font-size: 12px; line-height: 1.6;
}
.msg-ai-body pre code { background: none; border: none; padding: 0; color: inherit; }
.msg-ai-body ul, .msg-ai-body ol { padding-left: 20px; margin: 8px 0; }
.msg-ai-body li { margin-bottom: 5px; }
.msg-ai-body h1,.msg-ai-body h2,.msg-ai-body h3 {
  color: var(--navy); font-weight: 600; letter-spacing: -0.01em; margin: 14px 0 6px;
}
.msg-ai-footer {
  background: var(--surface2); border-top: 1px solid var(--border);
  padding: 7px 14px; display: flex; flex-wrap: wrap; gap: 5px; align-items: center;
}

/* ══════════════════════════════════════════════
   CHUNKS
══════════════════════════════════════════════ */
.chunks-section { margin: 6px 0 28px; }
.chunks-header  { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.chunks-title {
  font-size: 11px; font-weight: 600; color: var(--text3);
  text-transform: uppercase; letter-spacing: 0.10em; font-family: var(--mono);
}
.chunks-count {
  font-family: var(--mono); font-size: 10px; color: var(--accent);
  background: var(--accent-lt); border: 1px solid var(--accent-bd);
  border-radius: 10px; padding: 1px 7px;
}
.rewrite-note {
  background: var(--amber-lt); border: 1px solid var(--amber-bd);
  border-radius: var(--radius-sm); padding: 9px 13px; margin-bottom: 10px;
}
.rewrite-label {
  font-family: var(--mono); font-size: 9px; text-transform: uppercase;
  letter-spacing: 0.10em; color: var(--amber); margin-bottom: 2px;
}
.rewrite-text { font-size: 12px; color: var(--text2); line-height: 1.5; }
.chunk-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 12px 14px;
  margin-bottom: 8px; box-shadow: var(--shadow);
}
.chunk-top { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.chunk-num { font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--accent); min-width: 22px; }
.chunk-source {
  font-family: var(--mono); font-size: 11px; color: var(--text2);
  flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.chunk-score-wrap { display: flex; align-items: center; gap: 7px; flex-shrink: 0; }
.chunk-bar-bg { width: 60px; height: 3px; background: var(--surface3); border-radius: 2px; overflow: hidden; }
.chunk-bar-fill { height: 100%; border-radius: 2px; background: var(--accent); }
.chunk-score-pct {
  font-family: var(--mono); font-size: 11px; color: var(--accent);
  min-width: 30px; text-align: right; font-weight: 500;
}
.chunk-text {
  font-size: 12px; color: var(--text2); line-height: 1.7;
  overflow: hidden; display: -webkit-box;
  -webkit-line-clamp: 4; -webkit-box-orient: vertical;
}

/* ══════════════════════════════════════════════
   INPUT
══════════════════════════════════════════════ */
.input-section {
  margin-top: 28px; background: var(--surface);
  border: 1.5px solid var(--border2); border-radius: var(--radius);
  box-shadow: var(--shadow-md); overflow: hidden;
}
.input-section [data-testid="stTextInput"] > div,
.input-section [data-testid="stTextInput"] > div:focus-within,
.input-section [data-testid="stTextInput"] > div > div {
  background: transparent !important; border: none !important; box-shadow: none !important;
  border-radius: 0 !important;
}
.input-section [data-testid="stTextInput"] input {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: var(--text) !important; font-size: 14px !important;
  font-family: var(--sans) !important; caret-color: var(--accent) !important;
  padding: 14px 16px !important;
}
.input-section [data-testid="stTextInput"] input::placeholder { color: var(--text4) !important; }
.input-section [data-testid="stTextInput"] label { display: none !important; }
.input-footer {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 14px 10px; gap: 10px; border-top: 1px solid var(--border);
  flex-wrap: wrap;
}
.input-hint { font-size: 11px; color: var(--text4); font-family: var(--mono); }

/* Send button */
.main-area .stButton > button {
  background: var(--navy) !important; border: none !important;
  border-radius: var(--radius-sm) !important; color: #fff !important;
  font-family: var(--sans) !important; font-size: 13px !important;
  font-weight: 500 !important; padding: 8px 22px !important;
  height: auto !important; min-height: unset !important;
  transition: background 0.15s, transform 0.1s !important;
}
.main-area .stButton > button:hover  { background: var(--navy2) !important; }
.main-area .stButton > button:active { transform: scale(0.97) !important; }

/* Misc */
.stColumns { gap: 8px !important; }
[data-testid="column"] { overflow: visible !important; }
.stMarkdown { overflow: visible !important; word-break: break-word !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR TOGGLE JAVASCRIPT
# Injects a ☰ FAB that calls Streamlit's hidden collapse button.
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<button id="sb-open-fab" onclick="openSidebar()" title="Open sidebar">☰</button>

<script>
(function() {
  /* Find and click the hidden Streamlit collapse/expand button */
  function getSidebarBtn() {
    // Collapsed state: Streamlit renders an "Open sidebar" button in the main area
    var open = document.querySelector('button[aria-label="Open sidebar"]');
    if (open) return open;
    // Expanded state: collapse button inside sidebar
    var close = document.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (close) return close;
    return null;
  }

  window.openSidebar = function() {
    var btn = getSidebarBtn();
    if (btn) btn.click();
  };

  /* Poll every 250 ms to show/hide the FAB */
  function updateFab() {
    var fab = document.getElementById('sb-open-fab');
    if (!fab) return;
    // Streamlit adds aria-expanded="false" to the sidebar section when collapsed
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return;
    var collapsed = (
      sidebar.getAttribute('aria-expanded') === 'false' ||
      sidebar.getBoundingClientRect().width < 20
    );
    fab.style.display = collapsed ? 'flex' : 'none';
  }

  // Run immediately and then on interval
  updateFab();
  setInterval(updateFab, 250);
})();
</script>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR CONTENT
# ═══════════════════════════════════════════════════════════════
STARTERS = [
    "Why does chunk overlap become necessary when small chunks lose context?",
    "Why does hybrid retrieval outperform dense-only search?",
    "Why are FastAPI background tasks unreliable for heavy workloads?",
    "Why is SSG faster while SSR handles dynamic content better?",
    "How does observability catch fluent-but-wrong AI answers?",
    "What is 25 × 8 + 10?",
    "What date is 10 days after 2026-04-18?",
]

with st.sidebar:
    st.markdown("""
    <div class="sb-header">
      <div class="sb-logo-row">
        <div class="sb-icon">🔷</div>
        <div>
          <div class="sb-title">RAG Tutor</div>
          <div class="sb-tagline">Knowledge base · v2</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-label">Configuration</span>', unsafe_allow_html=True)
    new_s = st.selectbox(
        "Chunking strategy", ["recursive", "fixed"],
        index=0 if st.session_state.strategy == "recursive" else 1,
    )
    if new_s != st.session_state.strategy:
        st.session_state.strategy = new_s
        st.session_state.pipeline = None

    st.session_state.top_k = st.slider("Top-K documents", 1, 5, st.session_state.top_k)
    st.session_state.show_chunks = st.toggle("Show retrieved chunks", value=st.session_state.show_chunks)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">Conversation Memory</span>', unsafe_allow_html=True)

    mem: ConversationMemory = st.session_state.memory
    topics = mem.topic_summary()
    turns_cls = "stat-val active" if mem.turn_count else "stat-val"
    summ_cls  = "stat-val active" if mem.summary    else "stat-val"
    topic_cls = "stat-val active" if topics          else "stat-val"

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="{turns_cls}">{mem.turn_count}</div>
        <div class="stat-lbl">Turns</div>
      </div>
      <div class="stat-box">
        <div class="{summ_cls}">{"✓" if mem.summary else "—"}</div>
        <div class="stat-lbl">Summary</div>
      </div>
      <div class="stat-box">
        <div class="{topic_cls}">{len(topics)}</div>
        <div class="stat-lbl">Topics</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if mem.summary:
        st.markdown(f"""
        <div class="sb-summary">
          <div class="sb-summary-label">Rolling summary</div>
          <div class="sb-summary-text">{mem.summary}</div>
        </div>
        """, unsafe_allow_html=True)

    if topics:
        chips = "".join(f'<span class="topic-chip">{t}</span>' for t in topics[-6:])
        st.markdown(f'<div class="topic-row">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("↺  Clear conversation", key="clr"):
        st.session_state.messages    = []
        st.session_state.memory      = ConversationMemory()
        st.session_state.pipeline    = None
        st.session_state.last_output = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Example queries ──────────────────────────────────────────
    # Clicking sets pending_query = full question + submit_pending = True.
    # On next render: prefill is picked up by the text_input AND _run() fires.
    st.markdown('<span class="sb-label">Example Queries</span>', unsafe_allow_html=True)
    for q in STARTERS:
        if st.button(q, key=f"ex{hash(q)}"):
            st.session_state.pending_query  = q
            st.session_state.submit_pending = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="main-area">', unsafe_allow_html=True)

has_key  = bool(os.getenv("OPENAI_API_KEY", "").strip())
mode_lbl = "LLM + RAG" if has_key else "Retrieval only"
mode_cls = "b-green"   if has_key else "b-red"

st.markdown(f"""
<div class="page-header">
  <div class="ph-left">
    <div class="ph-status"></div>
    <div class="ph-title">Knowledge Tutor</div>
    <div class="ph-sep"></div>
    <div class="ph-sub">Personal AI study assistant</div>
  </div>
  <div class="badge-row">
    <span class="badge {mode_cls}">{mode_lbl}</span>
    <span class="badge b-accent">{st.session_state.strategy}</span>
    <span class="badge b-muted">k = {st.session_state.top_k}</span>
  </div>
</div>
""", unsafe_allow_html=True)

msgs = st.session_state.messages

if not msgs:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🔷</div>
      <div class="empty-title">What would you like to learn?</div>
      <div class="empty-desc">
        Ask any question about your knowledge base. I maintain full conversation context,
        detect repeated questions, and rewrite follow-ups for sharper retrieval.
      </div>
      <div class="feature-chips">
        <span class="feature-chip">RAG retrieval</span>
        <span class="feature-chip">Conversation memory</span>
        <span class="feature-chip">Duplicate detection</span>
        <span class="feature-chip">Query rewriting</span>
        <span class="feature-chip">Calculator &amp; dates</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in msgs:
        meta = msg.get("meta", {})
        t    = meta.get("time", "")

        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user-wrap">
              <div class="msg-user-inner">
                <div class="msg-user-meta">
                  <span class="msg-label lbl-user">You</span>
                  <span class="msg-time">{t}</span>
                </div>
                <div class="msg-user-bubble">{msg["content"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            route  = meta.get("route", "")
            badges = []
            if route == "tool":
                badges.append(f'<span class="badge b-green">⚙ {meta.get("tool","tool")}</span>')
            elif route == "rag":
                badges.append('<span class="badge b-accent">◆ RAG</span>')
            if meta.get("duplicate"):
                badges.append('<span class="badge b-red">↻ repeated question</span>')
            if meta.get("was_summarised"):
                badges.append('<span class="badge b-violet">⊟ summarised</span>')
            if meta.get("retrieval_query") and \
               meta["retrieval_query"] != meta.get("original_query", ""):
                badges.append('<span class="badge b-amber">✎ query rewritten</span>')
            if meta.get("turns"):
                badges.append(f'<span class="badge b-muted">turn {meta["turns"]}</span>')

            footer = (
                '<div class="msg-ai-footer">' + "".join(badges) + "</div>"
            ) if badges else ""

            st.markdown(f"""
            <div class="msg-ai-wrap">
              <div class="msg-ai-meta">
                <span class="msg-label lbl-ai">AI</span>
                <span class="msg-time">{t}</span>
              </div>
              <div class="msg-ai-card">
                <div class="msg-ai-body">
            """, unsafe_allow_html=True)

            st.markdown(msg["content"])

            st.markdown(f"""
                </div>{footer}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── Retrieved chunks ──────────────────────────────────────────
last = st.session_state.last_output
if last and last.get("route") == "rag" and st.session_state.show_chunks:
    results = last.get("results", [])
    if results:
        st.markdown(f"""
        <div class="chunks-section">
          <div class="chunks-header">
            <span class="chunks-title">Retrieved documents</span>
            <span class="chunks-count">{len(results)} chunks</span>
          </div>
        """, unsafe_allow_html=True)

        rq = last.get("retrieval_query", "")
        oq = last.get("original_query", "")
        if rq and rq != oq:
            st.markdown(f"""
            <div class="rewrite-note">
              <div class="rewrite-label">Query rewritten for retrieval</div>
              <div class="rewrite-text">{rq}</div>
            </div>
            """, unsafe_allow_html=True)

        for item in results:
            pct     = int(item["similarity"] * 100)
            preview = item["text"][:320] + ("…" if len(item["text"]) > 320 else "")
            st.markdown(f"""
            <div class="chunk-card">
              <div class="chunk-top">
                <span class="chunk-num">#{item['rank']}</span>
                <span class="chunk-source">{item['source']} · chunk {item['chunk_index']}</span>
                <div class="chunk-score-wrap">
                  <div class="chunk-bar-bg">
                    <div class="chunk-bar-fill" style="width:{pct}%"></div>
                  </div>
                  <span class="chunk-score-pct">{pct}%</span>
                </div>
              </div>
              <div class="chunk-text">{preview}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
# Pop pending state BEFORE rendering the text_input so value= gets the pre-fill
prefill     = st.session_state.pop("pending_query",  "") or ""
auto_submit = st.session_state.pop("submit_pending", False)

st.markdown('<div class="input-section">', unsafe_allow_html=True)

query = st.text_input(
    "q",
    value=prefill,
    placeholder="Ask a question about your knowledge base…",
    label_visibility="collapsed",
    key="qbox",
)

st.markdown('<div class="input-footer">', unsafe_allow_html=True)
c1, c2 = st.columns([7, 1])
with c1:
    st.markdown(
        '<span class="input-hint">Press Enter · click Send · or pick an example →</span>',
        unsafe_allow_html=True,
    )
with c2:
    send = st.button("Send →", key="send_btn", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # .input-section
st.markdown('</div>', unsafe_allow_html=True)  # .main-area


# ═══════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════
def _run(q: str):
    q = q.strip()
    if not q:
        return
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": q, "meta": {"time": now}})
    try:
        p = _pipe(st.session_state.strategy)
        p.memory = st.session_state.memory
        with st.spinner("Thinking…"):
            out = p.ask(query=q, top_k=st.session_state.top_k)
        st.session_state.memory = p.memory
        meta = {
            "time":            now,
            "route":           out.get("route"),
            "tool":            out.get("tool"),
            "duplicate":       out.get("duplicate", False),
            "was_summarised":  out.get("was_summarised", False),
            "retrieval_query": out.get("retrieval_query", q),
            "original_query":  q,
            "turns":           out.get("conversation_turns", 0),
        }
        st.session_state.messages.append(
            {"role": "assistant", "content": out["answer"], "meta": meta}
        )
        st.session_state.last_output = {**out, "original_query": q}
    except Exception as exc:
        err = (
            f"**Pipeline error:** `{exc}`\n\n"
            "Rebuild the index:\n"
            "```bash\npython ingest.py --strategy fixed\n"
            "python ingest.py --strategy recursive\n```"
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "meta": {"time": now, "route": "error"}}
        )
    st.rerun()


# ── Triggers ──────────────────────────────────────────────────
# 1. Manual: Send button clicked
if send and query:
    _run(query)

# 2. Auto-submit: sidebar example button was clicked.
#    `prefill` holds the full question (popped before text_input rendered),
#    `auto_submit` is True only when a sidebar example was clicked.
if auto_submit and prefill:
    _run(prefill)