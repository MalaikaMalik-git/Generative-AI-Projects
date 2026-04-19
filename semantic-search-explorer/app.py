import math
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "semantic_search_demo"
MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def distance_to_similarity(distance: float) -> float:
    return round(1 / (1 + distance), 4)


st.set_page_config(
    page_title="Semantic Search Explorer",
    page_icon="⬡",
    layout="wide"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #0a0c10 !important;
    color: #e8eaf0 !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
.main .block-container {
    background: #0a0c10 !important;
    padding-top: 1.5rem !important;
    max-width: 1200px !important;
}
[data-testid="stSidebar"] {
    background: #12151c !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * { color: #9ca3af !important; }
[data-testid="stTextInput"] input {
    background: #12151c !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-size: 15px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c6ef5 !important;
    box-shadow: 0 0 0 3px rgba(124,110,245,0.18) !important;
}
[data-testid="stTextInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: #6b7280 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetric"] {
    background: #12151c !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] p { color: #6b7280 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #a593ff !important; }
header[data-testid="stHeader"] { background: transparent !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stAlert"] {
    background: #12151c !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ Semantic Search")
    st.markdown("---")
    st.markdown("**PIPELINE**")
    for n, title, sub in [
        ("1", "Chunk documents", "Split into smaller passages"),
        ("2", "Embed chunks", "384-dim vectors via MiniLM"),
        ("3", "Store in Chroma", "Persistent vector index"),
        ("4", "Embed query", "Same model, same space"),
        ("5", "ANN search", "Return top-5 nearest chunks"),
    ]:
        st.markdown(f"**{n}.** {title}  \n*{sub}*")
    st.markdown("---")
    st.markdown("**EXAMPLE QUERIES**")
    for q in [
        "Cross-platform mobile apps?",
        "How do embeddings work?",
        "What does Chroma do?",
        "Cosine similarity?",
        "What is RAG?",
    ]:
        st.markdown(f"→ {q}")
    st.markdown("---")
    st.markdown("**WHY CHUNKING?**")
    st.markdown(
        "Full documents mix many topics into one embedding. "
        "Chunks let the system surface the exact relevant passage."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='font-family:DM Mono,monospace;font-size:11px;color:#a593ff;"
    "letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px;'>"
    "Vector Search Interface</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-family:DM Serif Display,serif;font-size:46px;line-height:1.1;"
    "color:#e8eaf0;margin-bottom:6px;'>Semantic "
    "<span style='color:#a593ff;font-style:italic;'>Search</span> Explorer</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-family:DM Mono,monospace;font-size:12px;color:#6b7280;"
    "margin-bottom:14px;'>// Embeddings × Chroma × all-MiniLM-L6-v2</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px;'>"
    "<span style='font-family:DM Mono,monospace;font-size:10px;padding:4px 12px;"
    "border-radius:100px;border:1px solid #7c6ef5;color:#a593ff;"
    "background:rgba(124,110,245,0.1);'>Chroma DB</span>"
    "<span style='font-family:DM Mono,monospace;font-size:10px;padding:4px 12px;"
    "border-radius:100px;border:1px solid #7c6ef5;color:#a593ff;"
    "background:rgba(124,110,245,0.1);'>Sentence Transformers</span>"
    "<span style='font-family:DM Mono,monospace;font-size:10px;padding:4px 12px;"
    "border-radius:100px;border:1px solid rgba(255,255,255,0.12);color:#6b7280;'>"
    "384-dim vectors</span>"
    "<span style='font-family:DM Mono,monospace;font-size:10px;padding:4px 12px;"
    "border-radius:100px;border:1px solid rgba(255,255,255,0.12);color:#6b7280;'>"
    "Cosine similarity</span></div>",
    unsafe_allow_html=True,
)

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_input("QUERY", placeholder="Example: How do embeddings work?")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Two columns ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")


# ─── RIGHT: diagram + info ────────────────────────────────────────────────────
with col_right:

    # SVG cosine diagram — rendered as HTML component
    cosine_svg = """
<div style="background:#12151c;border:1px solid rgba(255,255,255,0.08);
            border-radius:14px;padding:20px;margin-bottom:14px;">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6b7280;
              text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;">
    Cosine Similarity — Vector Space
  </div>
  <svg width="100%" viewBox="0 0 300 240" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5"
              markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
              stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
      <radialGradient id="glow" cx="50%" cy="90%" r="55%">
        <stop offset="0%" stop-color="#7c6ef5" stop-opacity="0.2"/>
        <stop offset="100%" stop-color="#7c6ef5" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <ellipse cx="150" cy="200" rx="110" ry="60" fill="url(#glow)"/>
    <circle cx="150" cy="192" r="4" fill="#7c6ef5" opacity="0.9"/>
    <text x="150" y="212" font-family="monospace" font-size="9"
          fill="#6b7280" text-anchor="middle">origin</text>
    <path d="M163 192 A18 18 0 0 0 150 175" stroke="#fbbf24" stroke-width="1"
          fill="none" stroke-dasharray="3 2" opacity="0.7"/>
    <path d="M163 192 A34 34 0 0 0 122 182" stroke="#4b5563" stroke-width="1"
          fill="none" stroke-dasharray="3 2" opacity="0.5"/>
    <line x1="150" y1="192" x2="150" y2="54" stroke="#7c6ef5" stroke-width="2"
          stroke-linecap="round" marker-end="url(#ah)"/>
    <rect x="122" y="37" width="56" height="19" rx="5"
          fill="rgba(124,110,245,0.15)" stroke="rgba(124,110,245,0.35)" stroke-width="0.8"/>
    <text x="150" y="50" font-family="monospace" font-size="9.5"
          fill="#a593ff" text-anchor="middle">Query</text>
    <line x1="150" y1="192" x2="205" y2="60" stroke="#2dd4bf" stroke-width="2"
          stroke-linecap="round" marker-end="url(#ah)"/>
    <rect x="176" y="44" width="66" height="19" rx="5"
          fill="rgba(45,212,191,0.12)" stroke="rgba(45,212,191,0.35)" stroke-width="0.8"/>
    <text x="209" y="57" font-family="monospace" font-size="9.5"
          fill="#2dd4bf" text-anchor="middle">Chunk A</text>
    <rect x="192" y="106" width="50" height="15" rx="4"
          fill="rgba(45,212,191,0.1)" stroke="rgba(45,212,191,0.25)" stroke-width="0.5"/>
    <text x="217" y="117" font-family="monospace" font-size="8.5"
          fill="#2dd4bf" text-anchor="middle">sim 0.91</text>
    <line x1="150" y1="192" x2="72" y2="88" stroke="#f472b6" stroke-width="2"
          stroke-linecap="round" opacity="0.7" marker-end="url(#ah)"/>
    <rect x="14" y="72" width="66" height="19" rx="5"
          fill="rgba(244,114,182,0.1)" stroke="rgba(244,114,182,0.3)" stroke-width="0.8"/>
    <text x="47" y="85" font-family="monospace" font-size="9.5"
          fill="#f472b6" text-anchor="middle">Chunk B</text>
    <rect x="62" y="132" width="50" height="15" rx="4"
          fill="rgba(244,114,182,0.08)" stroke="rgba(244,114,182,0.2)" stroke-width="0.5"/>
    <text x="87" y="143" font-family="monospace" font-size="8.5"
          fill="#f472b6" text-anchor="middle">sim 0.34</text>
    <text x="168" y="177" font-family="monospace" font-size="8.5"
          fill="#fbbf24" opacity="0.9">&#x3b8;&#x2081; small</text>
    <text x="108" y="174" font-family="monospace" font-size="8.5"
          fill="#6b7280" opacity="0.8" text-anchor="middle">&#x3b8;&#x2082; large</text>
  </svg>
  <div style="display:flex;gap:16px;margin-top:12px;">
    <div style="display:flex;align-items:center;gap:6px;">
      <div style="width:14px;height:2px;background:#2dd4bf;border-radius:2px;"></div>
      <span style="font-family:monospace;font-size:10px;color:#6b7280;">High similarity</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;">
      <div style="width:14px;height:2px;background:#f472b6;border-radius:2px;opacity:.7;"></div>
      <span style="font-family:monospace;font-size:10px;color:#6b7280;">Low similarity</span>
    </div>
  </div>
  <div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.07);
              font-size:12px;color:#6b7280;line-height:1.6;">
    Smaller angle &#x2192; higher cosine similarity &#x2192; more relevant result.
  </div>
</div>
"""
    st.markdown(cosine_svg, unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#12151c;border:1px solid rgba(255,255,255,0.08);"
        "border-radius:14px;padding:16px 18px;margin-bottom:14px;'>"
        "<div style='font-family:monospace;font-size:10px;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;'>How scoring works</div>"
        "<div style='font-size:12px;color:#6b7280;line-height:1.75;'>"
        "&#x2192; Each chunk is encoded as a 384-dim vector<br>"
        "&#x2192; Query is encoded in the same space<br>"
        "&#x2192; Chroma finds nearest neighbours<br>"
        "&#x2192; Score = <span style='font-family:monospace;color:#a593ff;'>1 / (1 + distance)</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='background:#12151c;border:1px solid rgba(255,255,255,0.08);"
        "border-radius:14px;padding:16px 18px;'>"
        "<div style='font-family:monospace;font-size:10px;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;'>Semantic search</div>"
        "<div style='font-size:12px;color:#6b7280;line-height:1.75;'>"
        "Finds <span style='color:#a593ff;'>meaning</span>, not just keywords. "
        "<span style='font-family:monospace;color:#e8eaf0;font-size:11px;'>"
        "&ldquo;mobile app framework&rdquo;</span> can surface a chunk about Flutter "
        "without an exact word match."
        "</div></div>",
        unsafe_allow_html=True,
    )


# ─── LEFT: Results ────────────────────────────────────────────────────────────
with col_left:
    st.markdown(
        "<div style='font-family:monospace;font-size:10px;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;'>Results</div>",
        unsafe_allow_html=True,
    )

    if not query:
        st.markdown(
            "<div style='background:#12151c;border:1px dashed rgba(255,255,255,0.12);"
            "border-radius:14px;padding:60px 24px;text-align:center;'>"
            "<div style='font-size:28px;margin-bottom:12px;opacity:.3;'>&#x2B21;</div>"
            "<div style='font-size:13px;color:#6b7280;line-height:1.7;'>"
            "Enter a query above to retrieve the top 5<br>semantically relevant chunks."
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            model = load_model()
            collection = load_collection()
            query_embedding = model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            if not documents:
                st.warning("No results found for this query.")
            else:
                # Metrics row
                best_sim = distance_to_similarity(distances[0])
                m1, m2, m3 = st.columns(3)
                m1.metric("Best similarity", f"{best_sim}")
                m2.metric("Chunks returned", len(documents))
                m3.metric("Best distance", f"{distances[0]:.4f}")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                for rank, (doc, meta, dist, item_id) in enumerate(
                        zip(documents, metadatas, distances, ids), start=1):
                    similarity = distance_to_similarity(dist)
                    pct = round(similarity * 100)

                    if rank == 1:
                        bar = "linear-gradient(180deg,#a593ff,#2dd4bf)"
                        badge_bg = "rgba(124,110,245,0.18)"
                        badge_col = "#a593ff"
                        card_border = "rgba(124,110,245,0.28)"
                    elif rank == 2:
                        bar, badge_bg, badge_col = "#7c6ef5", "rgba(124,110,245,0.1)", "#7c6ef5"
                        card_border = "rgba(255,255,255,0.1)"
                    else:
                        bar, badge_bg, badge_col = "#374151", "#1a1f2b", "#6b7280"
                        card_border = "rgba(255,255,255,0.07)"

                    score_col = "#2dd4bf" if pct >= 70 else "#fbbf24" if pct >= 45 else "#f472b6"

                    title = meta.get("title", "N/A")
                    source = meta.get("source", "N/A")
                    chunk_id = meta.get("chunk_id", item_id)

                    st.markdown(
                        f"<div style='background:#12151c;border:1px solid {card_border};"
                        f"border-radius:14px;padding:18px 20px 14px;margin-bottom:12px;"
                        f"position:relative;overflow:hidden;'>"
                        f"<div style='position:absolute;left:0;top:0;bottom:0;width:3px;"
                        f"background:{bar};border-radius:3px 0 0 3px;'></div>"
                        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>"
                        f"<div style='width:28px;height:28px;border-radius:8px;flex-shrink:0;"
                        f"background:{badge_bg};border:1px solid rgba(255,255,255,0.1);"
                        f"font-family:monospace;font-size:10px;color:{badge_col};"
                        f"display:flex;align-items:center;justify-content:center;'>#{rank}</div>"
                        f"<div style='flex:1;min-width:0;'>"
                        f"<div style='font-size:13px;font-weight:500;color:#e8eaf0;'>{title}</div>"
                        f"<div style='font-family:monospace;font-size:10px;color:#6b7280;margin-top:2px;'>"
                        f"{source} &nbsp;&middot;&nbsp; {chunk_id}</div></div>"
                        f"<div style='font-family:monospace;font-size:11px;font-weight:500;"
                        f"padding:3px 10px;border-radius:6px;flex-shrink:0;"
                        f"background:rgba(45,212,191,0.08);color:{score_col};'>{pct}%</div></div>"
                        f"<div style='font-size:13px;line-height:1.65;color:#9ca3af;"
                        f"border-top:1px solid rgba(255,255,255,0.06);padding-top:10px;'>{doc}</div>"
                        f"<div style='display:flex;gap:20px;margin-top:10px;'>"
                        f"<span style='font-family:monospace;font-size:10px;color:#4b5563;'>"
                        f"distance <span style='color:#6b7280;'>{dist:.4f}</span></span>"
                        f"<span style='font-family:monospace;font-size:10px;color:#4b5563;'>"
                        f"sim score <span style='color:#6b7280;'>{similarity}</span></span>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.markdown(
                f"<div style='background:#12151c;border:1px solid rgba(244,114,182,0.3);"
                f"border-radius:12px;padding:20px;color:#f472b6;font-size:13px;'>"
                f"&#x26a0; Error: {e}</div>",
                unsafe_allow_html=True,
            )


# ── Bottom info ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

b1, b2 = st.columns(2, gap="large")

with b1:
    st.markdown(
        "<div style='background:#12151c;border:1px solid rgba(255,255,255,0.07);"
        "border-radius:14px;padding:18px;'>"
        "<div style='font-family:monospace;font-size:10px;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;'>How embeddings work</div>"
        "<div style='font-size:12px;color:#6b7280;line-height:1.75;'>"
        "&#x2192; Each chunk converts to a 384-dim numeric vector<br>"
        "&#x2192; Vectors capture semantic meaning, not just words<br>"
        "&#x2192; Similar meanings cluster close in vector space<br>"
        "&#x2192; Cosine similarity measures closeness of two vectors"
        "</div></div>",
        unsafe_allow_html=True,
    )

with b2:
    st.markdown(
        "<div style='background:#12151c;border:1px solid rgba(255,255,255,0.07);"
        "border-radius:14px;padding:18px;'>"
        "<div style='font-family:monospace;font-size:10px;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;'>What I would improve</div>"
        "<div style='font-size:12px;color:#6b7280;line-height:1.75;'>"
        "&#x2192; Larger and more realistic corpus<br>"
        "&#x2192; Compare multiple embedding models<br>"
        "&#x2192; Hybrid keyword + semantic retrieval<br>"
        "&#x2192; Reranking for stronger final relevance<br>"
        "&#x2192; PCA / t-SNE embedding visualisation"
        "</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='text-align:center;font-family:monospace;font-size:11px;"
    "color:#374151;margin-top:20px;padding-top:20px;"
    "border-top:1px solid rgba(255,255,255,0.06);'>"
    "Model: all-MiniLM-L6-v2 &nbsp;&middot;&nbsp; DB: ChromaDB "
    "&nbsp;&middot;&nbsp; Framework: Streamlit</div>",
    unsafe_allow_html=True,
)