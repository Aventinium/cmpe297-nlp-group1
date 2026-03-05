from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from rag_local.app_core import answer_turn, cfg_with_overrides, init_index
from rag_local.eval_rag import default_eval_items, run_rag_eval
from rag_local.openalex_client import OpenAlexClient
from rag_local.openalex_fetch import materialize_openalex_selected
from rag_local.paper_rerank import rerank_by_cosine


st.set_page_config(page_title="Local RAG Chatbot", layout="wide")


# -----------------------------
# Cached OpenAlex search (API)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)  # 1 hour cache per (query, limit, mailto)
def cached_openalex_search(query: str, limit: int, mailto: str):
    client = OpenAlexClient()
    mailto_val = mailto.strip() or None
    return client.search_works(
        query=query,
        limit=int(limit),
        mailto=mailto_val,
        open_access_only=True,  # OA only
    )


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "cfg" not in st.session_state:
        st.session_state.cfg = {}

    # OpenAlex search state
    if "oa_ranked" not in st.session_state:
        st.session_state.oa_ranked = []  # list[RankedCandidate] from paper_rerank
    if "oa_selected" not in st.session_state:
        st.session_state.oa_selected = {}  # openalex_id -> bool
    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None


_init_state()


# -----------------------------
# Sidebar: settings
# -----------------------------
st.sidebar.title("Settings")

ollama_host = st.sidebar.text_input(
    "Ollama host",
    value=st.session_state.cfg.get("ollama_host", "http://localhost:11434"),
)
chat_model = st.sidebar.text_input(
    "Chat model",
    value=st.session_state.cfg.get("chat_model", "llama3.1:8b"),
)
system_prompt = st.sidebar.text_area(
    "System prompt",
    value=st.session_state.cfg.get(
    "system_prompt",
    """You are a RAG study assistant.

Core behavior:
- Use the provided SOURCES as the primary ground truth.
- If the answer is not in the sources, say “Not found in the provided sources” and suggest what to search for.
- Explain step-by-step, but keep it tight and structured.
- When you make a claim supported by sources, cite it inline like [S1], [S2].
- Prefer definitions, then intuition, then a worked example when helpful.
- End with 2–5 quick “check yourself” questions or a mini-quiz.

Output format:
1) Answer (with citations)
2) Key takeaways (bullets)
3) Mini-quiz (2–5 questions)

Be strict about citations when RAG is enabled."""
),
    height=140,
)

st.sidebar.markdown("---")
rag_enabled = st.sidebar.toggle(
    "Enable RAG",
    value=bool(st.session_state.cfg.get("rag_enabled", True)),
)

top_k = st.sidebar.slider(
    "top_k",
    min_value=1,
    max_value=15,
    value=int(st.session_state.cfg.get("top_k", 5)),
)
max_context_chars = st.sidebar.slider(
    "max_context_chars",
    min_value=2000,
    max_value=20000,
    value=int(st.session_state.cfg.get("max_context_chars", 6000)),
    step=500,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Index")

data_dir = st.sidebar.text_input(
    "Data dir",
    value=st.session_state.cfg.get("data_dir", "rag_local/Data"),
)
index_path = st.sidebar.text_input(
    "Index path",
    value=st.session_state.cfg.get("index_path", "rag_local/Data/.index/local_index.json"),
)
chunk_size = st.sidebar.number_input(
    "chunk_size",
    min_value=200,
    max_value=3000,
    value=int(st.session_state.cfg.get("chunk_size", 800)),
    step=50,
)
overlap = st.sidebar.number_input(
    "overlap",
    min_value=0,
    max_value=1000,
    value=int(st.session_state.cfg.get("overlap", 200)),
    step=25,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Embeddings")

embed_backend = st.sidebar.selectbox(
    "embed_backend",
    options=["ollama"],
    index=0,
)
embed_model = st.sidebar.text_input(
    "embed_model",
    value=st.session_state.cfg.get("embed_model", "nomic-embed-text"),
)

st.sidebar.markdown("---")
colA, colB = st.sidebar.columns(2)
load_idx = colA.button("Load index", use_container_width=True)
rebuild_idx = colB.button("Rebuild", use_container_width=True)
clear_chat = st.sidebar.button("Clear chat", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("RAG evaluation")
eval_n = st.sidebar.slider(
    "Eval questions",
    min_value=1,
    max_value=20,
    value=3,
)
eval_use_top_k_override = st.sidebar.checkbox("Override top_k for eval", value=False)
eval_top_k = st.sidebar.slider(
    "Eval top_k",
    min_value=1,
    max_value=15,
    value=int(st.session_state.cfg.get("top_k", 5)),
    disabled=not eval_use_top_k_override,
)
run_eval_btn = st.sidebar.button("Run eval", use_container_width=True)

# Update cfg in session_state
st.session_state.cfg = cfg_with_overrides(
    st.session_state.cfg,
    ollama_host=ollama_host,
    chat_model=chat_model,
    system_prompt=system_prompt,
    rag_enabled=rag_enabled,
    top_k=top_k,
    max_context_chars=max_context_chars,
    data_dir=data_dir,
    index_path=index_path,
    chunk_size=chunk_size,
    overlap=overlap,
    embed_backend=embed_backend,
    embed_model=embed_model,
)

if clear_chat:
    st.session_state.messages = []

if load_idx:
    try:
        idx, meta = init_index(st.session_state.cfg, force_rebuild=False)
        st.session_state.index = idx
        st.sidebar.success("Index loaded.")
    except Exception as e:
        st.sidebar.error(f"Load index failed: {e}")

import os
from pathlib import Path

if rebuild_idx:
    try:
        with st.sidebar.spinner("Rebuilding index (full reset)..."):
            # 1) Clear Streamlit caches that might hold old objects/results
            st.cache_data.clear()
            st.cache_resource.clear()

            data_dir_path = Path(st.session_state.cfg["data_dir"])
            index_path_path = Path(st.session_state.cfg["index_path"])

            # 2) Count how many ingested text files exist (sanity check)
            txt_files = list(data_dir_path.rglob("*.txt"))
            pdf_files = list(data_dir_path.rglob("*.pdf"))
            st.sidebar.caption(f"Data dir scan: {len(pdf_files)} PDFs, {len(txt_files)} TXT files")

            # 3) Hard-delete existing index file so init_index cannot reuse it
            if index_path_path.exists():
                index_path_path.unlink()

            # 4) Rebuild
            idx, meta = init_index(st.session_state.cfg, force_rebuild=True)

            # 5) Update session state with the freshly built in-memory index
            st.session_state.index = idx

        # Optional: show index stats if your index structure supports it
        try:
            n_chunks = len(idx.get("chunks", [])) if isinstance(idx, dict) else None
            if n_chunks is not None:
                st.sidebar.success(f"Index rebuilt. chunks={n_chunks}")
            else:
                st.sidebar.success("Index rebuilt.")
        except Exception:
            st.sidebar.success("Index rebuilt.")

    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

if run_eval_btn:
    try:
        with st.sidebar.spinner("Running RAG evaluation..."):
            if st.session_state.index is None:
                idx, _meta = init_index(st.session_state.cfg, force_rebuild=False)
                st.session_state.index = idx

            eval_cfg = st.session_state.cfg
            if eval_use_top_k_override:
                eval_cfg = cfg_with_overrides(st.session_state.cfg, top_k=int(eval_top_k))

            items = default_eval_items()[: max(0, int(eval_n))]
            st.session_state.eval_result = run_rag_eval(
                cfg=eval_cfg,
                index=st.session_state.index,
                items=items,
            )
        st.sidebar.success("Eval complete.")
    except Exception as e:
        st.sidebar.error(f"Eval failed: {e}")


# -----------------------------
# Sidebar: Paper search (OpenAlex)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Paper search (OpenAlex)")

oa_query = st.sidebar.text_input("Search query", key="oa_query", value=st.session_state.get("oa_query", ""))
oa_mailto = st.sidebar.text_input("mailto (recommended)", key="oa_mailto", value=st.session_state.get("oa_mailto", ""))
oa_limit = st.sidebar.slider("Candidates to fetch", 25, 200, 50, step=25, key="oa_limit")
oa_show_top = st.sidebar.slider("Show top ranked", 5, 25, 10, key="oa_show_top")

c1, c2 = st.sidebar.columns(2)
oa_search_btn = c1.button("Search", key="oa_search_btn", use_container_width=True)
oa_fetch_btn = c2.button("Fetch selected", key="oa_fetch_btn", use_container_width=True)

if oa_search_btn:
    if not oa_query.strip():
        st.sidebar.warning("Enter a query.")
    else:
        try:
            with st.sidebar.spinner("OpenAlex search..."):
                works = cached_openalex_search(
                    query=oa_query.strip(),
                    limit=int(oa_limit),
                    mailto=oa_mailto,
                )

            # Convert OpenAlexClient results -> dicts for reranker
            candidates: List[Dict[str, Any]] = []
            for w in works:
                candidates.append(
                    {
                        "id": w.id,
                        "title": w.title,
                        "abstract": w.abstract,
                        "year": w.year,
                        "url": w.url,
                        "pdf_url": w.pdf_url,
                        "raw": {
                            "source": "openalex",
                            "openalex_id": w.id,
                            "landing_url": w.url,
                            "pdf_url": w.pdf_url,
                        },
                    }
                )
            from rag_local.ollama_bootstrap import preflight_ollama_for_embeddings

            status = preflight_ollama_for_embeddings(
                host=st.session_state.cfg.get("ollama_host", "http://localhost:11434"),
                embed_model=st.session_state.cfg.get("embed_model", "nomic-embed-text"),
            )
            if not status.ok:
                st.sidebar.error(status.message)
                st.stop()
            else:
                st.sidebar.caption(status.message)
            with st.sidebar.spinner("Reranking (nomic-embed-text cosine)..."):
                ranked = rerank_by_cosine(
                    query=oa_query.strip(),
                    candidates=candidates,
                    ollama_host=st.session_state.cfg.get("ollama_host", "http://localhost:11434"),
                    embed_model=st.session_state.cfg.get("embed_model", "nomic-embed-text"),
                    id_key="id",
                    year_key="year",
                    url_key="url",
                    pdf_url_key="pdf_url",
                    source="openalex",
                    top_n=int(oa_show_top),
                )

            st.session_state.oa_ranked = ranked
            st.session_state.oa_selected = {}
            st.sidebar.success(f"Found {len(works)} OA candidates. Showing top {len(ranked)} reranked.")

        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")

ranked_results = st.session_state.get("oa_ranked", [])
if ranked_results:
    st.sidebar.caption("Candidates (OA only), reranked by cosine similarity:")

    for r in ranked_results:
        key = f"oa_pick_{r.id}"
        label = f"{r.score:.3f}  |  {r.title} ({r.year})" if r.year else f"{r.score:.3f}  |  {r.title}"
        picked = st.sidebar.checkbox(label, value=bool(st.session_state.oa_selected.get(r.id, False)), key=key)
        st.session_state.oa_selected[r.id] = picked

    if oa_fetch_btn:
        chosen = [r for r in ranked_results if st.session_state.oa_selected.get(r.id, False)]
        if not chosen:
            st.sidebar.warning("Select at least one paper.")
        else:
            payloads: List[Dict[str, Any]] = []
            for r in chosen:
                payloads.append(
                    {
                        "openalex_id": r.id,
                        "title": r.title,
                        "abstract": r.abstract,
                        "year": r.year,
                        "url": r.url,
                        "pdf_url": r.pdf_url,
                        "score": r.score,
                        "raw": {
                            "source": "openalex",
                            "openalex_id": r.id,
                            "landing_url": r.url,
                            "pdf_url": r.pdf_url,
                            "score": r.score,
                        },
                    }
                )

            try:
                with st.sidebar.spinner("Fetching PDFs (PDF-only)..."):
                    saved, skipped = materialize_openalex_selected(
                        selected=payloads,
                        data_dir=Path(st.session_state.cfg["data_dir"]),
                        query=oa_query.strip(),
                    )

                if saved:
                    st.sidebar.success(f"Fetched {len(saved)} paper(s). Now click Rebuild to ingest.")
                if skipped:
                    preview = "\n".join([f"- {pid}: {reason}" for pid, reason in skipped[:6]])
                    st.sidebar.warning(f"Skipped {len(skipped)}:\n{preview}")

            except Exception as e:
                st.sidebar.error(f"Fetch failed: {e}")


# -----------------------------
# Main: chat UI
# -----------------------------
st.title("Local RAG Chatbot")
st.caption("Streamlit UI (thin layer) · Backend in rag_local/")

eval_result = st.session_state.get("eval_result")
if eval_result:
    with st.expander("RAG evaluation results", expanded=True):
        summary = eval_result.get("summary", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Correctness", f"{float(summary.get('correctness_avg', 0.0)):.2f}")
        m2.metric("Relevance", f"{float(summary.get('relevance_avg', 0.0)):.2f}")
        m3.metric("Groundedness", f"{float(summary.get('groundedness_avg', 0.0)):.2f}")
        m4.metric("Retrieval rel.", f"{float(summary.get('retrieval_relevance_avg', 0.0)):.2f}")
        m5.metric("Latency (s)", f"{float(summary.get('latency_avg_s', 0.0)):.2f}")

        rows = eval_result.get("rows", [])
        if rows:
            st.dataframe(rows, use_container_width=True)

# Try to load index on startup (non-fatal)
if st.session_state.index is None:
    try:
        idx, meta = init_index(st.session_state.cfg, force_rebuild=False)
        st.session_state.index = idx
    except Exception:
        st.session_state.index = None

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("Sources", expanded=False):
                for i, s in enumerate(m["sources"], start=1):
                    st.markdown(f"**[S{i}]** `{s.get('file','')}`")
                    st.caption(f"chunk_id={s.get('chunk_id','')} · score={s.get('score','')}")
                    snippet = s.get("snippet", "")
                    if snippet:
                        st.code(snippet[:800])

prompt = st.chat_input("Ask a question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, sources = answer_turn(
                    history=st.session_state.messages,
                    user_text=prompt,
                    cfg=st.session_state.cfg,
                    index=st.session_state.index,
                )
            except Exception as e:
                reply, sources = f"Error: {e}", []

        st.markdown(reply)

        if sources:
            with st.expander("Sources", expanded=False):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"**[S{i}]** `{s.get('file','')}`")
                    st.caption(f"chunk_id={s.get('chunk_id','')} · score={s.get('score','')}")
                    snippet = s.get("snippet", "")
                    if snippet:
                        st.code(snippet[:800])

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})