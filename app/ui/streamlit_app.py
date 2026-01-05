# app/ui/streamlit_chatbot.py
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

from app.core.db import bootstrap_db, get_connection
from app.core.rag import bootstrap_rag, retrieve_context, format_citations
from app.core.logging_config import setup_logging

from app.core.llm_inference import generate_completion
from app.core.llm_rules import (
    PHASE1_ROUTER_SYSTEM,
    PHASE2_DB_SQL_SYSTEM,
    PHASE3_RAG_ANSWER_SYSTEM,
    PHASE4_FINAL_SYSTEM,
)

logger = setup_logging(__name__)

st.set_page_config(page_title="Fraud Chatbot", layout="wide")
st.title("🧠 Fraud Chatbot (DB + RAG)")
st.caption("Stateless. Phase1 router → Phase2 DB → Phase3 RAG → Phase4 final answer.")


# --------------------------
# Helpers: safe JSON parsing
# --------------------------
def safe_json_loads(raw: str) -> Optional[dict]:
    """
    Best-effort JSON parse:
    - tries direct json.loads
    - tries to extract first {...} block
    Returns dict or None.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try extract first JSON object block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def clamp_limit(n: int, default: int = 50, max_n: int = 200) -> int:
    if n <= 0:
        return default
    return min(n, max_n)


# --------------------------
# Artifacts (idempotent)
# --------------------------
@st.cache_resource(show_spinner=False)
def get_artifacts():
    db_path = bootstrap_db()
    rag = bootstrap_rag()
    return db_path, rag


# --------------------------
# Phase 1: Intent router
# --------------------------
def phase1_route(question: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PHASE1_ROUTER_SYSTEM},
        {"role": "user", "content": question},
    ]
    raw = generate_completion(messages, max_tokens=140, temperature=0.0, timeout_s=60, top_p=1.0)
    obj = safe_json_loads(raw) or {}
    route = obj.get("route", "both")
    if route not in ("db", "rag", "both", "general"):
        route = "both"
    return {
        "route": route,
        "confidence": float(obj.get("confidence", 0.0) or 0.0),
        "reason": obj.get("reason", "fallback"),
        "raw": raw,
    }


# --------------------------
# Phase 2: NL→SQL + execute
# --------------------------
def phase2_db(question: str, db_path) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PHASE2_DB_SQL_SYSTEM},
        {"role": "user", "content": question},
    ]
    raw = generate_completion(messages, max_tokens=2500, temperature=0.0, timeout_s=90, top_p=1.0)
    obj = safe_json_loads(raw) or {}
    sql = obj.get("sql", None)
    notes = obj.get("notes", "")

    if not sql:
        return {
            "ok": False,
            "sql": None,
            "notes": notes or "No SQL returned",
            "rows_preview": None,
            "error": None,
            "raw": raw,
        }

    # Execute SQL safely (read-only expectation)
    try:
        with get_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
            # preview: list[dict]
            preview = [dict(r) for r in rows[:50]]
    except sqlite3.Error as e:
        return {
            "ok": False,
            "sql": sql,
            "notes": notes,
            "rows_preview": None,
            "error": f"SQL error: {e}",
            "raw": raw,
        }

    return {
        "ok": True,
        "sql": sql,
        "notes": notes,
        "rows_preview": preview,
        "error": None,
        "raw": raw,
    }


# --------------------------
# Phase 3: RAG retrieve + (optional) LLM answer based on context
# --------------------------
def phase3_rag(question: str, rag, k: int = 5) -> Dict[str, Any]:
    hits = retrieve_context(rag, question, k=k)
    if not hits:
        return {
            "ok": False,
            "context": "",
            "sources": [],
            "answer": None,
            "notes": "No relevant context retrieved",
            "raw": None,
        }

    context = "\n\n---\n\n".join([t[:1200] for t, _ in hits])
    sources = format_citations([m for _, m in hits])

    # Optional: LLM summarization/answer strictly from context (Phase 3)
    messages = [
        {"role": "system", "content": PHASE3_RAG_ANSWER_SYSTEM},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
    ]
    raw = generate_completion(messages, max_tokens=1500, temperature=0.0, timeout_s=120, top_p=1.0)
    obj = safe_json_loads(raw) or {}
    ans = obj.get("answer")
    notes = obj.get("notes", "")

    return {
        "ok": True,
        "context": context,
        "sources": sources,
        "answer": ans,
        "notes": notes,
        "raw": raw,
    }


# --------------------------
# Phase 4: Final composer
# --------------------------
def phase4_final(
    question: str,
    route_decision: Dict[str, Any],
    db_out: Optional[Dict[str, Any]],
    rag_out: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Build a compact evidence blob
    evidence: Dict[str, Any] = {
        "question": question,
        "route": route_decision.get("route"),
        "route_confidence": route_decision.get("confidence"),
        "route_reason": route_decision.get("reason"),
        "db": None,
        "rag": None,
    }

    if db_out:
        evidence["db"] = {
            "ok": db_out.get("ok"),
            "sql": db_out.get("sql"),
            "notes": db_out.get("notes"),
            "error": db_out.get("error"),
            "rows_preview": db_out.get("rows_preview"),
        }

    if rag_out:
        evidence["rag"] = {
            "ok": rag_out.get("ok"),
            "notes": rag_out.get("notes"),
            "sources": rag_out.get("sources"),
            # Provide either Phase3 LLM answer or raw context (composer can use both)
            "phase3_answer": rag_out.get("answer"),
            "context": rag_out.get("context"),
        }

    messages = [
        {"role": "system", "content": PHASE4_FINAL_SYSTEM},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
    ]
    raw = generate_completion(messages, max_tokens=2500, temperature=0.0, timeout_s=180, top_p=1.0)
    obj = safe_json_loads(raw) or {}

    answer = obj.get("answer") or "Maaf, aku belum bisa menyusun jawaban dari evidence yang tersedia."
    citations = obj.get("citations") or []
    notes = obj.get("notes") or ""

    quality_score = float(obj.get("quality_score", 0.0) or 0.0)
    quality_reason = obj.get("quality_reason") or ""

    # If model forgot citations but we have RAG sources, attach them (non-hallucination)
    if not citations and rag_out and rag_out.get("sources"):
        citations = rag_out["sources"]

    return {
        "answer": answer,
        "citations": citations,
        "notes": notes,
        "quality_score": quality_score,
        "quality_reason": quality_reason,
        "raw": raw,
        "evidence": evidence,
    }


# --------------------------
# Orchestrator: Phase 1 → 4
# --------------------------
def run_pipeline(question: str, db_path, rag, k: int) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Phase 1
    route = phase1_route(question)

    # Phase 2 / 3 according to route
    db_out = None
    rag_out = None

    if route["route"] in ("db", "both"):
        db_out = phase2_db(question, db_path=db_path)
        # If Phase2 fails and route was db, fallback to RAG (your earlier plan)
        if route["route"] == "db" and not db_out.get("ok"):
            rag_out = phase3_rag(question, rag, k=k)

    if route["route"] in ("rag", "both"):
        rag_out = phase3_rag(question, rag, k=k)
        

    # Phase 4 always
    final = phase4_final(question, route, db_out=db_out, rag_out=rag_out)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "answer": final["answer"],
        "citations": final.get("citations", []),
        "mode": route["route"],
        "elapsed_ms": elapsed_ms,
        "quality_score": final.get("quality_score", 0.0),
        "quality_reason": final.get("quality_reason", ""),
        "debug": {
            "route": route,
            "db_out": db_out,
            "rag_out": rag_out,
            "final_notes": final.get("notes"),
            "final_raw": final.get("raw"),
            "evidence": final.get("evidence"),
        },
    }


# --------------------------
# UI
# --------------------------
with st.spinner("Loading artifacts (DB + RAG)..."):
    db_path, rag = get_artifacts()

with st.sidebar:
    st.header("⚙️ Settings")
    show_debug = st.checkbox("Show debug", value=True)
    k = st.slider("RAG top-k", 1, 10, 5)

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.session_state.last_debug = None
        st.session_state.last_out = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None
if "last_out" not in st.session_state:
    st.session_state.last_out = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Tulis pertanyaan kamu…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            out = run_pipeline(question, db_path=db_path, rag=rag, k=k)

            st.markdown(out["answer"])

            # show sources
            if out.get("citations"):
                with st.expander("Sources / Citations", expanded=False):
                    for c in out["citations"]:
                        st.write(f"- {c}")

            # show quality (phase4 output)
            qs = out.get("quality_score", None)
            qr = out.get("quality_reason", "")
            if qs is not None:
                st.caption(f"quality: `{qs:.2f}` — {qr}")

            # show route + timing
            st.caption(f"route: `{out['mode']}` | elapsed: `{out['elapsed_ms']:.0f} ms`")

            st.session_state.messages.append({"role": "assistant", "content": out["answer"]})
            st.session_state.last_debug = out["debug"]
            st.session_state.last_out = out  # <-- store for debug panel

if show_debug and st.session_state.last_debug:
    st.divider()
    st.subheader("🧪 Debug (last run)")

    dbg = st.session_state.last_debug
    last_out = st.session_state.last_out or {}
    qs = last_out.get("quality_score", None)
    qr = last_out.get("quality_reason", "")

    with st.expander("Phase 1: Router", expanded=True):
        st.json(dbg.get("route"))

    with st.expander("Phase 2: DB output", expanded=False):
        st.json(dbg.get("db_out"))

    with st.expander("Phase 3: RAG output", expanded=False):
        st.json(dbg.get("rag_out"))

    with st.expander("Phase 4: Evidence blob", expanded=False):
        st.json(dbg.get("evidence"))

    with st.expander("Phase 4: Raw LLM output", expanded=False):
        st.write(dbg.get("final_raw"))
        st.json({"quality_score": qs, "quality_reason": qr})
