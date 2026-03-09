from __future__ import annotations

import re
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Sequence

from rag_local.app_core import answer_turn, cfg_int


@dataclass(frozen=True)
class EvalItem:
    question: str
    reference_answer: str = ""


def default_eval_items() -> List[EvalItem]:
    return [
        EvalItem(
            question="What does perplexity represent in language modeling?",
            reference_answer=(
                "Perplexity measures how well a probability model predicts a sequence. "
                "Lower perplexity generally indicates better predictive performance."
            ),
        ),
        EvalItem(
            question="What are two pretraining reasons and one post-training reason for LLM hallucination?",
            reference_answer=(
                "Possible pretraining reasons include noisy or contradictory web data and incomplete coverage. "
                "A post-training reason is reward misalignment or over-optimization for plausible-sounding responses."
            ),
        ),
        EvalItem(
            question="How do skip-gram and CBOW differ?",
            reference_answer=(
                "Skip-gram predicts surrounding context words from a center word, while CBOW predicts the center word "
                "from surrounding context words."
            ),
        ),
    ]


# -------------------------------------------------------------------
# Heuristic scoring helpers
# -------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", (text or "").lower())


def _overlap_ratio(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _contains_citation(text: str) -> bool:
    return bool(re.search(r"\[S\d+\]", text or ""))


def _score_1_to_5(x: float) -> int:
    x = max(0.0, min(1.0, x))
    if x < 0.2:
        return 1
    if x < 0.4:
        return 2
    if x < 0.6:
        return 3
    if x < 0.8:
        return 4
    return 5


def _score_correctness(answer: str, reference: str) -> int:
    if not reference.strip():
        return 0  # N/A for live conversation mode
    return _score_1_to_5(_overlap_ratio(reference, answer))


def _score_relevance(question: str, answer: str) -> int:
    return _score_1_to_5(_overlap_ratio(question, answer))


def _score_groundedness(answer: str, sources: Sequence[Dict[str, Any]]) -> int:
    if not sources:
        return 1

    source_text = "\n".join((s.get("snippet", "") or "") for s in sources)
    overlap = _overlap_ratio(answer, source_text)

    # Slight bump if answer uses explicit source citations
    if _contains_citation(answer):
        overlap = min(1.0, overlap + 0.1)

    return _score_1_to_5(overlap)


def _score_retrieval_relevance(question: str, sources: Sequence[Dict[str, Any]]) -> int:
    if not sources:
        return 1

    source_text = "\n".join((s.get("snippet", "") or "") for s in sources)
    return _score_1_to_5(_overlap_ratio(question, source_text))


# -------------------------------------------------------------------
# Benchmark / regression eval (existing style)
# -------------------------------------------------------------------
def run_rag_eval(cfg: Any, index: Any, items: Sequence[EvalItem]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for item in items:
        t0 = time.perf_counter()
        answer, sources = answer_turn(
            history=[],
            user_text=item.question,
            cfg=cfg,
            index=index,
        )
        dt = time.perf_counter() - t0

        row = {
            "question": item.question,
            "correctness": _score_correctness(answer, item.reference_answer),
            "relevance": _score_relevance(item.question, answer),
            "groundedness": _score_groundedness(answer, sources),
            "retrieval_relevance": _score_retrieval_relevance(item.question, sources),
            "latency_s": round(dt, 4),
            "answer": answer,
            "sources": list(sources),
            "mode": "benchmark",
        }
        rows.append(row)

    return _summarize_rows(rows)


# -------------------------------------------------------------------
# Live GUI conversation eval
# -------------------------------------------------------------------
def run_conversation_eval(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates actual assistant turns already present in the GUI session.

    Expected message format:
      {"role": "user"|"assistant", "content": "...", ...}
    Assistant messages may contain:
      {"sources": [...], "latency_s": float}
    """
    rows: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        if i == 0:
            continue

        # Find the nearest previous user turn
        question = ""
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "user":
                question = messages[j].get("content", "")
                break

        if not question.strip():
            continue

        answer = msg.get("content", "") or ""
        sources = msg.get("sources", []) or []
        latency_s = float(msg.get("latency_s", 0.0) or 0.0)

        row = {
            "turn_index": i,
            "question": question,
            "correctness": 0,  # no gold answer in live chat mode
            "relevance": _score_relevance(question, answer),
            "groundedness": _score_groundedness(answer, sources),
            "retrieval_relevance": _score_retrieval_relevance(question, sources),
            "citation_coverage": 1 if _contains_citation(answer) else 0,
            "latency_s": round(latency_s, 4),
            "answer": answer,
            "sources": list(sources),
            "mode": "conversation",
        }
        rows.append(row)

    return _summarize_rows(rows)


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "rows": [],
            "summary": {
                "n": 0,
                "correctness_avg": 0.0,
                "relevance_avg": 0.0,
                "groundedness_avg": 0.0,
                "retrieval_relevance_avg": 0.0,
                "citation_coverage_avg": 0.0,
                "latency_avg_s": 0.0,
            },
        }

    correctness_vals = [r["correctness"] for r in rows if r.get("correctness", 0) > 0]
    relevance_vals = [r["relevance"] for r in rows]
    groundedness_vals = [r["groundedness"] for r in rows]
    retrieval_vals = [r["retrieval_relevance"] for r in rows]
    citation_vals = [r.get("citation_coverage", 0) for r in rows]
    latency_vals = [r["latency_s"] for r in rows]

    summary = {
        "n": len(rows),
        "correctness_avg": round(mean(correctness_vals), 2) if correctness_vals else 0.0,
        "relevance_avg": round(mean(relevance_vals), 2),
        "groundedness_avg": round(mean(groundedness_vals), 2),
        "retrieval_relevance_avg": round(mean(retrieval_vals), 2),
        "citation_coverage_avg": round(mean(citation_vals), 2),
        "latency_avg_s": round(mean(latency_vals), 2) if latency_vals else 0.0,
    }
    return {"rows": list(rows), "summary": summary}


def format_eval_report(result: Dict[str, Any]) -> str:
    rows = result.get("rows", [])
    summary = result.get("summary", {})

    lines = []
    mode = rows[0].get("mode", "unknown") if rows else "unknown"
    lines.append(f"[EVAL] mode={mode}")
    lines.append("[EVAL] # | corr rel grd ret cite | latency(s) | question")
    lines.append("[EVAL] " + "-" * 80)

    for idx, row in enumerate(rows, start=1):
        q = row.get("question", "").replace("\n", " ").strip()
        if len(q) > 42:
            q = q[:39] + "..."
        lines.append(
            f"[EVAL] {idx:<2} | "
            f"{row.get('correctness', 0):<4} "
            f"{row.get('relevance', 0):<3} "
            f"{row.get('groundedness', 0):<3} "
            f"{row.get('retrieval_relevance', 0):<3} "
            f"{row.get('citation_coverage', 0):<4} | "
            f"{row.get('latency_s', 0.0):>8.2f} | "
            f"{q}"
        )

    lines.append("")
    lines.append(
        "[EVAL] averages: "
        f"correctness={summary.get('correctness_avg', 0.0):.2f}, "
        f"relevance={summary.get('relevance_avg', 0.2):.2f}, "
        f"groundedness={summary.get('groundedness_avg', 0.0):.2f}, "
        f"retrieval_relevance={summary.get('retrieval_relevance_avg', 0.0):.2f}, "
        f"citation_coverage={summary.get('citation_coverage_avg', 0.0):.2f}, "
        f"latency={summary.get('latency_avg_s', 0.0):.2f}s"
    )
    return "\n".join(lines)