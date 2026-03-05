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
    reference_answer: str


@dataclass(frozen=True)
class EvalRow:
    question: str
    correctness: int
    relevance: int
    groundedness: int
    retrieval_relevance: int
    latency_s: float


def default_eval_items() -> List[EvalItem]:
    """
    Small deterministic eval set derived from local course-note files in rag_local/Data.
    This keeps evaluation lightweight and runnable inside `cmpe297-chat`.
    """
    return [
        EvalItem(
            question="What does perplexity represent in language modeling?",
            reference_answer=(
                "Perplexity is the exponential of cross-entropy loss and indicates "
                "how many choices the model feels it has on average."
            ),
        ),
        EvalItem(
            question="What are two pretraining reasons and one post-training reason for LLM hallucination?",
            reference_answer=(
                "In pretraining, hallucinations come from statistical inevitability "
                "and objective mismatch. In post-training, preference signals often "
                "reward answering over abstaining."
            ),
        ),
        EvalItem(
            question="How do skip-gram and CBOW differ?",
            reference_answer=(
                "Skip-gram predicts surrounding words from a center word, while CBOW "
                "predicts the center word from surrounding context words."
            ),
        ),
    ]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "while",
    "with",
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _keywords(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 2]


def _coverage_score(target: str, candidate: str) -> int:
    target_kws = set(_keywords(target))
    if not target_kws:
        return 0
    cand = set(_tokens(candidate))
    coverage = len(target_kws.intersection(cand)) / len(target_kws)

    if coverage >= 0.95:
        return 5
    if coverage >= 0.75:
        return 4
    if coverage >= 0.50:
        return 3
    if coverage >= 0.25:
        return 2
    if coverage > 0:
        return 1
    return 0


def run_rag_eval(*, cfg: Any, index: Any, items: Sequence[EvalItem] | None = None) -> Dict[str, Any]:
    eval_items = list(items or default_eval_items())
    if not eval_items:
        return {"rows": [], "summary": {}}

    rows: List[EvalRow] = []
    top_k = cfg_int(cfg, "top_k", 5)

    for it in eval_items:
        start = time.perf_counter()
        answer, _sources = answer_turn(history=[], user_text=it.question, cfg=cfg, index=index)
        latency_s = time.perf_counter() - start

        retrieved = index.search(it.question, top_k=top_k)
        retrieved_text = "\n".join((r.get("text") or "") for r in retrieved)

        correctness = _coverage_score(it.reference_answer, answer)
        relevance = _coverage_score(it.question, answer)
        groundedness = _coverage_score(answer, retrieved_text)
        retrieval_relevance = _coverage_score(it.reference_answer, retrieved_text)

        rows.append(
            EvalRow(
                question=it.question,
                correctness=correctness,
                relevance=relevance,
                groundedness=groundedness,
                retrieval_relevance=retrieval_relevance,
                latency_s=latency_s,
            )
        )

    summary = {
        "n": len(rows),
        "correctness_avg": mean(r.correctness for r in rows),
        "relevance_avg": mean(r.relevance for r in rows),
        "groundedness_avg": mean(r.groundedness for r in rows),
        "retrieval_relevance_avg": mean(r.retrieval_relevance for r in rows),
        "latency_avg_s": mean(r.latency_s for r in rows),
    }

    return {
        "rows": [
            {
                "question": r.question,
                "correctness": r.correctness,
                "relevance": r.relevance,
                "groundedness": r.groundedness,
                "retrieval_relevance": r.retrieval_relevance,
                "latency_s": r.latency_s,
            }
            for r in rows
        ],
        "summary": summary,
    }


def format_eval_report(eval_result: Dict[str, Any]) -> str:
    rows = eval_result.get("rows", [])
    summary = eval_result.get("summary", {})
    if not rows:
        return "[EVAL] No evaluation rows."

    lines: List[str] = []
    lines.append("[EVAL] RAG quick evaluation")
    lines.append("[EVAL] q# | corr rel grd ret | latency(s) | question")
    lines.append("[EVAL] ---+------------------+------------+-------------------------------")

    for i, r in enumerate(rows, start=1):
        q = (r.get("question") or "").strip().replace("\n", " ")
        if len(q) > 40:
            q = q[:37].rstrip() + "..."
        lines.append(
            f"[EVAL] {i:>2} |  {int(r.get('correctness', 0))}    {int(r.get('relevance', 0))}   {int(r.get('groundedness', 0))}   {int(r.get('retrieval_relevance', 0))}  |"
            f"   {float(r.get('latency_s', 0.0)):.2f}     | {q}"
        )

    lines.append("[EVAL]")
    lines.append(
        "[EVAL] averages: "
        f"correctness={summary.get('correctness_avg', 0.0):.2f}, "
        f"relevance={summary.get('relevance_avg', 0.0):.2f}, "
        f"groundedness={summary.get('groundedness_avg', 0.0):.2f}, "
        f"retrieval_relevance={summary.get('retrieval_relevance_avg', 0.0):.2f}, "
        f"latency={summary.get('latency_avg_s', 0.0):.2f}s"
    )
    return "\n".join(lines)
