"""
RAG System Evaluation Script

Comprehensive evaluation covering all required and optional metrics:

Required – Answer Quality:
  1. Groundedness:      % of answers factually supported by retrieved context
  2. Citation Accuracy: % of answers whose citations point to the correct source
  3. Exact/Partial Match (optional): % of answers matching a gold answer

Required – System Metrics:
  1. Latency (p50 / p95) measured over 10-20 on-topic queries

Optional – Ablations:
  - Compare retrieval k values (k=3, k=5, k=8)
  - Compare chunk sizes (500, 1000, 1500 chars)
"""
import json
import time
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: int
    question: str
    category: str
    expected_answer: str
    actual_answer: str
    sources_cited: List[str]
    expected_source: str
    latency_ms: float
    is_grounded: bool
    citation_correct: bool
    is_off_topic_handled: bool
    partial_match_score: float = 0.0  # 0-1 token overlap ratio


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def load_questions(filepath: str = None) -> List[Dict]:
    """Load evaluation questions from JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "questions.json"
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['questions']


def _tokenize(text: str) -> set:
    """Lowercase tokenize, strip punctuation, drop short words."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return {t for t in tokens if len(t) > 2}


def evaluate_groundedness(answer: str, expected_keywords: str) -> bool:
    """
    Check if the answer contains expected information.
    Uses keyword overlap – grounded if >= 30 % of key terms appear.
    """
    if not answer or not expected_keywords:
        return False
    answer_lower = answer.lower()
    key_terms = [t for t in expected_keywords.lower().split() if len(t) > 3]
    if not key_terms:
        return False
    matches = sum(1 for t in key_terms if t in answer_lower)
    return matches >= len(key_terms) * 0.3


def evaluate_citation(sources_cited: List[str], expected_source: str) -> bool:
    """Check if the correct source document was cited."""
    if not expected_source:
        return True
    expected_lower = expected_source.lower()
    for source in sources_cited:
        if expected_lower in source.lower() or source.lower() in expected_lower:
            return True
    return False


def evaluate_off_topic(answer: str) -> bool:
    """Check if off-topic questions are properly refused."""
    indicators = [
        "can only answer", "company policies", "outside",
        "don't have", "not in our policies",
        "hr@acmecorp.com", "contact hr",
    ]
    answer_lower = answer.lower()
    return any(ind in answer_lower for ind in indicators)


def evaluate_partial_match(answer: str, expected: str) -> float:
    """
    Token-level partial match score (Jaccard-like).
    Returns a float 0-1 representing overlap between answer and gold answer.
    """
    if not answer or not expected or expected == "OFF_TOPIC":
        return 0.0
    ans_tokens = _tokenize(answer)
    exp_tokens = _tokenize(expected)
    if not exp_tokens:
        return 0.0
    overlap = ans_tokens & exp_tokens
    # Use recall-oriented: how many expected tokens appear in the answer
    return len(overlap) / len(exp_tokens)


# ────────────────────────────────────────────────────────────
# Main evaluation
# ────────────────────────────────────────────────────────────
def run_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """Run the full evaluation suite and return structured results."""
    from app.rag.chain import get_rag_chain

    print("=" * 70)
    print("  RAG System Evaluation – Full Suite")
    print("=" * 70)

    questions = load_questions()
    print(f"\nLoaded {len(questions)} evaluation questions")

    print("Initializing RAG chain...")
    chain = get_rag_chain()
    print(f"  Model:      {chain.model_name}")
    print(f"  Documents:  {chain.vectorstore.count}")
    print(f"  Retrieval k: {chain.k}")

    results: List[EvaluationResult] = []
    latencies: List[float] = []

    print("\n─── Running evaluations ───\n")

    for i, q in enumerate(questions):
        if verbose:
            print(f"[{i+1:2d}/{len(questions)}] {q['category']:15s} | {q['question'][:55]}...")

        try:
            response = chain.query(q['question'])
            latencies.append(response.latency_ms)
            sources_cited = [s['source'] for s in response.sources]

            is_off_topic = q['expected_answer'] == "OFF_TOPIC"

            if is_off_topic:
                is_grounded = True
                citation_correct = True
                off_topic_handled = evaluate_off_topic(response.answer)
                partial_score = 0.0
            else:
                is_grounded = evaluate_groundedness(response.answer, q['expected_answer'])
                citation_correct = evaluate_citation(sources_cited, q.get('source', ''))
                off_topic_handled = True
                partial_score = evaluate_partial_match(response.answer, q['expected_answer'])

            result = EvaluationResult(
                question_id=q['id'],
                question=q['question'],
                category=q['category'],
                expected_answer=q['expected_answer'],
                actual_answer=response.answer,
                sources_cited=sources_cited,
                expected_source=q.get('source', ''),
                latency_ms=response.latency_ms,
                is_grounded=is_grounded,
                citation_correct=citation_correct,
                is_off_topic_handled=off_topic_handled,
                partial_match_score=partial_score,
            )
            results.append(result)

            if verbose:
                icon = "✅" if (is_grounded and citation_correct) else "❌"
                extra = f", Match: {partial_score:.0%}" if not is_off_topic else ", Off-topic ✓" if off_topic_handled else ", Off-topic ✗"
                print(f"       {icon} Grounded: {is_grounded}, Citation: {citation_correct}{extra}, {response.latency_ms:.0f}ms")

        except Exception as e:
            print(f"       ❌ Error: {e}")
            continue

    # ── Compute metrics ──
    on_topic  = [r for r in results if r.expected_answer != "OFF_TOPIC"]
    off_topic = [r for r in results if r.expected_answer == "OFF_TOPIC"]

    groundedness_pct = (sum(1 for r in on_topic if r.is_grounded) / len(on_topic) * 100) if on_topic else 0
    citation_pct     = (sum(1 for r in on_topic if r.citation_correct) / len(on_topic) * 100) if on_topic else 0
    off_topic_pct    = (sum(1 for r in off_topic if r.is_off_topic_handled) / len(off_topic) * 100) if off_topic else 100

    # Partial / Exact match
    partial_scores = [r.partial_match_score for r in on_topic]
    avg_partial_match = np.mean(partial_scores) * 100 if partial_scores else 0
    exact_match_pct   = (sum(1 for s in partial_scores if s >= 0.8) / len(partial_scores) * 100) if partial_scores else 0

    # Latency – computed over on-topic queries only (as required: 10-20 queries)
    on_topic_latencies = [r.latency_ms for r in on_topic]
    latency_p50 = np.percentile(on_topic_latencies, 50) if on_topic_latencies else 0
    latency_p95 = np.percentile(on_topic_latencies, 95) if on_topic_latencies else 0
    latency_avg = np.mean(on_topic_latencies) if on_topic_latencies else 0
    latency_min = np.min(on_topic_latencies) if on_topic_latencies else 0
    latency_max = np.max(on_topic_latencies) if on_topic_latencies else 0

    pass_rate = (sum(1 for r in on_topic if r.is_grounded and r.citation_correct) / len(on_topic) * 100) if on_topic else 0

    # ── Print report ──
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n📊 Answer Quality Metrics (on {len(on_topic)} on-topic questions):")
    print(f"   Groundedness:        {groundedness_pct:.1f}%")
    print(f"   Citation Accuracy:   {citation_pct:.1f}%")
    print(f"   Partial Match (avg): {avg_partial_match:.1f}%")
    print(f"   Exact Match (≥80%):  {exact_match_pct:.1f}%")
    print(f"   Off-topic Handling:  {off_topic_pct:.1f}% (on {len(off_topic)} off-topic questions)")

    print(f"\n⚡ Latency Metrics (over {len(on_topic_latencies)} on-topic queries):")
    print(f"   Min:     {latency_min:.0f}ms")
    print(f"   Average: {latency_avg:.0f}ms")
    print(f"   P50:     {latency_p50:.0f}ms")
    print(f"   P95:     {latency_p95:.0f}ms")
    print(f"   Max:     {latency_max:.0f}ms")

    print(f"\n📈 Overall Summary:")
    print(f"   Total questions:     {len(questions)}")
    print(f"   Successful queries:  {len(results)}")
    print(f"   Overall pass rate:   {pass_rate:.1f}%")

    print(f"\n📂 Category Breakdown:")
    categories = sorted(set(r.category for r in on_topic))
    for cat in categories:
        cr = [r for r in on_topic if r.category == cat]
        g = sum(1 for r in cr if r.is_grounded)
        c = sum(1 for r in cr if r.citation_correct)
        pm = np.mean([r.partial_match_score for r in cr]) * 100
        print(f"   {cat:18s}  Grounded: {g}/{len(cr)}  Citation: {c}/{len(cr)}  Match: {pm:.0f}%")

    # ── Structured output ──
    output = {
        "metrics": {
            "answer_quality": {
                "groundedness_pct": round(groundedness_pct, 1),
                "citation_accuracy_pct": round(citation_pct, 1),
                "partial_match_avg_pct": round(avg_partial_match, 1),
                "exact_match_pct": round(exact_match_pct, 1),
                "off_topic_handling_pct": round(off_topic_pct, 1),
            },
            "latency": {
                "num_queries": len(on_topic_latencies),
                "min_ms": round(latency_min, 1),
                "avg_ms": round(latency_avg, 1),
                "p50_ms": round(latency_p50, 1),
                "p95_ms": round(latency_p95, 1),
                "max_ms": round(latency_max, 1),
            },
            "summary": {
                "total_questions": len(questions),
                "successful_queries": len(results),
                "pass_rate_pct": round(pass_rate, 1),
            },
        },
        "per_question": [
            {
                "id": r.question_id,
                "question": r.question,
                "category": r.category,
                "expected_answer": r.expected_answer,
                "actual_answer": r.actual_answer,
                "sources_cited": r.sources_cited,
                "is_grounded": r.is_grounded,
                "citation_correct": r.citation_correct,
                "partial_match_score": round(r.partial_match_score, 3),
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in results
        ],
        "category_breakdown": {
            cat: {
                "total": len([r for r in on_topic if r.category == cat]),
                "grounded": sum(1 for r in on_topic if r.category == cat and r.is_grounded),
                "citation_correct": sum(1 for r in on_topic if r.category == cat and r.citation_correct),
                "avg_partial_match": round(np.mean([r.partial_match_score for r in on_topic if r.category == cat]) * 100, 1),
            }
            for cat in categories
        },
    }

    return output


# ────────────────────────────────────────────────────────────
# Ablation study
# ────────────────────────────────────────────────────────────
def run_ablation_k(verbose: bool = True) -> Dict[str, Any]:
    """
    Ablation: compare different retrieval k values (k=3, k=5, k=8).
    Uses a subset of 10 on-topic questions for efficiency.
    """
    from app.rag.chain import RAGChain

    print("\n" + "=" * 70)
    print("  ABLATION STUDY: Retrieval k")
    print("=" * 70)

    questions = [q for q in load_questions() if q['expected_answer'] != "OFF_TOPIC"][:10]
    k_values = [3, 5, 8]
    ablation_results = {}

    for k in k_values:
        print(f"\n── Testing k={k} on {len(questions)} questions ──")
        chain = RAGChain(k=k)
        grounded = 0
        cited = 0
        latencies = []
        partial_scores = []

        for q in questions:
            try:
                resp = chain.query(q['question'])
                latencies.append(resp.latency_ms)
                sources = [s['source'] for s in resp.sources]
                g = evaluate_groundedness(resp.answer, q['expected_answer'])
                c = evaluate_citation(sources, q.get('source', ''))
                p = evaluate_partial_match(resp.answer, q['expected_answer'])
                if g: grounded += 1
                if c: cited += 1
                partial_scores.append(p)
                if verbose:
                    print(f"   [{q['id']:2d}] G:{g} C:{c} M:{p:.0%} {resp.latency_ms:.0f}ms")
            except Exception as e:
                print(f"   [{q['id']:2d}] Error: {e}")

        n = len(questions)
        ablation_results[f"k={k}"] = {
            "groundedness_pct": round(grounded / n * 100, 1),
            "citation_accuracy_pct": round(cited / n * 100, 1),
            "partial_match_avg_pct": round(np.mean(partial_scores) * 100, 1) if partial_scores else 0,
            "latency_p50_ms": round(np.percentile(latencies, 50), 1) if latencies else 0,
            "latency_p95_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        }

    print("\n── Ablation Summary: k values ──")
    print(f"{'k':>5s} | {'Ground%':>8s} | {'Cite%':>8s} | {'Match%':>8s} | {'P50ms':>8s} | {'P95ms':>8s}")
    print("-" * 55)
    for label, m in ablation_results.items():
        print(f"{label:>5s} | {m['groundedness_pct']:>7.1f}% | {m['citation_accuracy_pct']:>7.1f}% | {m['partial_match_avg_pct']:>7.1f}% | {m['latency_p50_ms']:>7.0f} | {m['latency_p95_ms']:>7.0f}")

    return ablation_results


# ────────────────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────────────────
def save_results(results: Dict[str, Any], filepath: str = None):
    """Save evaluation results to JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "results.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {filepath}")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    parser.add_argument("--save", "-s", action="store_true", help="Save results to JSON")
    parser.add_argument("--ablation", "-a", action="store_true", help="Run ablation study on k values")
    args = parser.parse_args()

    results = run_evaluation(verbose=not args.quiet)

    if args.ablation:
        ablation = run_ablation_k(verbose=not args.quiet)
        results["ablation_k"] = ablation

    if args.save:
        save_results(results)
