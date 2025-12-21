"""
RAG System Evaluation Script

Evaluates the RAG system on:
- Groundedness: Are answers supported by retrieved context?
- Citation Accuracy: Do citations correctly point to source documents?
- Latency: Response time metrics (p50, p95)
"""
import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


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
    is_grounded: bool  # Answer is supported by context
    citation_correct: bool  # Correct source cited
    is_off_topic_handled: bool  # For off-topic questions


def load_questions(filepath: str = None) -> List[Dict]:
    """Load evaluation questions from JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "questions.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data['questions']


def evaluate_groundedness(answer: str, expected_keywords: str) -> bool:
    """
    Check if the answer contains expected information.
    Simple keyword matching for evaluation.
    """
    if not answer or not expected_keywords:
        return False
    
    answer_lower = answer.lower()
    expected_lower = expected_keywords.lower()
    
    # Check for key terms from expected answer
    key_terms = [term.strip() for term in expected_lower.split() if len(term) > 3]
    matches = sum(1 for term in key_terms if term in answer_lower)
    
    # Consider grounded if at least 30% of key terms match
    return matches >= len(key_terms) * 0.3 if key_terms else False


def evaluate_citation(sources_cited: List[str], expected_source: str) -> bool:
    """Check if the correct source was cited."""
    if not expected_source:
        return True  # No expected source for off-topic
    
    expected_lower = expected_source.lower()
    
    for source in sources_cited:
        if expected_lower in source.lower() or source.lower() in expected_lower:
            return True
    
    return False


def evaluate_off_topic(answer: str) -> bool:
    """Check if off-topic questions are properly handled."""
    off_topic_indicators = [
        "can only answer",
        "company policies",
        "outside",
        "don't have",
        "not in our policies",
        "hr@acmecorp.com",
        "contact hr"
    ]
    
    answer_lower = answer.lower()
    return any(indicator in answer_lower for indicator in off_topic_indicators)


def run_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """
    Run full evaluation on the RAG system.
    
    Returns:
        Dictionary with evaluation metrics and results
    """
    from app.rag.chain import get_rag_chain
    
    print("=" * 60)
    print("RAG System Evaluation")
    print("=" * 60)
    
    # Load questions
    questions = load_questions()
    print(f"\nLoaded {len(questions)} evaluation questions")
    
    # Initialize RAG chain
    print("Initializing RAG chain...")
    chain = get_rag_chain()
    print(f"Using model: {chain.model_name}")
    print(f"Documents indexed: {chain.vectorstore.count}")
    
    # Run evaluations
    results: List[EvaluationResult] = []
    latencies: List[float] = []
    
    print("\nRunning evaluations...\n")
    
    for i, q in enumerate(questions):
        if verbose:
            print(f"[{i+1}/{len(questions)}] {q['category']}: {q['question'][:50]}...")
        
        try:
            # Query the RAG system
            response = chain.query(q['question'])
            
            latencies.append(response.latency_ms)
            sources_cited = [s['source'] for s in response.sources]
            
            # Evaluate based on question type
            is_off_topic = q['expected_answer'] == "OFF_TOPIC"
            
            if is_off_topic:
                is_grounded = True  # N/A for off-topic
                citation_correct = True  # N/A for off-topic
                off_topic_handled = evaluate_off_topic(response.answer)
            else:
                is_grounded = evaluate_groundedness(response.answer, q['expected_answer'])
                citation_correct = evaluate_citation(sources_cited, q.get('source', ''))
                off_topic_handled = True  # N/A for on-topic
            
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
                is_off_topic_handled=off_topic_handled
            )
            results.append(result)
            
            if verbose:
                status = "✅" if (is_grounded and citation_correct) else "❌"
                print(f"    {status} Grounded: {is_grounded}, Citation: {citation_correct}, Latency: {response.latency_ms:.0f}ms")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    # Calculate metrics
    on_topic_results = [r for r in results if r.expected_answer != "OFF_TOPIC"]
    off_topic_results = [r for r in results if r.expected_answer == "OFF_TOPIC"]
    
    groundedness_pct = (sum(1 for r in on_topic_results if r.is_grounded) / len(on_topic_results) * 100) if on_topic_results else 0
    citation_pct = (sum(1 for r in on_topic_results if r.citation_correct) / len(on_topic_results) * 100) if on_topic_results else 0
    off_topic_pct = (sum(1 for r in off_topic_results if r.is_off_topic_handled) / len(off_topic_results) * 100) if off_topic_results else 100
    
    latency_p50 = np.percentile(latencies, 50) if latencies else 0
    latency_p95 = np.percentile(latencies, 95) if latencies else 0
    latency_avg = np.mean(latencies) if latencies else 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n📊 Answer Quality Metrics:")
    print(f"   Groundedness:      {groundedness_pct:.1f}%")
    print(f"   Citation Accuracy: {citation_pct:.1f}%")
    print(f"   Off-topic Handling: {off_topic_pct:.1f}%")
    
    print(f"\n⚡ Latency Metrics:")
    print(f"   Average: {latency_avg:.0f}ms")
    print(f"   P50:     {latency_p50:.0f}ms")
    print(f"   P95:     {latency_p95:.0f}ms")
    
    print(f"\n📈 Summary:")
    print(f"   Total questions: {len(questions)}")
    print(f"   Successful queries: {len(results)}")
    print(f"   Pass rate: {sum(1 for r in on_topic_results if r.is_grounded and r.citation_correct) / len(on_topic_results) * 100:.1f}%")
    
    # Category breakdown
    print(f"\n📂 By Category:")
    categories = set(r.category for r in on_topic_results)
    for cat in sorted(categories):
        cat_results = [r for r in on_topic_results if r.category == cat]
        cat_pass = sum(1 for r in cat_results if r.is_grounded)
        print(f"   {cat}: {cat_pass}/{len(cat_results)} ({cat_pass/len(cat_results)*100:.0f}%)")
    
    # Return structured results
    return {
        "metrics": {
            "groundedness_pct": groundedness_pct,
            "citation_accuracy_pct": citation_pct,
            "off_topic_handling_pct": off_topic_pct,
            "latency_avg_ms": latency_avg,
            "latency_p50_ms": latency_p50,
            "latency_p95_ms": latency_p95,
            "total_questions": len(questions),
            "successful_queries": len(results)
        },
        "results": [
            {
                "id": r.question_id,
                "question": r.question,
                "category": r.category,
                "is_grounded": r.is_grounded,
                "citation_correct": r.citation_correct,
                "latency_ms": r.latency_ms
            }
            for r in results
        ]
    }


def save_results(results: Dict[str, Any], filepath: str = None):
    """Save evaluation results to JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "results.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    parser.add_argument("--save", "-s", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    results = run_evaluation(verbose=not args.quiet)
    
    if args.save:
        save_results(results)
