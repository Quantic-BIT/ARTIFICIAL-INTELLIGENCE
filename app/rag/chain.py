"""
RAG Chain Module

Combines retrieval and generation for the complete RAG pipeline.
Uses Groq for fast, free LLM inference.
"""
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.rag.vectorstore import get_vectorstore


@dataclass
class RAGResponse:
    """Response from the RAG chain."""
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    query: str


class RAGChain:
    """Complete RAG chain for policy question answering."""
    
    SYSTEM_PROMPT = """You are a helpful HR assistant for Acme Corporation. Your role is to answer questions about company policies and procedures based ONLY on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the information in the context provided
2. If the context doesn't contain enough information to answer, say "I don't have enough information in our policies to answer that question."
3. Always cite which policy document(s) your answer comes from
4. Keep answers clear, concise, and professional
5. Do not make up information or policies
6. If asked about topics outside company policies, politely redirect to relevant HR contacts

Format your response as:
- A clear, direct answer to the question
- Citations in [Source: document_name] format at the end"""

    def __init__(self, model_name: str = None, k: int = 5):
        self.model_name = model_name or os.environ.get('LLM_MODEL', 'llama-3.1-8b-instant')
        self.k = k
        self.vectorstore = get_vectorstore()
        self.groq_client = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the Groq LLM client."""
        api_key = os.environ.get('GROQ_API_KEY')
        
        if not api_key:
            print("Warning: GROQ_API_KEY not set. LLM calls will fail.")
            return
        
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=api_key)
            print(f"Groq client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("groq package is required. Install with: pip install groq")
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for i, result in enumerate(results):
            source = result['metadata'].get('source', 'Unknown')
            title = result['metadata'].get('title', 'Unknown')
            content = result['content']
            
            context_parts.append(f"[Document: {title} ({source})]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the full prompt for the LLM."""
        return f"""Based on the following policy documents, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

Please provide a helpful, accurate answer based on the policy documents above. Remember to cite your sources."""

    def _call_llm(self, prompt: str) -> str:
        """Call the Groq LLM."""
        if not self.groq_client:
            return "Error: LLM not configured. Please set GROQ_API_KEY environment variable."
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1  # Low temperature for factual responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _is_off_topic(self, query: str, results: List[Dict[str, Any]]) -> bool:
        """Check if the query seems off-topic based on retrieval scores."""
        if not results:
            return True
        
        # If best match score is very low, likely off-topic
        best_score = results[0].get('score', 0)
        return best_score < 0.3
    
    def query(self, question: str) -> RAGResponse:
        """
        Process a user question through the RAG pipeline.
        
        Args:
            question: User's question about policies
        
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        results = self.vectorstore.search(question, k=self.k)
        
        # Check for off-topic queries
        if self._is_off_topic(question, results):
            answer = ("I can only answer questions about Acme Corporation's company policies "
                     "(such as PTO, benefits, remote work, security, expenses, etc.). "
                     "For other questions, please contact HR at hr@acmecorp.com.")
            
            return RAGResponse(
                answer=answer,
                sources=[],
                latency_ms=(time.time() - start_time) * 1000,
                query=question
            )
        
        # Build context from results
        context = self._build_context(results)
        
        # Generate prompt
        prompt = self._build_prompt(question, context)
        
        # Get LLM response
        answer = self._call_llm(prompt)
        
        # Format sources
        sources = [
            {
                "source": r['metadata'].get('source', 'Unknown'),
                "title": r['metadata'].get('title', 'Unknown'),
                "snippet": r['content'][:200] + "..." if len(r['content']) > 200 else r['content'],
                "score": r.get('score', 0)
            }
            for r in results
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            query=question
        )


# Singleton instance
_rag_chain = None


def get_rag_chain() -> RAGChain:
    """Get or create the singleton RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


if __name__ == "__main__":
    # Test RAG chain
    from dotenv import load_dotenv
    load_dotenv()
    
    chain = get_rag_chain()
    
    test_questions = [
        "How many vacation days do new employees get?",
        "What are the password requirements?",
        "Can I work from home?",
        "What is the best pizza place nearby?",  # Off-topic test
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        response = chain.query(question)
        
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nLatency: {response.latency_ms:.1f}ms")
        print(f"\nSources: {[s['source'] for s in response.sources]}")
