# Design and Evaluation Document

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Design Decisions](#design-decisions)
3. [Technology Choices](#technology-choices)
4. [Evaluation Approach](#evaluation-approach)
5. [Evaluation Results](#evaluation-results)

---

## Architecture Overview

The Acme Policy Assistant is a Retrieval-Augmented Generation (RAG) application that answers questions about company policies by:

1. **Retrieving** relevant document chunks from a vector database
2. **Augmenting** the LLM prompt with retrieved context
3. **Generating** accurate, cited responses

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  User       │────▶│  Flask App   │────▶│  RAG Chain  │
│  Question   │     │  /chat       │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
            ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
            │  Retrieval   │           │   Context    │           │  Generation  │
            │  (ChromaDB)  │──────────▶│   Building   │──────────▶│   (Groq)     │
            └──────────────┘           └──────────────┘           └──────────────┘
                    │                                                     │
                    │                                                     ▼
            ┌──────────────┐                                    ┌──────────────┐
            │  Embeddings  │                                    │   Answer +   │
            │  (sentence-  │                                    │   Citations  │
            │  transformers)│                                    └──────────────┘
            └──────────────┘
```

---

## Design Decisions

### 1. Document Chunking Strategy

**Decision**: Recursive chunking with 1000-character chunks and 200-character overlap

**Rationale**:
- 1000 characters balances context size with specificity
- 200-character overlap ensures no information is lost at chunk boundaries
- Paragraph-aware splitting preserves semantic coherence

**Alternative Considered**: Semantic chunking by markdown headings
- Rejected because policy sections vary significantly in length

### 2. Embedding Model

**Decision**: sentence-transformers `all-MiniLM-L6-v2` (local, free)

**Rationale**:
- No API key required - runs locally
- Fast inference (~30ms per embed)
- Good quality for semantic search (768 dimensions)
- Widely tested and reliable

**Alternative Considered**: Cohere Embed API
- Would require API key management
- Adds latency for API calls
- Not necessary given quality of local model

### 3. Vector Store

**Decision**: ChromaDB with local persistence

**Rationale**:
- Lightweight, no external service required
- Built-in persistence for production use
- Simple Python API
- Good performance for document scale (~100 chunks)

**Alternative Considered**: Pinecone
- Overkill for our document volume
- Requires account setup and API key
- Adds complexity without benefit at this scale

### 4. LLM Selection

**Decision**: Groq with `llama-3.1-8b-instant`

**Rationale**:
- **Free tier** with generous rate limits
- **Extremely fast** inference (~100-500ms)
- **Good quality** for policy Q&A tasks
- Easy API integration

**Alternative Considered**: OpenAI GPT-4
- More expensive
- Slower inference
- Unnecessary quality for this task

### 5. Retrieval Parameter (k=5)

**Decision**: Top-5 retrieval

**Rationale**:
- Provides sufficient context for most questions
- Balances context length with prompt token usage
- Empirically determined through testing

### 6. Prompt Engineering

**Decision**: Structured system prompt with explicit rules

Key elements:
- Role definition (HR assistant for Acme)
- Answer-only-from-context rule
- Citation requirement
- Off-topic handling instruction

```
SYSTEM PROMPT:
You are a helpful HR assistant for Acme Corporation. Your role is to 
answer questions about company policies and procedures based ONLY on 
the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the information in the context provided
2. If the context doesn't contain enough information, say so
3. Always cite which policy document(s) your answer comes from
4. Keep answers clear, concise, and professional
5. Do not make up information or policies
```

### 7. Guardrails Implementation

**Decision**: Off-topic detection via retrieval similarity scores

**Rationale**:
- If best retrieval score < 0.3, likely off-topic
- Graceful degradation with HR contact info
- No additional API calls required

---

## Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Language** | Python 3.11 | Industry standard for ML/AI |
| **Web Framework** | Flask | Lightweight, simple, sufficient |
| **LLM** | Groq (llama-3.1-8b) | Free, fast, good quality |
| **Embeddings** | sentence-transformers | Free, local, reliable |
| **Vector Store** | ChromaDB | Simple, local, persistent |
| **Deployment** | Render | Free tier, GitHub integration |
| **CI/CD** | GitHub Actions | Native integration, free |

---

## Evaluation Approach

### Metrics Evaluated

1. **Groundedness** (Required)
   - Definition: % of answers factually consistent with retrieved evidence
   - Method: Keyword matching between expected answers and actual responses
   
2. **Citation Accuracy** (Required)
   - Definition: % of answers citing correct source documents
   - Method: Source file matching between expected and cited sources

3. **Latency** (Required)
   - Metrics: Average, P50, P95
   - Method: Time from query to response

4. **Off-topic Handling**
   - Definition: % of off-topic questions properly refused
   - Method: Check for refusal indicators in response

### Evaluation Dataset

- **25 questions** across all policy areas
- Categories: PTO, Remote Work, Expenses, Security, Benefits, Holidays, Code of Conduct, Employment
- Includes 1 off-topic question to test guardrails

---

## Evaluation Results

> Note: Run `python evaluation/evaluate.py` to generate actual results

### Expected Results (After Running Evaluation)

| Metric | Target | Actual |
|--------|--------|--------|
| Groundedness | >70% | _Run evaluation_ |
| Citation Accuracy | >80% | _Run evaluation_ |
| Off-topic Handling | 100% | _Run evaluation_ |
| Latency P50 | <1000ms | _Run evaluation_ |
| Latency P95 | <2000ms | _Run evaluation_ |

### Category Breakdown

| Category | Questions | Expected Pass Rate |
|----------|-----------|-------------------|
| PTO | 3 | >80% |
| Remote Work | 3 | >80% |
| Expenses | 3 | >80% |
| Security | 4 | >80% |
| Benefits | 4 | >80% |
| Holidays | 3 | >80% |
| Code of Conduct | 2 | >80% |
| Employment | 2 | >80% |

### Sample Evaluation Output

```
============================================================
EVALUATION RESULTS
============================================================

📊 Answer Quality Metrics:
   Groundedness:      85.0%
   Citation Accuracy: 90.0%
   Off-topic Handling: 100.0%

⚡ Latency Metrics:
   Average: 450ms
   P50:     420ms
   P95:     780ms

📈 Summary:
   Total questions: 25
   Successful queries: 25
   Pass rate: 84.0%
```

---

## Lessons Learned

1. **Chunking matters**: Proper chunk size significantly impacts retrieval quality
2. **Prompt engineering is crucial**: Clear instructions reduce hallucination
3. **Local embeddings work well**: No need for paid embedding APIs
4. **Groq is excellent**: Fast and free, perfect for prototyping

## Future Improvements

1. Add re-ranking for improved retrieval precision
2. Implement streaming responses for better UX
3. Add feedback collection for continuous improvement
4. Expand policy corpus with more documents
