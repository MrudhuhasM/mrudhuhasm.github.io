---
title: "Building a Production-Grade Agentic RAG System"
date: 2024-12-01
tags: [rag, langgraph, nlp, llm, vector-databases]
description: "Deep dive into building a production-ready agentic RAG system with hybrid search, multi-step reasoning, and intelligent answer validation"
math: true
---
dg-home: false
dg-publish: true
dg-permalink: building-production-grade-agentic-rag-system
description: Deep dive into building a production-ready agentic RAG system with hybrid search, multi-step reasoning, and intelligent answer validation
---

# Building a Production-Grade Agentic RAG System

Basic RAG implementations—embed query, retrieve documents, pass to LLM, return response—break down when applied to complex domains requiring nuanced understanding. The standard pipeline provides no quality control, making it unsuitable for querying philosophical texts where accuracy and grounding matter.

Building a RAG system for Marcus Aurelius' "Meditations" exposed these limitations clearly. Stoic philosophy requires synthesizing insights across multiple passages and careful interpretation of context. A simple retrieval pipeline would either miss relevant passages or return superficially similar but contextually wrong content.

The solution is an agentic architecture that treats retrieval and generation as iterative processes with built-in quality validation. This system uses LangGraph to orchestrate multi-step reasoning, hybrid search to improve retrieval, and an evaluation loop to catch hallucinations before they reach users.

**Live Demo:** [https://meditations-rag-180347172582.asia-south1.run.app](https://meditations-rag-180347172582.asia-south1.run.app)
**GitHub:** [https://github.com/MrudhuhasM/meditations-rag](https://github.com/MrudhuhasM/meditations-rag)

## Agentic Architecture with LangGraph

The system operates through four specialized nodes that form a feedback loop rather than a linear pipeline.

### Controller Node
Analyzes incoming queries to determine retrieval strategy. For factual lookups ("What does Marcus say about anger?"), the controller routes directly to dense vector search. For philosophical synthesis ("How do Stoic principles apply to modern work?"), it enables hybrid search and sets higher thresholds for the evaluator.

### Retriever Node: Hybrid Search

The retriever combines two search methods through weighted score fusion:

**Dense Vector Search** uses semantic embeddings to find conceptually similar passages. This captures meaning beyond exact keyword matches—"dealing with difficult people" retrieves passages about "managing relationships" even without shared vocabulary.

**Question-Based Retrieval** matches user queries against synthetic questions generated during document ingestion. For each text chunk, an LLM generates questions the passage could answer. User query "How should I handle anger?" matches generated question "What techniques does Marcus suggest for managing anger?"

Score fusion combines both methods:

```python
final_score = (alpha * dense_score) + (beta * question_score)
```

Setting $\alpha = 0.6$ and $\beta = 0.4$ weights semantic similarity higher while still capturing question-format matches. This prevents missing passages where users phrase questions differently than the source text.

### Generator Node
Synthesizes information across retrieved passages while maintaining source attribution. The generator receives ranked passages and must cite specific text when making claims. This constraint reduces hallucination by forcing grounding in retrieved content.

### Evaluator Node

The evaluator validates three properties:
- **Groundedness**: Claims must be supported by retrieved passages
- **Relevance**: Response must address the query
- **Completeness**: All query aspects must be covered

Evaluation uses an LLM to assess these properties against retrieved context. If any check fails, the system can request additional context, reformulate the response, or return to the controller for a different strategy. This feedback loop catches roughly 15-20% of responses that would otherwise be irrelevant or hallucinated.

## Production Engineering

Deploying RAG systems exposes issues invisible during local development. Rate limiting prevents abuse, authentication controls access, and structured logging enables debugging production failures.

### Rate Limiting
SlowAPI implements token bucket rate limiting at 10 requests per minute per IP:

```python
@limiter.limit("10/minute")
async def query_endpoint(request: Request):
    # Handle query
    pass
```

This prevents resource exhaustion while allowing burst traffic from legitimate users.

### Authentication
Middleware validates API keys before processing requests. The system supports three tiers: demo keys (limited queries), authenticated keys (standard rate limits), and premium keys (higher limits, priority queue). Usage tracking per key enables cost analysis and abuse detection.

### Logging
Structured logging captures query metadata for debugging and performance analysis:

```python
logger.info(
    "Query processed",
    extra={
        "query": query_text,
        "retrieval_time_ms": retrieval_time,
        "num_chunks": len(chunks),
        "evaluation_score": eval_score
    }
)
```

This data reveals retrieval patterns, identifies slow queries, and tracks evaluation scores over time.

### Error Handling
The system handles three failure modes: LLM service unavailability triggers fallback to alternative providers (OpenAI → Gemini → local models), vector DB connection failures serve cached responses when available, and malformed queries return clear error messages with suggested reformulations.

---

## Data Processing: Semantic Chunking and Metadata

Retrieval quality depends on how text is chunked during ingestion. Fixed-size chunks (512 tokens) break mid-sentence or mid-thought, degrading semantic coherence. Semantic chunking addresses this by grouping sentences based on embedding similarity.

### Semantic Chunking

The algorithm:
1. Split text into sentences
2. Generate embeddings for each sentence
3. Compute cosine similarity between adjacent sentences
4. Group sentences where similarity $> 0.7$
5. Create chunks from groups

This produces chunks that maintain topical coherence. A passage discussing anger management stays together rather than being split across chunks based on arbitrary token counts.

### Metadata Extraction

For each chunk, an LLM extracts structured metadata:

```python
metadata = {
    "book": 3,
    "chapter": 12,
    "topics": ["anger", "self-control", "virtue"],
    "philosophical_concepts": ["Stoicism", "reason"],
    "synthetic_questions": [
        "How should one handle anger?",
        "What does Marcus say about self-control?"
    ]
}
```

This metadata enables filtered retrieval ("passages about virtue from Book 3"), question matching (user query → synthetic questions), and topic exploration. The upfront cost of metadata extraction—about 2 seconds per chunk using GPT-3.5—pays off through improved retrieval precision.

## Implementation Details

**LangGraph** orchestrates the agentic workflow with state management. Each node modifies shared state containing the query, retrieved passages, generated response, and evaluation scores.

**Qdrant** serves as the vector database, providing sub-100ms similarity search with metadata filtering. The system stores ~500 chunks from "Meditations" with 1536-dimensional embeddings (OpenAI ada-002).

**Language model support** spans cloud providers (OpenAI GPT-4/3.5, Gemini) and local inference (llama.cpp, vLLM). Multi-provider support enables cost/quality tradeoffs: GPT-4 for generation, GPT-3.5 for evaluation, local models for development.

**FastAPI** handles HTTP requests with async support. Docker containerization enables reproducible deployments across development and production environments.

## Lessons from Building Production RAG

**Hybrid search is essential.** Pure semantic search misses exact matches; pure keyword search misses conceptual similarity. Weighted fusion delivers better results than either method alone.

**Evaluation catches hallucinations.** The evaluator node catches 15-20% of responses that would have been irrelevant or hallucinated. This quality gate justifies the extra latency.

**Metadata drives retrieval quality.** Rich metadata transforms retrieval from "find similar text" to "find relevant knowledge." The investment in metadata extraction—2 seconds per chunk during ingestion—pays off in retrieval precision.

**Simplicity wins.** The initial workflow had 7 nodes. Testing revealed that 4 nodes cover essential decision points without unnecessary complexity. Over-engineering the agent introduced latency without improving quality.

**Production features require time.** Rate limiting, authentication, logging, and error handling consumed 30% of development time but differentiate a usable system from a demo.

## Performance Characteristics

**Query Latency:**
- p50: 2.5 seconds
- p95: 4.5 seconds
- p99: 7 seconds

**Retrieval Accuracy (evaluated on 100 test queries):**
- Relevant chunks in top 5: 87%
- Answer grounded in sources: 92%
- Answer addresses question: 89%

**Scalability:**
The stateless API design enables horizontal scaling. Current deployment handles 50 concurrent requests with Qdrant providing sub-100ms vector search.

## Future Work

**Multi-Document Support:** Extend beyond "Meditations" to query across Stoic texts (Epictetus, Seneca) with comparative analysis.

**Conversation Memory:** Maintain dialogue history to handle follow-up questions and enable deeper exploration.

**Citation Visualization:** Build UI showing which passages contributed to each part of the answer.

**Fine-Tuned Embeddings:** Train a custom embedding model on philosophical texts for improved semantic understanding.

**Feedback Loop:** Collect user ratings on answer quality to improve evaluation criteria.

## Conclusion

Moving from basic RAG to production systems requires addressing quality control, retrieval precision, and operational reliability. The agentic architecture with LangGraph enables multi-step reasoning and evaluation loops that catch hallucinations. Hybrid search improves retrieval accuracy. Production engineering—rate limiting, authentication, logging, error handling—makes the system deployable.

The complete implementation is available at:

**Live Demo:** [https://meditations-rag-180347172582.asia-south1.run.app](https://meditations-rag-180347172582.asia-south1.run.app)
**GitHub:** [https://github.com/MrudhuhasM/meditations-rag](https://github.com/MrudhuhasM/meditations-rag)
