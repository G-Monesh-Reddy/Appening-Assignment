# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from an **Agentic AI eBook**. The system uses **Gemini embeddings, Pinecone vector search, LangGraph, and Streamlit** to retrieve relevant document context and generate grounded responses.

If the retrieved context does not contain enough information to answer the question, the chatbot refuses to answer rather than generating unsupported information.

---

## Features

- PDF document ingestion and text chunking
- Semantic vector retrieval using Pinecone
- Gemini embeddings for document and query vectors
- LangGraph-based RAG workflow
- Strict context-grounded generation
- Relevance threshold for out-of-domain questions
- Top-K retrieval with configurable K
- Retrieved page numbers and similarity scores
- Streamlit interactive UI
- Out-of-domain question refusal
- Benchmark queries for evaluating grounding and retrieval

---

## Architecture

```text
                 Agentic AI eBook
                        │
                        ▼
                PDF Text Extraction
                        │
                        ▼
              Recursive Text Splitting
                        │
                        ▼
              Gemini Embeddings
                        │
                        ▼
                  Pinecone Index
                        │
                        │
                  User Question
                        │
                        ▼
                Gemini Embedding
                        │
                        ▼
               Pinecone Retrieval
                    Top-K = 5
                        │
                        ▼
                Relevance Check
                        │
             ┌──────────┴──────────┐
             │                     │
        Relevant Context      Insufficient
             │                  Context
             ▼                     ▼
       LangGraph Generate       Refusal
             │
             ▼
       Gemini LLM Response
             │
             ▼
          Streamlit UI
```

---

## Tech Stack

| Component       | Technology                     |
| --------------- | ------------------------------ |
| Language        | Python 3.10+                   |
| LLM             | Gemini                         |
| Embeddings      | `gemini-embedding-001`         |
| Vector Database | Pinecone                       |
| Retrieval       | Semantic Vector Search         |
| Orchestration   | LangGraph                      |
| PDF Processing  | PyPDF                          |
| Text Splitting  | RecursiveCharacterTextSplitter |
| UI              | Streamlit                      |
| Configuration   | python-dotenv                  |

---

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── graph.py
│   └── config.py
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── tests_sample_queries.py
```

---

## How It Works

### 1. Document Ingestion

The Agentic AI eBook is loaded from PDF and divided into smaller overlapping chunks.

Current configuration:

```text
Chunk size:       1000
Chunk overlap:    150
Total pages:      59
Total chunks:     114
```

Each chunk is converted into a vector embedding using:

```text
gemini-embedding-001
```

The vectors and associated metadata such as page number and chunk text are stored in Pinecone.

---

### 2. Query Retrieval

When a user submits a question:

1. The question is converted into an embedding.
2. Pinecone performs semantic similarity search.
3. The top 5 most relevant chunks are retrieved.
4. The highest similarity score is used as the retrieval relevance signal.

The current configuration is:

```env
TOP_K=5
MIN_RELEVANCE_SCORE=0.35
```

Top-K was set to 5 to provide sufficient context while limiting irrelevant or redundant chunks.

---

### 3. LangGraph Workflow

The RAG pipeline is orchestrated using LangGraph.

```text
START
  │
  ▼
retrieve
  │
  ▼
generate
  │
  ▼
 END
```

The graph maintains state containing the question, retrieved context, relevance score, answer, and grounding information.

---

### 4. Strict Grounding

The generation prompt instructs Gemini to answer **only using the retrieved eBook context**.

If the retrieved context is insufficient, the chatbot responds with a refusal such as:

```text
I don't have enough information in the provided Agentic AI eBook
to answer this question.
```

This prevents the model from using general world knowledge to answer questions outside the document.

---

## Out-of-Domain Handling

Retrieval and answerability are treated as separate concepts.

For example, asking:

```text
What is the capital city of Australia?
```

may still return semantically related Agentic AI chunks because Pinecone always searches for the closest vectors.

However, if those chunks do not contain enough information to answer the question, the generation layer refuses to answer.

Therefore:

```text
Semantic similarity ≠ Answer availability
```

This is important for reducing hallucination in document-grounded RAG systems.

---

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

PINECONE_INDEX_NAME=rag-agentic-ai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

GEMINI_LLM_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
MIN_RELEVANCE_SCORE=0.35
```

**Never commit `.env` or API keys to GitHub.**

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-agentic-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Pinecone Index

The project uses a Pinecone serverless index configured with:

```text
Index:      rag-agentic-ai
Dimension:  768
Metric:     cosine
Cloud:      AWS
Region:     us-east-1
```

The index contains the embeddings generated from the Agentic AI eBook.

---

## Run Ingestion

If the Pinecone index needs to be created or populated:

```bash
python -m src.ingestion
```

The ingestion pipeline:

```text
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Gemini embeddings
 ↓
Pinecone upsert
```

The current dataset contains **59 pages and 114 chunks**.

---

## Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Testing

Run the benchmark queries:

```bash
python tests_sample_queries.py
```

The benchmark includes questions such as:

```text
1. What is Agentic AI according to the eBook?

2. How do AI agents differ from traditional automation systems?

3. What are the core components of an Agentic Architecture?

4. What role does memory play in Agentic AI workflows?

5. Who won the 2022 FIFA World Cup?

6. What are the main characteristics of Agentic AI systems?
```

The fifth query is an intentional out-of-domain test. The expected behavior is refusal because the answer is not contained in the eBook.

---

## Evaluation Results

The normal Pinecone RAG pipeline was evaluated against six benchmark questions.

| Test                       | Top Pinecone Score | Result            |
| -------------------------- | -----------------: | ----------------- |
| Agentic AI definition      |             0.8680 | Grounded          |
| Agents vs automation       |             0.7514 | Grounded          |
| Agentic Architecture       |             0.7838 | Grounded          |
| Memory in Agentic AI       |             0.7863 | Grounded          |
| FIFA World Cup             |             0.4432 | Correctly refused |
| Agentic AI characteristics |             0.8250 | Grounded          |

The out-of-domain FIFA query produced a lower retrieval score and was correctly refused rather than answered using external knowledge.

---

## Why Semantic Retrieval?

The final implementation uses Pinecone semantic retrieval rather than adding a lexical BM25 retrieval layer.

A BM25 + Pinecone + Reciprocal Rank Fusion experiment was also evaluated against the benchmark queries. It did not demonstrate a consistent improvement in retrieval or answer quality for this relatively small 114-chunk document.

Therefore, the final implementation uses the simpler semantic retrieval pipeline:

```text
Gemini Embedding
       ↓
Pinecone
       ↓
Top-K Semantic Retrieval
       ↓
LangGraph
       ↓
Grounded Gemini Generation
```

This keeps the production implementation focused while avoiding unnecessary retrieval complexity.

---

## Grounding Strategy

The system uses multiple signals to prevent unsupported answers:

1. Semantic retrieval from Pinecone
2. Top-K context selection
3. Minimum relevance threshold
4. Strict generation instructions
5. Explicit refusal when the context is insufficient

The system therefore does not assume:

```text
Retrieved chunk = Answer exists
```

Instead:

```text
Retrieved chunks
      ↓
Are they sufficiently relevant?
      ↓
Can the question be answered from them?
      ↓
YES → Generate grounded answer
NO  → Refuse
```

---

## Key Design Decisions

### Top-K = 5

Five chunks provide multiple pieces of evidence while keeping the context focused.

Increasing Top-K does not necessarily improve answer quality because additional retrieved chunks can introduce irrelevant information.

### Relevance Threshold = 0.35

A minimum similarity threshold is used to identify weak retrieval results and prevent unsupported generation.

### Gemini

Gemini is used for both embeddings and generation in the final implementation.

The original assignment reference specifies OpenAI models; this implementation uses Gemini as an alternative API provider.

### Streamlit

Streamlit was selected instead of FastAPI for a lightweight interactive demonstration interface.

The UI exposes:

- Generated answer
- Retrieval relevance score
- Retrieved context
- Page numbers
- Individual chunk similarity scores

---

## Security

Do not commit credentials to the repository.

The following should remain local:

```text
.env
```

Use environment variables or deployment-platform secrets for:

```text
GEMINI_API_KEY
PINECONE_API_KEY
```

---

## Future Improvements

Potential improvements include:

- Reranking retrieved chunks
- Better answerability classification
- Query rewriting
- Citation-aware generation
- Evaluation using retrieval precision/recall
- Automated RAG evaluation metrics
- Conversation memory
- Streaming generation

These are not required for the current implementation.

---

## Assignment Alignment

The implementation covers the core requirements:

- Python-based RAG implementation
- PDF ingestion
- Chunking and embeddings
- Pinecone vector storage
- LangGraph retrieval and generation workflow
- Strict document grounding
- Streamlit interface
- Retrieved context visibility
- Relevance score visibility
- 5–6 benchmark queries
- Out-of-domain refusal
- Architecture documentation

---

## Author

**Monesh Reddy Gurram**

B.Tech Computer Science & Engineering — 2026
