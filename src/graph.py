from typing import TypedDict, List, Dict, Any

from google import genai
from google.genai import types

from pinecone import Pinecone

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from src.config import (
    GEMINI_API_KEY,
    GEMINI_LLM_MODEL,
    GEMINI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
    TOP_K,
    MIN_RELEVANCE_SCORE
)


# ---------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)


# ---------------------------------------------------------
# GRAPH STATE
# ---------------------------------------------------------

class RAGState(TypedDict, total=False):

    question: str

    query_embedding: List[float]

    retrieved_context: List[Dict[str, Any]]

    relevance_score: float

    answer: str

    grounded: bool


# ---------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------

def create_query_embedding(
    question: str
) -> List[float]:

    response = gemini_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values


# ---------------------------------------------------------
# RETRIEVAL NODE
# ---------------------------------------------------------

def retrieve_node(
    state: RAGState
) -> RAGState:

    question = state["question"]

    query_embedding = create_query_embedding(
        question
    )

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    retrieved_context = []

    scores = []

    for match in results.matches:

        score = float(match.score)

        scores.append(score)

        metadata = match.metadata or {}

        retrieved_context.append(
            {
                "id": match.id,
                "text": metadata.get(
                    "text",
                    ""
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "score": score
            }
        )

    if scores:
        relevance_score = max(scores)
    else:
        relevance_score = 0.0

    return {
        **state,
        "query_embedding": query_embedding,
        "retrieved_context": retrieved_context,
        "relevance_score": relevance_score
    }


# ---------------------------------------------------------
# GENERATION NODE
# ---------------------------------------------------------

def generate_node(
    state: RAGState
) -> RAGState:

    question = state["question"]

    contexts = state.get(
        "retrieved_context",
        []
    )

    relevance_score = state.get(
        "relevance_score",
        0.0
    )

    # No useful context
    if (
        not contexts
        or relevance_score < MIN_RELEVANCE_SCORE
    ):

        return {
            **state,
            "answer": (
                "I don't have enough information "
                "in the provided Agentic AI eBook "
                "to answer this question."
            ),
            "grounded": False
        }

    context_text = "\n\n".join(
        [
            (
                f"[Page {item['page']}]\n"
                f"{item['text']}"
            )
            for item in contexts
        ]
    )

    system_instruction = """
You are a strict document-grounded RAG assistant.

Your ONLY knowledge source is the retrieved context
provided below from the Agentic AI eBook.

Rules:

1. Answer ONLY using the provided context.
2. Do NOT use your general knowledge.
3. Do NOT invent facts.
4. If the context does not contain enough information
   to answer the question, say exactly:

   "I don't have enough information in the provided
   Agentic AI eBook to answer this question."

5. Keep the answer concise but informative.
6. Do not mention information that is not supported
   by the retrieved context.

Retrieved Context:
------------------
""" + context_text

    prompt = f"""
{system_instruction}

User Question:
{question}
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=prompt
    )

    answer = response.text.strip()

    return {
        **state,
        "answer": answer,
        "grounded": True
    }


# ---------------------------------------------------------
# BUILD LANGGRAPH
# ---------------------------------------------------------

def build_graph():

    workflow = StateGraph(
        RAGState
    )

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()


# Compiled application
rag_graph = build_graph()

# PUBLIC FUNCTION
def ask_question(
    question: str
) -> RAGState:

    initial_state: RAGState = {
        "question": question
    }

    result = rag_graph.invoke(
        initial_state
    )

    return result