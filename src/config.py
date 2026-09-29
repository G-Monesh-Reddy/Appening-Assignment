import os
from pathlib import Path

from dotenv import load_dotenv

# PROJECT ROOT
BASE_DIR = Path(__file__).resolve().parent.parent

# LOAD LOCAL .env
load_dotenv(BASE_DIR / ".env")

# STREAMLIT SECRETS

try:
    import streamlit as st

    STREAMLIT_SECRETS = st.secrets

except Exception:
    STREAMLIT_SECRETS = {}


def get_config(key, default=None):
    """
    Get configuration value from:
    1. Environment variable (.env locally)
    2. Streamlit Secrets (Streamlit Cloud)
    3. Default value
    """

    value = os.getenv(key)

    if value is not None:
        return value

    try:
        value = STREAMLIT_SECRETS.get(key)
    except Exception:
        value = None

    if value is not None:
        return value

    return default

# API KEYS
GEMINI_API_KEY = get_config("GEMINI_API_KEY")

PINECONE_API_KEY = get_config("PINECONE_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

# PINECONE
PINECONE_INDEX_NAME = get_config(
    "PINECONE_INDEX_NAME",
    "rag-agentic-ai"
)

PINECONE_CLOUD = get_config(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = get_config(
    "PINECONE_REGION",
    "us-east-1"
)

# GEMINI
GEMINI_LLM_MODEL = get_config(
    "GEMINI_LLM_MODEL",
    "gemini-3.5-flash-lite"
)

GEMINI_EMBEDDING_MODEL = get_config(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)

# RAG CONFIGURATION
EMBEDDING_DIMENSION = int(
    get_config(
        "EMBEDDING_DIMENSION",
        "768"
    )
)

CHUNK_SIZE = int(
    get_config(
        "CHUNK_SIZE",
        "1000"
    )
)

CHUNK_OVERLAP = int(
    get_config(
        "CHUNK_OVERLAP",
        "150"
    )
)

TOP_K = int(
    get_config(
        "TOP_K",
        "5"
    )
)

MIN_RELEVANCE_SCORE = float(
    get_config(
        "MIN_RELEVANCE_SCORE",
        "0.35"
    )
)


PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"