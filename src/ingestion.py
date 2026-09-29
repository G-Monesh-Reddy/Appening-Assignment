from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google import genai
from google.genai import types

from pinecone import Pinecone, ServerlessSpec

from src.config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_PATH,
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


# ---------------------------------------------------------
# PDF LOADING
# ---------------------------------------------------------

def load_pdf(pdf_path: Path):
    """
    Load PDF pages and return text with page numbers.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        text = text.strip()

        if text:
            documents.append(
                {
                    "text": text,
                    "page": page_number
                }
            )

    print(
        f"Loaded {len(documents)} pages from PDF."
    )

    return documents


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def create_chunks(documents):
    """
    Split PDF text into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        split_texts = splitter.split_text(
            document["text"]
        )

        for index, text in enumerate(split_texts):

            chunks.append(
                {
                    "id": f"page-{document['page']}-chunk-{index}",
                    "text": text,
                    "page": document["page"]
                }
            )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# ---------------------------------------------------------
# GEMINI EMBEDDINGS
# ---------------------------------------------------------

def generate_embedding(text: str) -> List[float]:
    """
    Generate a single Gemini embedding.
    """

    response = gemini_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values


# ---------------------------------------------------------
# PINECONE INDEX
# ---------------------------------------------------------

def create_or_get_index():

    existing_indexes = [
        index.name
        for index in pinecone_client.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:

        print(
            f"Creating Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

    else:

        print(
            f"Pinecone index already exists: "
            f"{PINECONE_INDEX_NAME}"
        )

    return pinecone_client.Index(
        PINECONE_INDEX_NAME
    )


# ---------------------------------------------------------
# UPSERT
# ---------------------------------------------------------

def upsert_chunks(index, chunks):

    vectors = []

    total = len(chunks)

    for position, chunk in enumerate(chunks, start=1):

        print(
            f"Embedding chunk "
            f"{position}/{total}"
        )

        embedding = generate_embedding(
            chunk["text"]
        )

        vectors.append(
            {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "page": chunk["page"]
                }
            }
        )

    # Pinecone supports batches.
    batch_size = 50

    for start in range(
        0,
        len(vectors),
        batch_size
    ):

        batch = vectors[
            start:start + batch_size
        ]

        index.upsert(
            vectors=batch
        )

        print(
            f"Uploaded "
            f"{min(start + batch_size, len(vectors))}"
            f"/{len(vectors)} vectors"
        )


# ---------------------------------------------------------
# MAIN INGESTION PIPELINE
# ---------------------------------------------------------

def run_ingestion():

    print("\n==============================")
    print("RAG INGESTION STARTED")
    print("==============================\n")

    documents = load_pdf(
        PDF_PATH
    )

    chunks = create_chunks(
        documents
    )

    index = create_or_get_index()

    upsert_chunks(
        index,
        chunks
    )

    print("\n==============================")
    print("INGESTION COMPLETED")
    print("==============================")

    print(
        f"\nIndex: {PINECONE_INDEX_NAME}"
    )

    print(
        f"Vectors inserted: {len(chunks)}"
    )


if __name__ == "__main__":
    run_ingestion()