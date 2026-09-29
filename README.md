# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from an Agentic AI eBook.

The project uses:

- Python
- Gemini API
- Gemini Embeddings
- Pinecone
- LangGraph
- Streamlit

---

## Architecture

```text
                    Agentic AI PDF
                          |
                          v
                  PDF Text Extraction
                          |
                          v
                      Chunking
                          |
                          v
                  Gemini Embeddings
                          |
                          v
                       Pinecone
                          |
                          |
User Question ------------+
       |
       v
   LangGraph
       |
       v
   Retrieval
       |
       v
Relevant Context
       |
       v
   Gemini LLM
       |
       v
Grounded Response
       |
       v
    Streamlit
```
