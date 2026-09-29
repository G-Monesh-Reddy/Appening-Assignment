import streamlit as st

from src.graph import ask_question




st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)



st.title("🤖 Agentic AI RAG Chatbot")

st.markdown(
    """
Ask questions about the **Agentic AI eBook**.

The chatbot is strictly grounded in the retrieved
document context and will refuse questions that
cannot be answered from the eBook.
"""
)



with st.sidebar:

    st.header("RAG Configuration")

    st.write(
        "Retrieval: Pinecone"
    )

    st.write(
        "Embeddings: Gemini"
    )

    st.write(
        "LLM: Gemini"
    )

    st.write(
        "Orchestration: LangGraph"
    )

    st.divider()

    st.info(
        "The model is instructed not to use "
        "general knowledge outside the eBook."
    )


question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is Agentic AI?"
    )
)



if st.button(
    "Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Retrieving context and generating answer..."
        ):

            try:

                result = ask_question(
                    question.strip()
                )

                # -----------------------------------------
                # ANSWER
                # -----------------------------------------

                st.subheader(
                    "Answer"
                )

                st.write(
                    result.get(
                        "answer",
                        "No answer generated."
                    )
                )

                # -----------------------------------------
                # SCORE
                # -----------------------------------------

                score = result.get(
                    "relevance_score",
                    0.0
                )

                st.subheader(
                    "Retrieval Relevance"
                )

                st.metric(
                    "Top Context Score",
                    f"{score:.4f}"
                )

                # -----------------------------------------
                # GROUNDING STATUS
                # -----------------------------------------

                grounded = result.get(
                    "grounded",
                    False
                )

                if grounded:

                    st.success(
                        "Answer generated from retrieved document context."
                    )

                else:

                    st.warning(
                        "The question could not be grounded "
                        "in the provided document."
                    )

                # -----------------------------------------
                # CONTEXT
                # -----------------------------------------

                st.subheader(
                    "Retrieved Context"
                )

                contexts = result.get(
                    "retrieved_context",
                    []
                )

                if not contexts:

                    st.info(
                        "No relevant context retrieved."
                    )

                else:

                    for i, context in enumerate(
                        contexts,
                        start=1
                    ):

                        score = context.get(
                            "score",
                            0.0
                        )

                        page = context.get(
                            "page",
                            "Unknown"
                        )

                        with st.expander(
                            f"Context {i} | "
                            f"Page {page} | "
                            f"Score {score:.4f}"
                        ):

                            st.write(
                                context.get(
                                    "text",
                                    ""
                                )
                            )

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )