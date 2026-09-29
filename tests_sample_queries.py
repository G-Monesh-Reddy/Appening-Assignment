from src.graph import ask_question


QUESTIONS = [

    "What is Agentic AI according to the eBook?",

    "How do AI agents differ from traditional "
    "automation systems?",

    "What are the core components of an "
    "Agentic Architecture?",

    "What role does memory play in Agentic AI "
    "workflows?",

    "Who won the 2022 FIFA World Cup?",

    "What are the main characteristics of "
    "Agentic AI systems?"
]


def run_tests():

   

    for number, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"TEST {number}"
        )

        print(
            f"{'=' * 70}"
        )

        print(
            f"\nQuestion:\n{question}"
        )

        try:

            result = ask_question(
                question
            )

            print(
                "\nAnswer:"
            )

            print(
                result.get(
                    "answer",
                    "No answer."
                )
            )

            print(
                "\nRelevance Score:"
            )

            print(
                f"{result.get('relevance_score', 0.0):.4f}"
            )

            print(
                "\nGrounded:"
            )

            print(
                result.get(
                    "grounded",
                    False
                )
            )

            print()

        except Exception as e:

            print(
                f"\nERROR: {e}"
            )


if __name__ == "__main__":

    run_tests()