"""Prompts shared with the offline evaluation (scripts/inference/run_conflict_eval.py)."""

CLOSED_BOOK = (
    "Instruction: Answer the following question in 1-3 words based on your internal knowledge. "
    "If the answer is unknown, strictly respond with 'I don't know'.\n\n"
    "Question: {question}\nAnswer:"
)

RAG_INSTRUCTION = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer based on the context, just say you don't know. "
    "Keep the answer as short as possible."
)


def closed_book_prompt(question: str) -> str:
    return CLOSED_BOOK.format(question=question)


def rag_prompt(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(contexts)
    return f"{RAG_INSTRUCTION}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"


# Same wording as scripts/detectors/collect_signals.py, where thresholds were tuned.
VERIFY = (
    "Question: {question}\nProposed answer: {answer}\n"
    "Based on your own knowledge, is the proposed answer correct? Reply with Yes or No.\nReply:"
)


def verify_prompt(question: str, answer: str) -> str:
    return VERIFY.format(question=question, answer=answer)
