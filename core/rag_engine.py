from langchain_ollama import OllamaLLM
from constants import MODEL_TYPE, TARGET_SOURCE_CHUNKS


def build_prompt(context, question, chat_history=None):

    history_text = ""

    if chat_history:
        for msg in chat_history[-4:]:
            history_text += f'{msg["role"]}: {msg["message"]}\n'

    prompt = f"""
Use the conversation history and context to answer.

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


# ⭐ NORMAL (NON-STREAM)
def ask_question(vectorstore, question, chat_history=None):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TARGET_SOURCE_CHUNKS}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    llm = OllamaLLM(model=MODEL_TYPE, temperature=0)

    prompt = build_prompt(context, question, chat_history)

    answer = llm.invoke(prompt)

    return answer, docs


# ⭐ REAL STREAMING FUNCTION
def stream_answer(vectorstore, question, chat_history=None, reuse_sources=None):

    # ⭐ If follow-up question → reuse previous sources
    if reuse_sources:
        docs = reuse_sources
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": TARGET_SOURCE_CHUNKS}
        )
        docs = retriever.invoke(question)

    # Build context from docs
    context = "\n\n".join(d.page_content for d in docs)

    llm = OllamaLLM(model=MODEL_TYPE, temperature=0)

    prompt = build_prompt(context, question, chat_history)

    # 👇 REAL TOKEN STREAMING
    stream = llm.stream(prompt)

    return stream, docs
