import time
import re
import hashlib
import numpy as np
from langchain_ollama import OllamaLLM
from constants import MODEL_TYPE, TARGET_SOURCE_CHUNKS


# ---------------- DOCUMENT HASH (FOR CACHING) ----------------
def docs_hash(docs):
    combined = "".join(d.page_content[:200] for d in docs[:5])
    return hashlib.md5(combined.encode()).hexdigest()


# ---------------- QUESTION GENERATION ----------------
def generate_questions(docs, llm, n=2):
    qs = []

    for d in docs[:n]:

        q = llm.invoke(
            f"""
Generate ONE short factual question.
Return ONLY the question.

Text:
{d.page_content[:600]}
"""
        ).strip()

        qs.append(q)

    return qs


# ---------------- SAFE LLM SCORE PARSER ----------------
def llm_score(llm, prompt):

    strict_prompt = f"""
Return ONLY a single number between 0 and 1.
No explanation.

{prompt}
"""

    result = llm.invoke(strict_prompt)

    match = re.search(r"\d*\.?\d+", result)

    if not match:
        return 0.0

    val = float(match.group())

    # clamp score safely
    return max(0.0, min(val, 1.0))


# ---------------- RAGAS STYLE SELF EVALUATION ----------------
def self_evaluation(vectorstore, docs, cache=None):

    if not docs:
        return {"status": "No docs for evaluation"}

    current_hash = docs_hash(docs)

    # ⭐ RETURN CACHE IF DOCS SAME
    if cache and cache.get("hash") == current_hash:
        return cache

    llm = OllamaLLM(model=MODEL_TYPE, temperature=0)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TARGET_SOURCE_CHUNKS}
    )

    questions = generate_questions(docs, llm)

    faithfulness = []
    relevancy = []
    context_precision = []
    context_recall = []

    total_latency = 0

    # ---------------- EVALUATION LOOP ----------------
    for q in questions:

        start = time.time()

        retrieved = retriever.invoke(q)

        if not retrieved:
            continue

        context = "\n\n".join(d.page_content for d in retrieved)

        answer = llm.invoke(
            f"Context:\n{context}\n\nQuestion:{q}\nAnswer:"
        )

        total_latency += time.time() - start

        # 🧠 FAITHFULNESS (SIMPLIFIED PROMPT)
        faithfulness.append(llm_score(llm, f"""
Rate between 0 and 1.

If the answer mostly matches the context give HIGH score.
If unrelated give LOW score.

Context:
{context}

Answer:
{answer}
"""))

        # 🎯 ANSWER RELEVANCY (SIMPLIFIED)
        relevancy.append(llm_score(llm, f"""
Rate between 0 and 1.

If the answer clearly responds to the question give HIGH score.

Question:
{q}

Answer:
{answer}
"""))

        # 📄 CONTEXT PRECISION (FIXED PROMPT — NO MORE 0.0 ISSUE)
        context_precision.append(llm_score(llm, f"""
Rate between 0 and 1.

Does the context look useful for answering the question?

Question:
{q}

Context:
{context}
"""))

        # 📚 CONTEXT RECALL (SIMPLIFIED)
        context_recall.append(llm_score(llm, f"""
Rate between 0 and 1.

Does the context contain enough details to answer?

Question:
{q}

Context:
{context}
"""))

    # ---------------- FINAL RESULTS ----------------
    results = {
        "Auto Questions": len(questions),
        "Faithfulness": round(np.mean(faithfulness), 3) if faithfulness else 0.0,
        "Answer Relevancy": round(np.mean(relevancy), 3) if relevancy else 0.0,
        "Context Precision": round(np.mean(context_precision), 3) if context_precision else 0.0,
        "Context Recall": round(np.mean(context_recall), 3) if context_recall else 0.0,
        "Avg Latency (sec)": round(total_latency / max(len(questions), 1), 2),
        "Generated Questions": questions
    }

    return {
        "hash": current_hash,
        "results": results
    }
