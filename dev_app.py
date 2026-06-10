import streamlit as st
from io import BytesIO
import time

from core.loaders import load_single_document
from core.vectorstore import get_vectorstore
from core.rag_engine import ask_question
from core.evaluation import self_evaluation

# ---------------- PAGE CONFIG ----------------
st.set_page_config("DocuMind Dev Console", "🧠", layout="wide")
st.title("🧠 DocuMind — Developer Console")

# ---------------- SESSION ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "docs" not in st.session_state:
    st.session_state.docs = []

if "eval_cache" not in st.session_state:
    st.session_state.eval_cache = None

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ---------------- METRIC CARD ----------------
def metric_card(label, value):

    if isinstance(value, (int, float)):
        if value >= 0.75:
            color = "#2ecc71"
        elif value >= 0.5:
            color = "#f1c40f"
        else:
            color = "#e74c3c"
    else:
        color = "#3498db"

    st.markdown(f"""
    <div style="
        padding:14px;
        border-radius:10px;
        background-color:#1e1e1e;
        border-left:6px solid {color};
        margin-bottom:10px;">
        <strong>{label}</strong><br>
        <span style="font-size:20px">{value}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("📂 Developer Tools")

    files = st.file_uploader(
        "Upload Files",
        accept_multiple_files=True
    )

    # ⭐ ULTRA PRO → CLEAN DEV DATABASE EACH UPLOAD
    if st.button("Process Documents") and files:

        docs = []

        for f in files:
            docs.extend(load_single_document(BytesIO(f.read()), f.name))

        # ⭐ DEV CONSOLE USES SEPARATE DB
        db, status = get_vectorstore(
            docs,
            persist_dir="dev_db"
        )

        st.session_state.docs = docs
        st.session_state.vectorstore = db

        st.success("✅ Dev Documents Loaded")

    st.divider()
    st.subheader("🧠 Evaluation Panel")

    if st.button("Run Self Evaluation"):

        if st.session_state.vectorstore and st.session_state.docs:

            with st.spinner("Running evaluation..."):

                cache_output = self_evaluation(
                    st.session_state.vectorstore,
                    st.session_state.docs,
                    st.session_state.eval_cache
                )

            st.session_state.eval_cache = cache_output
            st.session_state.eval_results = cache_output["results"]

        else:
            st.warning("Upload documents first.")

# ---------------- SHOW EVALUATION ----------------
if st.session_state.eval_results:

    results = st.session_state.eval_results

    st.markdown("### 📊 Evaluation Metrics")

    metric_card("🧠 Faithfulness", results["Faithfulness"])
    metric_card("🎯 Answer Relevancy", results["Answer Relevancy"])
    metric_card("📄 Context Precision", results["Context Precision"])
    metric_card("📚 Context Recall", results["Context Recall"])
    metric_card("⚡ Avg Latency (sec)", results["Avg Latency (sec)"])

    with st.expander("📋 Auto Generated Evaluation Questions"):
        for i, q in enumerate(results["Generated Questions"], 1):
            st.write(f"{i}. {q}")

# ---------------- CHAT ----------------
st.subheader("💬 Developer Query Test")

q = st.text_input("Ask a question")

if q and st.session_state.vectorstore:

    start_time = time.time()

    answer, sources = ask_question(
        st.session_state.vectorstore,
        q,
        []
    )

    latency = round(time.time() - start_time, 2)

    st.markdown("### ⚡ Performance")
    st.info(f"Latency: {latency} sec")

    st.markdown("### 🤖 Answer")
    st.write(answer)

    st.divider()
    st.subheader("📄 Retrieved Chunks")

    for i, d in enumerate(sources, 1):
        st.markdown(f"**Chunk {i}**")
        st.code(d.page_content[:400])

    st.divider()
    st.subheader("📂 Source Files")

    # ⭐ REMOVE DUPLICATE SOURCES
    unique_sources = list(set(
        d.metadata.get("source", "unknown") for d in sources
    ))

    for src in unique_sources:
        st.write("-", src)

# ---------------- EMPTY STATE ----------------
if not st.session_state.vectorstore:
    st.info("Upload documents to begin developer testing.")
