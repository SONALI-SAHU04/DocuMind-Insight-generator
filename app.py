import streamlit as st # type: ignore
from io import BytesIO
from langchain_ollama import OllamaLLM # type: ignore
from constants import MODEL_TYPE # type: ignore

from core.loaders import load_single_document
from core.vectorstore import get_vectorstore
from core.rag_engine import stream_answer

# ---------------- PAGE CONFIG ----------------
st.set_page_config("DocuMind", "📘", layout="wide")

# ---------------- SESSION ----------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None


# ⭐ LIVE AUTO SCROLL SCRIPT
AUTO_SCROLL = """
<script>
var main = window.parent.document.querySelector(".main");
if(main){
    main.scrollTop = main.scrollHeight;
}
</script>
"""


# ---------------- DOCUMENT SUMMARY FUNCTION ----------------
def summarize_documents(vectorstore):

    filename = st.session_state.get("last_file", None)

    if not filename:
        return "No recent document found."

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8, "filter": {"source": filename}}
    )

    docs = retriever.invoke("Give overview of this document")

    if not docs:
        return "No content found for summary."

    context = "\n\n".join(d.page_content for d in docs)

    llm = OllamaLLM(model=MODEL_TYPE)

    summary = llm.invoke(f"""
Create a clear summary of this document.

Context:
{context}

Give:
- Key topics
- Main ideas
- Important points
""")

    return summary


# ---------------- EXPORT CHAT FUNCTION ----------------
def export_chat_text(chat):

    text = ""

    for msg in chat:
        role = "User" if msg["role"]=="user" else "Assistant"
        text += f"{role}: {msg['message']}\n\n"

    return text


# ---------------- FOLLOWUP BUTTONS ----------------
def followup_buttons():

    # ⭐ Disable buttons until first bot reply exists
    disabled = not any(m["role"]=="bot" for m in st.session_state.chat)

    col1, col2, col3 = st.columns(3)

    if col1.button("🔎 Explain Simply", disabled=disabled):
        return "Explain the previous answer in simple terms."

    if col2.button("📌 Key Points", disabled=disabled):
        return "Give key bullet points from the previous answer."

    if col3.button("📖 More Details", disabled=disabled):
        return "Provide more detailed explanation."

    return None


# ---------------- UI STYLE ----------------
css = """
<style>
.chat-message {
    padding: 1.2rem;
    border-radius: 0.6rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.chat-message.user { background-color: #2b313e; }
.chat-message.bot { background-color: #475063; }
.chat-message img {
    max-width: 55px;
    border-radius: 50%;
    margin-right: 12px;
}
.chat-message .message {
    color: white;
    font-size: 15px;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
<img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif">
<div class="message">{}</div>
</div>
"""

user_template = """
<div class="chat-message user">
<img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa">
<div class="message">{}</div>
</div>
"""

st.markdown(css, unsafe_allow_html=True)

# ---------------- UI ----------------
st.title("📘 DocuMind — Offline Document Insights Generator")

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("## 📂 Documents")

    files = st.file_uploader(
        "Upload PDF / TXT / Word files",
        type=["pdf", "txt", "csv", "doc", "docx", "xls", "xlsx"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    with col1:
        process_btn = st.button("Process")

    with col2:
        summary_btn = st.button("Summary")

    if process_btn and files:

        docs = []

        for f in files:
            docs.extend(load_single_document(BytesIO(f.read()), f.name))
            st.session_state.last_file = f.name   

        db, status = get_vectorstore(docs)
        st.session_state.vectorstore = db

        if status == "overwritten":
            st.info("♻️ File updated")
        elif status == "added":
            st.success("🟢 Added")
        else:
            st.warning("⚠️ No documents processed")

    if summary_btn:

        if st.session_state.vectorstore:

            with st.spinner("Generating summary..."):
                summary = summarize_documents(st.session_state.vectorstore)

            st.session_state.chat.append({
                "role":"bot",
                "message":f"📄 Document Summary:\n\n{summary}"
            })

            st.rerun()

        else:
            st.warning("Upload documents first.")

    if st.session_state.chat:

        st.markdown("### ⚡ Actions")

        chat_text = export_chat_text(st.session_state.chat)

        st.download_button(
            "📥 Download Chat",
            data=chat_text,
            file_name="documind_chat.txt",                              
            mime="text/plain"
        )

    st.markdown("### ⚙️ Tools")

    DEV_MODE = st.toggle("🧠 Check Sources")


# ---------------- DISPLAY CHAT HISTORY ----------------
chat_container = st.container()

with chat_container:

    for item in st.session_state.chat:

        if item["role"] == "user":
            st.markdown(user_template.format(item["message"]), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.format(item["message"]), unsafe_allow_html=True)

            if DEV_MODE and "sources" in item:
                with st.expander("📄 Sources"):

                    # ⭐ remove duplicate filenames
                    unique_sources = list(dict.fromkeys(
                        d.metadata.get("source", "unknown")
                        for d in item["sources"]
                    ))

                    for src in unique_sources:
                        st.write("-", src)

# ⭐ FOLLOW-UP BUTTONS ONLY AFTER CHAT EXISTS
if st.session_state.chat:
    suggestion = followup_buttons()
else:
    suggestion = None


# ⭐ FOLLOWUP STREAMING
if suggestion and st.session_state.vectorstore:

    st.session_state.last_question = suggestion

    reuse_sources = None
    for msg in reversed(st.session_state.chat):
        if msg["role"] == "bot" and "sources" in msg:
            reuse_sources = msg["sources"]
            break

    st.session_state.chat.append({"role": "user","message": suggestion})

    stream_container = st.empty()
    full_text = ""

    stream, sources = stream_answer(
        st.session_state.vectorstore,
        suggestion,
        st.session_state.chat,
        reuse_sources=reuse_sources
    )

    for chunk in stream:
        full_text += chunk
        stream_container.markdown(
            bot_template.format(full_text) + AUTO_SCROLL,
            unsafe_allow_html=True
        )

    st.session_state.chat.append({
        "role": "bot",
        "message": full_text,
        "sources": reuse_sources if reuse_sources else sources
    })

    st.rerun()


# ---------------- CHAT INPUT ----------------
st.subheader("💬 Ask a Question")

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Type your question")
    submitted = st.form_submit_button("Send")


# ⭐ MAIN STREAMING WITH AUTO SCROLL
if submitted and question and st.session_state.vectorstore:

    st.session_state.last_question = question

    st.session_state.chat.append({"role": "user","message": question})

    stream_container = st.empty()
    full_text = ""

    stream, sources = stream_answer(
        st.session_state.vectorstore,
        question,
        st.session_state.chat
    )

    for chunk in stream:
        full_text += chunk
        stream_container.markdown(
            bot_template.format(full_text) + AUTO_SCROLL,
            unsafe_allow_html=True
        )

    st.session_state.chat.append({
        "role": "bot",
        "message": full_text,
        "sources": sources
    })

    if DEV_MODE:
        st.sidebar.subheader("📄 Retrieved Chunks")
        for d in sources:
            st.sidebar.code(d.page_content[:300])

    st.rerun()


# ---------------- EMPTY STATE ----------------
if not st.session_state.vectorstore:
    st.info("Upload documents to get started.")
