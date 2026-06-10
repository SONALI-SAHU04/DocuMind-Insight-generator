from collections import defaultdict

from langchain_chroma import Chroma # type: ignore
from langchain_ollama import OllamaEmbeddings # type: ignore

from constants import ( # type: ignore
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    EMBEDDINGS_MODEL,
    CHROMA_SETTINGS
)

from core.loaders import split_documents # type: ignore


# ---------------- VECTORSTORE ----------------
def get_vectorstore(documents, persist_dir=PERSIST_DIRECTORY):
    """
    Main vectorstore builder.

    app.py      → uses default db
    dev_app.py  → can pass persist_dir="dev_db"

    Supports:
    ✔ multi-file upload
    ✔ overwrite same file
    ✔ grouped metadata
    """

    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,   # ⭐ now dynamic
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    if not documents:
        return db, "empty"

    # ⭐ split docs into chunks
    chunks = split_documents(documents)

    # ⭐ GROUP CHUNKS BY FILENAME
    grouped_chunks = defaultdict(list)

    for c in chunks:
        filename = c.metadata.get("source", "unknown")
        grouped_chunks[filename].append(c)

    status = "added"

    # ⭐ PROCESS EACH FILE SEPARATELY
    for filename, file_chunks in grouped_chunks.items():

        # get existing metadata
        existing = db.get(include=["metadatas"])

        existing_files = set()

        if existing and existing.get("metadatas"):
            for m in existing["metadatas"]:
                if m and "source" in m:
                    existing_files.add(m["source"])

        # ⭐ overwrite if file already exists
        if filename in existing_files:
            print(f"♻️ Overwriting existing file: {filename}")
            db.delete(where={"source": filename})
            status = "overwritten"

        # ⭐ ensure metadata present
        for c in file_chunks:
            if not c.metadata:
                c.metadata = {}
            c.metadata["source"] = filename

        db.add_documents(file_chunks)

    return db, status
