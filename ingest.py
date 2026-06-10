#!/usr/bin/env python3

# DocuMind — Ingestion Pipeline
import os
import glob
from typing import List
from multiprocessing import Pool

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# ---- Load ENV ----
load_dotenv()

# ---- Import Project Constants ----
from constants import (
    PERSIST_DIRECTORY,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDINGS_MODEL,
)

SOURCE_DIRECTORY = os.getenv("SOURCE_DIRECTORY", "source_documents")

# ----Custom Email Loader Fix----
class MyElmLoader(UnstructuredEmailLoader):
    def load(self) -> List[Document]:
        try:
            try:
                return super().load()
            except ValueError as e:
                if "text/html content not found" in str(e):
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    return super().load()
                raise
        except Exception as e:
            raise type(e)(f"{self.file_path}: {e}") from e


# ---- Loader Mapping ----
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


# ----Document Loading----
def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()

    if ext not in LOADER_MAPPING:
        raise ValueError(f"Unsupported file type: {ext}")

    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)

    return loader.load()


def load_documents(directory: str) -> List[Document]:

    files = []

    # Collect files recursively
    for ext in LOADER_MAPPING:
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext.upper()}"), recursive=True))

    if not files:
        print("❌ No supported files found.")
        return []

    documents: List[Document] = []

    # Multiprocessing for faster loading
    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(files), desc="📂 Loading documents") as bar:
            for d in pool.imap_unordered(load_single_document, files):
                documents.extend(d)
                bar.update()

    return documents


# ----Chunking----

def process_documents() -> List[Document]:

    print(f"📂 Loading documents from: {SOURCE_DIRECTORY}")

    documents = load_documents(SOURCE_DIRECTORY)

    if not documents:
        print("❌ No documents loaded.")
        exit(0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️ Created {len(chunks)} chunks")

    return chunks


# ----Main Ingestion Pipeline----
def main():

    print("🚀 Starting DocuMind ingestion pipeline...")

    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

    documents = process_documents()

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    db.add_documents(documents)

    print("✅ INGESTION COMPLETE")
    print("👉 Next step: streamlit run app.py")


if __name__ == "__main__":
    main()
