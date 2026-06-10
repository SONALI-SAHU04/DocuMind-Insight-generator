import os
import tempfile
from io import BytesIO
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)

from constants import CHUNK_SIZE, CHUNK_OVERLAP


# ---------------- SUPPORTED FILE TYPES ----------------
LOADER_MAPPING = {
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
}


# ---------------- LOAD SINGLE DOCUMENT ----------------
def load_single_document(file: BytesIO, filename: str) -> List[Document]:

    ext = os.path.splitext(filename)[1].lower()

    if ext not in LOADER_MAPPING:
        raise ValueError(f"Unsupported file type: {ext}")

    # Create temporary file because loaders require file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.getvalue())
        temp_path = tmp.name

    loader_class, loader_args = LOADER_MAPPING[ext]

    loader = loader_class(temp_path, **loader_args)
    documents = loader.load()

    # Attach original filename as metadata
    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["source"] = filename

    os.remove(temp_path)

    return documents


# ---------------- SPLIT INTO CHUNKS ----------------
def split_documents(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(docs)
