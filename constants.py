import os
from chromadb.config import Settings # type: ignore
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "db")
COLLECTION_NAME = "documind"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TARGET_SOURCE_CHUNKS = int(os.environ.get("TARGET_SOURCE_CHUNKS", 5))

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL_NAME", "nomic-embed-text")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "llama3")

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False
)
