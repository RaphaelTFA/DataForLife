from pathlib import Path
import os

BASE_DIR = Path.cwd()
SRC_ROOT = Path.cwd()  # if running with PYTHONPATH=src this is repo root
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_METADATA_DIR = DATA_DIR / "raw_metadata"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

for p in (RAW_DIR, RAW_METADATA_DIR, EXTRACTED_DIR, CHUNKS_DIR, VECTOR_DB_DIR):
    p.mkdir(parents=True, exist_ok=True)

# default embedder (change via env RAG_EMBEDDER)
DEFAULT_EMBEDDER = os.getenv("RAG_EMBEDDER", "intfloat/multilingual-e5-base")

# Chroma persistent directory
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(VECTOR_DB_DIR))

# OpenAI (or other) defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
