from pathlib import Path
import os

BASE_DIR = Path("math_rag_pipeline")
SRC_ROOT = Path.cwd() 
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_METADATA_DIR = DATA_DIR / "raw_metadata"
EXTRACTED_DIR = DATA_DIR / "cleaned"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

for p in (RAW_DIR, RAW_METADATA_DIR, EXTRACTED_DIR, CHUNKS_DIR, VECTOR_DB_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBEDDER = os.getenv("RAG_EMBEDDER", "intfloat/multilingual-e5-base")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(VECTOR_DB_DIR))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "anthropic/claude-3.7-sonnet")
