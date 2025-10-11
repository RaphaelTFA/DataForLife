from chromadb.config import Settings
import chromadb
from math_rag_pipeline.src.rag_toan.config import CHROMA_PERSIST_DIR
from pathlib import Path

class ChromaIndex:
    def __init__(self, collection_name: str = "rag_mathematics", persist_dir: str = CHROMA_PERSIST_DIR):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        # use duckdb+parquet backend for persistence
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
        self.col = self.client.get_or_create_collection(name=collection_name)

    def add(self, ids: list, texts: list, embeddings, metadatas: list = None):
        """
        Add docs to chroma. embeddings may be numpy array -> convert to list.
        """
        emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self.col.add(ids=ids, documents=texts, embeddings=emb_list, metadatas=metadatas)

    def query(self, query_embedding, n_results: int = 3):
        res = self.col.query(query_embeddings=[query_embedding], n_results=n_results)
        return res

    def persist(self):
        self.client.persist()
