from math_rag_pipeline.src.rag_toan.embedding.embedder import Embedder
from math_rag_pipeline.src.rag_toan.indexer.chroma_index import ChromaIndex
from math_rag_pipeline.src.rag_toan.retriever.reranker import Reranker

class Retriever:
    def __init__(self, embedder: Embedder = None, index: ChromaIndex = None):
        self.embedder = embedder or Embedder()
        self.index = index or ChromaIndex()

    def retrieve(self, query: str, top_k: int = 3):
        q_emb = self.embedder.encode([query])[0]
        res = self.index.query(q_emb, n_results=top_k)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0] if "metadatas" in res else [None]*len(docs)
        distances = res.get("distances", [[]])[0] if "distances" in res else None
        out = []
        for i, (did, doc, md) in enumerate(zip(ids, docs, metadatas)):
            out.append({"id": did, "text": doc, "metadata": md})
        return out

def query_vector_db(query: str, top_k=10):
    retriever = Retriever()
    docs = retriever.retrieve(query, top_k=top_k*2)
    reranker = Reranker()
    reranked_docs = reranker.rerank(query, docs, top_k=top_k)
    return reranked_docs
