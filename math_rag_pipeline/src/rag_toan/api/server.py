from fastapi import FastAPI, Query
from math_rag_pipeline.src.rag_toan.retriever.retriever import query_vector_db
from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

app = FastAPI(title="Math RAG API", description="RAG pipeline for math documents")

@app.get("/query")
def query(q: str = Query(..., description="Câu hỏi cần hỏi")):
    docs = query_vector_db(q)
    context = "\n\n---\n\n".join([d["text"] for d in docs])
    answer = ask_llm(context, q)
    return {"query": q, "answer": answer, "context": [d["text"] for d in docs]}
