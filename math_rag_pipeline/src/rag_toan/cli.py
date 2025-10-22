import typer
from pathlib import Path
import uuid
from math_rag_pipeline.src.rag_toan.ingestion.pdf_loader import save_extracted_text
from math_rag_pipeline.src.rag_toan.ingestion.text_cleaner import clean_text
from math_rag_pipeline.src.rag_toan.chunking.splitter import split_by_headings, chunk_by_words
from math_rag_pipeline.src.rag_toan.embedding.embedder import Embedder
from math_rag_pipeline.src.rag_toan.indexer.chroma_index import ChromaIndex
from math_rag_pipeline.src.rag_toan.retriever.retriever import Retriever
from math_rag_pipeline.src.rag_toan.llm.client import ask_llm
from math_rag_pipeline.src.rag_toan.config import EXTRACTED_DIR, CHUNKS_DIR
from math_rag_pipeline.src.rag_toan.utils.io import write_text_file, append_jsonl, read_text_file, safe_mkdir
import sys
import shutil
import logging

sys.tracebacklimit = 0 
app = typer.Typer(help="RAG Toán — CLI: ingest / index / query")

@app.command()
def ingest(pdf_path: str, out_name: str = None):
    """
    Extract + clean text from PDF and save to data/extracted.
    """
    name = out_name if out_name else pdf_path.split("/")[-1].replace(".pdf", "")
    destination = f"math_rag_pipeline/data/raw/{name}.pdf"
    safe_mkdir("math_rag_pipeline/data/raw/")
    safe_mkdir("math_rag_pipeline/data/cleaned/")
    shutil.copy(pdf_path, destination)
    extracted_path = save_extracted_text(pdf_path, out_basename=out_name)

    raw = read_text_file(extracted_path)
    cleaned = clean_text(raw)
    cleaned_path = f"math_rag_pipeline/data/cleaned/{name}.clean.txt"
    write_text_file(cleaned_path, cleaned)

@app.command()
def index(chunk_size: int = 400, overlap: int = 50, embed_model: str = None):
    """
    Chunk -> embed -> index into Chroma. 
    Uses the latest cleaned file in data/extracted.
    """
    # 1. Lấy file clean mới nhất
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    files = sorted(Path("math_rag_pipeline/data/cleaned").glob("*.clean.txt"), key=lambda p: p.stat().st_mtime)
    if not files:
        typer.echo("❌ No cleaned files found in extracted directory.")
        raise typer.Exit(code=1)
    latest = files[-1]

    # 2. Đọc văn bản, chunk
    text = read_text_file(str(latest))
    chunks = split_by_headings(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        typer.echo("❌ No chunks generated.")
        raise typer.Exit(code=1)

    # 3. Encode
    embedder = Embedder(model_name=embed_model) if embed_model else Embedder()
    embeddings = embedder.encode(chunks)

    # 4. Tạo metadata cho từng chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": latest.name, "chunk_index": i} for i in range(len(chunks))]

    # 5. Lưu vào Chroma
    chroma = ChromaIndex()
    chroma.add(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)
    chroma.persist()

    # 6. Ghi ra file .jsonl để xem chunk
    out_path = Path(CHUNKS_DIR) / (latest.stem + ".chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)

    for id_, chunk, meta in zip(ids, chunks, metadatas):
        append_jsonl(str(out_path), {"id": id_, "text": chunk, "metadata": meta})

    typer.echo(f"✅ Indexed and saved chunks to: {out_path}")

@app.command()
def apply_data(pdf_path: str, out_name: str = None, chunk_size: int = 400, overlap: int = 50, embed_model: str = None):
    """
    Ingest + index in one command.
    """
    ingest(pdf_path=pdf_path, out_name=out_name)
    index(chunk_size=chunk_size, overlap=overlap, embed_model=embed_model);

@app.command()
def query(q: str, top_k: int = 3):
    """
    Retrieve top_k relevant chunks and ask LLM to answer.
    """
    retriever = Retriever()
    docs = retriever.retrieve(q, top_k=top_k)
    if not docs:
        raise typer.Exit(code=1)
    context = "\n\n---\n\n".join([d["text"] for d in docs])
    answer = ask_llm(context, q)

if __name__ == "__main__":
    app()

# command:
# python -m math_rag_pipeline.src.rag_toan.cli apply-data "path/to/file.pdf"
# python -m math_rag_pipeline.src.rag_toan.cli query --q "<query>"