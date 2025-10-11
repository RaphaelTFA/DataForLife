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
from math_rag_pipeline.src.rag_toan.utils.io import write_text_file, append_jsonl, read_text_file
from math_rag_pipeline.src.rag_toan.qg.concept_tagger import tag_concept
from math_rag_pipeline.src.rag_toan.qg.qa_pair_generator import generate_qa_pairs
from math_rag_pipeline.src.rag_toan.qg.evaluator import evaluate_question
import sys

app = typer.Typer(help="RAG Toán — CLI: ingest / index / query")

@app.command()
def ingest(pdf_path: str, out_name: str = None):
    """
    Extract + clean text from PDF and save to data/extracted.
    """
    typer.echo("🔎 Extracting PDF ...")
    extracted_path = save_extracted_text(pdf_path, out_basename=out_name)
    typer.echo(f"✅ Extracted saved: {extracted_path}")

    typer.echo("🧼 Cleaning text ...")
    raw = read_text_file(extracted_path)
    cleaned = clean_text(raw)
    cleaned_path = Path(extracted_path).with_name(Path(extracted_path).stem + ".clean.txt")
    write_text_file(str(cleaned_path), cleaned)
    typer.echo(f"✅ Cleaned saved: {cleaned_path}")

@app.command()
def index(chunk_size: int = 400, overlap: int = 50, embed_model: str = None):
    """
    Chunk -> embed -> index into Chroma. Uses latest cleaned file in data/extracted.
    """
    files = sorted(Path(EXTRACTED_DIR).glob("*.clean.txt"))
    if not files:
        typer.echo("Không tìm thấy file cleaned trong data/extracted. Chạy `ingest` trước.")
        raise typer.Exit(code=1)
    latest = files[-1]
    text = read_text_file(str(latest))
    typer.echo(f"📄 Chunking {latest.name} ...")
    chunks = split_by_headings(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        typer.echo("Không có chunk nào được tạo — kiểm tra file.")
        raise typer.Exit(code=1)
    typer.echo(f"➡️ {len(chunks)} chunks created.")

    typer.echo("📥 Generating embeddings ...")
    embedder = Embedder(model_name=embed_model) if embed_model else Embedder()
    embeddings = embedder.encode(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": latest.name, "chunk_index": i} for i in range(len(chunks))]

    typer.echo("💾 Writing to Chroma DB ...")
    index = ChromaIndex()
    index.add(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)
    index.persist()

    out_chunks = Path(CHUNKS_DIR) / (latest.stem + ".chunks.jsonl")
    for _id, chunk, md in zip(ids, chunks, metadatas):
        append_jsonl(str(out_chunks), {"id": _id, "text": chunk, "metadata": md})
    typer.echo(f"✅ Indexed and saved chunk metadata to {out_chunks}")

@app.command()
def query(q: str, top_k: int = 3):
    """
    Retrieve top_k relevant chunks and ask LLM to answer.
    """
    retriever = Retriever()
    typer.echo("🔎 Retrieving relevant chunks ...")
    docs = retriever.retrieve(q, top_k=top_k)
    if not docs:
        typer.echo("Không có tài liệu phù hợp trong index.")
        raise typer.Exit(code=1)
    context = "\n\n---\n\n".join([d["text"] for d in docs])
    typer.echo(f"🧠 Sending {len(docs)} chunks to LLM ...")
    answer = ask_llm(context, q)
    typer.echo("\n===== LLM ANSWER =====\n")
    typer.echo(answer)
    typer.echo("\n======================\n")

app = typer.Typer(help="📘 Math Generator CLI")

qg_app = typer.Typer(help="🎓 Question Generation pipeline")
app.add_typer(qg_app, name="qg")

@qg_app.command("run")
def run_qg(input_file: str, out_file: str = "data/generated_questions.jsonl", num: int = 5):
    """
    Chạy toàn bộ pipeline sinh câu hỏi toán học từ 1 file đã extract/clean.
    """
    text = read_text_file(input_file)
    typer.echo("🔍 Xác định chủ đề ...")
    concept = tag_concept(text)
    typer.echo(f"📘 Chủ đề: {concept}")

    typer.echo("🧠 Sinh câu hỏi & đáp án ...")
    qa_pairs = generate_qa_pairs(text, concept, num=num)

    typer.echo("✅ Đánh giá câu hỏi ...")
    for qa in qa_pairs:
        qa["evaluation"] = evaluate_question(qa["question"], qa.get("answer", ""))
        append_jsonl(out_file, qa)

    typer.echo(f"✅ Hoàn thành! Lưu {len(qa_pairs)} câu hỏi vào {out_file}")

@app.command()
def generate(
    subject: str = typer.Option(
        ..., 
        help="Chủ đề toán học (algebra, geometry, calculus, statistics, trigonometry)"
    ),
    difficulty: str = typer.Option(
        ..., 
        help="Độ khó (easy, medium, hard, challenging)"
    ),
    education_level: str = typer.Option(
        "high_school", 
        help="Cấp độ giáo dục (elementary, middle_school, high_school, university)"
    ),
    num: int = typer.Option(
        5, 
        help="Số lượng cặp câu hỏi-đáp án cần tạo"
    ),
    use_rag: bool = typer.Option(
        True, 
        help="Sử dụng RAG để tìm ngữ cảnh liên quan"
    )
):
    """
    Sinh câu hỏi và đáp án dựa vào chủ đề và độ khó.
    """
    from math_rag_pipeline.src.rag_toan.qg.qa_pair_generator import generate_qa_pairs as gen_qa, save_qa_pairs

    typer.echo(f"🧠 Đang sinh {num} câu hỏi {subject} với độ khó {difficulty} cho {education_level}...")
    
    try:
        qa_pairs = gen_qa(
            subject=subject,
            difficulty=difficulty,
            education_level=education_level,
            num=num,
            use_rag=use_rag
        )
        
        # Save the generated QA pairs
        output_path = save_qa_pairs(
            qa_pairs=qa_pairs,
            subject=subject,
            difficulty=difficulty,
            education_level=education_level
        )
        
        # Display results
        typer.echo("\n===== GENERATED QUESTIONS =====\n")
        for i, qa in enumerate(qa_pairs, 1):
            typer.echo(f"[{i}] Question: {qa['question']}")
            typer.echo(f"    Answer: {qa['answer'][:100]}..." if len(qa['answer']) > 100 else f"    Answer: {qa['answer']}")
            typer.echo("")
        
        typer.echo(f"\n✅ Generated {len(qa_pairs)} QA pairs and saved to: {output_path}")
        
    except ValueError as e:
        typer.echo(f"❌ Error: {str(e)}")
        typer.echo("\nAvailable options:")
        typer.echo("- subjects: algebra, geometry, calculus, statistics, trigonometry")
        typer.echo("- difficulty: easy, medium, hard, challenging")
        typer.echo("- education_level: elementary, middle_school, high_school, university")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error generating questions: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
