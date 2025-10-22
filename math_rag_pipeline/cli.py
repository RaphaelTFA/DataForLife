import typer
from pathlib import Path
import uuid
import sys

app = typer.Typer(help="CLI: Generate_question")

@app.command()
def gen_answer(
    question: str = typer.Option(
        ..., 
        help="Câu hỏi toán học cần trả lời"
    )
):
    """
    Sinh đáp án cho câu hỏi toán học.
    """
    from math_rag_pipeline.src.prompt.answer_generator import generate_answer
    from math_rag_pipeline.src.rag_toan.cli import query

    try: 
        content = query(question, top_k=3)
        answer_item = generate_answer(
            question=question,
            context=content
        )
        return answer_item[0]
        
    except Exception as e:
        typer.echo(f"[ERROR] Exception: {str(e)}")
        raise typer.Exit(code=1)

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
    num: int = typer.Option(
        5, 
        help="Số lượng cặp câu hỏi-đáp án cần tạo"
    )
):
    """
    Sinh câu hỏi và đáp án dựa vào chủ đề và độ khó.
    """
    from math_rag_pipeline.src.prompt.question_generator import generate_question as genT2R
    from math_rag_pipeline.src.SLM.inference import paraphrase as paraR2Q
    from math_rag_pipeline.src.rag_toan.cli import query

    query = "Nêu ra các kiến thức của chủ đề {subject}, tóm tắt chi tiết"

    try: 
        content = query(query, top_k=7)
        
        raw_question = genT2R(
            text=content, 
            concept=subject, 
            num=num, 
            level=difficulty
        )

        for item in raw_question:
            item["question"] = paraR2Q(item["question"])
            answer = gen_answer(item["question"])
            item["solution"]= answer["solution"]
            item["answer"]= answer["answer"]

    except ValueError as e:
        typer.echo(f"[ERROR] ValueError: {str(e)}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[ERROR] Exception: {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
