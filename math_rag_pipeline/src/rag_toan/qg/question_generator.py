from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def generate_questions(text: str, concept: str, num: int = 3, level: str = "medium"):
    prompt = f"""
    Sinh {num} câu hỏi Toán học ở mức độ {level}, 
    dựa trên nội dung sau (thuộc chủ đề "{concept}").

    Trả về danh sách câu hỏi đánh số.
    Nội dung:
    {text[:2000]}
    """
    return ask_llm(prompt)
