from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def generate_questions(text: str, concept: str, num: int = 3, level: str = "medium"):
    """
    Sinh câu hỏi Toán học dựa trên đoạn văn và chủ đề cụ thể.
    
    Args:
        text (str): Nội dung hoặc đoạn văn nguồn.
        concept (str): Chủ đề hoặc khái niệm toán học.
        num (int): Số lượng câu hỏi cần sinh.
        level (str): Mức độ câu hỏi ('easy', 'medium', 'hard').

    Returns:
        List[Dict[str, str]]: Danh sách các câu hỏi ở dạng JSON [{ "question": "..." }, ...]
    """
    
    prompt = f"""
    Hãy sinh {num} câu hỏi Toán học ở mức độ {level}, dựa trên nội dung sau (thuộc chủ đề "{concept}"):

    --- Nội dung ---
    {text}
    ----------------

    Yêu cầu:
    - Mỗi câu hỏi nên rõ ràng, không lặp ý.
    - Tập trung vào kiến thức của chủ đề "{concept}".
    - Không cần lời giải.
    - Đáp án phải được trả về đúng định dạng JSON như sau:
    [
        {{"question": "..."}},
        {{"question": "..."}},
        ...
    ]
    """
    response = ask_llm(prompt)
    try:
        questions = json.loads(response)
        if isinstance(questions, list) and all("question" in q for q in questions):
            return questions
        else:
            raise ValueError("JSON không hợp lệ hoặc thiếu trường 'question'")
    except Exception:
        return [{"question": response.strip()}]