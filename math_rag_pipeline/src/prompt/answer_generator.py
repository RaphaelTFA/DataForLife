from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def generate_answer(text: str, question: str):
    """
    Sinh câu trả lời cho câu hỏi Toán học dựa trên đoạn văn và câu hỏi cụ thể.

    Args:
        text (str): Nội dung hoặc đoạn văn nguồn.
        question (str): Câu hỏi cần trả lời

    Returns:
        List[Dict[str, str]]: Danh sách câu hỏi ở dạng JSON [{ "question": "...", "solution": "...", "answer": "..." }]
    """
    
    prompt = f"""
    Hãy sinh câu trả lời cho câu hỏi sau, dựa trên nội dung sau:

    Câu hỏi: {question}

    --- Nội dung ---
    {text}
    ----------------

    Yêu cầu:
    - Đáp án phải được trả về đúng định dạng JSON như sau:
    [
        {{"question": "{question}", "solution": "...", "answer": "..."}}
    ]

    Yêu cầu:
    - Mỗi câu hỏi nên rõ ràng, không lặp ý.
    - Solution là lời giải đầy đủ, chi tiết từng bước.
    - Answer là câu trả lời số thập phân ngắn gọn, làm tròn đến 2 chữ số sau phần thập phân.
    - Đáp án phải được trả về đúng định dạng JSON như sau:
    [
        {{"question": "...", "solution": "...", "answer": "..."}}
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