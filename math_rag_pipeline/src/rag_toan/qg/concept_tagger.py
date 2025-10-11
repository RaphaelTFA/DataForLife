from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def tag_concept(text: str) -> str:
    prompt = f"""
    Xác định chủ đề Toán học chính của đoạn sau.
    Trả kết quả ngắn gọn, ví dụ: "Đạo hàm", "Tích phân", "Hình học không gian".

    Đoạn văn:
    {text[:1500]}  # tránh prompt quá dài
    """
    return ask_llm(prompt).strip()
