from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def evaluate_question(question: str, answer: str):
    prompt = f"""
    Đánh giá câu hỏi sau về độ rõ ràng, độ khó (dễ/trung bình/khó), và tính chính xác của đáp án.
    Trả kết quả JSON: {{"clarity": ..., "difficulty": ..., "correctness": ...}}

    Câu hỏi: {question}
    Đáp án: {answer}
    """
    return ask_llm(prompt)
