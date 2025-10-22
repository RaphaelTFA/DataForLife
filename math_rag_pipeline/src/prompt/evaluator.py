from math_rag_pipeline.src.rag_toan.llm.client import ask_llm
def evaluate_question(question: str, answer: str):
    with open("math_rag_pipeline/src/prompt/math_test.example.py", "r", encoding="utf-8") as f:
        example_content = f.read()
    prompt = f"""
    Bạn là chuyên gia ra đề và đánh giá câu hỏi Toán học.
    Hãy đánh giá câu hỏi sau dựa trên các tiêu chí sau và trả về kết quả dưới dạng JSON với đúng các trường:
    {{
        "clarity": "rõ ràng / chưa rõ ràng", 
        "difficulty": "dễ / trung bình / khó",
        "correctness": "đúng / sai / không đủ thông tin để đánh giá",
        "is_real_math_problem": true/false,   // Câu hỏi có phải là bài toán thực tế, hợp lệ không?
        "vietnamese_math_style": true/false,  // Câu hỏi có giống phong cách đề toán thi tốt nghiệp trung học phổ thông tại Việt Nam không?
        "useful_for_students": true/false,    // Câu hỏi có phù hợp và hữu ích để học sinh luyện tập không?
        "feedback": "Giải thích ngắn gọn lý do cho các đánh giá trên."
    }}
    Câu hỏi: {question}
    Đáp án đề xuất: {answer}
    Chỉ trả về JSON hợp lệ, không giải thích ngoài JSON.
    Nội dung đề thi Toán tốt nghiệp trung học phổ thông Việt Nam (tham khảo):
    ---------
    {example_content} 
    ---------
    """
    return ask_llm(prompt)
