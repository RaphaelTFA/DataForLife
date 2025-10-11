import json
import os
from typing import List, Dict, Any, Optional
from math_rag_pipeline.src.rag_toan.llm.client import ask_llm
from math_rag_pipeline.src.rag_toan.retriever.retriever import query_vector_db
from math_rag_pipeline.src.rag_toan.config import OPENAI_MODEL
from pathlib import Path

# Dictionary mapping education levels to their appropriate grade ranges
EDUCATION_LEVELS = {
    "elementary": "lớp 1-5",
    "middle_school": "lớp 6-9",
    "high_school": "lớp 10-12",
    "university": "đại học"
}

# Dictionary mapping subjects to their corresponding topics
MATH_SUBJECTS = {
    "algebra": ["biểu thức đại số", "phương trình", "bất phương trình", "hàm số", "logarit", "số mũ"],
    "geometry": ["hình học phẳng", "hình học không gian", "vector", "tọa độ", "góc", "tam giác"],
    "calculus": ["đạo hàm", "tích phân", "giới hạn", "cực trị", "hàm số liên tục"],
    "statistics": ["xác suất", "thống kê", "tổ hợp", "chỉnh hợp", "phân phối"],
    "trigonometry": ["lượng giác", "sin", "cos", "tan", "hàm số lượng giác"]
}

# Dictionary mapping difficulty levels to descriptions
DIFFICULTY_LEVELS = {
    "easy": "dễ, cơ bản, kiến thức đơn giản",
    "medium": "trung bình, yêu cầu hiểu kiến thức",
    "hard": "khó, yêu cầu phân tích sâu và kỹ năng giải toán tốt",
    "challenging": "thách thức, câu hỏi phức tạp, yêu cầu sáng tạo và tổng hợp nhiều kiến thức"
}

def generate_qa_pairs_from_text(text: str, concept: str, num: int = 5) -> List[Dict[str, str]]:
    """Generate QA pairs based on provided text and concept"""
    prompt = f"""
    Sinh {num} cặp (Câu hỏi, Đáp án) Toán học thuộc chủ đề "{concept}".
    Trả về JSON dạng:
    [
      {{"question": "...", "answer": "..."}}
    ]
    Dựa trên nội dung:
    {text[:3000]}
    """
    raw = ask_llm(prompt, "")
    try:
        return json.loads(raw)
    except Exception:
        lines = [l for l in raw.split("\n") if l.strip()]
        return [{"question": l, "answer": ""} for l in lines]

def generate_qa_pairs(subject: str, difficulty: str, 
                     education_level: str = "high_school", 
                     num: int = 5, 
                     use_rag: bool = True) -> List[Dict[str, str]]:
    """
    Generate QA pairs based on subject and difficulty without requiring text input.
    
    Args:
        subject: One of the math subjects (algebra, geometry, calculus, statistics, trigonometry)
        difficulty: Difficulty level (easy, medium, hard, challenging)
        education_level: Education level (elementary, middle_school, high_school, university)
        num: Number of QA pairs to generate
        use_rag: Whether to use RAG to find relevant context before generating questions
        
    Returns:
        List of dictionaries containing question and answer pairs
    """
    # Validate inputs
    if subject not in MATH_SUBJECTS:
        raise ValueError(f"Subject must be one of: {', '.join(MATH_SUBJECTS.keys())}")
    
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"Difficulty must be one of: {', '.join(DIFFICULTY_LEVELS.keys())}")
    
    if education_level not in EDUCATION_LEVELS:
        raise ValueError(f"Education level must be one of: {', '.join(EDUCATION_LEVELS.keys())}")
    
    # Get topic details
    topics = MATH_SUBJECTS[subject]
    difficulty_desc = DIFFICULTY_LEVELS[difficulty]
    grade_range = EDUCATION_LEVELS[education_level]
    
    # Use RAG to find relevant context if available and requested
    context = ""
    if use_rag:
        try:
            # Query for relevant context from the vector DB
            search_query = f"{subject} {topics[0]} {grade_range}"
            docs = query_vector_db(search_query, top_k=3)
            if docs:
                context = "\n\n---\n\n".join([d["text"] for d in docs])
        except Exception as e:
            print(f"Warning: RAG retrieval failed: {e}. Proceeding without context.")
    
    # Construct the prompt
    system_prompt = """Bạn là một giáo viên toán giỏi. Nhiệm vụ của bạn là tạo ra các câu hỏi và đáp án toán học 
    chất lượng cao, phù hợp với cấp độ giáo dục và độ khó yêu cầu. Câu hỏi phải rõ ràng, có cấu trúc tốt. 
    Đáp án phải đầy đủ với các bước giải, công thức và lời giải thích."""
    
    prompt = f"""
    Vui lòng tạo {num} câu hỏi toán học và đáp án chi tiết về chủ đề: {subject} ({', '.join(topics)}).
    
    Yêu cầu:
    - Câu hỏi dành cho {grade_range}
    - Độ khó: {difficulty} ({difficulty_desc})
    - Mỗi câu hỏi phải khác nhau và có tính ứng dụng
    - Đáp án phải chi tiết, có đầy đủ các bước và công thức
    
    Trả về kết quả theo định dạng JSON sau:
    [
        {{
            "question": "Nội dung câu hỏi...",
            "answer": "Đáp án chi tiết với các bước giải..."
        }},
        ...
    ]
    """
    
    # Add context if available
    if context:
        prompt += f"\n\nTham khảo nội dung sau để tạo câu hỏi liên quan:\n{context[:2000]}"
    
    # Call LLM
    raw_response = ask_llm(prompt, system_prompt, model=OPENAI_MODEL, max_tokens=2048)
    
    # Process response
    try:
        # Try to parse JSON response
        qa_pairs = json.loads(raw_response)
        return qa_pairs
    except json.JSONDecodeError:
        # Fallback parsing if JSON parsing fails
        print("Warning: Failed to parse JSON. Using fallback parsing method.")
        result = []
        current_qa = {"question": "", "answer": ""}
        
        for line in raw_response.split('\n'):
            line = line.strip()
            if line.startswith('{"question"') or line.startswith('[{"question"'):
                try:
                    # Try parsing the whole response again
                    if '[' in line:
                        start_idx = line.find('[')
                        return json.loads(line[start_idx:])
                    else:
                        start_idx = line.find('{')
                        result.append(json.loads(line[start_idx:]))
                except:
                    pass
            elif "question" in line.lower() and ":" in line:
                # Save previous QA pair if it exists
                if current_qa["question"] and current_qa["answer"]:
                    result.append(current_qa.copy())
                # Start new QA pair
                current_qa = {
                    "question": line.split(":", 1)[1].strip().strip('"'),
                    "answer": ""
                }
            elif "answer" in line.lower() and ":" in line and not current_qa["answer"]:
                current_qa["answer"] = line.split(":", 1)[1].strip().strip('"')
            elif current_qa["question"] and not current_qa["answer"]:
                # Still collecting the question
                current_qa["question"] += " " + line
            elif current_qa["question"] and current_qa["answer"]:
                # Collecting the answer
                current_qa["answer"] += " " + line
        
        # Add the last QA pair if it exists
        if current_qa["question"] and current_qa["answer"]:
            result.append(current_qa)
            
        return result if result else [
            {"question": "Không thể sinh câu hỏi. Vui lòng thử lại với chủ đề khác.", 
             "answer": "Không có đáp án."}
        ]

def save_qa_pairs(qa_pairs: List[Dict[str, str]], 
                 subject: str, 
                 difficulty: str,
                 education_level: str = "high_school", 
                 output_dir: str = None) -> str:
    """Save generated QA pairs to a JSON file"""
    if output_dir is None:
        # Default to a 'generated_qa' directory in the project
        output_dir = Path.cwd() / "data" / "generated_qa"
        
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on parameters
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{subject}_{education_level}_{difficulty}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "subject": subject,
                "difficulty": difficulty,
                "education_level": education_level,
                "generated_at": timestamp,
                "count": len(qa_pairs)
            },
            "qa_pairs": qa_pairs
        }, f, ensure_ascii=False, indent=2)
    
    return filepath
