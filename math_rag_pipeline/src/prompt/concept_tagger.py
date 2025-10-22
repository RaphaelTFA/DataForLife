from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def tag_concept(text: str) -> str:
    prompt = f"""
    Xác định chủ đề Toán học chính của đoạn sau.
    Trả kết quả ngắn gọn, thuộc các chủ đề sau (định dạng JSON):
    {
        "10": [
            "Mệnh đề - Tập hợp",
            "Bất phương trình và hệ bất phương trình bậc nhất hai ẩn",
            "Hệ thức lượng trong tam giác",
            "Vector",
            "Các số đặc trưng của mẫu số liệu, không ghép nhóm",
            "Hàm số, đồ thị và ứng dụng",
            "Phương pháp tọa độ trong mặt phẳng",
            "Đại số tổ hợp": ["Quy tắc đếm", "Hoán vị, chỉnh hợp và tổ hợp", "Nhị thức Newton"],
            "Tính xác suất theo định nghĩa cổ điển": ["Biến cố và định nghĩa cổ điển của xác suất", "Thực hành tính xác suất theo định nghĩa cổ điển"]
        ]
        "11": [
            "Hàm số lượng giác - Phương trình lượng giác",
            "Dãy số - Cấp số cộng - Cấp số nhân",
            "Các số đặc trưng đo xu thế trung tâm của mẫu số liệu ghép nhóm",
            "Quan hệ song song trong không gian",
            "Giới hạn - Hàm số liên tục",
            "Hàm số mũ - Hàm số Logarit",
            "Quan hệ vuông góc trong không gian",
            "Đạo hàm"
        ],
        "12": [
            "Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số",
            "Vector và hệ trục tọa độ trong không gian",
            "Các số đặc trưng đo mức độ phân tán của mẫu số liệu ghép nhóm",
            "Nguyên hàm - Tích phân",
            "Phương pháp tọa độ trong không gian",
            "Xác suất có điều kiện"
        ]
    } Đoạn văn:
    {text[:1500]} 
    """
    return ask_llm(prompt).strip()
