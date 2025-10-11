import os
import openai
from math_rag_pipeline.src.rag_toan.config import OPENAI_API_KEY, OPENAI_MODEL

# configure api key
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_llm(context: str, query: str, system_prompt: str = None, model: str = None, max_tokens: int = 512) -> str:
    model = model or OPENAI_MODEL
    if system_prompt is None:
        system_prompt = "Bạn là trợ lý toán học: trả lời rõ ràng, ngắn gọn, giải thích các bước nếu cần."

    prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}\n\nHãy trả lời bằng tiếng Việt, có giải thích khi cần."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()
