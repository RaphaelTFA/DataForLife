import os
from openai import OpenAI
from math_rag_pipeline.src.rag_toan.config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1/", # Mention! Openrouter is the best option for now
    api_key = OPENAI_API_KEY if OPENAI_API_KEY else os.getenv("OPENAI_API_KEY"),
)

def ask_llm(prompt: str, model: str = OPENAI_MODEL, max_tokens: int = 65536) -> str:
    model = model or OPENAI_MODEL
    system_prompt = "Chỉ đưa ra câu trả lời theo yêu cầu của câu hỏi, không được phép không được đưa ra câu đánh giá hoặc lời nói bên lề"
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()