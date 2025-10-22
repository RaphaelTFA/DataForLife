from transformers import AutoTokenizer, AutoModelForCausalLM
from SLM.fine_tune import math_paraphrase
import torch

model_name = "gpt2-math-paraphraser"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def paraphrase(prompt: str, max_new_tokens=65536):
    if not find_dir("SLM_Q2Q/gpt2-math-paraphraser"):
        math_paraphrase()
    original_prompt = "Hãy viết lại câu hỏi sau mà không làm ảnh hưởng đến con số và ý nghĩa:\n" + (prompt if prompt else "A tank is filled by a pipe in 4 hours and emptied by another in 6 hours. How long will it take to fill the tank if both are open?\n---\n")
    inputs = tokenizer(original_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

