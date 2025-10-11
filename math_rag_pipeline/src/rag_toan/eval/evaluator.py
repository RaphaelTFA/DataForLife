import json
from math_rag_pipeline.src.rag_toan.retriever.retriever import query_vector_db
from math_rag_pipeline.src.rag_toan.llm.client import ask_llm

def evaluate(dataset_path: str = "data/eval/sample_qa.json", top_k=5):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    correct = 0
    for item in dataset:
        q, gold = item["question"], item["answer"]
        docs = query_vector_db(q, top_k=top_k)
        llm_ans = ask_llm(q, docs)
        if gold.lower() in llm_ans.lower():
            correct += 1
        print(f"\nQ: {q}\nA(gold): {gold}\nA(pred): {llm_ans[:200]}")

    acc = correct / total * 100
    print(f"\n✅ Accuracy: {acc:.2f}% ({correct}/{total})")
