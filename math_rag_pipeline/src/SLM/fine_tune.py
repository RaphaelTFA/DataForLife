from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch

def math_paraphrase():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset("json", data_files={"train": "SLM/data/math_phrase.json"})

    max_length = 256

    def preprocess_function(examples):
        texts = []
        for content, difficulty, question in zip(examples["content"], examples["difficulty"], examples["question"]):
            text = f"Độ khó: {difficulty}\nCâu hỏi: {question}\nNội dung: {content}"
            texts.append(text)

        outputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir="math-paraphraser",
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"].shuffle(seed=42),
    )

    trainer.train()

    save_dir = "math-paraphraser"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
