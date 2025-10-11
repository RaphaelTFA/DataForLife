from pathlib import Path
import json
import os

def write_text_file(path: str, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_json(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# These functions are duplicates of read_text_file and write_text_file
# Keeping them for backward compatibility
def read_text(path: str) -> str:
    return read_text_file(path)

def write_text(path: str, content: str):
    write_text_file(path, content)