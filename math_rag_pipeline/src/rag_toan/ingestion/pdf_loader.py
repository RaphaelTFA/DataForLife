#using dots.ocr
import pdfplumber
from pathlib import Path
from math_rag_pipeline.src.rag_toan.utils.io import write_text_file
from math_rag_pipeline.src.rag_toan.config import EXTRACTED_DIR

def load_pdf_to_pages(pdf_path: str):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text = p.extract_text()
            if text:
                pages.append(text)
    return pages

def save_extracted_text(pdf_path: str, out_basename: str = None) -> str:
    pdf_path = Path(pdf_path)
    if out_basename is None:
        out_basename = pdf_path.stem + ".txt"
    pages = load_pdf_to_pages(str(pdf_path))
    # join pages with explicit marker
    content = "\n\n<!-- PAGE_BREAK -->\n\n".join(pages)
    out_path = EXTRACTED_DIR / out_basename
    write_text_file(str(out_path), content)
    return str(out_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print("Saved to:", save_extracted_text(sys.argv[1]))
    else:
        print("Usage: python pdf_loader.py file.pdf")
