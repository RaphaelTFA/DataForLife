import re

def clean_text(text: str) -> str:
    # Normalize spaces/newlines and remove control chars except common punctuation and Vietnamese letters
    text = text.replace("\u00a0", " ")
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Keep Unicode range for Vietnamese letters and basic punctuation
    text = re.sub(r'[^\n\x20-\x7EÀ-ỹ\.,;:!?()“”"\'%+\-=/\u2000-\u206F\u2E00-\u2E7F]', '', text)
    return text.strip()
