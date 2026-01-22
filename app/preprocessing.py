import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def detect_language(text: str) -> str:
    devanagari = sum('\u0900' <= ch <= '\u097F' for ch in text)
    latin = sum('a' <= ch.lower() <= 'z' for ch in text)

    if devanagari and latin:
        return "Hinglish"
    if devanagari:
        return "Hindi"
    if latin:
        return "English"
    return "Unknown"