import re
import torch

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ----------------------------
# Language Detection
# ----------------------------
def detect_language(text: str) -> str:
    devanagari = sum('\u0900' <= ch <= '\u097F' for ch in text)
    latin = sum('a' <= ch.lower() <= 'z' for ch in text)

    if devanagari > 0 and latin > 0:
        return "Hinglish"
    elif devanagari > 0:
        return "Hindi"
    elif latin > 0:
        return "English"
    else:
        return "Unknown"


# ----------------------------
# Intent Prediction
# (model & tokenizer passed from api.py)
# ----------------------------
def predict_intent(text, tokenizer, model, id2label):
    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return {
        "predicted_intent": id2label[prediction.item()],
        "language": detect_language(text),
        "confidence": round(confidence.item(), 4)
    }
