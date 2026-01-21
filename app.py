from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = Flask(__name__, static_folder="static")
CORS(app)

# -----------------------------
# Lazy model loading (IMPORTANT)
# -----------------------------
tokenizer = None
model = None
id2label = None

def load_model():
    global tokenizer, model, id2label
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained("muril_model")
        model = AutoModelForSequenceClassification.from_pretrained("muril_model")
        model.eval()
        id2label = model.config.id2label

# -----------------------------
# Utilities
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def detect_language(text):
    devanagari = sum('\u0900' <= ch <= '\u097F' for ch in text)
    latin = sum('a' <= ch.lower() <= 'z' for ch in text)
    if devanagari > 0 and latin > 0:
        return "Hinglish"
    elif devanagari > 0:
        return "Hindi"
    elif latin > 0:
        return "English"
    return "Unknown"

def predict_intent(text):
    load_model()  # lazy load here
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "text_input": text,
        "language_detected": detect_language(text),
        "predicted_intent": id2label[pred.item()],
        "confidence_score": round(conf.item(), 2)
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "API running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()

    invalid_inputs = [
        "hi", "hello", "how are you", "who are you", "what is your name", "thanks",
        "thank you", "good morning", "good night", "ok", "bye", "help", "?", "!"
    ]

    if text.lower() in invalid_inputs or len(text.split()) < 3:
        return jsonify({
            "message": "कृपया अपनी समस्या या शिकायत स्पष्ट रूप से लिखें।"
        }), 200

    result = predict_intent(text)
    return jsonify(result)

@app.route("/")
def serve_html():
    return send_from_directory("static", "index.html")

# ❌ DO NOT use app.run() in production