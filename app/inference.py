import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self):
        self.device = torch.device("cpu")

        # EXACT local model path
        self.model_path = r"D:\upsida-chatbot - Copy\model\muril"

        self.id2label = {
            0: "Infrastructure_Road_Condition",
            1: "Waste_Management_Concern",
            2: "Land_Allotment_Query",
            3: "Infrastructure_Water_Supply_Issue",
            4: "Infrastructure_Power_Outage"
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)

        self.model.eval()

    # -------------------------
    # Language detection
    # -------------------------
    def detect_language(self, text: str) -> str:
        devanagari = sum('\u0900' <= ch <= '\u097F' for ch in text)
        latin = sum('a' <= ch.lower() <= 'z' for ch in text)

        if devanagari > 0 and latin > 0:
            return "Hinglish"
        elif devanagari > 0:
            return "Hindi"
        elif latin > 0:
            return "English"
        return "Unknown"

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "language": "unknown",
                "confidence": 0.0
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        return {
            "intent": self.id2label[pred.item()],
            "language": self.detect_language(text),
            "confidence": round(float(confidence.item()), 4)
        }