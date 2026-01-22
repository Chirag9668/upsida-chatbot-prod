from flask import Flask
from app.api import register_routes
from app.inference import IntentClassifier


def create_app():
    app = Flask(__name__)

    # ✅ NO arguments here (classifier already knows model path)
    classifier = IntentClassifier()

    register_routes(app, classifier)
    return app


# ✅ WSGI entry point
app = create_app()
