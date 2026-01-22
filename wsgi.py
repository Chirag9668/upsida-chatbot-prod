from flask import Flask
from app.api import register_routes
from app.inference import IntentClassifier

ID2LABEL = {
    0: "Infrastructure_Road_Condition",
    1: "Waste_Management_Concern",
    2: "Land_Allotment_Query",
    3: "Infrastructure_Water_Supply_Issue",
    4: "Infrastructure_Power_Outage"
}

def create_app():
    app = Flask(__name__)

    classifier = IntentClassifier(
        model_path="model/muril",
        id2label=ID2LABEL
    )

    register_routes(app, classifier)
    return app


# WSGI entry point (IMPORTANT)
app = create_app()