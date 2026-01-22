from flask import Blueprint, request, jsonify, send_from_directory

api = Blueprint("api", __name__)

def register_routes(app, classifier):

    @api.route("/", methods=["GET"])
    def home():
        return send_from_directory("static", "index.html")

    @api.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "intent": "unknown",
                "language": "unknown",
                "confidence": 0.0
            })

        # ✅ CALL CLASSIFIER (ONLY THIS)
        result = classifier.predict(text)

        # ✅ RETURN SAME KEYS (NO RENAMING)
        return jsonify(result)

    app.register_blueprint(api)