from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import os


app = Flask(__name__)


# --------------------------------------------------
# MLflow server
# --------------------------------------------------

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)


# --------------------------------------------------
# MLflow Run ID
# --------------------------------------------------

RUN_ID = "f874b7a0630f4479ad391e3c1e3efa58"


# --------------------------------------------------
# Model URI
# --------------------------------------------------

MODEL_URI = f"runs:/{RUN_ID}/iris_model"


# --------------------------------------------------
# Load model from MLflow
# --------------------------------------------------

model = mlflow.sklearn.load_model(
    MODEL_URI
)


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Iris ML Model API is running"
    })


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]]

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000
    )
