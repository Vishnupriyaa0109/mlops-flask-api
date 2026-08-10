from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn


app = Flask(__name__)


# --------------------------------------------------
# MLflow server
# --------------------------------------------------

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


# --------------------------------------------------
# MLflow Run ID
# --------------------------------------------------

RUN_ID = "919a106797a84a059464bb19d4846690"


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
