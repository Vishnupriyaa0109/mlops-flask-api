import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-flask-api")


# Train model
with mlflow.start_run() as run:

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)


    # Prediction
    predictions = model.predict(X_test)


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)


    # Log metric
    mlflow.log_metric("accuracy", accuracy)


    # Log model
    mlflow.sklearn.log_model(
        model,
        name="iris_model"
    )


    print("Model trained successfully")
    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)
