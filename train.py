import mlflow
import mlflow.sklearn

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------
# 1. Pull UCI Iris dataset
# --------------------------------------------------

iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets


# --------------------------------------------------
# 2. Encode target labels
# --------------------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y.iloc[:, 0]
)


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "mlops-flask-api"
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

with mlflow.start_run() as run:

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # 6. Prediction
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------
    # 7. Accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # --------------------------------------------------
    # 8. Log parameters
    # --------------------------------------------------

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "dataset",
        "UCI Iris"
    )


    # --------------------------------------------------
    # 9. Log metric
    # --------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # --------------------------------------------------
    # 10. Log model
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="iris_model"
    )


    # --------------------------------------------------
    # 11. Display results
    # --------------------------------------------------

    print("UCI Iris model trained successfully")
    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)
    print("Classes:", list(label_encoder.classes_))
