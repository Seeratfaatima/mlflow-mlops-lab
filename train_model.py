import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Connect to SQLite tracking backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("sklearn-classifier")

# Hyperparameters (we will tweak these to compare runs)
params = {
    "n_estimators": 50,
    "max_depth": 3,
    "min_samples_split": 2,
    "random_state": 42
}

with mlflow.start_run():
    # --- 1. Load Data ---
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    mlflow.log_param("dataset", "breast_cancer")
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test",  len(X_test))

    # --- 2. Log Hyperparameters ---
    mlflow.log_params(params)

    # --- 3. Train Model ---
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # --- 4. Evaluate ---
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
    }
    mlflow.log_metrics(metrics)

    # --- 5. Feature Importance Chart ---
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([data.feature_names[i] for i in top_idx], importances[top_idx])
    ax.set_title("Top 10 feature importances")
    fig.tight_layout()
    fig.savefig("feature_importance.png", dpi=100)
    mlflow.log_artifact("feature_importance.png", artifact_path="plots")
    plt.close()

    # --- 6. Log the Trained Model ---
    mlflow.sklearn.log_model(
    model, 
    artifact_path="model", 
    serialization_format="cloudpickle"
)
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
