import mlflow
from mlflow.tracking import MlflowClient

# Connect to our SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Your champion run_id from Task 6
BEST_RUN_ID = "1613206a8bf347078d8b6dd4d8b334ff"
MODEL_NAME = "BreastCancerClassifier"

# 1. Register the model from the run's artifacts
# (creates version 1, or increments to v2, v3 if it already exists)
model_uri = f"runs:/{BEST_RUN_ID}/model"
result = mlflow.register_model(model_uri, MODEL_NAME)

print(f"Registered: {MODEL_NAME} v{result.version}")
print(f"Initial stage: {result.current_stage}")

# 2. Add documentation/description to this version
client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description="Random Forest on Breast Cancer dataset. Accuracy: 0.965, ROC-AUC: 0.997."
)

# 3. Add governance tags
client.set_registered_model_tag(MODEL_NAME, "framework", "sklearn")
client.set_registered_model_tag(MODEL_NAME, "owner", "mlops-student")

# 4. Transition stage: Move to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True  # Automatically archives older Production versions!
)
print(f"🎉 Successfully promoted {MODEL_NAME} v{result.version} to Production!")
