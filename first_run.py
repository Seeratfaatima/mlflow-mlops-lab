import mlflow

# Tell MLflow to use local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Set the experiment name
mlflow.set_experiment("my-first-experiment")

# 2. Start a tracked run
with mlflow.start_run(run_name="baseline"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Simulate training loop — log metrics over epochs
    for epoch in range(10):
        mlflow.log_metric("train_loss", 1.0 - epoch * 0.08, step=epoch)
        mlflow.log_metric("val_loss",   1.0 - epoch * 0.06, step=epoch)

    # Log tags
    mlflow.set_tag("team", "mlops-lab")
    mlflow.set_tag("data_version", "v1.0")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
