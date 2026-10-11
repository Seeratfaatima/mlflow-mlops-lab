import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Point client to our SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Get the experiment by name (from Task 4)
exp = client.get_experiment_by_name("sklearn-classifier")
print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")

# 2. Search runs using SQL-like filter criteria
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.accuracy > 0.90",
    order_by=["metrics.roc_auc DESC"],
    max_results=10
)

print(f"\nFound {len(runs)} qualifying runs:\n")

# 3. Build a clean comparison summary table
rows = []
for run in runs:
    rows.append({
        "run_id":       run.info.run_id[:8],  # First 8 chars for easy display
        "status":       run.info.status,
        "n_estimators": run.data.params.get("n_estimators"),
        "max_depth":    run.data.params.get("max_depth"),
        "accuracy":     round(run.data.metrics.get("accuracy", 0), 4),
        "roc_auc":      round(run.data.metrics.get("roc_auc", 0), 4),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# 4. Print the best run's full run_id (save this for Task 7!)
if runs:
    best_run = runs[0]
    print(f"\n🏆 Best run_id: {best_run.info.run_id}")
    print(f"Best params: {best_run.data.params}")
