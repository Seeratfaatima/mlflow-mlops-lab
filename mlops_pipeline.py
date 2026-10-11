import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Pipeline Configuration
EXPERIMENT    = "mlops-pipeline"
MODEL_NAME    = "ProductionClassifier"
ACCURACY_GATE = 0.93   # Quality gate: model must reach at least 93% accuracy to deploy!

# Connect tracking and registry client to SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

# 2. Data Preparation
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# 3. Model Tournament: 3 Candidate Architectures
candidates = [
    ("RandomForest",  RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), False),
    ("GradBoost",     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), False),
    ("LogisticReg",   LogisticRegression(C=1.0, max_iter=1000, random_state=42), True),  # Needs scaled data
]

print("🏁 Starting Model Tournament...\n")
run_results = []

for name, model, scaled in candidates:
    Xtr_use = X_tr_s if scaled else X_tr
    Xte_use = X_te_s if scaled else X_te

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("scaled", scaled)
        
        model.fit(Xtr_use, y_tr)
        
        acc = accuracy_score(y_te, model.predict(Xte_use))
        auc = roc_auc_score(y_te, model.predict_proba(Xte_use)[:, 1])
        
        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
        
        run_results.append({
            "name": name, 
            "run_id": run.info.run_id, 
            "accuracy": acc, 
            "roc_auc": auc
        })
        print(f"  ⚡ {name:15s} -> Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

# 4. Automated Evaluation & Promotion Gate
best = max(run_results, key=lambda r: r["roc_auc"])
print(f"\n🏆 Champion Candidate: {best['name']} (ROC-AUC={best['roc_auc']:.4f}, Accuracy={best['accuracy']:.4f})")

if best["accuracy"] >= ACCURACY_GATE:
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 QUALITY GATE PASSED! Auto-promoted {best['name']} v{mv.version} -> PRODUCTION!")
else:
    print(f"❌ QUALITY GATE FAILED: Accuracy {best['accuracy']:.4f} < {ACCURACY_GATE}. Deployment aborted.")
