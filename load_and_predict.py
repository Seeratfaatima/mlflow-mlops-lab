import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Point to our SQLite registry database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "BreastCancerClassifier"
MODEL_STAGE = "Production"   # Or "1"

# --- Option A: Universal Python API (mlflow.pyfunc) ---
# pyfunc is framework-agnostic: this same code works whether the model is
# scikit-learn, PyTorch, TensorFlow, or XGBoost!
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# Load a sample of 5 rows from the Breast Cancer dataset
data = load_breast_cancer()
sample = pd.DataFrame(data.data[:5], columns=data.feature_names)

# Make predictions
preds = pyfunc_model.predict(sample)
print("🎯 pyfunc predictions (0=malignant, 1=benign):", preds)

# --- Option B: Framework-Specific (mlflow.sklearn) ---
# Useful when you need framework-specific methods like predict_proba()
sk_model = mlflow.sklearn.load_model(model_uri)
proba = sk_model.predict_proba(sample)[:, 1]
print("📊 Probability of benign:", proba.round(3))
