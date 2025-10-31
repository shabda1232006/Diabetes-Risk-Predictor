import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from pathlib import Path
import pandas as pd

# Set paths (relative to reports/)
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "data" / "processed" / "metadata.json"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
ROC_PATH = ROOT / "reports" / "roc_curve.png"

# Load metadata for features/target
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
features = meta["features"]
target = meta["target"]

# Load test data
print(f"Loading test data from {TEST_PATH}")
test = pd.read_csv(TEST_PATH)
X_test = test[features]
y_test = test[target]

# Load model
print(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)["model"]

# Get predicted probabilities
probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)

# Plot ROC curve
plt.figure(figsize=(6,6))
roc_disp = RocCurveDisplay.from_predictions(y_test, probs)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig(ROC_PATH)
plt.close()

print(f"ROC curve saved to {ROC_PATH}")
