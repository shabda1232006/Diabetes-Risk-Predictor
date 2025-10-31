import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	classification_report,
	f1_score,
	roc_auc_score,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
META_PATH = PROCESSED_DIR / "metadata.json"
MODEL_PATH = MODELS_DIR / "model.joblib"


def load_splits():
	train = pd.read_csv(PROCESSED_DIR / "train.csv")
	val = pd.read_csv(PROCESSED_DIR / "val.csv")
	test = pd.read_csv(PROCESSED_DIR / "test.csv")
	with open(META_PATH, "r", encoding="utf-8") as f:
		meta = json.load(f)
	features = meta["features"]
	target = meta["target"]
	X_train, y_train = train[features], train[target]
	X_val, y_val = val[features], val[target]
	X_test, y_test = test[features], test[target]
	return (X_train, y_train, X_val, y_val, X_test, y_test, features, target)


def build_models(random_state: int = 42):
	# Two strong baselines for tabular data
	logit = Pipeline([
		("scaler", StandardScaler()),
		("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
	])
	hgb = HistGradientBoostingClassifier(
		learning_rate=0.08,
		max_depth=None,
		max_iter=300,
		l2_regularization=0.0,
		random_state=random_state,
	)
	return {"logistic_regression": logit, "hist_gbdt": hgb}


def evaluate_and_save(best_name: str, model, X_test, y_test, features):
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	REPORTS_DIR.mkdir(parents=True, exist_ok=True)

	probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
	preds = (probs >= 0.5).astype(int)

	metrics = {
		"roc_auc": float(roc_auc_score(y_test, probs)),
		"avg_precision": float(average_precision_score(y_test, probs)),
		"f1": float(f1_score(y_test, preds)),
		"accuracy": float(accuracy_score(y_test, preds)),
	}

	# Save model and metrics
	artifact = {
		"model_name": best_name,
		"features": features,
		"metrics": metrics,
	}
	joblib.dump({"model": model, "artifact": artifact}, MODEL_PATH)

	with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
		json.dump(artifact, f, indent=2)

	# Calibration plot
	plt.figure(figsize=(6, 6))
	CalibrationDisplay.from_predictions(y_test, probs, n_bins=10, strategy="quantile")
	plt.title("Calibration Curve")
	plt.tight_layout()
	plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=150)
	plt.close()

	print("Saved model to", MODEL_PATH)
	print("Metrics:", json.dumps(metrics, indent=2))


def main(train_flag: bool, eval_flag: bool, seed: int):
	X_train, y_train, X_val, y_val, X_test, y_test, features, _ = load_splits()
	models = build_models(random_state=seed)

	best_name = None
	best_model = None
	best_score = -np.inf

	if train_flag:
		for name, model in models.items():
			model.fit(X_train, y_train)
			# Use ROC AUC on validation to select
			probs_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
			score = roc_auc_score(y_val, probs_val)
			print(f"Validation ROC AUC - {name}: {score:.4f}")
			if score > best_score:
				best_score = score
				best_name = name
				best_model = model

	if eval_flag:
		if best_model is None:
			# Load existing
			bundle = joblib.load(MODEL_PATH)
			best_model = bundle["model"]
			best_name = bundle["artifact"]["model_name"]
		evaluate_and_save(best_name, best_model, X_test, y_test, features)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train and evaluate diabetes risk model")
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--evaluate", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	main(args.train, args.evaluate, args.seed)
