"""
Standalone script to generate model comparison visualizations for PPT presentation.
Does not modify existing code.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
META_PATH = PROCESSED_DIR / "metadata.json"


def load_data():
	"""Load train, validation, and test splits."""
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
	"""Build both models."""
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
	return {"Logistic Regression": logit, "Histogram GBDT": hgb}


def plot_confusion_matrices(models_results, reports_dir):
	"""Plot confusion matrices for both models."""
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	
	for idx, (name, results) in enumerate(models_results.items()):
		cm = confusion_matrix(results["y_test"], results["predictions"])
		sns.heatmap(
			cm,
			annot=True,
			fmt="d",
			cmap="Blues",
			ax=axes[idx],
			cbar_kws={"shrink": 0.8},
			xticklabels=["No Diabetes", "Diabetes"],
			yticklabels=["No Diabetes", "Diabetes"],
		)
		axes[idx].set_title(f"Confusion Matrix - {name}", fontsize=12, fontweight="bold")
		axes[idx].set_ylabel("True Label", fontsize=10)
		axes[idx].set_xlabel("Predicted Label", fontsize=10)
	
	plt.tight_layout()
	plt.savefig(reports_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
	plt.close()
	print("✓ Saved confusion_matrices.png")


def plot_roc_curves(models_results, reports_dir):
	"""Plot ROC curves comparing both models."""
	plt.figure(figsize=(8, 6))
	
	for name, results in models_results.items():
		fpr, tpr, _ = roc_curve(results["y_test"], results["probabilities"])
		roc_auc = roc_auc_score(results["y_test"], results["probabilities"])
		plt.plot(
			fpr,
			tpr,
			label=f"{name} (AUC = {roc_auc:.3f})",
			linewidth=2,
		)
	
	plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", linewidth=1)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate", fontsize=11)
	plt.ylabel("True Positive Rate", fontsize=11)
	plt.title("ROC Curves - Model Comparison", fontsize=13, fontweight="bold")
	plt.legend(loc="lower right", fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(reports_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
	plt.close()
	print("✓ Saved roc_curves.png")


def plot_feature_importance(hgb_model, features, reports_dir):
	"""Plot feature importance for Histogram GBDT model."""
	if hasattr(hgb_model, "feature_importances_"):
		importances = hgb_model.feature_importances_
	else:
		return
	
	# Sort features by importance
	indices = np.argsort(importances)[::-1]
	sorted_features = [features[i] for i in indices]
	sorted_importances = importances[indices]
	
	plt.figure(figsize=(9, 6))
	colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
	plt.barh(range(len(sorted_features)), sorted_importances, color=colors)
	plt.yticks(range(len(sorted_features)), sorted_features)
	plt.xlabel("Feature Importance", fontsize=11)
	plt.title("Feature Importance - Histogram GBDT Model", fontsize=13, fontweight="bold")
	plt.gca().invert_yaxis()
	plt.grid(True, alpha=0.3, axis="x")
	plt.tight_layout()
	plt.savefig(reports_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
	plt.close()
	print("✓ Saved feature_importance.png")


def generate_comparison_summary(models_results, reports_dir):
	"""Generate a brief summary for PPT."""
	summary = []
	summary.append("=" * 60)
	summary.append("MODEL COMPARISON SUMMARY - Diabetes Prediction")
	summary.append("=" * 60)
	summary.append("")
	
	# Metrics comparison
	summary.append("PERFORMANCE METRICS (Test Set):")
	summary.append("-" * 60)
	summary.append(f"{'Metric':<20} {'Logistic Regression':<25} {'Histogram GBDT':<25}")
	summary.append("-" * 60)
	
	metrics = ["ROC AUC", "Accuracy", "F1 Score", "Avg Precision"]
	for metric in metrics:
		lr_val = f"{models_results['Logistic Regression'][metric.lower().replace(' ', '_')]:.4f}"
		gbdt_val = f"{models_results['Histogram GBDT'][metric.lower().replace(' ', '_')]:.4f}"
		summary.append(f"{metric:<20} {lr_val:<25} {gbdt_val:<25}")
	
	summary.append("")
	summary.append("KEY FINDINGS:")
	summary.append("-" * 60)
	
	best_model = "Histogram GBDT" if models_results["Histogram GBDT"]["roc_auc"] > models_results["Logistic Regression"]["roc_auc"] else "Logistic Regression"
	
	summary.append(f"• Best Model: {best_model}")
	summary.append(f"  - ROC AUC: {models_results[best_model]['roc_auc']:.4f}")
	summary.append(f"  - Accuracy: {models_results[best_model]['accuracy']:.4f}")
	summary.append("")
	summary.append("• Model Characteristics:")
	summary.append("  - Logistic Regression: Interpretable, fast, linear relationships")
	summary.append("  - Histogram GBDT: Non-linear patterns, better performance")
	summary.append("")
	summary.append("• Feature Set: BMI, Age, GenHlth, HighBP, HighChol, PhysActivity")
	summary.append("=" * 60)
	
	summary_text = "\n".join(summary)
	
	# Save to file
	with open(reports_dir / "comparison_summary.txt", "w", encoding="utf-8") as f:
		f.write(summary_text)
	
	print("✓ Saved comparison_summary.txt")
	return summary_text


def main():
	"""Main function to generate all comparison visualizations."""
	print("Generating model comparison visualizations...")
	print("-" * 60)
	
	# Load data
	X_train, y_train, X_val, y_val, X_test, y_test, features, _ = load_data()
	
	# Build models
	models = build_models(random_state=42)
	
	# Train and evaluate both models
	models_results = {}
	
	for name, model in models.items():
		print(f"\nTraining {name}...")
		model.fit(X_train, y_train)
		
		# Predictions
		if hasattr(model, "predict_proba"):
			probabilities = model.predict_proba(X_test)[:, 1]
		else:
			probabilities = model.predict(X_test)
		predictions = (probabilities >= 0.5).astype(int)
		
		# Calculate metrics
		metrics = {
			"roc_auc": roc_auc_score(y_test, probabilities),
			"accuracy": accuracy_score(y_test, predictions),
			"f1_score": f1_score(y_test, predictions),
			"avg_precision": average_precision_score(y_test, probabilities),
		}
		
		print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
		print(f"  Accuracy: {metrics['accuracy']:.4f}")
		
		models_results[name] = {
			"model": model,
			"y_test": y_test,
			"predictions": predictions,
			"probabilities": probabilities,
			**metrics,
		}
	
	# Ensure reports directory exists
	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	
	# Generate visualizations
	print("\n" + "-" * 60)
	print("Generating visualizations...")
	plot_confusion_matrices(models_results, REPORTS_DIR)
	plot_roc_curves(models_results, REPORTS_DIR)
	
	# Feature importance (only for GBDT)
	if "Histogram GBDT" in models_results:
		hgb_model = models_results["Histogram GBDT"]["model"]
		plot_feature_importance(hgb_model, features, REPORTS_DIR)
	
	# Generate summary
	print("\n" + "-" * 60)
	summary = generate_comparison_summary(models_results, REPORTS_DIR)
	
	print("\n" + "=" * 60)
	print("COMPARISON GENERATION COMPLETE!")
	print("=" * 60)
	print(f"\nGenerated files in '{REPORTS_DIR}':")
	print("  • confusion_matrices.png")
	print("  • roc_curves.png")
	print("  • feature_importance.png")
	print("  • comparison_summary.txt")
	print("\n" + summary)


if __name__ == "__main__":
	main()



