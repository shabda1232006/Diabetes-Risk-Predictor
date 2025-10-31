import argparse
import json
import os
from pathlib import Path
from typing import List

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_binary_health_indicators_BRFSS2015.csv"
FALLBACK_URLS = [
	UCI_URL,
	"https://raw.githubusercontent.com/akmand/datasets/master/diabetes_binary_health_indicators_BRFSS2015.csv",
]
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
META_PATH = PROCESSED_DIR / "metadata.json"

# Selected features from BRFSS 2015 dataset
# Reduced feature set prioritizing predictive power and simpler user input
FEATURES: List[str] = [
	"BMI",                  # Body Mass Index
	"Age",                  # Age band (1..13)
	"GenHlth",              # General health (1..5)
	"HighBP",               # High blood pressure (0/1)
	"HighChol",             # High cholesterol (0/1)
	"PhysActivity",         # Physical activity (0/1)
]
TARGET = "Diabetes_binary"


def ensure_dirs() -> None:
	RAW_DIR.mkdir(parents=True, exist_ok=True)
	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
	"""Download the BRFSS 2015 diabetes indicators dataset to data/raw with fallbacks."""
	ensure_dirs()
	csv_path = RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
	if csv_path.exists():
		return csv_path
	last_error: Exception | None = None
	for url in FALLBACK_URLS:
		try:
			# Use pandas to download and standardize write
			df = pd.read_csv(url)
			df.to_csv(csv_path, index=False)
			return csv_path
		except Exception as e:
			last_error = e
			continue
	# If all sources failed, raise a helpful error
	raise RuntimeError(
		"Failed to download dataset from all known sources. "
		"Check your internet connection or manually download the CSV and place it at "
		f"{csv_path}. Last error: {last_error}"
	)


def preprocess_dataset(raw_csv: Path | None = None, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> None:
	"""Preprocess and split the dataset, saving train/val/test CSVs and metadata."""
	from sklearn.model_selection import train_test_split

	ensure_dirs()
	if raw_csv is None:
		raw_csv = RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
		if not raw_csv.exists():
			raw_csv = download_dataset()

	df = pd.read_csv(raw_csv)

	# Harmonize target: allow either Diabetes_binary or Diabetes_012 (map 1/2 -> 1)
	if TARGET not in df.columns:
		if "Diabetes_012" in df.columns:
			df[TARGET] = (df["Diabetes_012"] >= 1).astype(int)
		else:
			raise ValueError("Expected target column 'Diabetes_binary' or 'Diabetes_012' not found")

	# Keep only selected features and target
	missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
	if missing_columns:
		raise ValueError(f"Missing expected columns: {missing_columns}")

	df = df[FEATURES + [TARGET]].copy()

	# Handle any NA by simple imputation (dataset mostly has 0/NA); use median
	df = df.fillna(df.median(numeric_only=True))

	X = df[FEATURES]
	y = df[TARGET]

	# First split off test, then split train into train/val
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)
	val_rel = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=val_rel, random_state=random_state, stratify=y_train
	)

	# Save splits
	X_train.assign(**{TARGET: y_train}).to_csv(PROCESSED_DIR / "train.csv", index=False)
	X_val.assign(**{TARGET: y_val}).to_csv(PROCESSED_DIR / "val.csv", index=False)
	X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DIR / "test.csv", index=False)

	# Save metadata
	metadata = {
		"features": FEATURES,
		"target": TARGET,
		"random_state": random_state,
		"splits": {"test_size": test_size, "val_size": val_size},
	}
	with open(META_PATH, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BRFSS Diabetes data utilities")
    parser.add_argument("--download", action="store_true", help="Download raw dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess and split dataset")
    parser.add_argument("--raw_csv", type=str, default=None, help="Path to a local raw CSV (skip download)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.download:
        path = download_dataset()
        print(f"Downloaded to {path}")

    if args.preprocess:
        raw_csv_path = Path(args.raw_csv) if args.raw_csv else None
        preprocess_dataset(raw_csv=raw_csv_path, test_size=args.test_size, val_size=args.val_size, random_state=args.seed)
        print("Preprocessed data saved in data/processed")
