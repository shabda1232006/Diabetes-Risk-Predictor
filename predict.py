import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
import sys

MODEL_PATH = Path("models/model.joblib")
PROCESSED_META = Path("data/processed/metadata.json")


def load_model_and_schema():
	bundle = joblib.load(MODEL_PATH)
	with open(PROCESSED_META, "r", encoding="utf-8") as f:
		meta = json.load(f)
	features: List[str] = meta["features"]
	return bundle["model"], features


def read_input(input_path: Optional[str], features: List[str]) -> pd.DataFrame:
	if input_path is None:
		# Read JSON from stdin, one object or array
		payload = json.load(sys.stdin)
		if isinstance(payload, dict):
			frame = pd.DataFrame([payload])
		else:
			frame = pd.DataFrame(payload)
	else:
		path = Path(input_path)
		if path.suffix.lower() == ".csv":
			frame = pd.read_csv(path)
		else:
			frame = pd.read_json(path)
	missing = [c for c in features if c not in frame.columns]
	if missing:
		raise ValueError(f"Missing required feature columns: {missing}")
	return frame[features]


def main(input_path: Optional[str], output_path: Optional[str]):
	model, features = load_model_and_schema()
	X = read_input(input_path, features)
	probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
	out = X.copy()
	out["diabetes_risk_proba"] = probs
	if output_path:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		out.to_csv(output_path, index=False)
		print(f"Wrote predictions to {output_path}")
	else:
		print(out.to_csv(index=False))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Predict diabetes risk from CSV/JSON")
	parser.add_argument("--input", type=str, default=None, help="Path to input CSV/JSON. If omitted, read JSON from stdin.")
	parser.add_argument("--output", type=str, default=None, help="Path to output CSV. If omitted, prints to stdout.")
	args = parser.parse_args()
	main(args.input, args.output)
