from pathlib import Path
from typing import Dict, Any

import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "data" / "processed" / "metadata.json"

app = FastAPI(title="Diabetes Risk Predictor")

# Serve static web directory
if WEB_DIR.exists():
	app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


def _load_model_and_features():
	if not MODEL_PATH.exists():
		raise FileNotFoundError("Trained model not found. Please run training first.")
	bundle = joblib.load(MODEL_PATH)
	with open(META_PATH, "r", encoding="utf-8") as f:
		meta = json.load(f)
	features = meta["features"]
	model = bundle["model"]
	return model, features


@app.get("/")
async def index() -> FileResponse:
	index_path = WEB_DIR / "index.html"
	if not index_path.exists():
		raise HTTPException(status_code=404, detail="index.html not found. Did you create the frontend?")
	return FileResponse(index_path)


@app.post("/predict")
async def predict(payload: Dict[str, Any]) -> JSONResponse:
	try:
		model, features = _load_model_and_features()
		frame = pd.DataFrame([payload])
		missing = [c for c in features if c not in frame.columns]
		if missing:
			raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")
		X = frame[features]
		probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
		return JSONResponse({"probability": float(probs[0])})
	except FileNotFoundError as e:
		raise HTTPException(status_code=500, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")





