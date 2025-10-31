# Diabetes Risk Prediction (BRFSS 2015)

Predict risk of diabetes using health and lifestyle indicators from the CDC BRFSS 2015 "Diabetes Health Indicators" dataset. The pipeline downloads data, preprocesses, trains a model, evaluates metrics, and exposes a simple CLI for batch predictions.

## Dataset
- Source: UCI Machine Learning Repository — Diabetes Health Indicators (BRFSS 2015)
- Direct CSV: https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_binary_health_indicators_BRFSS2015.csv
- Target: `Diabetes_binary` (0 = no diabetes, 1 = diabetes)
- Features include: BMI, age category, physical activity, smoking, alcohol use, general health, mental/physical health days, high BP/cholesterol, and more.

Why BRFSS? It is large (250k+ rows) and includes lifestyle and health behavior features, making it better-suited than the classic Pima dataset for population risk screening.

## Project layout
```
.
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
├─ reports/
├─ src/
│  ├─ data.py          # download + preprocess
│  ├─ train.py         # train and evaluate model
│  └─ predict.py       # CLI predictions
├─ web/
│  └─ index.html       # simple frontend form
└─ requirements.txt
```

## Quickstart
1) Create a virtual environment (recommended) and install dependencies:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

2) Download and preprocess the dataset, then train the model (or provide a local CSV via `--raw_csv`):
```bash
python -m src.data --download --preprocess
# OR
python -m src.data --preprocess --raw_csv "C:/path/to/diabetes_012_health_indicators_BRFSS2015.csv"
python -m src.train --train --evaluate
```

3) Make predictions on new records (CSV with header). Outputs a CSV with probabilities:
```bash
python -m src.predict --input sample_inputs.csv --output predictions.csv
```

## Web UI (FastAPI)
- Start the server:
```bash
uvicorn src.server:app --reload
```
- Open the form at `http://127.0.0.1:8000/`. Fill in fields and click Predict to see the risk probability.

## Features used
The pipeline uses a curated subset of informative variables from BRFSS 2015, all numeric or binary:
- HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

## Metrics reported
- ROC AUC, Average Precision (PR AUC), F1, Accuracy
- Calibration curve plot saved to `reports/`

## Reproducibility
- Fixed random seed across splits and training
- Model stored at `models/model.joblib`

## Notes
- All inputs expected as numeric (same encoding as source). See `src/data.py` for mappings and schema.
- This project is designed for quick experimentation and education; for production, consider model monitoring, bias checks, and secure data handling.
