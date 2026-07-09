# Intelligent Data Platform (IDP)

Upload any CSV and get back a cleaned, explained, and deployment-ready machine learning model — with a business-facing PDF report — in minutes. No manual data science required.

IDP automates the judgment calls a data scientist normally makes: cleaning, problem-type detection, model selection, explainability, drift monitoring, and reporting.

## What it does

1. **Auto-preprocessing** — detects and fills missing values, encodes categoricals, scales numerics
2. **Problem detection** — automatically classifies the task as classification, regression, or clustering
3. **Imbalance-aware splitting** — stratified train/test split + stratified k-fold, with `scale_pos_weight` computed for imbalanced data
4. **Cost-aware AutoML** — trains up to 5 models (Logistic Regression, Random Forest, XGBoost, Neural Network, SVM) and selects the winner by **expected business cost**, not just raw metrics
5. **Recall-floor guardrail** — prevents a pure cost-minimizer from degenerating into a "predict nothing" model
6. **Explainability** — SHAP (adaptive explainer selection: Tree / Linear / Kernel), LIME cross-checks, and calibration curves with Brier score
7. **Drift detection** — PSI and KS-test to flag when incoming data has shifted from training
8. **LLM reporting** — Groq LLaMA 3.3 writes a plain-English business report
9. **PDF export** — combines leaderboard, decision log, charts, and narrative into a downloadable report

## Tech stack

**Backend:** FastAPI, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, LIME, Groq LLaMA 3.3, ReportLab, MongoDB Atlas
**Frontend:** React
**Demo dataset:** Kaggle Credit Card Fraud (284,807 transactions, 0.17% fraud rate)

## Key engineering decisions

- **SMOTE runs inside each CV fold**, never before splitting — avoids data leakage from synthetic samples contaminating held-out folds
- **Adaptive SHAP explainer** — TreeExplainer for tree models, LinearExplainer for linear models, bounded KernelExplainer otherwise (KernelSHAP on 284K rows would never finish)
- **Data-size-aware speed scaling** — SMOTE and 5-fold CV on small data where they're cheap; `scale_pos_weight` and 3-fold on large data where SMOTE becomes impractical
- **SVM auto-excluded above 20K rows** due to O(n²) training complexity — stated explicitly in the decision log, not silently dropped
- **AUC-PR reported alongside AUC-ROC** — on a 0.17%-positive dataset, ROC-AUC can look deceptively strong

## Running locally

### Backend
```bash
conda create -n idp python=3.11 -y
conda activate idp
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY and MONGODB_URI
uvicorn backend.main:app --reload --port 8000
```
Interactive API docs: http://127.0.0.1:8000/docs

### Frontend
```bash
cd frontend
npm install
npm start
```
App runs at http://localhost:3000

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload-dataset` | Upload a CSV |
| POST | `/preprocess` | Clean + split + detect problem type |
| POST | `/train` | Cost-aware AutoML across models |
| GET | `/results/{job_id}` | Leaderboard + decision log |
| GET | `/explain/{job_id}` | SHAP importance + calibration |
| GET | `/detect-drift/{job_id}` | PSI / KS drift report |
| GET | `/report/{job_id}` | LLM-generated report text |
| GET | `/download-report/{job_id}` | Full PDF report |

## Author

Smit Velani — MS Data Science, Northeastern University