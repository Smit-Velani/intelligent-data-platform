---

## Setup & Installation

**Clone the repository:**
```bash
git clone https://github.com/Smit-Velani/intelligent-data-platform.git
cd intelligent-data-platform
```

**Backend:**
```bash
conda create -n idp python=3.11 -y
conda activate idp
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY and MONGODB_URI
uvicorn backend.main:app --reload --port 8000
```
Interactive API docs: `http://127.0.0.1:8000/docs`

**Frontend:**
```bash
cd frontend
npm install
npm start
```
Open browser at: `http://localhost:3000`

Get free API keys:
- Groq: https://console.groq.com
- MongoDB Atlas (optional): https://www.mongodb.com/atlas

---

## Running Tests

```bash
pip install pytest
pytest -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload-dataset` | Upload a CSV |
| POST | `/preprocess` | Clean, split, and detect problem type |
| POST | `/train` | Cost-aware AutoML across models |
| GET | `/results/{job_id}` | Leaderboard and decision log |
| GET | `/explain/{job_id}` | SHAP importance and calibration |
| GET | `/detect-drift/{job_id}` | PSI / KS drift report |
| GET | `/report/{job_id}` | LLM-generated report text |
| GET | `/view-report/{job_id}` | PDF report (inline preview) |
| GET | `/download-report/{job_id}` | PDF report (download) |

---

## ML Design Decisions

**Cost-Aware Model Selection**
- Converts each model's confusion matrix into an expected dollar cost
- Uses a business cost matrix (false-negative vs false-positive cost)
- Recall-floor guardrail disqualifies models below a minimum recall

**Leakage-Free SMOTE**
- SMOTE runs inside each cross-validation fold via an imblearn Pipeline
- Synthetic samples never touch held-out folds, preventing inflated metrics

**Adaptive SHAP Explainer**
- TreeExplainer for tree models, LinearExplainer for linear models
- Bounded KernelExplainer otherwise (KernelSHAP on 284K rows is impractical)

**Data-Size-Aware Speed Scaling**
- SMOTE and 5-fold CV on small data where they are cheap
- `scale_pos_weight` and 3-fold CV on large data — cut 284K-row training from ~20 min to ~3 min
- SVM auto-excluded above 20K rows due to O(n²) complexity, logged explicitly

---

## Known Limitations

- In-memory job store — results are lost on server restart. A production version would persist to Redis or disk.
- No authentication layer.
- Not yet deployed to a public URL (runs locally).

---

## Author

**Smitkumar Velani**
MS Data Science — Northeastern University, Boston

[GitHub](https://github.com/Smit-Velani) | [LinkedIn](https://linkedin.com/in/smit-velani) | [Portfolio](https://smit-velani.github.io)

---

*Built with Python · FastAPI · React · XGBoost · SHAP · Groq LLaMA · Scikit-Learn*