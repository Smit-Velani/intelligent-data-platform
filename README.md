_Example markdown once you add images:_
`![Results dashboard](docs/screenshots/results.png)`

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

## Running tests

```bash
pip install pytest
pytest -v
```

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

## Known limitations

- In-memory job store (results lost on server restart) — a production version would persist to Redis/disk
- No authentication layer
- Not yet deployed to a public URL

## Author

Smit Velani — MS Data Science, Northeastern University
[github.com/Smit-Velani](https://github.com/Smit-Velani)