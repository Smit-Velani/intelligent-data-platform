"""
main.py — FastAPI application. Wires preprocessor -> model_selector ->
explainer -> drift_detector -> reporter into REST endpoints.
"""
from __future__ import annotations

import io
import os
import uuid
import traceback
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.preprocessor import load_csv, preprocess
from backend.model_selector import run_selection, CostMatrix
from backend.explainer import generate_explainability_report, explain_single_prediction
from backend.drift_detector import detect_drift
from backend.reporter import generate_llm_report, generate_pdf_report

try:
    from backend import db
    HAS_DB = True
except Exception:
    HAS_DB = False
    db = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
MODEL_STORE_DIR = os.getenv("MODEL_STORE_DIR", os.path.join(ROOT_DIR, "model_store"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_STORE_DIR, exist_ok=True)

app = FastAPI(title="Intelligent Data Platform API", version="1.0.0")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict = {}


def _safe_db_call(fn_name, *args, **kwargs):
    if not HAS_DB or db is None:
        return
    try:
        fn = getattr(db, fn_name)
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[db warning] {fn_name} failed (continuing without persistence): {e}")


class PreprocessRequest(BaseModel):
    job_id: str
    target_col: Optional[str] = None
    test_size: float = 0.2
    n_splits: int = 5


class TrainRequest(BaseModel):
    job_id: str
    false_negative_cost: float = 500.0
    false_positive_cost: float = 15.0
    min_recall: float = 0.5
    use_smote: bool = True


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Intelligent Data Platform API"}


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    job_id = str(uuid.uuid4())
    contents = await file.read()

    save_path = os.path.join(UPLOAD_DIR, f"{job_id}.csv")
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        df = load_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    JOBS[job_id] = {"df": df, "csv_path": save_path, "filename": file.filename}

    dataset_meta = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }
    _safe_db_call("save_dataset_meta", job_id, file.filename, dataset_meta["n_rows"], dataset_meta["n_cols"], dataset_meta["columns"])

    return {"job_id": job_id, **dataset_meta}


@app.post("/preprocess")
def preprocess_dataset(req: PreprocessRequest):
    job = JOBS.get(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found. Upload a dataset first.")

    try:
        result = preprocess(
            job["df"], target_col=req.target_col,
            test_size=req.test_size, n_splits=req.n_splits,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    job["preprocess_result"] = result
    job["target_col"] = req.target_col

    return {
        "job_id": req.job_id,
        "problem_type": result.problem_type,
        "class_balance": result.class_balance,
        "scale_pos_weight": result.scale_pos_weight,
        "n_train": int(result.X_train.shape[0]),
        "n_test": int(result.X_test.shape[0]),
        "feature_names": result.feature_names,
        "dropped_columns": result.dropped_columns,
    }


@app.post("/train")
def train_models(req: TrainRequest):
    job = JOBS.get(req.job_id)
    if job is None or "preprocess_result" not in job:
        raise HTTPException(status_code=404, detail="Run /preprocess for this job_id first.")

    result = job["preprocess_result"]
    cost_matrix = CostMatrix(
        false_negative_cost=req.false_negative_cost,
        false_positive_cost=req.false_positive_cost,
        min_recall=req.min_recall,
    )

    try:
        selection = run_selection(result, cost_matrix=cost_matrix, use_smote=req.use_smote)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

    job["selection_result"] = selection

    model_path = os.path.join(MODEL_STORE_DIR, f"{req.job_id}_best_model.joblib")
    joblib.dump(selection.best_estimator, model_path)
    job["model_path"] = model_path

    _safe_db_call("save_run", req.job_id, selection.problem_type, {"leaderboard": selection.leaderboard}, selection.best_model_name)

    return {
        "job_id": req.job_id,
        "best_model": selection.best_model_name,
        "leaderboard": selection.leaderboard,
        "decision_log": selection.decision_log,
    }


@app.get("/results/{job_id}")
def get_results(job_id: str):
    job = JOBS.get(job_id)
    if job is None or "selection_result" not in job:
        raise HTTPException(status_code=404, detail="No trained model for this job_id yet.")
    sel = job["selection_result"]
    return {
        "job_id": job_id,
        "best_model": sel.best_model_name,
        "leaderboard": sel.leaderboard,
        "decision_log": sel.decision_log,
    }


@app.get("/explain/{job_id}")
def explain_dataset(job_id: str):
    job = JOBS.get(job_id)
    if job is None or "selection_result" not in job:
        raise HTTPException(status_code=404, detail="Train a model for this job_id first.")

    result = job["preprocess_result"]
    sel = job["selection_result"]

    report = generate_explainability_report(
        sel.best_estimator, result.X_train, result.X_test, result.y_test,
        result.feature_names, result.problem_type,
    )
    job["explainability_report"] = report

    return {
        "job_id": job_id,
        "explainer_type": report.explainer_type,
        "global_importance": report.global_importance,
        "summary_plot_base64": report.summary_plot_base64,
        "bar_plot_base64": report.bar_plot_base64,
        "calibration_plot_base64": report.calibration_plot_base64,
        "brier_score": report.brier_score,
        "notes": report.notes,
    }


@app.get("/explain/{job_id}/instance/{row_index}")
def explain_instance(job_id: str, row_index: int):
    job = JOBS.get(job_id)
    if job is None or "selection_result" not in job:
        raise HTTPException(status_code=404, detail="Train a model for this job_id first.")

    result = job["preprocess_result"]
    sel = job["selection_result"]

    if row_index < 0 or row_index >= len(result.X_test):
        raise HTTPException(status_code=400, detail=f"row_index must be between 0 and {len(result.X_test) - 1}.")

    explanation = explain_single_prediction(
        sel.best_estimator, result.X_train, result.X_test[row_index], result.feature_names,
    )
    return {"job_id": job_id, "row_index": row_index, **explanation}


@app.get("/detect-drift/{job_id}")
def check_drift(job_id: str):
    job = JOBS.get(job_id)
    if job is None or "preprocess_result" not in job:
        raise HTTPException(status_code=404, detail="Run /preprocess for this job_id first.")

    result = job["preprocess_result"]
    report = detect_drift(result.X_train, result.X_test, result.feature_names)

    return {
        "job_id": job_id,
        "overall_status": report.overall_status,
        "drifted_feature_count": report.drifted_feature_count,
        "total_feature_count": report.total_feature_count,
        "features": [f.__dict__ for f in report.features],
    }


@app.get("/report/{job_id}")
def get_report(job_id: str):
    job = JOBS.get(job_id)
    if job is None or "selection_result" not in job:
        raise HTTPException(status_code=404, detail="Train a model for this job_id first.")

    sel = job["selection_result"]
    df = job["df"]
    explain_report = job.get("explainability_report")
    top_features = explain_report.global_importance if explain_report else []

    drift = detect_drift(job["preprocess_result"].X_train, job["preprocess_result"].X_test, job["preprocess_result"].feature_names)
    drift_summary = {
        "overall_status": drift.overall_status,
        "drifted_feature_count": drift.drifted_feature_count,
        "total_feature_count": drift.total_feature_count,
    }

    dataset_summary = {"n_rows": len(df), "n_cols": len(df.columns)}
    report_text = generate_llm_report(
        dataset_summary, sel.problem_type, sel.leaderboard, sel.decision_log, top_features, drift_summary,
    )
    job["report_text"] = report_text

    return {"job_id": job_id, "report_text": report_text}


@app.get("/download-report/{job_id}")
def download_report(job_id: str):
    job = JOBS.get(job_id)
    if job is None or "selection_result" not in job:
        raise HTTPException(status_code=404, detail="Train a model for this job_id first.")

    sel = job["selection_result"]
    df = job["df"]
    explain_report = job.get("explainability_report")

    report_text = job.get("report_text")
    if report_text is None:
        top_features = explain_report.global_importance if explain_report else []
        dataset_summary = {"n_rows": len(df), "n_cols": len(df.columns)}
        report_text = generate_llm_report(dataset_summary, sel.problem_type, sel.leaderboard, sel.decision_log, top_features)

    pdf_path = os.path.join(MODEL_STORE_DIR, f"{job_id}_report.pdf")
    generate_pdf_report(
        pdf_path,
        {"n_rows": len(df), "n_cols": len(df.columns)},
        sel.problem_type,
        sel.leaderboard,
        report_text,
        bar_plot_base64=explain_report.bar_plot_base64 if explain_report else None,
        calibration_plot_base64=explain_report.calibration_plot_base64 if explain_report else None,
    )
    _safe_db_call("save_report", job_id, report_text, pdf_path)

    return FileResponse(pdf_path, media_type="application/pdf", filename=f"idp_report_{job_id[:8]}.pdf")