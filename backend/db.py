"""
db.py — MongoDB Atlas connection layer.

Stores three collections:
  - datasets   : metadata about each uploaded CSV (rows, cols, dtypes, job_id)
  - runs       : one document per training run (metrics, best_model, timestamps)
  - reports    : generated LLM report text + PDF path, keyed by job_id
"""
import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)
db = client["idp"]

datasets_col = db["datasets"]
runs_col = db["runs"]
reports_col = db["reports"]


def save_dataset_meta(job_id: str, filename: str, n_rows: int, n_cols: int, columns: list):
    doc = {
        "job_id": job_id,
        "filename": filename,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "created_at": datetime.utcnow(),
    }
    datasets_col.insert_one(doc)
    return doc


def save_run(job_id: str, problem_type: str, results: dict, best_model: str):
    doc = {
        "job_id": job_id,
        "problem_type": problem_type,
        "results": results,
        "best_model": best_model,
        "created_at": datetime.utcnow(),
    }
    runs_col.insert_one(doc)
    return doc


def save_report(job_id: str, report_text: str, pdf_path: str):
    doc = {
        "job_id": job_id,
        "report_text": report_text,
        "pdf_path": pdf_path,
        "created_at": datetime.utcnow(),
    }
    reports_col.insert_one(doc)
    return doc


def get_run(job_id: str):
    return runs_col.find_one({"job_id": job_id}, {"_id": 0})


def get_report(job_id: str):
    return reports_col.find_one({"job_id": job_id}, {"_id": 0})
