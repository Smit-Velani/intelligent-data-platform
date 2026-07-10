"""
reporter.py — Turns raw pipeline results (dataset stats, model leaderboard,
decision log, SHAP importances, drift status) into:
  1. A plain-English analysis written by Groq LLaMA 3.3
  2. A downloadable PDF report combining that narrative with the key charts
"""
from __future__ import annotations

import os
import base64
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_groq_client() -> Optional["Groq"]:
    if not HAS_GROQ:
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_your"):
        return None
    return Groq(api_key=api_key)


def build_llm_prompt(
    dataset_summary: dict,
    problem_type: str,
    leaderboard: list,
    decision_log: str,
    top_features: list,
    drift_summary: Optional[dict] = None,
) -> str:
    leaderboard_str = "\n".join(
        f"- {row['model']}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "model")
        for row in leaderboard
    )
    features_str = ", ".join(f"{f['feature']} ({f['mean_abs_shap']:.4f})" for f in top_features[:8]) if top_features else "not available"
    drift_str = f"Drift status: {drift_summary['overall_status']} ({drift_summary['drifted_feature_count']}/{drift_summary['total_feature_count']} features drifted)" if drift_summary else "No drift check run yet."

    return f"""You are a senior data scientist writing a report for a business stakeholder audience.
Be concise, plain-English, and avoid jargon where possible. Do not invent numbers not given below.

DATASET SUMMARY:
- Rows: {dataset_summary.get('n_rows')}
- Columns: {dataset_summary.get('n_cols')}
- Problem type: {problem_type}

MODEL LEADERBOARD (each model's cross-validated performance):
{leaderboard_str}

MODEL SELECTION DECISION LOG:
{decision_log}

TOP FEATURES BY SHAP IMPORTANCE (feature name, mean absolute SHAP value):
{features_str}

{drift_str}

Write a report with these sections, each with a short heading:
1. Executive Summary (2-3 sentences, plain English, for a non-technical reader)
2. Key Findings (3-5 bullet points on what drives the predictions)
3. Model Performance Analysis (explain why the winning model was chosen, referencing the decision log)
4. Business Recommendations (2-3 concrete, actionable suggestions)

Keep the whole report under 400 words."""


def generate_llm_report(
    dataset_summary: dict,
    problem_type: str,
    leaderboard: list,
    decision_log: str,
    top_features: list,
    drift_summary: Optional[dict] = None,
) -> str:
    client = _get_groq_client()
    prompt = build_llm_prompt(dataset_summary, problem_type, leaderboard, decision_log, top_features, drift_summary)

    if client is None:
        return (
            "[Groq API key not configured — this is a placeholder report.]\n\n"
            "Add GROQ_API_KEY to your .env file to generate a real narrative report.\n\n"
            "--- Prompt that would have been sent ---\n" + prompt
        )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[LLM report generation failed: {e}]\n\nFalling back to raw decision log:\n\n{decision_log}"


def _b64_to_image_flowable(b64_string: Optional[str], width=5.5 * inch):
    if not b64_string:
        return None
    img_bytes = base64.b64decode(b64_string)
    import io
    img_buf = io.BytesIO(img_bytes)
    img = Image(img_buf)
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def generate_pdf_report(
    output_path: str,
    dataset_summary: dict,
    problem_type: str,
    leaderboard: list,
    llm_report_text: str,
    bar_plot_base64: Optional[str] = None,
    calibration_plot_base64: Optional[str] = None,
    drift_summary: Optional[dict] = None,
) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER)
    story = []

    story.append(Paragraph("GlassBox ML — Analysis Report", title_style))
    story.append(Paragraph(datetime.utcnow().strftime("Generated %B %d, %Y %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Dataset Overview", styles["Heading2"]))
    story.append(Paragraph(
        f"Rows: {dataset_summary.get('n_rows')} &nbsp;|&nbsp; "
        f"Columns: {dataset_summary.get('n_cols')} &nbsp;|&nbsp; "
        f"Problem type: {problem_type}", styles["Normal"]
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Model Leaderboard", styles["Heading2"]))
    if leaderboard:
        cols = [k for k in leaderboard[0].keys()]
        table_data = [cols] + [[str(row.get(c, "")) for c in cols] for row in leaderboard]
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ]))
        story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    for line in llm_report_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
        elif line.startswith("#") or (len(line) < 60 and line.endswith(":")):
            story.append(Paragraph(line.lstrip("# "), styles["Heading3"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))

    bar_img = _b64_to_image_flowable(bar_plot_base64)
    if bar_img:
        story.append(PageBreak())
        story.append(Paragraph("Feature Importance (SHAP)", styles["Heading2"]))
        story.append(bar_img)

    cal_img = _b64_to_image_flowable(calibration_plot_base64)
    if cal_img:
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("Model Calibration", styles["Heading2"]))
        story.append(cal_img)

    if drift_summary:
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("Data Drift Status", styles["Heading2"]))
        story.append(Paragraph(
            f"Overall status: {drift_summary['overall_status']} "
            f"({drift_summary['drifted_feature_count']}/{drift_summary['total_feature_count']} features drifted)",
            styles["Normal"]
        ))

    doc.build(story)
    return output_path
