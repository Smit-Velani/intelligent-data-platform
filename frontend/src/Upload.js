import React, { useState } from "react";
import { uploadDataset, preprocessDataset } from "./api";

export default function Upload({ onPreprocessed }) {
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [targetCol, setTargetCol] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const res = await uploadDataset(file);
      setUploadResult(res.data);
      if (res.data.columns && res.data.columns.length > 0) {
        setTargetCol(res.data.columns[res.data.columns.length - 1]);
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const handlePreprocess = async () => {
    if (!uploadResult) return;
    setLoading(true);
    setError("");
    try {
      const res = await preprocessDataset(uploadResult.job_id, targetCol);
      onPreprocessed({ jobId: uploadResult.job_id, preprocessResult: res.data, uploadResult });
    } catch (err) {
      setError(err.response?.data?.detail || "Preprocessing failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2>1. Upload Dataset</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button style={styles.button} onClick={handleUpload} disabled={!file || loading}>
        {loading && !uploadResult ? "Uploading..." : "Upload CSV"}
      </button>

      {error && <p style={styles.error}>{error}</p>}

      {uploadResult && (
        <div style={styles.section}>
          <p>
            <strong>{uploadResult.filename || "Dataset"}</strong> — {uploadResult.n_rows} rows,{" "}
            {uploadResult.n_cols} columns
          </p>

          <label style={styles.label}>
            Target column (what should the model predict?):
            <select
              style={styles.select}
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
            >
              {uploadResult.columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </label>

          <button style={styles.button} onClick={handlePreprocess} disabled={loading}>
            {loading ? "Processing..." : "Preprocess Dataset"}
          </button>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    background: "#fff",
    borderRadius: 8,
    padding: 24,
    boxShadow: "0 1px 4px rgba(0,0,0,0.1)",
    marginBottom: 24,
  },
  button: {
    marginTop: 12,
    marginLeft: 12,
    padding: "8px 16px",
    borderRadius: 6,
    border: "none",
    background: "#4C72B0",
    color: "#fff",
    cursor: "pointer",
  },
  section: { marginTop: 16 },
  label: { display: "block", marginTop: 12, marginBottom: 8 },
  select: { marginLeft: 8, padding: 4 },
  error: { color: "#c0392b", marginTop: 8 },
};