import React, { useState, useEffect, useRef } from "react";
import { trainModels } from "./api";

function estimateTime(nRows, nModels) {
  const perModelSec = Math.max(3, (nRows / 1000) * 0.15 * 3);
  const totalSec = perModelSec * nModels;
  const lowMin = Math.max(0.2, (totalSec * 0.7) / 60);
  const highMin = (totalSec * 1.4) / 60;
  const fmt = (m) => (m < 1 ? `${Math.round(m * 60)} sec` : `${m.toFixed(1)} min`);
  return `${fmt(lowMin)} – ${fmt(highMin)}`;
}

export default function Dashboard({ jobId, preprocessResult, onTrained }) {
  const [costParams, setCostParams] = useState({
    falseNegativeCost: 500,
    falsePositiveCost: 15,
    minRecall: 0.3,
    useSmote: true,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  const nRows = preprocessResult.n_train || 0;
  const nModels = nRows > 20000 ? 4 : 5;
  const estimate = estimateTime(nRows, nModels);

  useEffect(() => {
    if (loading) {
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      setElapsed(0);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  const update = (field) => (e) => {
    const value = e.target.type === "checkbox" ? e.target.checked : Number(e.target.value);
    setCostParams((prev) => ({ ...prev, [field]: value }));
  };

  const handleTrain = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await trainModels(jobId, costParams);
      onTrained(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Training failed.");
    } finally {
      setLoading(false);
    }
  };

  const fmtElapsed = (s) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  if (loading) {
    return (
      <div style={styles.card}>
        <div style={styles.loadingBox}>
          <div style={styles.spinner} />
          <h2 style={{ marginTop: 20 }}>Training models…</h2>
          <p style={styles.loadingDetail}>
            {nModels} models on {nRows.toLocaleString()} rows with cross-validation
          </p>
          <p style={styles.estimate}>Estimated time: {estimate}</p>
          <p style={styles.elapsed}>Elapsed: {fmtElapsed(elapsed)}</p>
          <p style={styles.hint}>Please keep this tab open — training is running on the server.</p>
        </div>
        <style>{keyframes}</style>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <h2>2. Configure & Train</h2>

      <p style={styles.meta}>
        Problem type: <strong>{preprocessResult.problem_type}</strong> &nbsp;|&nbsp; Train rows:{" "}
        {preprocessResult.n_train} &nbsp;|&nbsp; Test rows: {preprocessResult.n_test}
      </p>
      {preprocessResult.scale_pos_weight && (
        <p style={styles.meta}>
          Class imbalance detected — scale_pos_weight: {preprocessResult.scale_pos_weight.toFixed(2)}
        </p>
      )}
      <p style={styles.estimatePreview}>⏱ Estimated training time: {estimate}</p>

      <div style={styles.grid}>
        <label style={styles.label}>
          False negative cost ($)
          <input style={styles.input} type="number" value={costParams.falseNegativeCost} onChange={update("falseNegativeCost")} />
        </label>
        <label style={styles.label}>
          False positive cost ($)
          <input style={styles.input} type="number" value={costParams.falsePositiveCost} onChange={update("falsePositiveCost")} />
        </label>
        <label style={styles.label}>
          Minimum recall floor
          <input style={styles.input} type="number" step="0.05" min="0" max="1" value={costParams.minRecall} onChange={update("minRecall")} />
        </label>
        <label style={styles.checkboxLabel}>
          <input type="checkbox" checked={costParams.useSmote} onChange={update("useSmote")} />
          Use SMOTE (oversample minority class)
        </label>
      </div>

      <button style={styles.button} onClick={handleTrain} disabled={loading}>
        Train Models
      </button>

      {error && <p style={styles.error}>{error}</p>}
    </div>
  );
}

const keyframes = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;

const styles = {
  card: { background: "#fff", borderRadius: 8, padding: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.1)", marginBottom: 24 },
  meta: { color: "#555", fontSize: 14 },
  estimatePreview: { color: "#4C72B0", fontSize: 14, fontWeight: "bold", marginTop: 8 },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16, marginBottom: 16 },
  label: { display: "flex", flexDirection: "column", fontSize: 14 },
  checkboxLabel: { display: "flex", alignItems: "center", gap: 8, fontSize: 14 },
  input: { marginTop: 4, padding: 6, borderRadius: 4, border: "1px solid #ccc" },
  button: { padding: "8px 16px", borderRadius: 6, border: "none", background: "#4C72B0", color: "#fff", cursor: "pointer" },
  error: { color: "#c0392b", marginTop: 8 },
  loadingBox: { textAlign: "center", padding: "40px 20px" },
  spinner: {
    width: 48, height: 48, margin: "0 auto", border: "5px solid #e0e0e0",
    borderTop: "5px solid #4C72B0", borderRadius: "50%", animation: "spin 1s linear infinite",
  },
  loadingDetail: { color: "#555", fontSize: 15 },
  estimate: { color: "#4C72B0", fontSize: 16, fontWeight: "bold" },
  elapsed: { color: "#2e7d32", fontSize: 18, fontWeight: "bold", fontFamily: "monospace" },
  hint: { color: "#999", fontSize: 13, marginTop: 16 },
};