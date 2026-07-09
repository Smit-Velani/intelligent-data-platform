import React, { useState } from "react";
import { uploadDataset, preprocessDataset } from "./api";
import { T } from "./theme";

export default function Upload({ onPreprocessed }) {
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [targetCol, setTargetCol] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);

  const pickFile = (f) => {
    setFile(f);
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
      if (res.data.columns?.length) {
        setTargetCol(res.data.columns[res.data.columns.length - 1]);
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed. Check the file is a valid CSV.");
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
      <div style={styles.eyebrow}>STEP 01 — INGEST</div>
      <h2 style={styles.h2}>Upload your dataset</h2>
      <p style={styles.lead}>Any tabular CSV. IDP detects the problem type and cleans it automatically.</p>

      <label
        style={{ ...styles.drop, ...(dragOver ? styles.dropActive : {}), ...(file ? styles.dropFilled : {}) }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (e.dataTransfer.files?.[0]) pickFile(e.dataTransfer.files[0]);
        }}
      >
        <input type="file" accept=".csv" style={{ display: "none" }} onChange={(e) => pickFile(e.target.files[0])} />
        <div style={styles.dropIcon}>{file ? "📄" : "⬆"}</div>
        <div style={styles.dropText}>
          {file ? file.name : "Drop a CSV here or click to browse"}
        </div>
        {file && <div style={styles.dropHint}>{(file.size / 1024 / 1024).toFixed(1)} MB · ready to upload</div>}
      </label>

      <button style={{ ...styles.btnPrimary, opacity: !file || loading ? 0.5 : 1 }} onClick={handleUpload} disabled={!file || loading}>
        {loading && !uploadResult ? "Uploading…" : "Upload CSV"}
      </button>

      {error && <div style={styles.error}>{error}</div>}

      {uploadResult && (
        <div style={styles.reveal}>
          <div style={styles.statRow}>
            <Stat label="Rows" value={uploadResult.n_rows.toLocaleString("en-US")} />
            <Stat label="Columns" value={uploadResult.n_cols} />
            <Stat label="File" value={uploadResult.filename || "dataset.csv"} mono={false} />
          </div>

          <label style={styles.selectLabel}>Target column — what should the model predict?</label>
          <select style={styles.select} value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
            {uploadResult.columns.map((c) => (
              <option key={c} value={c} style={{ background: T.panel }}>{c}</option>
            ))}
          </select>

          <button style={{ ...styles.btnPrimary, opacity: loading ? 0.5 : 1 }} onClick={handlePreprocess} disabled={loading}>
            {loading ? "Processing…" : "Preprocess & continue →"}
          </button>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, mono = true }) {
  return (
    <div style={styles.stat}>
      <div style={styles.statLabel}>{label}</div>
      <div style={{ ...styles.statValue, fontFamily: mono ? T.mono : T.sans, fontSize: mono ? 22 : 14 }}>{value}</div>
    </div>
  );
}

const styles = {
  card: { background: T.panel, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32, boxShadow: "0 20px 60px rgba(0,0,0,0.4)" },
  eyebrow: { fontFamily: T.mono, fontSize: 11, letterSpacing: 2, color: T.cyan, marginBottom: 10 },
  h2: { margin: "0 0 6px", fontSize: 24, fontWeight: 700 },
  lead: { margin: "0 0 24px", color: T.textDim, fontSize: 14 },
  drop: {
    display: "block", border: `2px dashed ${T.border}`, borderRadius: 16, padding: "44px 20px",
    textAlign: "center", cursor: "pointer", transition: "all 0.2s ease", background: "rgba(56,189,248,0.02)",
  },
  dropActive: { borderColor: T.cyan, background: "rgba(56,189,248,0.08)" },
  dropFilled: { borderColor: T.mint, background: "rgba(52,211,153,0.05)" },
  dropIcon: { fontSize: 34, marginBottom: 10 },
  dropText: { fontSize: 15, fontWeight: 500, color: T.text },
  dropHint: { fontFamily: T.mono, fontSize: 12, color: T.mint, marginTop: 8 },
  btnPrimary: {
    marginTop: 20, width: "100%", padding: "14px", borderRadius: 12, border: "none",
    background: T.gradient, color: "#0B1220", fontWeight: 700, fontSize: 15, cursor: "pointer",
    boxShadow: "0 8px 24px rgba(56,189,248,0.3)",
  },
  error: { marginTop: 16, padding: "12px 16px", borderRadius: 10, background: "rgba(248,113,113,0.1)", border: `1px solid ${T.red}`, color: T.red, fontSize: 13 },
  reveal: { marginTop: 28, paddingTop: 28, borderTop: `1px solid ${T.border}` },
  statRow: { display: "flex", gap: 12, marginBottom: 24 },
  stat: { flex: 1, background: T.panelHi, border: `1px solid ${T.border}`, borderRadius: 12, padding: "14px 16px" },
  statLabel: { fontFamily: T.mono, fontSize: 10, letterSpacing: 1.5, color: T.textFaint, textTransform: "uppercase" },
  statValue: { fontWeight: 700, color: T.text, marginTop: 6, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },
  selectLabel: { display: "block", fontSize: 13, color: T.textDim, marginBottom: 8 },
  select: {
    width: "100%", padding: "12px 14px", borderRadius: 10, border: `1px solid ${T.border}`,
    background: T.panelHi, color: T.text, fontSize: 14, fontFamily: T.mono, cursor: "pointer",
  },
};