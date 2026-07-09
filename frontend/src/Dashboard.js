import React, { useState, useEffect, useRef } from "react";
import { trainModels } from "./api";
import { T } from "./theme";

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
    falseNegativeCost: 500, falsePositiveCost: 15, minRecall: 0.3, useSmote: true,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  const nRows = preprocessResult.n_train || 0;
  const nModels = nRows > 20000 ? 4 : 5;
  const estimate = estimateTime(nRows, nModels);

  useEffect(() => {
    if (loading) timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    else { clearInterval(timerRef.current); setElapsed(0); }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  const update = (field) => (e) => {
    const value = e.target.type === "checkbox" ? e.target.checked : Number(e.target.value);
    setCostParams((prev) => ({ ...prev, [field]: value }));
  };

  const handleTrain = async () => {
    setLoading(true); setError("");
    try {
      const res = await trainModels(jobId, costParams);
      onTrained(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Training failed.");
    } finally { setLoading(false); }
  };

  const fmtElapsed = (s) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  if (loading) {
    return (
      <div style={styles.card}>
        <div style={styles.loadingBox}>
          <div style={styles.ring} />
          <h2 style={styles.h2}>Training models</h2>
          <p style={styles.loadDetail}>
            {nModels} models · {nRows.toLocaleString("en-US")} rows · cross-validation
          </p>
          <div style={styles.readouts}>
            <div style={styles.readout}>
              <div style={styles.roLabel}>ESTIMATE</div>
              <div style={{ ...styles.roValue, color: T.cyan }}>{estimate}</div>
            </div>
            <div style={styles.readout}>
              <div style={styles.roLabel}>ELAPSED</div>
              <div style={{ ...styles.roValue, color: T.mint }}>{fmtElapsed(elapsed)}</div>
            </div>
          </div>
          <p style={styles.hint}>Keep this tab open — training runs on the server.</p>
        </div>
        <style>{keyframes}</style>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.eyebrow}>STEP 02 — CONFIGURE</div>
      <h2 style={styles.h2}>Set the business cost, then train</h2>

      <div style={styles.pills}>
        <Pill label="Problem" value={preprocessResult.problem_type} />
        <Pill label="Train rows" value={preprocessResult.n_train?.toLocaleString("en-US")} />
        <Pill label="Test rows" value={preprocessResult.n_test?.toLocaleString("en-US")} />
        {preprocessResult.scale_pos_weight && (
          <Pill label="Imbalance (pos wt)" value={preprocessResult.scale_pos_weight.toFixed(1)} accent />
        )}
      </div>

      <div style={styles.estBanner}>Estimated training time: <strong>{estimate}</strong></div>

      <div style={styles.grid}>
        <Field label="False negative cost ($)" hint="Cost of a missed fraud">
          <input style={styles.input} type="number" value={costParams.falseNegativeCost} onChange={update("falseNegativeCost")} />
        </Field>
        <Field label="False positive cost ($)" hint="Cost of a false alarm">
          <input style={styles.input} type="number" value={costParams.falsePositiveCost} onChange={update("falsePositiveCost")} />
        </Field>
        <Field label="Minimum recall floor" hint="Reject models below this recall">
          <input style={styles.input} type="number" step="0.05" min="0" max="1" value={costParams.minRecall} onChange={update("minRecall")} />
        </Field>
        <label style={styles.checkField}>
          <input type="checkbox" checked={costParams.useSmote} onChange={update("useSmote")} style={styles.checkbox} />
          <div>
            <div style={styles.fieldLabel}>Use SMOTE</div>
            <div style={styles.fieldHint}>Oversample minority (auto-off on large data)</div>
          </div>
        </label>
      </div>

      <button style={styles.btnPrimary} onClick={handleTrain}>Train models →</button>
      {error && <div style={styles.error}>{error}</div>}
    </div>
  );
}

function Pill({ label, value, accent }) {
  return (
    <div style={{ ...styles.pill, ...(accent ? styles.pillAccent : {}) }}>
      <span style={styles.pillLabel}>{label}</span>
      <span style={styles.pillValue}>{value}</span>
    </div>
  );
}
function Field({ label, hint, children }) {
  return (
    <div>
      <div style={styles.fieldLabel}>{label}</div>
      <div style={styles.fieldHint}>{hint}</div>
      {children}
    </div>
  );
}

const keyframes = `@keyframes spin { to { transform: rotate(360deg); } }`;

const styles = {
  card: { background: T.panel, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32, boxShadow: "0 20px 60px rgba(0,0,0,0.4)" },
  eyebrow: { fontFamily: T.mono, fontSize: 11, letterSpacing: 2, color: T.cyan, marginBottom: 10 },
  h2: { margin: "0 0 20px", fontSize: 24, fontWeight: 700 },
  pills: { display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 20 },
  pill: { display: "flex", flexDirection: "column", background: T.panelHi, border: `1px solid ${T.border}`, borderRadius: 10, padding: "8px 14px" },
  pillAccent: { borderColor: T.amber, background: "rgba(245,158,11,0.08)" },
  pillLabel: { fontFamily: T.mono, fontSize: 10, letterSpacing: 1, color: T.textFaint, textTransform: "uppercase" },
  pillValue: { fontFamily: T.mono, fontSize: 16, fontWeight: 600, color: T.text, marginTop: 2 },
  estBanner: { padding: "10px 14px", borderRadius: 10, background: "rgba(56,189,248,0.08)", border: `1px solid rgba(56,189,248,0.3)`, color: T.cyan, fontSize: 13, marginBottom: 24, fontFamily: T.mono },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 28 },
  fieldLabel: { fontSize: 13, fontWeight: 600, color: T.text },
  fieldHint: { fontSize: 11, color: T.textFaint, marginBottom: 8, marginTop: 2 },
  input: { width: "100%", padding: "10px 12px", borderRadius: 10, border: `1px solid ${T.border}`, background: T.panelHi, color: T.text, fontSize: 14, fontFamily: T.mono, boxSizing: "border-box" },
  checkField: { display: "flex", gap: 12, alignItems: "flex-start", background: T.panelHi, border: `1px solid ${T.border}`, borderRadius: 10, padding: 12, cursor: "pointer" },
  checkbox: { marginTop: 2, width: 18, height: 18, accentColor: T.cyan, cursor: "pointer" },
  btnPrimary: { width: "100%", padding: 14, borderRadius: 12, border: "none", background: T.gradient, color: "#0B1220", fontWeight: 700, fontSize: 15, cursor: "pointer", boxShadow: "0 8px 24px rgba(56,189,248,0.3)" },
  error: { marginTop: 16, padding: "12px 16px", borderRadius: 10, background: "rgba(248,113,113,0.1)", border: `1px solid ${T.red}`, color: T.red, fontSize: 13 },
  loadingBox: { textAlign: "center", padding: "40px 20px" },
  ring: { width: 56, height: 56, margin: "0 auto 24px", borderRadius: "50%", border: `4px solid ${T.border}`, borderTopColor: T.cyan, animation: "spin 0.9s linear infinite" },
  loadDetail: { color: T.textDim, fontSize: 14, fontFamily: T.mono, marginTop: 4 },
  readouts: { display: "flex", gap: 16, justifyContent: "center", margin: "28px 0 20px" },
  readout: { background: T.panelHi, border: `1px solid ${T.border}`, borderRadius: 12, padding: "16px 28px", minWidth: 120 },
  roLabel: { fontFamily: T.mono, fontSize: 10, letterSpacing: 2, color: T.textFaint },
  roValue: { fontFamily: T.mono, fontSize: 26, fontWeight: 700, marginTop: 6 },
  hint: { color: T.textFaint, fontSize: 12 },
};