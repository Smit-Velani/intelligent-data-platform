import React, { useEffect, useState } from "react";
import { getExplainability, getDrift, downloadReportUrl, viewReportUrl } from "./api";
import { T } from "./theme";

export default function Results({ jobId, trainResult }) {
  const [explain, setExplain] = useState(null);
  const [drift, setDrift] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(true);
  const [loadingDrift, setLoadingDrift] = useState(true);
  const [showPdf, setShowPdf] = useState(false);

  useEffect(() => {
    getExplainability(jobId).then((r) => setExplain(r.data)).catch(() => setExplain({ error: true })).finally(() => setLoadingExplain(false));
    getDrift(jobId).then((r) => setDrift(r.data)).catch(() => setDrift({ error: true })).finally(() => setLoadingDrift(false));
  }, [jobId]);

  const leaderboard = trainResult.leaderboard || [];
  const columns = leaderboard.length ? Object.keys(leaderboard[0]) : [];
  const downloadUrl = downloadReportUrl(jobId);
  const previewUrl = viewReportUrl(jobId);

  const driftColor = drift?.overall_status === "stable" ? T.mint : drift?.overall_status === "significant_drift" ? T.red : T.amber;

  return (
    <div style={styles.card}>
      <div style={styles.eyebrow}>STEP 03 — ANALYZE</div>
      <div style={styles.winnerBanner}>
        <div>
          <div style={styles.winnerLabel}>BEST MODEL</div>
          <div style={styles.winnerName}>{trainResult.best_model}</div>
        </div>
        <div style={styles.trophy}>🏆</div>
      </div>

      <h3 style={styles.h3}>Leaderboard</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={styles.table}>
          <thead>
            <tr>{columns.map((c) => <th key={c} style={styles.th}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {leaderboard.map((row, i) => (
              <tr key={i} style={row.model === trainResult.best_model ? styles.winRow : {}}>
                {columns.map((c) => (
                  <td key={c} style={styles.td}>
                    {typeof row[c] === "number" ? row[c].toLocaleString("en-US") : row[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 style={styles.h3}>Decision log</h3>
      <pre style={styles.pre}>{trainResult.decision_log}</pre>

      <h3 style={styles.h3}>Explainability · SHAP</h3>
      {loadingExplain && <p style={styles.muted}>Computing SHAP explanations…</p>}
      {explain && !explain.error && (
        <div>
          <div style={styles.tag}>explainer: {explain.explainer_type}</div>
          {explain.bar_plot_base64 && <img src={`data:image/png;base64,${explain.bar_plot_base64}`} alt="SHAP importance" style={styles.chart} />}
          {explain.calibration_plot_base64 && <img src={`data:image/png;base64,${explain.calibration_plot_base64}`} alt="Calibration" style={styles.chart} />}
        </div>
      )}
      {explain?.error && <p style={styles.errText}>Could not load explainability report.</p>}

      <h3 style={styles.h3}>Data drift</h3>
      {loadingDrift && <p style={styles.muted}>Checking for drift…</p>}
      {drift && !drift.error && (
        <div style={{ ...styles.driftBadge, borderColor: driftColor, color: driftColor }}>
          <span style={{ ...styles.driftDot, background: driftColor }} />
          {drift.overall_status.replace("_", " ")} · {drift.drifted_feature_count}/{drift.total_feature_count} features drifted
        </div>
      )}

      <h3 style={styles.h3}>Full report</h3>
      <div style={styles.btnRow}>
        <button style={styles.btnGhost} onClick={() => setShowPdf((s) => !s)}>
          {showPdf ? "Hide preview" : "View report"}
        </button>
        <a href={downloadUrl} target="_blank" rel="noreferrer" style={{ textDecoration: "none" }}>
          <button style={styles.btnPrimary}>Download PDF</button>
        </a>
      </div>
      {showPdf && (
        <div style={styles.pdfWrap}>
          <iframe src={previewUrl} title="Analysis Report" style={styles.pdfFrame} />
        </div>
      )}
    </div>
  );
}

const styles = {
  card: { background: T.panel, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32, boxShadow: "0 20px 60px rgba(0,0,0,0.4)" },
  eyebrow: { fontFamily: T.mono, fontSize: 11, letterSpacing: 2, color: T.cyan, marginBottom: 16 },
  winnerBanner: { display: "flex", alignItems: "center", justifyContent: "space-between", background: "linear-gradient(135deg, rgba(56,189,248,0.12), rgba(167,139,250,0.12))", border: `1px solid rgba(56,189,248,0.3)`, borderRadius: 16, padding: "20px 24px", marginBottom: 28 },
  winnerLabel: { fontFamily: T.mono, fontSize: 11, letterSpacing: 2, color: T.textDim },
  winnerName: { fontSize: 28, fontWeight: 700, marginTop: 4, background: T.gradient, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" },
  trophy: { fontSize: 36 },
  h3: { fontSize: 15, fontWeight: 600, margin: "28px 0 12px", color: T.text },
  table: { borderCollapse: "collapse", width: "100%", fontSize: 12.5, fontFamily: T.mono },
  th: { textAlign: "left", padding: "10px 12px", background: T.panelHi, color: T.textDim, borderBottom: `1px solid ${T.border}`, fontWeight: 600, whiteSpace: "nowrap" },
  td: { padding: "10px 12px", borderBottom: `1px solid ${T.border}`, color: T.textDim, whiteSpace: "nowrap" },
  winRow: { background: "rgba(56,189,248,0.06)" },
  pre: { background: "#0A0F1A", border: `1px solid ${T.border}`, padding: 16, borderRadius: 12, fontSize: 12, fontFamily: T.mono, color: T.textDim, overflowX: "auto", whiteSpace: "pre-wrap", lineHeight: 1.6 },
  muted: { color: T.textFaint, fontSize: 13 },
  tag: { display: "inline-block", fontFamily: T.mono, fontSize: 12, color: T.violet, background: "rgba(167,139,250,0.1)", border: `1px solid rgba(167,139,250,0.3)`, borderRadius: 8, padding: "4px 10px", marginBottom: 12 },
  chart: { maxWidth: "100%", borderRadius: 12, border: `1px solid ${T.border}`, marginBottom: 12, background: "#fff" },
  errText: { color: T.red, fontSize: 13 },
  driftBadge: { display: "inline-flex", alignItems: "center", gap: 8, border: "1px solid", borderRadius: 10, padding: "8px 14px", fontSize: 13, fontFamily: T.mono, textTransform: "capitalize" },
  driftDot: { width: 8, height: 8, borderRadius: "50%" },
  btnRow: { display: "flex", gap: 12 },
  btnGhost: { padding: "12px 20px", borderRadius: 10, border: `1px solid ${T.border}`, background: T.panelHi, color: T.text, cursor: "pointer", fontSize: 14, fontWeight: 500 },
  btnPrimary: { padding: "12px 20px", borderRadius: 10, border: "none", background: T.gradient, color: "#0B1220", fontWeight: 700, fontSize: 14, cursor: "pointer" },
  pdfWrap: { marginTop: 16, border: `1px solid ${T.border}`, borderRadius: 12, overflow: "hidden" },
  pdfFrame: { width: "100%", height: 640, border: "none", background: "#fff" },
};