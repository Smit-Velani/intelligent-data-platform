import React, { useEffect, useState } from "react";
import { getExplainability, getDrift, downloadReportUrl } from "./api";

export default function Results({ jobId, trainResult }) {
  const [explain, setExplain] = useState(null);
  const [drift, setDrift] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(true);
  const [loadingDrift, setLoadingDrift] = useState(true);
  const [showPdf, setShowPdf] = useState(false);

  useEffect(() => {
    getExplainability(jobId)
      .then((res) => setExplain(res.data))
      .catch(() => setExplain({ error: true }))
      .finally(() => setLoadingExplain(false));

    getDrift(jobId)
      .then((res) => setDrift(res.data))
      .catch(() => setDrift({ error: true }))
      .finally(() => setLoadingDrift(false));
  }, [jobId]);

  const leaderboard = trainResult.leaderboard || [];
  const columns = leaderboard.length > 0 ? Object.keys(leaderboard[0]) : [];
  const pdfUrl = downloadReportUrl(jobId);

  return (
    <div style={styles.card}>
      <h2>3. Results</h2>

      <p style={styles.winner}>
        🏆 Best model: <strong>{trainResult.best_model}</strong>
      </p>

      <h3>Leaderboard</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={styles.table}>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c} style={styles.th}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((row, i) => (
              <tr key={i} style={row.model === trainResult.best_model ? styles.winnerRow : {}}>
                {columns.map((c) => (
                  <td key={c} style={styles.td}>
                    {typeof row[c] === "number" ? row[c].toLocaleString() : row[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Decision Log</h3>
      <pre style={styles.pre}>{trainResult.decision_log}</pre>

      <h3>Explainability (SHAP)</h3>
      {loadingExplain && <p>Computing SHAP explanations…</p>}
      {explain && !explain.error && (
        <div>
          <p style={styles.meta}>Explainer type: {explain.explainer_type}</p>
          {explain.notes?.map((n, i) => (
            <p key={i} style={styles.note}>ℹ️ {n}</p>
          ))}
          {explain.bar_plot_base64 && (
            <img src={`data:image/png;base64,${explain.bar_plot_base64}`} alt="SHAP feature importance" style={styles.chart} />
          )}
          {explain.calibration_plot_base64 && (
            <img src={`data:image/png;base64,${explain.calibration_plot_base64}`} alt="Calibration curve" style={styles.chart} />
          )}
        </div>
      )}
      {explain?.error && <p style={styles.error}>Could not load explainability report.</p>}

      <h3>Data Drift</h3>
      {loadingDrift && <p>Checking for drift…</p>}
      {drift && !drift.error && (
        <p>
          Status: <strong>{drift.overall_status}</strong> ({drift.drifted_feature_count}/
          {drift.total_feature_count} features drifted)
        </p>
      )}
      {drift?.error && <p style={styles.error}>Could not load drift report.</p>}

      <h3>Full Report</h3>
      <div style={styles.buttonRow}>
        <button style={styles.viewButton} onClick={() => setShowPdf((s) => !s)}>
          {showPdf ? "Hide Report Preview" : "View Report"}
        </button>
        <a href={pdfUrl} target="_blank" rel="noreferrer">
          <button style={styles.button}>Download PDF</button>
        </a>
      </div>

      {showPdf && (
        <div style={styles.pdfWrapper}>
          <iframe src={pdfUrl} title="Analysis Report" style={styles.pdfFrame} />
        </div>
      )}
    </div>
  );
}

const styles = {
  card: { background: "#fff", borderRadius: 8, padding: 24, boxShadow: "0 1px 4px rgba(0,0,0,0.1)", marginBottom: 24 },
  winner: { fontSize: 16 },
  table: { borderCollapse: "collapse", width: "100%", fontSize: 13 },
  th: { textAlign: "left", padding: "6px 10px", background: "#4C72B0", color: "#fff" },
  td: { padding: "6px 10px", borderBottom: "1px solid #eee" },
  winnerRow: { background: "#eef3fb", fontWeight: "bold" },
  pre: { background: "#f7f7f7", padding: 12, borderRadius: 6, fontSize: 12, overflowX: "auto", whiteSpace: "pre-wrap" },
  meta: { color: "#555", fontSize: 14 },
  note: { color: "#8a6d3b", fontSize: 13 },
  chart: { maxWidth: "100%", marginTop: 12, marginBottom: 12, border: "1px solid #eee" },
  buttonRow: { display: "flex", gap: 12, marginTop: 8 },
  button: { padding: "8px 16px", borderRadius: 6, border: "none", background: "#2e7d32", color: "#fff", cursor: "pointer" },
  viewButton: { padding: "8px 16px", borderRadius: 6, border: "1px solid #4C72B0", background: "#fff", color: "#4C72B0", cursor: "pointer" },
  pdfWrapper: { marginTop: 16, border: "1px solid #ccc", borderRadius: 6, overflow: "hidden" },
  pdfFrame: { width: "100%", height: 600, border: "none" },
  error: { color: "#c0392b" },
};