import React, { useState } from "react";
import Upload from "./Upload";
import Dashboard from "./Dashboard";
import Results from "./Results";
import { T } from "./theme";

const STEPS = [
  { id: 1, label: "Ingest", sub: "Upload & clean" },
  { id: 2, label: "Configure", sub: "Cost & train" },
  { id: 3, label: "Analyze", sub: "Explain & report" },
];

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [preprocessResult, setPreprocessResult] = useState(null);
  const [trainResult, setTrainResult] = useState(null);

  const handlePreprocessed = ({ jobId, preprocessResult }) => {
    setJobId(jobId);
    setPreprocessResult(preprocessResult);
    setTrainResult(null);
  };
  const handleTrained = (result) => setTrainResult(result);
  const handleReset = () => {
    setJobId(null);
    setPreprocessResult(null);
    setTrainResult(null);
  };

  const currentStep = !jobId ? 1 : !trainResult ? 2 : 3;

  return (
    <div style={styles.page}>
      <div style={styles.ambient} />
      <header style={styles.header}>
        <div style={styles.brandRow}>
          <div style={styles.logo}>
            <span style={styles.logoMark}>IDP</span>
          </div>
          <div>
            <h1 style={styles.title}>Intelligent Data Platform</h1>
            <p style={styles.subtitle}>
              Upload a CSV. Get a trained, explained, deployable model back.
            </p>
          </div>
          {jobId && (
            <button style={styles.resetButton} onClick={handleReset}>
              ↺ New analysis
            </button>
          )}
        </div>

        <div style={styles.stepper}>
          {STEPS.map((step, i) => {
            const active = step.id === currentStep;
            const done = step.id < currentStep;
            return (
              <React.Fragment key={step.id}>
                <div style={styles.step}>
                  <div
                    style={{
                      ...styles.stepNum,
                      ...(active ? styles.stepNumActive : {}),
                      ...(done ? styles.stepNumDone : {}),
                    }}
                  >
                    {done ? "✓" : String(step.id).padStart(2, "0")}
                  </div>
                  <div>
                    <div style={{ ...styles.stepLabel, color: active || done ? T.text : T.textFaint }}>
                      {step.label}
                    </div>
                    <div style={styles.stepSub}>{step.sub}</div>
                  </div>
                </div>
                {i < STEPS.length - 1 && (
                  <div style={{ ...styles.stepLine, background: done ? T.gradient : T.border }} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </header>

      <main style={styles.main}>
        {!jobId && <Upload onPreprocessed={handlePreprocessed} />}
        {jobId && preprocessResult && !trainResult && (
          <Dashboard jobId={jobId} preprocessResult={preprocessResult} onTrained={handleTrained} />
        )}
        {jobId && trainResult && <Results jobId={jobId} trainResult={trainResult} />}
      </main>

      <footer style={styles.footer}>
        Built by Smit Velani · FastAPI · scikit-learn · XGBoost · SHAP · Groq LLaMA
      </footer>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: T.bg,
    color: T.text,
    fontFamily: T.sans,
    position: "relative",
    overflow: "hidden",
  },
  ambient: {
    position: "fixed",
    top: "-30%",
    left: "50%",
    transform: "translateX(-50%)",
    width: 900,
    height: 900,
    background: "radial-gradient(circle, rgba(56,189,248,0.10) 0%, rgba(167,139,250,0.06) 40%, transparent 70%)",
    pointerEvents: "none",
    zIndex: 0,
  },
  header: {
    position: "relative",
    zIndex: 1,
    maxWidth: 960,
    margin: "0 auto",
    padding: "40px 20px 0",
  },
  brandRow: { display: "flex", alignItems: "center", gap: 16, marginBottom: 36, flexWrap: "wrap" },
  logo: {
    width: 52, height: 52, borderRadius: 14,
    background: T.gradient,
    display: "flex", alignItems: "center", justifyContent: "center",
    boxShadow: "0 8px 24px rgba(56,189,248,0.35)",
    flexShrink: 0,
  },
  logoMark: { fontFamily: T.mono, fontWeight: 700, fontSize: 15, color: "#0B1220", letterSpacing: 0.5 },
  title: { margin: 0, fontSize: 26, fontWeight: 700, letterSpacing: -0.5 },
  subtitle: { margin: "4px 0 0", color: T.textDim, fontSize: 14 },
  resetButton: {
    marginLeft: "auto",
    padding: "9px 16px",
    borderRadius: 10,
    border: `1px solid ${T.border}`,
    background: T.panel,
    color: T.text,
    cursor: "pointer",
    fontSize: 13,
    fontFamily: T.mono,
  },
  stepper: { display: "flex", alignItems: "center", gap: 8, paddingBottom: 8 },
  step: { display: "flex", alignItems: "center", gap: 12 },
  stepNum: {
    width: 40, height: 40, borderRadius: 12,
    border: `1px solid ${T.border}`,
    background: T.panel,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontFamily: T.mono, fontSize: 14, fontWeight: 600, color: T.textFaint,
    flexShrink: 0,
    transition: "all 0.3s ease",
  },
  stepNumActive: {
    background: T.gradient, color: "#0B1220", borderColor: "transparent",
    boxShadow: "0 6px 18px rgba(56,189,248,0.35)",
  },
  stepNumDone: { borderColor: T.mint, color: T.mint, background: "rgba(52,211,153,0.08)" },
  stepLabel: { fontSize: 14, fontWeight: 600 },
  stepSub: { fontSize: 11, color: T.textFaint, fontFamily: T.mono },
  stepLine: { flex: 1, height: 2, borderRadius: 2, margin: "0 4px" },
  main: { position: "relative", zIndex: 1, maxWidth: 960, margin: "32px auto", padding: "0 20px" },
  footer: {
    position: "relative", zIndex: 1,
    textAlign: "center", padding: "24px", color: T.textFaint,
    fontSize: 12, fontFamily: T.mono,
  },
};