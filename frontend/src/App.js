import React, { useState } from "react";
import Upload from "./Upload";
import Dashboard from "./Dashboard";
import Results from "./Results";

export default function App() {
  const [jobId, setJobId] = useState(null);
  const [preprocessResult, setPreprocessResult] = useState(null);
  const [trainResult, setTrainResult] = useState(null);

  const handlePreprocessed = ({ jobId, preprocessResult }) => {
    setJobId(jobId);
    setPreprocessResult(preprocessResult);
    setTrainResult(null);
  };

  const handleTrained = (result) => {
    setTrainResult(result);
  };

  const handleReset = () => {
    setJobId(null);
    setPreprocessResult(null);
    setTrainResult(null);
  };

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>Intelligent Data Platform</h1>
        <p style={styles.subtitle}>
          Upload a CSV. Get a cleaned, explained, and deployed ML model back.
        </p>
        {jobId && (
          <button style={styles.resetButton} onClick={handleReset}>
            Start Over
          </button>
        )}
      </header>

      <main style={styles.main}>
        {!jobId && <Upload onPreprocessed={handlePreprocessed} />}

        {jobId && preprocessResult && !trainResult && (
          <Dashboard
            jobId={jobId}
            preprocessResult={preprocessResult}
            onTrained={handleTrained}
          />
        )}

        {jobId && trainResult && <Results jobId={jobId} trainResult={trainResult} />}
      </main>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#f0f2f5",
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
  },
  header: {
    background: "#1a2733",
    color: "#fff",
    padding: "24px 32px",
    textAlign: "center",
    position: "relative",
  },
  title: { margin: 0, fontSize: 28 },
  subtitle: { margin: "8px 0 0", color: "#b0c4d8" },
  resetButton: {
    position: "absolute",
    top: 24,
    right: 32,
    padding: "6px 14px",
    borderRadius: 6,
    border: "1px solid #fff",
    background: "transparent",
    color: "#fff",
    cursor: "pointer",
  },
  main: { maxWidth: 900, margin: "32px auto", padding: "0 16px" },
};