import axios from "axios";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

const api = axios.create({ baseURL: API_BASE });

export const uploadDataset = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return api.post("/upload-dataset", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const preprocessDataset = (jobId, targetCol, testSize = 0.2, nSplits = 5) =>
  api.post("/preprocess", {
    job_id: jobId,
    target_col: targetCol,
    test_size: testSize,
    n_splits: nSplits,
  });

export const trainModels = (jobId, costParams) =>
  api.post("/train", {
    job_id: jobId,
    false_negative_cost: costParams.falseNegativeCost,
    false_positive_cost: costParams.falsePositiveCost,
    min_recall: costParams.minRecall,
    use_smote: costParams.useSmote,
  });

export const getResults = (jobId) => api.get(`/results/${jobId}`);

export const getExplainability = (jobId) => api.get(`/explain/${jobId}`);

export const getDrift = (jobId) => api.get(`/detect-drift/${jobId}`);

export const getReport = (jobId) => api.get(`/report/${jobId}`);

export const downloadReportUrl = (jobId) => `${API_BASE}/download-report/${jobId}`;

export const viewReportUrl = (jobId) => `${API_BASE}/view-report/${jobId}`;

export default api;