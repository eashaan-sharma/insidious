import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleAnalyse = async () => {
    if (!file) {
      alert("Please upload an image first");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      setResult({
        label:
          data.prediction === 0
            ? "Normal"
            : data.prediction === 1
            ? "Pneumonia"
            : "Other",
        confidence: data.confidence,
        heatmap_path: data.heatmap_path,
        overlay_path:data.overlay_path,
      });
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header-section">
        <h1>Pneumonia Detection Tool</h1>
        <p>Upload an X-ray for AI-assisted analysis.</p>
      </div>

      <div className="upload-section">
        <label className="file-upload">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
          <span>Browse and upload X-ray Image</span>
        </label>
      </div>

      {preview && (
        <div className="preview-section">
          <img src={preview} alt="X-ray preview" />
        </div>
      )}

      <div className="action-section">
        <button onClick={handleAnalyse} disabled={loading}>
          {loading ? "Analysing..." : "Analyse"}
        </button>
      </div>

      <div className="result-section">
        {loading && <p>Processing image...</p>}

        {result && (
          <>
            <p>
              <b>Prediction:</b> {result.label}
            </p>
            <p>
              <b>Confidence:</b> {Math.round(result.confidence * 100)}%
            </p>

            {result.heatmap_path && (
              <div className="heatmap-section">
                <img
                  src={`http://127.0.0.1:8000/${result.overlay_path}`}
                  alt="Grad-CAM Heatmap with overlay"
                />
                <img
                src={`http://127.0.0.1:8000/${result.heatmap_path}`}
                alt="Heatmap"
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
