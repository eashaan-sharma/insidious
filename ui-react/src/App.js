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
      setResult(null); // reset previous result
    }
  };

  const handleAnalyse = () => {
    if (!file) {
      alert("Please upload an image first");
      return;
    }

    setLoading(true);
    setResult(null);

    // Simulated backend response
    setTimeout(() => {
      setResult({
        label: "Pneumonia",
        confidence: 0.87,
      });
      setLoading(false);
    }, 1500);
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
          </>
        )}
      </div>
    </div>
  );
}

export default App;
