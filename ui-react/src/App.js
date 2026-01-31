import React, { useState } from "react";
import "./App.css";

function App() {
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  return (
    <div className="app-container">
      <header className="top-bar">
        <h1>Radiology AI Console</h1>
        <span className="status-dot">● System Ready</span>
      </header>

      <div className="main-grid">

        {/* LEFT PANEL */}
        <div className="panel input-panel">
          <h2>Input X-ray</h2>

          <label className="file-upload">
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <span>Upload Chest X-ray</span>
          </label>

          {preview && (
            <div className="preview-box">
              <img src={preview} alt="X-ray preview" />
            </div>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div className="panel analysis-panel">
          <h2>AI Analysis Summary</h2>

          <div className="metric">
            <span className="metric-label">Prediction</span>
            <span className="metric-value">Normal</span>
          </div>

          <div className="metric">
            <span className="metric-label">Confidence</span>
            <span className="metric-value warning">58%</span>
          </div>

          <div className="alert-box">
            ⚠️ Low confidence detected.  
            Manual review is recommended.
          </div>

          <div className="model-info">
            Model: Swin Transformer  
            <br />
            Mode: Uncertainty-aware screening
          </div>
        </div>

      </div>

      <footer className="footer-note">
        Decision-support only • Not a diagnostic system
      </footer>
    </div>
  );
}

export default App;
