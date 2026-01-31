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
      
      <div className="header-section">
        <h1>Pneumonia Detection Tool</h1>
        <p>Upload an X-ray for AI-Assisted analysis.</p>
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
        <button disabled={!preview}>Analyse</button>
      </div>

      <div className="result-section">
        <p>
          <b>Prediction:</b> Pneumonia (Demo)
        </p>
        <p>
          <b>Model:</b> Swin Transformer
        </p>
      </div>

    </div>
  );
}

export default App;
