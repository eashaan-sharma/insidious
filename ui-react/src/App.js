import React from "react";

function App() {
  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>AI-Assisted Pneumonia Detection</h1>

      <p>Upload a chest X-ray image for analysis.</p>

      <input type="file" accept="image/*" />
      <br /><br />

      <button>Predict</button>

      <p style={{ marginTop: "20px" }}>
        <b>Prediction:</b> Pneumonia (Demo)
      </p>

      <p>
        <b>Model:</b> Swin Transformer
      </p>
    </div>
  );
}

export default App;

